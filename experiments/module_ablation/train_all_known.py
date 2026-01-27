"""
消融实验: 移除"模拟未知攻击"机制
所有目标工况都视为已知攻击进行训练，不进行 Support/Query 划分
"""
import os
import sys
import yaml
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from copy import deepcopy

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torch.utils.data import ConcatDataset, DataLoader
from src.data.dataloader import NpyDataset
from src.data.dataloader import get_dataloader
from src.models.encoder import MechanicEncoder
from src.models.decoder import MechanicDecoder
from src.models.classifier import MechanicClassifier
from src.utils.metrics import MetricRecorder


# 消融实验输出目录
ABLATION_OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_infinite_loader(loader):
    """将DataLoader转换为无限循环生成器"""
    while True:
        for batch in loader:
            yield batch


def load_data_split(config):
    """
    根据配置文件加载数据:
    1. source_iter:  Teacher 专用 (来自 source_wc / ["WC1"])
    2. target_iters: Student 专用 (来自 target_wcs / ["WC2", "WC3"])
    """
    data_cfg = config['data']
    batch_size = config['training']['batch_size']

    # === 1. 加载 Teacher 专用的源域数据 (Source WC) ===
    source_wcs = data_cfg['source_wc']
    if isinstance(source_wcs, str):
        source_wcs = [source_wcs]

    source_datasets = []
    for wc in source_wcs:
        path = os.path.join(data_cfg['root_dir'], wc, 'train')
        if not os.path.exists(path):
            raise FileNotFoundError(f"源工况路径不存在: {path}")
        source_datasets.append(NpyDataset(path))

    combined_source = ConcatDataset(source_datasets)
    source_loader = DataLoader(combined_source, batch_size=batch_size, shuffle=True, pin_memory=True)
    source_iter = get_infinite_loader(source_loader)

    # === 2. 加载 Student 用的目标工况数据 (Target WCs) ===
    target_iters = {}
    target_wcs = data_cfg['target_wcs']

    for wc in target_wcs:
        path = os.path.join(data_cfg['root_dir'], wc, 'train')
        if not os.path.exists(path):
            print(f"警告: 目标工况路径不存在，跳过: {path}")
            continue

        loader = get_dataloader(path, batch_size, shuffle=True)
        target_iters[wc] = get_infinite_loader(loader)

    return source_iter, target_iters


def load_teacher(config, device):
    """加载冻结的教师模型"""
    cfg = config['model']
    encoder = MechanicEncoder(cfg['input_channels'], cfg['base_filters'], cfg['feature_dim']).to(device)
    decoder = MechanicDecoder(cfg['feature_dim'], cfg['input_channels'], cfg['base_filters']).to(device)
    classifier = MechanicClassifier(cfg['feature_dim'], cfg['num_classes'], cfg['dropout']).to(device)

    source_list = config['data']['source_wc']
    nums = sorted(["".join(filter(str.isdigit, wc)) for wc in source_list], key=int)
    wc_tag = "_".join(nums)
    filename = f"train_{wc_tag}_best_model.pth"
    ckpt_path = os.path.join(
        config['teacher']['checkpoint'],
        config['data']['dataset_name'],
        filename
    )

    if os.path.exists(ckpt_path) and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        classifier.load_state_dict(ckpt['classifier_state_dict'])
        decoder.load_state_dict(ckpt['decoder_state_dict'])
    else:
        raise FileNotFoundError(f"教师模型文件未找到: {ckpt_path}")

    encoder.eval()
    classifier.eval()
    decoder.eval()
    for p in encoder.parameters(): p.requires_grad = False
    for p in decoder.parameters(): p.requires_grad = False

    return encoder, classifier, decoder


def compute_loss(encoder_s, classifier, encoder_t, decoder_t, x_s, y_s, x_t, y_t, config):
    """计算通用损失函数 (L_AC + L_CC + L_LC)"""
    loss_cfg = config['loss']

    # 1. 学生前向传播
    feat_s = encoder_s(x_s)
    logits = classifier(feat_s)

    # 2. 教师前向传播 (作为Ground Truth，不传梯度)
    with torch.no_grad():
        feat_t = encoder_t(x_t).detach()

    # === L_AC: Adversarial/Alignment Consistency ===
    # 对齐"同类数据的特征中心"
    l_ac = torch.tensor(0.0, device=x_s.device)
    valid_classes = 0

    classes_s = torch.unique(y_s)
    classes_t = torch.unique(y_t)
    common_classes = [c for c in classes_s if c in classes_t]

    for c in common_classes:
        proto_s = feat_s[y_s == c].mean(dim=0)
        proto_t = feat_t[y_t == c].mean(dim=0)
        l_ac += nn.MSELoss()(proto_s, proto_t)
        valid_classes += 1

    if valid_classes > 0:
        l_ac = l_ac / valid_classes

    # L_CC: 循环一致性 (特征 -> 教师解码 -> 教师编码)
    x_recon = decoder_t(feat_s, target_length=x_s.shape[-1])
    with torch.no_grad():
        feat_cycle = encoder_t(x_recon).detach()

    if feat_cycle.shape[0] == feat_t.shape[0]:
        l_cc = nn.MSELoss()(feat_cycle, feat_t)
    else:
        l_cc = torch.tensor(0.0, device=feat_cycle.device)

    # L_LC: 标签一致性
    l_lc = nn.CrossEntropyLoss()(logits, y_s)

    total = loss_cfg['lambda_ac'] * l_ac + loss_cfg['lambda_cc'] * l_cc + loss_cfg['lambda_lc'] * l_lc
    return total, {'ac': l_ac.item(), 'cc': l_cc.item(), 'lc': l_lc.item()}


def inner_update(encoder, loss, inner_lr, first_order=True):
    """内循环更新: θ' = θ - α * ∇L"""
    encoder_prime = deepcopy(encoder)

    grads = grad(loss,
                 encoder.parameters(),
                 create_graph=not first_order,
                 retain_graph=True,
                 allow_unused=True)

    for p, g in zip(encoder_prime.parameters(), grads):
        if g is not None:
            p.data = p.data - inner_lr * g

    return encoder_prime


def train_step_all_known(source_iter, target_iters, encoder_s, classifier, encoder_t, decoder_t, config, device):
    """
    消融实验版本: 所有目标工况都视为已知攻击
    不进行 Support/Query 划分，所有 target_wcs 都用于训练

    与原版 meta_train_step 的区别:
    - 原版: 将 target_wcs 划分为 support_wcs (N-1个) 和 query_wc (1个)
           在 support 上做内循环更新，在 query 上验证泛化
    - 消融版: 所有 target_wcs 都作为已知攻击，直接计算损失并更新
    """
    meta_cfg = config['meta']
    wc_list = list(target_iters.keys())

    # === 关键差异: 不再划分 Support/Query，所有工况都是已知的 ===
    random.shuffle(wc_list)

    total_loss = torch.tensor(0.0, device=device)
    loss_count = 0

    # 遍历所有目标工况 (全部视为已知攻击)
    for wc in wc_list:
        # A. Student 获取当前工况数据
        x_s, y_s = next(target_iters[wc])
        x_s, y_s = x_s.to(device), y_s.to(device)

        # B. Teacher 获取源工况数据 (作为对齐的 Anchor)
        x_t, y_t = next(source_iter)
        x_t, y_t = x_t.to(device), y_t.to(device)

        # C. 计算损失
        l_step, _ = compute_loss(
            encoder_s, classifier, encoder_t, decoder_t,
            x_s, y_s, x_t, y_t, config
        )
        total_loss += l_step
        loss_count += 1

    # 计算平均损失
    if loss_count > 0:
        total_loss = total_loss / loss_count

    # === 可选: 仍然使用内循环更新机制 (但没有 Query 验证) ===
    # 这里我们保留内循环更新，但不再有 meta-test 阶段
    # 如果想完全移除元学习机制，可以直接返回 total_loss

    use_inner_update = config.get('ablation', {}).get('use_inner_update', True)

    if use_inner_update and isinstance(total_loss, torch.Tensor) and total_loss.item() != 0:
        # 进行一次内循环更新
        encoder_prime = inner_update(encoder_s, total_loss, meta_cfg['inner_lr'], meta_cfg['first_order'])

        # 用更新后的参数再计算一次损失 (作为最终损失)
        # 随机选一个工况验证
        val_wc = random.choice(wc_list)
        x_val_s, y_val_s = next(target_iters[val_wc])
        x_val_s, y_val_s = x_val_s.to(device), y_val_s.to(device)

        x_val_t, y_val_t = next(source_iter)
        x_val_t, y_val_t = x_val_t.to(device), y_val_t.to(device)

        l_val, _ = compute_loss(
            encoder_prime, classifier, encoder_t, decoder_t,
            x_val_s, y_val_s, x_val_t, y_val_t, config
        )

        # 组合损失 (类似原版，但这里不区分 support/query)
        final_loss = total_loss + meta_cfg['beta'] * l_val
        return final_loss, total_loss.item(), l_val.item()
    else:
        # 不使用内循环更新，直接返回总损失
        return total_loss, total_loss.item() if isinstance(total_loss, torch.Tensor) else 0, 0.0


def evaluate(encoder, classifier, config, device, recorder=None):
    """在测试集上评估泛化能力"""
    data_cfg = config['data']
    batch_size = config['training']['batch_size']

    results = {}
    total_acc = 0

    encoder.eval()
    classifier.eval()

    if recorder is not None:
        recorder.reset()

    for wc in data_cfg['test_wcs']:
        path = os.path.join(data_cfg['root_dir'], wc, 'test')
        loader = get_dataloader(path, batch_size, shuffle=False)

        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                feat = encoder(x)
                pred = classifier(feat).argmax(1)

                if recorder is not None:
                    recorder.update(wc, pred, y)

                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = 100. * correct / total
        results[wc] = acc
        total_acc += acc
        print(f"  工况 {wc}: {acc:.2f}%")

    avg_acc = total_acc / len(data_cfg['test_wcs'])
    print(f"  平均准确率: {avg_acc:.2f}%")
    return avg_acc


def main(config_path):
    config = load_config(config_path)
    set_seed(config['seed'])

    device = torch.device(f"cuda:{config['device']['gpu_id']}"
                          if config['device']['use_cuda'] and torch.cuda.is_available()
                          else "cpu")
    print(f"[消融实验] 训练设备: {device}")
    print(f"[消融实验] 模式: 所有目标工况视为已知攻击 (无 Support/Query 划分)")

    # === 消融实验输出目录 ===
    ablation_ckpt_dir = os.path.join(ABLATION_OUTPUT_DIR, "ckpts")
    ablation_log_dir = os.path.join(ABLATION_OUTPUT_DIR, "log")
    os.makedirs(ablation_ckpt_dir, exist_ok=True)
    os.makedirs(ablation_log_dir, exist_ok=True)

    # 模型初始化
    encoder_t, classifier_t, decoder_t = load_teacher(config, device)

    encoder_s = MechanicEncoder(
        input_channels=config['model']['input_channels'],
        base_filters=config['model']['base_filters'],
        output_feature_dim=config['model']['feature_dim']
    ).to(device)
    encoder_s.load_state_dict(encoder_t.state_dict())

    classifier = classifier_t
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    # 数据加载
    source_iter, target_iters = load_data_split(config)
    print(f"数据加载完成。Target: {len(target_iters)}个工况 (全部视为已知)")

    # 修改配置，指向消融实验目录
    ablation_config = config.copy()
    ablation_config['output'] = {'save_dir': ablation_ckpt_dir}

    metric_recorder = MetricRecorder(
        save_dir=ablation_log_dir,
        config=ablation_config,
        class_names=[str(i) for i in range(config['model']['num_classes'])],
    )
    metric_recorder.save_config(ablation_config)

    # 优化器
    params = encoder_s.parameters()
    optimizer = optim.Adam(params, lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])

    iterations_per_epoch = config['training'].get('iterations_per_epoch', 100)
    best_acc = 0

    print("开始消融实验训练 (All Known)...")

    for epoch in range(1, config['training']['epochs'] + 1):
        encoder_s.train()

        total_loss, total_train, total_val = 0, 0, 0

        for _ in range(iterations_per_epoch):
            optimizer.zero_grad()

            loss, l_train, l_val = train_step_all_known(
                source_iter,
                target_iters,
                encoder_s,
                classifier,
                encoder_t,
                decoder_t,
                config,
                device
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_train += l_train
            total_val += l_val

        print(f"Epoch {epoch}: Loss={total_loss/iterations_per_epoch:.4f}, "
              f"Train={total_train/iterations_per_epoch:.4f}, "
              f"Val={total_val/iterations_per_epoch:.4f}")

        if epoch % 1 == 0:
            print(f"Epoch {epoch} 评估:")
            avg_acc = evaluate(encoder_s, classifier, config, device, metric_recorder)

            if avg_acc > best_acc:
                best_acc = avg_acc

                # 生成文件名
                src_list = config['data']['source_wc']
                src_nums = sorted(["".join(filter(str.isdigit, x)) for x in src_list], key=int)
                src_tag = "_".join(src_nums)

                tgt_list = config['data']['target_wcs']
                tgt_nums = sorted(["".join(filter(str.isdigit, x)) for x in tgt_list], key=int)
                tgt_tag = "_".join(tgt_nums)

                # 保存到消融实验目录
                file_name = f"ablation_all_known_train_{src_tag}_target_{tgt_tag}_best.pth"
                save_path = os.path.join(ablation_ckpt_dir, config['data']['dataset_name'], file_name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                torch.save({
                    'encoder_state_dict': encoder_s.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'best_acc': best_acc,
                    'ablation_type': 'all_known'
                }, save_path)

                print(f"[消融] 保存最佳模型: {save_path} (Acc: {best_acc:.2f}%)")
                metric_recorder.calculate_and_save(epoch)

    print(f"\n[消融实验] 训练结束，最佳平均准确率: {best_acc:.2f}%")

    # 保存最终结果汇总
    summary_path = os.path.join(ablation_log_dir, config['data']['dataset_name'], "summary.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'a', encoding='utf-8') as f:
        src_tag = "_".join(sorted(["".join(filter(str.isdigit, x)) for x in config['data']['source_wc']], key=int))
        tgt_tag = "_".join(sorted(["".join(filter(str.isdigit, x)) for x in config['data']['target_wcs']], key=int))
        f.write(f"train_{src_tag}_target_{tgt_tag}: Best Acc = {best_acc:.2f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="消融实验: 所有目标工况视为已知攻击")
    parser.add_argument("--config", default="configs/mcid.yaml", help="配置文件路径")
    args = parser.parse_args()
    main(args.config)
