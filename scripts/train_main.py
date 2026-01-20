import os
from pyexpat import model
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
from itertools import cycle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import ConcatDataset, DataLoader
from src.data.dataloader import NpyDataset
from src.data.dataloader import get_dataloader
from src.models.encoder import MechanicEncoder
from src.models.decoder import MechanicDecoder
from src.models.classifier import MechanicClassifier
from src.utils.metrics import MetricRecorder


# torch.backends.cudnn.benchmark = False

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data_split(config):
    """
    根据配置文件加载数据:
    1. source_iter:  Teacher 专用 (来自 source_wc / ["WC1"])
    2. target_iters: Student 专用 (来自 target_wcs / ["WC2", "WC3"])
    """
    data_cfg = config['data']
    batch_size = config['training']['batch_size']

    # === 1. 加载 Teacher 专用的源域数据 (Source WC) ===
    source_wcs = data_cfg['source_wc']  # e.g., ["WC1"]
    if isinstance(source_wcs, str):
        source_wcs = [source_wcs]
    # print(f"[Data] Teacher 加载源工况: {source_wcs}")

    # source_path = os.path.join(data_cfg['root_dir'], "_".join(source_wcs), 'train')
    # # 检查路径是否存在
    # if not os.path.exists(source_path):
    #     raise FileNotFoundError(f"源工况路径不存在: {source_path}")

    # source_loader = get_dataloader(source_path, batch_size, shuffle=True)
    # source_iter = get_infinite_loader(source_loader)

    source_datasets = []
    for wc in source_wcs:
        path = os.path.join(data_cfg['root_dir'], wc, 'train')
        if not os.path.exists(path):
            raise FileNotFoundError(f"源工况路径不存在: {path}")
        source_datasets.append(NpyDataset(path))

    # 合并多个数据集并转换为无限迭代器
    combined_source = ConcatDataset(source_datasets)
    source_loader = DataLoader(combined_source, batch_size=batch_size, shuffle=True, pin_memory=True)
    source_iter = get_infinite_loader(source_loader)

    # === 2. 加载 Student 用的目标工况数据 (Target WCs / Attacker Pool) ===
    target_iters = {}
    target_wcs = data_cfg['target_wcs'] # e.g., ["WC2", "WC3"]
    # print(f"[Data] Student 加载目标工况 (作为攻击): {target_wcs}")

    for wc in target_wcs:
        path = os.path.join(data_cfg['root_dir'], wc, 'train')
        if not os.path.exists(path):
            print(f"警告: 目标工况路径不存在，跳过: {path}")
            continue

        loader = get_dataloader(path, batch_size, shuffle=True)
        target_iters[wc] = get_infinite_loader(loader)

    if len(target_iters) < 2:
        print("警告: target_wcs 数量少于2，元学习无法进行 Support/Query 划分！")

    return source_iter, target_iters


def load_teacher(config, device):
    """加载冻结的教师模型"""
    cfg = config['model']
    encoder = MechanicEncoder(cfg['input_channels'], cfg['base_filters'], cfg['feature_dim']).to(device)
    decoder = MechanicDecoder(cfg['feature_dim'], cfg['input_channels'], cfg['base_filters']).to(device)
    classifier = MechanicClassifier(cfg['feature_dim'], cfg['num_classes'], cfg['dropout']).to(device)

    # # 确保加载具体的pth文件
    # ckpt_path = os.path.join(config['teacher']['checkpoint'], config['data']['dataset_name'], 'best_model.pth')

    source_list = config['data']['source_wc']
    nums = sorted(["".join(filter(str.isdigit, wc)) for wc in source_list], key=int)
    wc_tag = "_".join(nums)
    filename = f"train_{wc_tag}_best_model.pth"
    ckpt_path = os.path.join(
        config['teacher']['checkpoint'],
        config['data']['dataset_name'],
        filename
    )

    # print(f"正在加载教师模型: {ckpt_path}")

    if os.path.exists(ckpt_path) and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        classifier.load_state_dict(ckpt['classifier_state_dict'])
        decoder.load_state_dict(ckpt['decoder_state_dict'])
        # print(f"教师模型已加载: {ckpt_path}")
    else:
        raise FileNotFoundError(f"教师模型文件未找到: {ckpt_path}")

    encoder.eval()
    classifier.eval()
    decoder.eval()
    for p in encoder.parameters(): p.requires_grad = False
    for p in decoder.parameters(): p.requires_grad = False

    return encoder, classifier, decoder


def build_student(config, device):
    """构建学生模型"""
    cfg = config['model']
    encoder = MechanicEncoder(cfg['input_channels'], cfg['base_filters'], cfg['feature_dim']).to(device)
    # classifier = MechanicClassifier(cfg['feature_dim'], cfg['num_classes'], cfg['dropout']).to(device)
    # return encoder, classifier
    return encoder


def get_infinite_loader(loader):
    """将DataLoader转换为无限循环生成器"""
    while True:
        for batch in loader:
            yield batch


def load_all_wc_data(config):
    """加载所有目标工况数据，返回无限迭代器"""
    data_cfg = config['data']
    batch_size = config['training']['batch_size']

    wc_iters = {}
    for wc in data_cfg['target_wcs']:
        path = os.path.join(data_cfg['root_dir'], wc, 'train')
        loader = get_dataloader(path, batch_size, shuffle=True)
        wc_iters[wc] = get_infinite_loader(loader)

    return wc_iters


# def compute_loss(encoder_s, classifier, encoder_t, decoder_t, x, labels, config):
def compute_loss(encoder_s, classifier, encoder_t, decoder_t, x_s,y_s,x_t,y_t, config):
    """计算通用损失函数 (L_AC + L_CC + L_LC)"""
    loss_cfg = config['loss']

    # 1. 学生前向传播
    feat_s = encoder_s(x_s)
    logits = classifier(feat_s)

    # 2. 教师前向传播 (作为Ground Truth，不传梯度)
    with torch.no_grad():
        feat_t = encoder_t(x_t).detach()

    # # L_AC: 对抗/特征一致性
    # l_ac = nn.MSELoss()(feat_s, feat_t)
        # === 2. L_AC: Adversarial/Alignment Consistency (核心修改) ===
    # 既然数据不成对，我们对齐"同类数据的特征中心"
    l_ac = torch.tensor(0.0, device=x_s.device)
    valid_classes = 0

    # 获取当前 Batch 中两边都存在的类别
    classes_s = torch.unique(y_s)
    classes_t = torch.unique(y_t)
    # 取交集，只对齐两边都有的类别
    common_classes = [c for c in classes_s if c in classes_t]

    for c in common_classes:
        # 计算 Student 在该类别下的特征均值 (Prototype)
        proto_s = feat_s[y_s == c].mean(dim=0)
        # 计算 Teacher 在该类别下的特征均值
        proto_t = feat_t[y_t == c].mean(dim=0)

        # 最小化两者的距离
        l_ac += nn.MSELoss()(proto_s, proto_t)
        valid_classes += 1

    # 如果当前 Batch 没有任何重叠类别 (概率很小)，则 loss 为 0
    if valid_classes > 0:
        l_ac = l_ac / valid_classes


    # L_CC: 循环一致性 (特征 -> 教师解码 -> 教师编码)
    x_recon = decoder_t(feat_s, target_length=x_s.shape[-1])
    with torch.no_grad():
        feat_cycle = encoder_t(x_recon).detach()

    # print(f"feat_s shape: {feat_s.shape}, feat_t shape: {feat_t.shape}, feat_cycle shape: {feat_cycle.shape}")
    # l_cc = nn.MSELoss()(feat_cycle, feat_t)

    if feat_cycle.shape[0] == feat_t.shape[0]:
        l_cc = nn.MSELoss()(feat_cycle, feat_t)
    else:
        # 形状对不上（比如最后一次batch），直接 loss=0，跳过计算避免报错
        l_cc = torch.tensor(0.0, device=feat_cycle.device)

    # L_LC: 标签一致性
    l_lc = nn.CrossEntropyLoss()(logits, y_s)

    total = loss_cfg['lambda_ac'] * l_ac + loss_cfg['lambda_cc'] * l_cc + loss_cfg['lambda_lc'] * l_lc
    return total, {'ac': l_ac.item(), 'cc': l_cc.item(), 'lc': l_lc.item()}


def inner_update(encoder, loss, inner_lr, first_order=True):
    """内循环更新: θ' = θ - α * ∇L_support"""
    encoder_prime = deepcopy(encoder)

    # 计算梯度 (retain_graph=True 确保不释放计算图，以便后续计算Meta Loss)
    grads = grad(loss,
                 encoder.parameters(),
                 create_graph=not first_order,
                 retain_graph=True,
                 allow_unused=True)

    # 更新临时参数
    for p, g in zip(encoder_prime.parameters(), grads):
        if g is not None:
            p.data = p.data - inner_lr * g

    return encoder_prime


# def meta_train_step(wc_iters, encoder_s, classifier, encoder_t, decoder_t, config, device):
#     """
#     标准的元学习步骤 (DG-Meta / MLDG 风格):
#     1. Task Sampling: 将工况划分为 Meta-Train (Support) 和 Meta-Test (Query)
#     2. Inner Loop: 在 Support Set 上计算损失并获得临时参数 θ'
#     3. Outer Loop: 在 Query Set 上使用 θ' 计算损失，并结合 Support Loss 进行最终更新
#     """
#     meta_cfg = config['meta']
#     wc_list = list(wc_iters.keys())

#     # 随机划分工况任务
#     random.shuffle(wc_list)
#     # N-1个工况用于内循环更新 (模拟已知工况)
#     support_wcs = wc_list[:-1]
#     # 1个工况用于外循环测试 (模拟未知工况)
#     query_wc = wc_list[-1]

#     # ========== 1. Meta-Train / Support Set 阶段 ==========
#     # 从支持集工况中随机抽取一个 Batch
#     train_wc = random.choice(support_wcs)
#     x_sup, y_sup = next(wc_iters[train_wc])
#     x_sup, y_sup = x_sup.to(device), y_sup.to(device)

#     # 计算 Support Loss
#     l_sup, _ = compute_loss(encoder_s, classifier, encoder_t, decoder_t, x_sup, y_sup, config)

#     # 获取临时参数 θ' (Fast Weights)
#     encoder_prime = inner_update(encoder_s, l_sup, meta_cfg['inner_lr'], meta_cfg['first_order'])

#     # ========== 2. Meta-Test / Query Set 阶段 ==========
#     # 从查询集工况中抽取一个 Batch
#     x_qry, y_qry = next(wc_iters[query_wc])
#     x_qry, y_qry = x_qry.to(device), y_qry.to(device)

#     # 使用临时参数 θ' 计算 Query Loss
#     # 注意：这里使用 encoder_prime (θ')，但分类器 classifier 共享 (或视具体算法而定)
#     l_qry, metrics = compute_loss(encoder_prime, classifier, encoder_t, decoder_t, x_qry, y_qry, config)

#     # ========== 3. 最终 Meta Loss ==========
#     # MLDG 常用: Total Loss = L_support + beta * L_query
#     l_total = l_sup + meta_cfg['beta'] * l_qry

#     return l_total, l_sup.item(), l_qry.item()



def meta_train_step(source_iter, target_iters, encoder_s, classifier, encoder_t, decoder_t, config, device):
    """
    source_iter:  对应 config['source_wc'] (WC1)
    target_iters: 对应 config['target_wcs'] (WC2, WC3)
    """
    meta_cfg = config['meta']

    # 获取所有目标工况名称
    wc_list = list(target_iters.keys())

    # === 1. 任务划分 (Task Sampling) ===
    # 必须至少有2个工况才能做元学习划分 (Unknown vs Known)
    # 如果只有1个工况，代码会报错，建议在 load_data_split 里检查
    random.shuffle(wc_list)

    query_wc = wc_list[-1]       # 模拟未知攻击 (Unknown Attack)
    support_wcs = wc_list[:-1]   # 模拟已知攻击 (Known Attacks)

    # === 2. Meta-Train / Support Set 阶段 ===
    total_sup_loss = 0

    # 遍历所有已知攻击工况
    for wc in support_wcs:
        # A. Student 获取当前攻击工况数据 (如 WC2)
        x_s, y_s = next(target_iters[wc])
        x_s, y_s = x_s.to(device), y_s.to(device)

        # B. Teacher 获取源工况数据 (WC1) - 作为对齐的 Anchor
        x_t, y_t = next(source_iter)
        x_t, y_t = x_t.to(device), y_t.to(device)

        # C. 计算损失 (使用基于类别的对齐)
        l_step, _ = compute_loss(
            encoder_s, classifier, encoder_t, decoder_t,
            x_s, y_s, x_t, y_t, config
        )
        total_sup_loss += l_step

    # 计算平均 Support Loss
    if len(support_wcs) > 0:
        total_sup_loss = total_sup_loss / len(support_wcs)
    else:
        # 极端情况：如果 target_wcs 只有1个，无法划分，这里会是0
        # 为了代码不崩，直接用 query 做 support (退化为普通训练)
        total_sup_loss = torch.tensor(0.0, device=device)

    # === 3. Inner Loop Update (获取临时参数 θ') ===
    # 注意：如果 total_sup_loss 是 0 (例如 target_wcs 只有1个)，encoder_prime 就是 encoder_s
    if isinstance(total_sup_loss, torch.Tensor) and total_sup_loss.item() != 0:
        encoder_prime = inner_update(encoder_s, total_sup_loss, meta_cfg['inner_lr'], meta_cfg['first_order'])
    else:
        encoder_prime = encoder_s # 不更新

    # === 4. Meta-Test / Query Set 阶段 ===
    # A. Student 获取未知攻击工况数据 (如 WC3)
    x_q_s, y_q_s = next(target_iters[query_wc])
    x_q_s, y_q_s = x_q_s.to(device), y_q_s.to(device)

    # B. Teacher 依然获取源工况数据 (WC1)
    x_q_t, y_q_t = next(source_iter)
    x_q_t, y_q_t = x_q_t.to(device), y_q_t.to(device)

    # 计算 Query Loss (验证泛化能力)
    l_qry, metrics = compute_loss(
        encoder_prime, classifier, encoder_t, decoder_t,
        x_q_s, y_q_s, x_q_t, y_q_t, config
    )

    # === 5. 总损失组合 ===
    # 最终更新使用: L_support + beta * L_query
    l_total = total_sup_loss + meta_cfg['beta'] * l_qry

    return l_total, total_sup_loss.item() if isinstance(total_sup_loss, torch.Tensor) else 0, l_qry.item()


def evaluate(encoder, classifier, config, device,recorder=None):
    """在测试集上评估泛化能力"""
    data_cfg = config['data']
    batch_size = config['training']['batch_size']

    results = {}
    total_acc = 0

    encoder.eval()
    classifier.eval()

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
    print(f"训练设备: {device}")

    os.makedirs(config['output']['save_dir'], exist_ok=True)

    # 模型初始化
    encoder_t, classifier_t, decoder_t = load_teacher(config, device)
    # encoder_s, classifier = build_student(config, device)

    encoder_s = MechanicEncoder(
        input_channels=config['model']['input_channels'],
        base_filters=config['model']['base_filters'],
        output_feature_dim=config['model']['feature_dim']
    ).to(device)
    encoder_s.load_state_dict(encoder_t.state_dict())
# 分类器直接用 teacher 的，或者把 teacher 的分类器提取出来
    classifier = classifier_t
    classifier.eval() # 必须 Eval 模式
    for p in classifier.parameters():
        p.requires_grad = False # 必须冻结


    # 数据加载 (使用无限迭代器，摒弃 source_loader)
    # wc_iters = load_all_wc_data(config)


    source_iter, target_iters = load_data_split(config)

    # 打印一下确认数据加载成功
    print(f"数据加载完成。Source: Target: {len(target_iters)}个")


    metric_recorder = MetricRecorder(
        save_dir="log",
        # experiment_name=f"{config['data']['dataset_name']}",
        config=config,
        class_names=[str(i) for i in range(config['model']['num_classes'])],
    )
    metric_recorder.save_config(config)

    # 优化器
    # params = list(encoder_s.parameters()) + list(classifier.parameters())
    params = encoder_s.parameters()  # 只优化编码器参数
    optimizer = optim.Adam(params, lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])

    # 训练参数
    iterations_per_epoch = config['training'].get('iterations_per_epoch', 100) # 每 Epoch 迭代次数
    best_acc = 0

    print("开始 MCID 元学习训练...")

    for epoch in range(1, config['training']['epochs'] + 1):
        encoder_s.train()
        # classifier.train()

        total_loss, total_sup, total_qry = 0, 0, 0

        for _ in range(iterations_per_epoch):
            optimizer.zero_grad()

            # 执行一步元训练
            # loss, l_sup, l_qry = meta_train_step(
            #     wc_iters,
            #     encoder_s, classifier, encoder_t, decoder_t, config, device
            # )

            # 可能少传了 device，或者因为换行导致 device 被漏掉了

            # loss.backward()
            # optimizer.step()

            # total_loss += loss.item()

            loss, l_sup, l_qry = meta_train_step(
                source_iter,    # 1. 对应定义中的 source_iter
                target_iters,   # 2. 对应定义中的 target_iters
                encoder_s,      # 3. 对应 encoder_s
                classifier,     # 4. 对应 classifier
                encoder_t,      # 5. 对应 encoder_t
                decoder_t,      # 6. 对应 decoder_t
                config,         # 7. 对应 config
                device          # 8. <--- 之前报错就是缺了这个
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_sup += l_sup
            total_qry += l_qry

        # 日志
        print(f"Epoch {epoch}: Loss={total_loss/iterations_per_epoch:.4f}, "
              f"Support={total_sup/iterations_per_epoch:.4f}, "
              f"Query={total_qry/iterations_per_epoch:.4f}")

        # 评估与保存
        # if epoch % 5 == 0:
        #     print(f"Epoch {epoch} 评估:")
        #     avg_acc = evaluate(encoder_s, classifier, config, device, metric_recorder)

        #     if avg_acc > best_acc:
        #         best_acc = avg_acc
        #         save_path = os.path.join(config['output']['save_dir'],
        #                                f"{config['data']['dataset_name']}_mcid_best.pth")
        #         torch.save({
        #             'encoder_state_dict': encoder_s.state_dict(),
        #             'classifier_state_dict': classifier.state_dict(),
        #             'best_acc': best_acc
        #         }, save_path)
        #         print(f"保存最佳模型: {save_path} (Acc: {best_acc:.2f}%)")

        #         metric_recorder.calculate_and_save(epoch)

        # if epoch% 1==5:
        if epoch % 1 == 0:
            print(f"Epoch {epoch} 评估:")
            avg_acc = evaluate(encoder_s, classifier, config, device, metric_recorder)

            if avg_acc > best_acc:
                best_acc = avg_acc

                # --- 核心修改：动态生成包含 source 和 target 编号的文件名 ---
                # 1. 处理 Source 编号 (例如 ["WC1"] -> "1")
                src_list = config['data']['source_wc']
                src_nums = sorted(["".join(filter(str.isdigit, x)) for x in src_list], key=int)
                src_tag = "_".join(src_nums)

                # 2. 处理 Target 编号 (例如 ["WC2", "WC3"] -> "2_3")
                tgt_list = config['data']['target_wcs']
                tgt_nums = sorted(["".join(filter(str.isdigit, x)) for x in tgt_list], key=int)
                tgt_tag = "_".join(tgt_nums)

                # 3. 组合最终路径
                file_name = f"mcid_train_{src_tag}_meta_{tgt_tag}_best.pth"
                save_path = os.path.join(config['output']['save_dir'],config['data']['dataset_name'], file_name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'encoder_state_dict': encoder_s.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'best_acc': best_acc
                }, save_path)

                print(f"保存最佳模型: {save_path} (Acc: {best_acc:.2f}%)")
                metric_recorder.calculate_and_save(epoch)

    print(f"\n训练结束，最佳平均准确率: {best_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mcid.yaml")
    args = parser.parse_args()
    main(args.config)
