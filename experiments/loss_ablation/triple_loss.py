import os
import sys
import yaml
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from copy import deepcopy
from tqdm import tqdm

# === 路径修正：确保能导入 src ===
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir) # 假设脚本在 scripts/ 下，根目录在上级
sys.path.append(root_dir)

# 导入你的模型和数据定义
from src.data.dataloader import get_dataloader, NpyDataset
from torch.utils.data import ConcatDataset, DataLoader
from src.models.encoder import MechanicEncoder
from src.models.decoder import MechanicDecoder
from src.models.classifier import MechanicClassifier

# ================= 配置区域 =================

# 1. 基础配置文件 (只读，用于获取数据路径)
CONFIG_PATH = "configs/mcid_PU_train_1_meta_2_4.yaml" # <--- 改成你实际的配置文件路径

# 2. 结果保存路径 (只输出 CSV)
RESULT_CSV = "experiments/loss_results.csv"

# 3. 实验设置
TOTAL_SUM = 1.5   # 损失系数之和
STEP_SIZE = 0.25  # 步长
EPOCHS = 10       # 每组参数跑多少轮 (消融实验不需要跑太久，能看出趋势即可)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===========================================

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# === 1. 你的原始 Loss 逻辑 (为了独立性，这里复制一份并修改为接受动态系数) ===

def compute_loss_dynamic(encoder_s, classifier, encoder_t, decoder_t, x_s, y_s, x_t, y_t, lambdas):
    # 1. Forward
    feat_s = encoder_s(x_s)
    logits = classifier(feat_s)

    with torch.no_grad():
        feat_t = encoder_t(x_t).detach()

    # 2. L_AC (Prototype Alignment)
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

    # 3. L_CC (Cyclic Consistency)
    x_recon = decoder_t(feat_s, target_length=x_s.shape[-1])
    with torch.no_grad():
        feat_cycle = encoder_t(x_recon).detach()

    # 【严格保留你的逻辑】：比较 feat_cycle 和 feat_t
    if feat_cycle.shape[0] == feat_t.shape[0]:
        l_cc = nn.MSELoss()(feat_cycle, feat_t)
    else:
        l_cc = torch.tensor(0.0, device=feat_cycle.device)

    # 4. L_LC
    l_lc = nn.CrossEntropyLoss()(logits, y_s)

    # 5. 动态加权
    total = lambdas['ac'] * l_ac + lambdas['cc'] * l_cc + lambdas['lc'] * l_lc
    return total

def inner_update(encoder, loss, inner_lr, first_order=True):
    encoder_prime = deepcopy(encoder)
    grads = grad(loss, encoder.parameters(), create_graph=not first_order, retain_graph=True, allow_unused=True)
    for p, g in zip(encoder_prime.parameters(), grads):
        if g is not None:
            p.data = p.data - inner_lr * g
    return encoder_prime

# === 2. 模型与数据准备 (只读模式) ===

def setup_env(config):
    # 1. 准备数据
    data_cfg = config['data']
    batch_size = config['training']['batch_size']

    # Source
    src_list = data_cfg['source_wc']
    if isinstance(src_list, str): src_list = [src_list]
    src_datasets = [NpyDataset(os.path.join(data_cfg['root_dir'], wc, 'train')) for wc in src_list]
    src_loader = DataLoader(ConcatDataset(src_datasets), batch_size=batch_size, shuffle=True)

    # Target
    tgt_wcs = data_cfg['target_wcs']
    tgt_loaders = {}
    for wc in tgt_wcs:
        path = os.path.join(data_cfg['root_dir'], wc, 'train')
        tgt_loaders[wc] = get_dataloader(path, batch_size, shuffle=True)

    # Test
    test_loaders = {}
    for wc in data_cfg['test_wcs']:
        path = os.path.join(data_cfg['root_dir'], wc, 'test')
        test_loaders[wc] = get_dataloader(path, batch_size, shuffle=False)

    # 2. 准备 Teacher 模型 (只加载一次)
    cfg_m = config['model']
    enc_t = MechanicEncoder(cfg_m['input_channels'], cfg_m['base_filters'], cfg_m['feature_dim']).to(DEVICE)
    dec_t = MechanicDecoder(cfg_m['feature_dim'], cfg_m['input_channels'], cfg_m['base_filters']).to(DEVICE)
    cls_t = MechanicClassifier(cfg_m['feature_dim'], cfg_m['num_classes'], cfg_m['dropout']).to(DEVICE)

    # 推导路径
    src_nums = sorted(["".join(filter(str.isdigit, wc)) for wc in src_list], key=int)
    wc_tag = "_".join(src_nums)
    ckpt_path = os.path.join(config['teacher']['checkpoint'], config['data']['dataset_name'], f"train_{wc_tag}_best_model.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"教师模型未找到: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    enc_t.load_state_dict(ckpt['encoder_state_dict'])
    cls_t.load_state_dict(ckpt['classifier_state_dict'])
    if 'decoder_state_dict' in ckpt:
        dec_t.load_state_dict(ckpt['decoder_state_dict'])

    # 冻结
    for m in [enc_t, dec_t, cls_t]:
        m.eval()
        for p in m.parameters(): p.requires_grad = False

    return enc_t, dec_t, cls_t, src_loader, tgt_loaders, test_loaders

def get_infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

# === 3. 核心消融循环 ===

def run_ablation_training(config, lambdas, models, data_loaders):
    """
    独立运行一次训练过程
    models: (enc_t, dec_t, cls_t)
    """
    enc_t, dec_t, cls_t = models
    src_loader, tgt_loaders, test_loaders = data_loaders

    # 1. 初始化学生 (每次都从 Teacher 继承)
    cfg_m = config['model']
    enc_s = MechanicEncoder(cfg_m['input_channels'], cfg_m['base_filters'], cfg_m['feature_dim']).to(DEVICE)
    enc_s.load_state_dict(enc_t.state_dict()) # Reset weights

    optimizer = optim.Adam(enc_s.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])

    # 2. 准备迭代器
    src_iter = get_infinite_loader(src_loader)
    tgt_iters = {k: get_infinite_loader(v) for k, v in tgt_loaders.items()}
    tgt_keys = list(tgt_iters.keys())

    iters_per_epoch = 50 # 固定步数，加快实验
    best_acc = 0.0

    # 3. 训练循环
    # 为了不刷屏，把这一层的 tqdm 关掉或者简化
    for epoch in range(EPOCHS):
        enc_s.train()
        for _ in range(iters_per_epoch):
            random.shuffle(tgt_keys)
            query_wc = tgt_keys[-1]
            support_wcs = tgt_keys[:-1]

            optimizer.zero_grad()
            total_sup_loss = 0

            # Meta-Train
            for wc in support_wcs:
                x_s, y_s = next(tgt_iters[wc])
                x_t, y_t = next(src_iter)
                loss = compute_loss_dynamic(enc_s, cls_t, enc_t, dec_t,
                                          x_s.to(DEVICE), y_s.to(DEVICE), x_t.to(DEVICE), y_t.to(DEVICE),
                                          lambdas)
                total_sup_loss += loss

            if len(support_wcs) > 0:
                total_sup_loss /= len(support_wcs)
                enc_prime = inner_update(enc_s, total_sup_loss, config['meta']['inner_lr'])
            else:
                enc_prime = enc_s
                total_sup_loss = torch.tensor(0.0).to(DEVICE)

            # Meta-Test
            x_qs, y_qs = next(tgt_iters[query_wc])
            x_qt, y_qt = next(src_iter)
            l_qry = compute_loss_dynamic(enc_prime, cls_t, enc_t, dec_t,
                                       x_qs.to(DEVICE), y_qs.to(DEVICE), x_qt.to(DEVICE), y_qt.to(DEVICE),
                                       lambdas)

            loss_final = total_sup_loss + config['meta']['beta'] * l_qry
            loss_final.backward()
            optimizer.step()

        # 4. 验证 (每2轮测一次)
        if (epoch + 1) % 2 == 0:
            enc_s.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for loader in test_loaders.values():
                    for x, y in loader:
                        pred = cls_t(enc_s(x.to(DEVICE))).argmax(1)
                        correct += (pred == y.to(DEVICE)).sum().item()
                        total += y.size(0)
            acc = 100 * correct / total
            if acc > best_acc: best_acc = acc

    return best_acc

# === 主程序 ===

def main():
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: 配置文件 {CONFIG_PATH} 不存在")
        return

    # 1. 生成组合
    combos = []
    for ac in np.arange(0, TOTAL_SUM + STEP_SIZE/1000, STEP_SIZE):
        remaining = TOTAL_SUM - ac
        for cc in np.arange(0, remaining + STEP_SIZE/1000, STEP_SIZE):
            lc = remaining - cc
            if lc < 0: continue
            combos.append({'ac': round(ac, 2), 'cc': round(cc, 2), 'lc': round(lc, 2)})

    combos.sort(key=lambda x: x['lc'], reverse=True)

    print(f"=== 独立消融实验 ===")
    print(f"Config: {CONFIG_PATH}")
    print(f"Params: Sum={TOTAL_SUM}, Step={STEP_SIZE}, Total={len(combos)}")
    print(f"Output: {RESULT_CSV}")
    print("-" * 30)

    # 2. 准备环境 (只读)
    config = load_yaml(CONFIG_PATH)
    set_seed(config['seed'])
    print("Pre-loading data and teacher...")
    enc_t, dec_t, cls_t, src_loader, tgt_loaders, test_loaders = setup_env(config)
    models = (enc_t, dec_t, cls_t)
    loaders = (src_loader, tgt_loaders, test_loaders)
    print("Done.")

    results = []

    # 3. 循环测试
    pbar = tqdm(combos, desc="Ablation Progress")
    for params in pbar:
        pbar.set_description(f"Testing AC={params['ac']} CC={params['cc']}")

        try:
            acc = run_ablation_training(config, params, models, loaders)

            res = params.copy()
            res['Accuracy'] = acc
            results.append(res)

            # 实时保存
            pd.DataFrame(results).to_csv(RESULT_CSV, index=False)

        except Exception as e:
            print(f"\n[Error] {params}: {e}")
            continue

    print(f"\n完成！结果已保存在 {RESULT_CSV}")

if __name__ == "__main__":
    main()
