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
import matplotlib.pyplot as plt

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataloader import NpyDataset, get_dataloader
from torch.utils.data import ConcatDataset, DataLoader
from src.models.encoder import MechanicEncoder
from src.models.decoder import MechanicDecoder
from src.models.classifier import MechanicClassifier

# ================= 工具函数 =================

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_sparsity_index(model):
    """计算 Encoder 参数的 L2 范数平方和"""
    total_norm_sq = 0.0
    for param in model.parameters():
        total_norm_sq += param.norm(2).item() ** 2
    return total_norm_sq

def get_infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

# ================= 核心计算逻辑 (完全复用 main.py) =================

def compute_loss(encoder_s, classifier, encoder_t, decoder_t, x_s, y_s, x_t, y_t, config):
    loss_cfg = config['loss']

    feat_s = encoder_s(x_s)
    logits = classifier(feat_s)
    with torch.no_grad():
        feat_t = encoder_t(x_t).detach()

    # L_AC
    l_ac = torch.tensor(0.0, device=x_s.device)
    valid_classes = 0
    classes_s = torch.unique(y_s); classes_t = torch.unique(y_t)
    common_classes = [c for c in classes_s if c in classes_t]
    for c in common_classes:
        proto_s = feat_s[y_s == c].mean(dim=0)
        proto_t = feat_t[y_t == c].mean(dim=0)
        l_ac += nn.MSELoss()(proto_s, proto_t)
        valid_classes += 1
    if valid_classes > 0: l_ac /= valid_classes

    # L_CC
    x_recon = decoder_t(feat_s, target_length=x_s.shape[-1])
    with torch.no_grad():
        feat_cycle = encoder_t(x_recon).detach()

    if feat_cycle.shape[0] == feat_t.shape[0]:
        l_cc = nn.MSELoss()(feat_cycle, feat_t) # 保持你的原始逻辑
    else:
        l_cc = torch.tensor(0.0, device=feat_cycle.device)

    # L_LC
    l_lc = nn.CrossEntropyLoss()(logits, y_s)

    total = loss_cfg['lambda_ac'] * l_ac + loss_cfg['lambda_cc'] * l_cc + loss_cfg['lambda_lc'] * l_lc
    return total

def inner_update(encoder, loss, inner_lr, first_order=True):
    encoder_prime = deepcopy(encoder)
    grads = grad(loss, encoder.parameters(), create_graph=not first_order, retain_graph=True, allow_unused=True)
    for p, g in zip(encoder_prime.parameters(), grads):
        if g is not None:
            p.data = p.data - inner_lr * g
    return encoder_prime

def meta_train_step(source_iter, target_iters, encoder_s, classifier, encoder_t, decoder_t, config, device):
    meta_cfg = config['meta']
    wc_list = list(target_iters.keys())
    random.shuffle(wc_list)

    # 至少要有2个工况才能分 Support/Query，否则回退到普通训练
    if len(wc_list) >= 2:
        query_wc = wc_list[-1]
        support_wcs = wc_list[:-1]
    else:
        query_wc = wc_list[0]
        support_wcs = wc_list

    # Meta-Train
    total_sup_loss = 0
    for wc in support_wcs:
        x_s, y_s = next(target_iters[wc])
        x_t, y_t = next(source_iter)
        x_s, y_s, x_t, y_t = x_s.to(device), y_s.to(device), x_t.to(device), y_t.to(device)
        l_step = compute_loss(encoder_s, classifier, encoder_t, decoder_t, x_s, y_s, x_t, y_t, config)
        total_sup_loss += l_step

    if len(support_wcs) > 0:
        total_sup_loss /= len(support_wcs)

    # Inner Update
    if isinstance(total_sup_loss, torch.Tensor) and total_sup_loss.item() != 0:
        encoder_prime = inner_update(encoder_s, total_sup_loss, meta_cfg['inner_lr'], meta_cfg['first_order'])
    else:
        encoder_prime = encoder_s

    # Meta-Test
    x_qs, y_qs = next(target_iters[query_wc])
    x_qt, y_qt = next(source_iter)
    x_qs, y_qs, x_qt, y_qt = x_qs.to(device), y_qs.to(device), x_qt.to(device), y_qt.to(device)

    l_qry = compute_loss(encoder_prime, classifier, encoder_t, decoder_t, x_qs, y_qs, x_qt, y_qt, config)

    # Final
    l_total = total_sup_loss + meta_cfg['beta'] * l_qry
    return l_total

# ================= 主程序 =================

def main(config_path):
    config = load_config(config_path)
    set_seed(config['seed'])
    device = torch.device(f"cuda:{config['device']['gpu_id']}" if config['device']['use_cuda'] and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. 准备数据
    data_cfg = config['data']
    batch_size = config['training']['batch_size']

    # Source
    src_list = data_cfg['source_wc'] if isinstance(data_cfg['source_wc'], list) else [data_cfg['source_wc']]
    src_datasets = [NpyDataset(os.path.join(data_cfg['root_dir'], wc, 'train')) for wc in src_list]
    source_loader = DataLoader(ConcatDataset(src_datasets), batch_size=batch_size, shuffle=True)
    source_iter = get_infinite_loader(source_loader)

    # Target
    target_iters = {}
    for wc in data_cfg['target_wcs']:
        path = os.path.join(data_cfg['root_dir'], wc, 'train')
        target_iters[wc] = get_infinite_loader(get_dataloader(path, batch_size, shuffle=True))

    # 2. 准备 Teacher (加载权重)
    print("Loading Teacher...")
    cfg_m = config['model']
    enc_t = MechanicEncoder(cfg_m['input_channels'], cfg_m['base_filters'], cfg_m['feature_dim']).to(device)
    dec_t = MechanicDecoder(cfg_m['feature_dim'], cfg_m['input_channels'], cfg_m['base_filters']).to(device)
    cls_t = MechanicClassifier(cfg_m['feature_dim'], cfg_m['num_classes'], cfg_m['dropout']).to(device)

    # 路径推导
    src_nums = sorted(["".join(filter(str.isdigit, wc)) for wc in src_list], key=int)
    wc_tag = "_".join(src_nums)
    ckpt_path = os.path.join(config['teacher']['checkpoint'], config['data']['dataset_name'], f"train_{wc_tag}_best_model.pth")

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        enc_t.load_state_dict(ckpt['encoder_state_dict'])
        cls_t.load_state_dict(ckpt['classifier_state_dict'])
        if 'decoder_state_dict' in ckpt: dec_t.load_state_dict(ckpt['decoder_state_dict'])
    else:
        raise FileNotFoundError(f"Teacher Not Found: {ckpt_path}")

    for m in [enc_t, dec_t, cls_t]:
        m.eval()
        for p in m.parameters(): p.requires_grad = False

    # 3. 准备 Student (继承 Teacher)
    print("Initializing Student...")
    enc_s = MechanicEncoder(cfg_m['input_channels'], cfg_m['base_filters'], cfg_m['feature_dim']).to(device)
    enc_s.load_state_dict(enc_t.state_dict()) # 继承权重

    optimizer = optim.Adam(enc_s.parameters(), lr=config['training']['lr'])

    # 4. 训练并记录
    sparsity_history = []
    epochs = config['training']['epochs']
    iterations = config['training'].get('iterations_per_epoch', 100)

    print(f"Start Analyzing Sparsity (MID Student), Epochs: {epochs}")

    for epoch in range(1, epochs + 1):
        enc_s.train()
        for _ in range(iterations):
            optimizer.zero_grad()
            loss = meta_train_step(source_iter, target_iters, enc_s, cls_t, enc_t, dec_t, config, device)
            loss.backward()
            optimizer.step()

        # === 核心：记录 Student Encoder 参数范数 ===
        current_sparsity = calculate_sparsity_index(enc_s)
        sparsity_history.append(current_sparsity)

        print(f"Epoch [{epoch}/{epochs}] - MID Sparsity Index: {current_sparsity:.4f}")

    # 5. 保存结果
    data_save_path = "experiments/sparsity_data_mid.npy"
    plot_save_path = "experiments/sparsity_curve_mid.png"

    np.save(data_save_path, np.array(sparsity_history))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), sparsity_history, label='MID (Meta-Learning)', color='blue', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('L2 Norm of Parameters')
    plt.title('MID Sparsity Curve')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(plot_save_path)

    print("\n" + "="*40)
    print("MID 稀疏性分析完成！")
    print(f"1. 数据已保存: {data_save_path}")
    print(f"2. 预览图已保存: {plot_save_path}")
    print("\n[下一步] 请运行绘图脚本，将 'sparsity_data_baseline.npy' 和 'sparsity_data_mid.npy' 画在一起。")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mcid.yaml")
    args = parser.parse_args()
    main(args.config)
