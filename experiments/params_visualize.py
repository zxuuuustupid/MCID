import os
import sys
import yaml
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  # 引入绘图库

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataloader import NpyDataset
from src.models.encoder import MechanicEncoder
from src.models.decoder import MechanicDecoder
from src.models.classifier import MechanicClassifier

# ================= 核心工具函数 =================

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_sparsity_index(model):
    """
    计算模型的稀疏度指标 (Sparsity Index)
    论文定义: I_sparse = sum(||theta||^2)
    这里我们计算 Encoder 所有参数的 L2 范数平方和
    """
    total_norm_sq = 0.0
    for param in model.parameters():
        # param.norm(2) 计算 L2 范数，.item() 转标量，**2 平方
        total_norm_sq += param.norm(2).item() ** 2
    return total_norm_sq

def train_one_epoch(encoder, classifier, decoder, train_loader, criterion, optimizer, device):
    encoder.train()
    classifier.train()
    decoder.train()

    # 简单的 Recon Loss
    recon_criterion = nn.MSELoss()

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Forward
        features = encoder(data)
        outputs = classifier(features)

        # 简单起见，这里不需要太关注 loss 的具体值，只要能训练起来让参数更新即可
        # 只要参数在更新，我们就能观测到 Norm 的变化
        loss_cls = criterion(outputs, target)

        # 你的代码里有 decoder，为了保持一致性保留它，但它不是稀疏性的主角
        recon_data = decoder(features, target_length=data.shape[-1])
        # loss_recon = recon_criterion(recon_data, data)

        loss = loss_cls #+ 0.5 * loss_recon
        loss.backward()
        optimizer.step()

def main(config_path):
    config = load_config(config_path)
    set_seed(config['seed'])

    device = torch.device(f"cuda:{config['device']['gpu_id']}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. 构建模型
    print("构建模型...")
    enc_cfg = config['model']['encoder']
    cls_cfg = config['model']['classifier']

    encoder = MechanicEncoder(
        input_channels=enc_cfg['input_channels'],
        base_filters=enc_cfg['base_filters'],
        output_feature_dim=enc_cfg['output_feature_dim']
    ).to(device)

    classifier = MechanicClassifier(
        feature_dim=cls_cfg['feature_dim'],
        num_classes=cls_cfg['num_classes'],
        dropout_rate=cls_cfg['dropout_rate']
    ).to(device)

    decoder = MechanicDecoder(
        feature_dim=enc_cfg['output_feature_dim'],
        output_channels=enc_cfg['input_channels'],
        base_filters=enc_cfg['base_filters']
    ).to(device)

    # 2. 优化器
    params = list(encoder.parameters()) + list(classifier.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # 3. 数据加载 (简化版)
    data_cfg = config['data']
    train_wcs = data_cfg['train_wc'] if isinstance(data_cfg['train_wc'], list) else [data_cfg['train_wc']]

    datasets = []
    for wc in train_wcs:
        path = os.path.join(data_cfg['root_dir'], wc, 'train')
        datasets.append(NpyDataset(path))

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(datasets),
        batch_size=2*config['training']['batch_size'],
        # batch_size=8,
        shuffle=True
    )

    # 4. 记录器
    sparsity_history = []
    epochs = config['training']['epochs']

    print(f"开始分析稀疏性 (Baseline模式), 共 {epochs} 轮...")

    # 5. 循环
    for epoch in range(1, epochs + 1):
        # 训练一轮
        train_one_epoch(encoder, classifier, decoder, train_loader, criterion, optimizer, device)

        # === 核心：记录 Encoder 的参数范数 ===
        current_sparsity = calculate_sparsity_index(encoder)
        sparsity_history.append(current_sparsity)

        print(f"Epoch [{epoch}/{epochs}] - Encoder Param Norm: {current_sparsity:.4f}")

    # 6. 可视化并保存结果
    plot_save_path = "experiments/sparsity_curve_baseline.png"
    data_save_path = "experiments/sparsity_data_baseline.npy"

    # 保存数据以便后续对比
    np.save(data_save_path, np.array(sparsity_history))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), sparsity_history, label='Baseline (Standard Training)', color='red', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('L2 Norm of Parameters (Sparsity Index)')
    plt.title('Variation of Parameters\' Norm (Reproduction of Fig.10)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(plot_save_path)

    print("\n" + "="*40)
    print(f"分析完成！")
    print(f"1. 趋势图已保存至: {plot_save_path}")
    print(f"2. 原始数据已保存至: {data_save_path}")
    print("提示: 要复现完整的图10，请对 MID (Student) 代码也运行类似的记录逻辑，然后将两条曲线画在一起。")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/teacher.yaml")
    args = parser.parse_args()
    main(args.config)
