import os
import sys
import yaml
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# 将项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataloader import get_dataloader
from src.models.encoder import MechanicEncoder


class FeatureVisualizer:
    def __init__(self, config, device_id=0):
        """
        Args:
            config: 配置字典（已解析的 YAML）
        """
        self.config = config
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.encoder = None
        self._init_model()

    def _init_model(self):
        """初始化模型结构（兼容 teacher 和 mcid 两种配置格式）"""
        cfg = self.config['model']

        # 兼容两种配置格式:
        # teacher: model.encoder.input_channels
        # mcid: model.input_channels
        if 'encoder' in cfg:
            enc_cfg = cfg['encoder']
            input_channels = enc_cfg['input_channels']
            base_filters = enc_cfg['base_filters']
            feature_dim = enc_cfg['output_feature_dim']
        else:
            input_channels = cfg['input_channels']
            base_filters = cfg['base_filters']
            feature_dim = cfg['feature_dim']

        self.encoder = MechanicEncoder(
            input_channels=input_channels,
            base_filters=base_filters,
            output_feature_dim=feature_dim
        ).to(self.device)
        self.encoder.eval()

    def load_checkpoint(self, checkpoint_path):
        """加载模型权重"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 兼容不同的保存格式
        if 'encoder_state_dict' in checkpoint:
            state_dict = checkpoint['encoder_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 移除 module. 前缀 (DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v

        self.encoder.load_state_dict(new_state_dict, strict=False)
        print("Model loaded successfully.")

    def extract_features(self, dataloader, max_per_class=None):
        """提取特征和标签，支持先采样再提取（加速）"""

        # 先收集所有数据和标签（不过编码器）
        all_x, all_y = [], []
        for x, y in dataloader:
            all_x.append(x)
            all_y.append(y)
        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0).numpy()

        # 采样（在提取特征之前）
        if max_per_class is not None:
            sampled_idx = self._get_sampled_indices(all_y, max_per_class)
            all_x = all_x[sampled_idx]
            all_y = all_y[sampled_idx]
            print(f"[Info] Sampled {len(all_y)} points before feature extraction")

        # 只对采样后的数据提取特征
        print("Extracting features...")
        with torch.no_grad():
            all_x = all_x.to(self.device)
            features = self.encoder(all_x).cpu().numpy()

        return features, all_y

    def _get_sampled_indices(self, labels, max_per_class):
        """获取分层采样的索引"""
        unique_labels = np.unique(labels)
        sampled_idx = []

        for lbl in unique_labels:
            idx = np.where(labels == lbl)[0]
            if len(idx) > max_per_class:
                idx = np.random.choice(idx, max_per_class, replace=False)
            sampled_idx.extend(idx)

        sampled_idx = np.array(sampled_idx)
        np.random.shuffle(sampled_idx)
        return sampled_idx

    def run_tsne(self, features, random_state=42):
        """执行 t-SNE 降维"""
        print(f"Running t-SNE on {features.shape[0]} samples...")
        tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=random_state)
        return tsne.fit_transform(features)

    def plot_scatter(self, embeddings, labels, save_path, title=None):
        """绘制学术风格散点图"""
        plt.figure(figsize=(10, 8))
        sns.set_style("whitegrid")

        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        palette = sns.color_palette("bright", num_classes)

        sns.scatterplot(
            x=embeddings[:, 0], y=embeddings[:, 1],
            hue=labels, palette=palette, style=labels,
            s=60, alpha=0.8, edgecolor="w", legend='full'
        )

        if title:
            plt.title(title, fontsize=15, weight='bold')

        plt.xticks([])
        plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                   title="Fault Types", frameon=False)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
        plt.close()


def parse_checkpoint_path(checkpoint_path):
    """
    从 checkpoint 路径解析信息
    路径格式: ckpts/<model_type>/<dataset>/<filename>.pth
    返回: (dataset, model_type, ckpt_name)
    """
    ckpt_path = os.path.normpath(checkpoint_path)
    parts = ckpt_path.split(os.sep)
    filename = os.path.basename(ckpt_path)

    # 提取 model_type 和 dataset
    dataset, model_type = None, None
    for i, part in enumerate(parts):
        if part in ['mcid', 'teacher']:
            model_type = part
            if i + 1 < len(parts) and not parts[i + 1].endswith('.pth'):
                dataset = parts[i + 1]
            break

    if not dataset or not model_type:
        raise ValueError(f"Cannot parse checkpoint path: {checkpoint_path}\n"
                         f"Expected format: ckpts/<mcid|teacher>/<dataset>/<file>.pth")

    # 提取 ckpt_name (去掉 _best.pth 后缀)
    ckpt_name = filename.replace('_best.pth', '').replace('.pth', '')

    return dataset, model_type, ckpt_name


def find_config(dataset, model_type, ckpt_name):
    """根据信息查找对应的配置文件"""
    # 构建精确配置文件名
    # ckpt_name: teacher_train_1 或 mcid_train_1_meta_2_3
    config_name = f"{ckpt_name.replace(f'{model_type}_', f'{model_type}_{dataset}_')}.yaml"
    config_path = os.path.join("configs", config_name)

    if os.path.exists(config_path):
        return config_path

    # 模糊匹配
    config_dir = "configs"
    prefix = f"{model_type}_{dataset}"
    candidates = [f for f in os.listdir(config_dir) if f.startswith(prefix) and f.endswith('.yaml')]

    if candidates:
        config_path = os.path.join(config_dir, candidates[0])
        print(f"[Auto] Using fallback config: {config_path}")
        return config_path

    raise FileNotFoundError(f"No config found for {model_type}/{dataset}")


def get_unseen_wc(config, model_type):
    """
    获取测试工况
    - 默认使用 test_wcs 的最后一个（通常是完全未见的工况）
    """
    data_cfg = config['data']

    # 获取见过的工况（用于打印信息）
    if model_type == 'teacher':
        seen_wcs = {data_cfg.get('train_wc', 'WC1')}
    else:
        seen_wcs = set(data_cfg.get('source_wc', []))
        seen_wcs.update(data_cfg.get('target_wcs', []))

    test_wcs = data_cfg.get('test_wcs', [])

    # 默认取最后一个
    if test_wcs:
        return test_wcs[-1], seen_wcs
    else:
        return 'WC4', seen_wcs  # 兜底默认值


def main():
    parser = argparse.ArgumentParser(
        description="t-SNE Feature Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/SNE.py --checkpoint ckpts/teacher/PU/teacher_train_1_best.pth
  python experiments/SNE.py --checkpoint ckpts/mcid/PU/mcid_train_1_meta_2_3_best.pth
        """
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=str, default=None, help="Config file (auto-inferred if omitted)")
    parser.add_argument("--target_wc", type=str, default=None, help="Target WC (auto-selects unseen WC if omitted)")
    parser.add_argument("--save_dir", type=str, default="experiments/SNE", help="Output directory")
    parser.add_argument("--max_per_class", type=int, default=100, help="Max samples per class (default: 500)")

    args = parser.parse_args()

    # 1. 解析 checkpoint 路径
    dataset, model_type, ckpt_name = parse_checkpoint_path(args.checkpoint)
    print(f"[Info] Dataset: {dataset}, Model: {model_type}")

    # 2. 查找配置文件
    if args.config is None:
        args.config = find_config(dataset, model_type, ckpt_name)
    print(f"[Info] Config: {args.config}")

    # 3. 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 4. 确定测试工况（选择未见过的）
    if args.target_wc:
        target_wc = args.target_wc
        _, seen_wcs = get_unseen_wc(config, model_type)
        if target_wc in seen_wcs:
            print(f"[Warning] {target_wc} was seen during training!")
    else:
        target_wc, seen_wcs = get_unseen_wc(config, model_type)

    print(f"[Info] Seen WCs: {seen_wcs}")
    print(f"[Info] Testing on unseen WC: {target_wc}")

    # 5. 初始化并加载模型
    visualizer = FeatureVisualizer(config)
    visualizer.load_checkpoint(args.checkpoint)

    # 6. 加载数据
    data_cfg = config['data']
    data_path = os.path.join(data_cfg['root_dir'], target_wc, 'test')
    if not os.path.exists(data_path):
        data_path = os.path.join(data_cfg['root_dir'], target_wc, 'train')
        print(f"[Info] Using train split: {data_path}")

    batch_size = config['training']['batch_size']
    loader = get_dataloader(data_path, batch_size, shuffle=False)

    # 7. 提取特征 + 采样 + t-SNE
    features, labels = visualizer.extract_features(loader, max_per_class=args.max_per_class)
    print(f"[Info] Using {len(labels)} points for t-SNE")

    embeddings = visualizer.run_tsne(features)

    # 8. 保存图片
    # 格式: experiments/SNE/<dataset>/tsne_<dataset>_<ckpt_name>_test_<wc>.png
    file_name = f"tsne_{dataset}_{ckpt_name}_test_{target_wc}.png"
    save_path = os.path.join(args.save_dir, dataset, file_name)

    model_label = "Teacher" if model_type == 'teacher' else "MCID"
    title = f"{dataset} {model_label} (Unseen: {target_wc})"

    visualizer.plot_scatter(embeddings, labels, save_path, title=title)


if __name__ == "__main__":
    main()



"""
How to run:

python experiments/SNE.py --checkpoint ckpts/teacher/PU/train_1_best_model.pth
python experiments/SNE.py --checkpoint ckpts/mcid/PU/mcid_train_1_meta_2_3_best.pth
"""
