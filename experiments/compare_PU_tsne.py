"""
PU数据集 t-SNE 对比可视化脚本

生成 2x4 子图网格：
- 第一行：Teacher网络在四种工况下的特征可视化
- 第二行：MCID方法在四种工况下的特征可视化

Usage:
    python experiments/compare_PU_tsne.py
    python experiments/compare_PU_tsne.py --teacher_ckpt ckpts/teacher/PU/train_1_best_model.pth --mcid_ckpt ckpts/mcid/PU/mcid_train_1_meta_2_3_best.pth
    python experiments/compare_PU_tsne.py --max_per_class 200 --save_name my_comparison
"""

import os
import sys
import yaml
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm

# 将项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataloader import get_dataloader
from src.models.encoder import MechanicEncoder


class FeatureExtractor:
    """特征提取器"""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.encoder = None
        self._init_model()

    def _init_model(self):
        """初始化模型结构（兼容 teacher 和 mcid 两种配置格式）"""
        cfg = self.config['model']

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

        # checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

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

    def extract_features(self, dataloader, max_per_class=None, sampled_idx=None):
        """提取特征

        Args:
            dataloader: 数据加载器
            max_per_class: 每类最大采样数
            sampled_idx: 预计算的采样索引（如果提供，则使用此索引而不是重新采样）

        Returns:
            features: 特征数组
            labels: 标签数组
            sampled_idx: 使用的采样索引（用于在不同模型间保持一致）
        """
        all_x, all_y = [], []
        for x, y in dataloader:
            all_x.append(x)
            all_y.append(y)
        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0).numpy()

        # 采样
        if sampled_idx is not None:
            # 使用提供的采样索引
            all_x = all_x[sampled_idx]
            all_y = all_y[sampled_idx]
        elif max_per_class is not None:
            sampled_idx = self._get_sampled_indices(all_y, max_per_class)
            all_x = all_x[sampled_idx]
            all_y = all_y[sampled_idx]

        # 提取特征
        with torch.no_grad():
            all_x = all_x.to(self.device)
            features = self.encoder(all_x).cpu().numpy()

        return features, all_y, sampled_idx

    def _get_sampled_indices(self, labels, max_per_class):
        """分层采样"""
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


def run_tsne(features, random_state=42, perplexity=30):
    """执行 t-SNE 降维"""
    # 确保perplexity不超过样本数
    n_samples = features.shape[0]
    perplexity = min(perplexity, n_samples - 1)

    tsne = TSNE(
        n_components=2,
        init='pca',
        learning_rate='auto',
        random_state=random_state,
        perplexity=perplexity,
        n_iter=1000
    )
    return tsne.fit_transform(features)


# ============================================================================
# 顶刊级别绘图配置类
# ============================================================================

class PlotConfig:
    """
    绘图配置类 - 封装所有可调参数

    使用示例:
        config = PlotConfig()
        config.marker.size = 30
        config.marker.style = 'o'
        config.colors.scheme = 'nature'
        config.title.row1_labels = ['WC1 (Train)', 'WC2 (Unseen)', ...]
    """

    def __init__(self):
        # =====================================================================
        # 字体设置
        # =====================================================================
        self.font = FontConfig()

        # =====================================================================
        # 图形尺寸与DPI
        # =====================================================================
        self.figure = FigureConfig()

        # =====================================================================
        # 散点/标记设置
        # =====================================================================
        self.marker = MarkerConfig()

        # =====================================================================
        # 颜色设置
        # =====================================================================
        self.colors = ColorConfig()

        # =====================================================================
        # 布局设置
        # =====================================================================
        self.layout = LayoutConfig()

        # =====================================================================
        # 边框/轴线设置
        # =====================================================================
        self.spine = SpineConfig()

        # =====================================================================
        # 图例设置
        # =====================================================================
        self.legend = LegendConfig()

        # =====================================================================
        # 标题设置
        # =====================================================================
        self.title = TitleConfig()

        # =====================================================================
        # 标签映射
        # =====================================================================
        self.labels = LabelConfig()


class FontConfig:
    """字体配置"""
    def __init__(self):
        self.family = 'Times New Roman'
        self.title_size = 14
        self.label_size = 16
        self.legend_size = 10
        self.legend_title_size = 11
        self.suptitle_size = 16
        self.weight = 'bold'  # 'normal', 'bold', 'light'


class FigureConfig:
    """图形配置"""
    def __init__(self):
        self.width = 16
        self.height = 8
        self.dpi = 300
        self.background_color = 'white'
        self.save_formats = ['png', 'pdf', 'svg']  # 要保存的格式列表


class MarkerConfig:
    """散点/标记配置"""
    def __init__(self):
        # 标记样式: 'o'圆形, 's'方形, '^'三角, 'D'菱形, 'p'五边形, '*'星形, 'h'六边形
        self.style = 'o'
        self.size = 20
        self.alpha = 0.85

        # 边缘设置
        self.edge_color = 'white'
        self.edge_width = 0.3

        # 是否显示边缘
        self.show_edge = True


class ColorConfig:
    """颜色配置"""
    def __init__(self):
        # 配色方案: 'nature', 'science', 'ieee', 'elegant', 'tableau', 'custom'
        self.scheme = 'nature'

        # 自定义颜色列表（当scheme='custom'时使用）
        self.custom_colors = [
            '#E64B35', '#4DBBD5', '#00A087', '#3C5488',
            '#F39B7F', '#8491B4', '#91D1C2', '#DC0000'
        ]

    def get_palette(self, num_classes):
        """获取配色方案"""
        palettes = {
            'nature': [
                '#E64B35', '#4DBBD5', '#00A087', '#3C5488',
                '#F39B7F', '#8491B4', '#91D1C2', '#DC0000',
                '#7E6148', '#B09C85',
            ],
            'science': [
                '#3B4992', '#EE0000', '#008B45', '#631879',
                '#008280', '#BB0021', '#5F559B', '#A20056',
                '#808180', '#1B1919',
            ],
            'ieee': [
                '#0072BD', '#D95319', '#EDB120', '#7E2F8E',
                '#77AC30', '#4DBEEE', '#A2142F', '#FF6F61',
                '#6B5B95', '#88B04B',
            ],
            'elegant': [
                '#264653', '#2A9D8F', '#E9C46A', '#F4A261',
                '#E76F51', '#606C38', '#283618', '#DDA15E',
                '#BC6C25', '#9B2226',
            ],
            'tableau': [
                '#4E79A7', '#F28E2B', '#E15759', '#76B7B2',
                '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7',
                '#9C755F', '#BAB0AC',
            ],
            'custom': self.custom_colors,
        }

        palette = palettes.get(self.scheme, palettes['nature'])

        if num_classes > len(palette):
            palette = (palette * (num_classes // len(palette) + 1))[:num_classes]

        return palette[:num_classes]


class LayoutConfig:
    """布局配置"""
    def __init__(self):
        self.wspace = 0.08          # 子图水平间距
        self.hspace = 0.15          # 子图垂直间距
        self.left_margin = 0.05     # 左边距
        self.right_margin = 0.92    # 右边距（为图例留空间）
        self.top_margin = 0.95      # 上边距
        self.bottom_margin = 0.05   # 下边距


class SpineConfig:
    """边框/轴线配置"""
    def __init__(self):
        self.visible = True
        self.linewidth = 1.0
        self.color = '#333333'

        # 可单独控制四边
        self.top = True
        self.bottom = True
        self.left = True
        self.right = True

        # 网格
        self.grid_visible = False
        self.grid_style = '--'
        self.grid_alpha = 0.3
        self.grid_color = '#CCCCCC'


class LegendConfig:
    """图例配置"""
    def __init__(self):
        self.show = True
        self.position = 'right'      # 'right', 'bottom', 'none'
        self.bbox_x = 0.95           # x位置
        self.bbox_y = 0.75            # y位置

        self.title = 'Fault Types'
        self.title_fontweight = 'bold'

        self.frameon = True
        self.frame_alpha = 0.95
        self.frame_edgecolor = '#CCCCCC'
        self.fancybox = False

        self.marker_size = 8
        self.borderpad = 0.8
        self.labelspacing = 0.6
        self.handletextpad = 0.5

        # 列数（用于底部图例）
        self.ncol = 1


class TitleConfig:
    """标题配置"""
    def __init__(self):
        # ----- 总标题 -----
        self.show_suptitle = False
        self.suptitle_text = 'PU Dataset: Teacher vs CTSL Feature Visualization'
        self.suptitle_y = 1.05

        # ----- 行标签 -----
        self.row_labels = ['Teacher', 'CTSL']
        self.row_label_pad = 15

        # ----- 子图标题 -----
        # 设为None则自动生成，或手动指定
        # 格式: [[row1_col1, row1_col2, ...], [row2_col1, row2_col2, ...]]
        self.subplot_titles = None

        # 自动标题的格式模板
        # {wc}: 工况名, {tag}: 标签(Train/Unseen/Source/Target)
        self.auto_title_format = '{wc} ({tag})'

        # 标题padding
        self.title_pad = 8

        # ----- 自定义标签文字 -----
        self.train_tag = 'Train'
        self.unseen_tag = 'Unseen'
        self.source_tag = 'Source'
        self.target_tag = 'Target'


class LabelConfig:
    """类别标签配置"""
    def __init__(self):
        # 类别名称映射 {类别索引: 显示名称}
        # 例如: {0: 'Normal', 1: 'Inner Fault', 2: 'Outer Fault', ...}
        self.class_names = None

        # 默认标签格式
        self.default_format = 'Class {}'


def setup_plot_style(config: PlotConfig):
    """设置全局绑定样式"""
    import matplotlib as mpl

    # 设置字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = [config.font.family, 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'

    # 设置其他参数
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # 线宽
    plt.rcParams['axes.linewidth'] = config.spine.linewidth


def plot_comparison_grid(
    teacher_embeddings_list,
    mcid_embeddings_list,
    labels_list,
    wcs,
    save_path,
    teacher_train_wc='WC1',
    mcid_source_wc='WC1',
    mcid_target_wcs=None,
    config: PlotConfig = None
):
    """
    绑制 2x4 对比图（顶刊级别）

    Args:
        teacher_embeddings_list: Teacher模型在各工况下的t-SNE embeddings
        mcid_embeddings_list: MCID模型在各工况下的t-SNE embeddings
        labels_list: 各工况对应的标签列表
        wcs: 工况名称列表 ['WC1', 'WC2', 'WC3', 'WC4']
        save_path: 保存路径
        teacher_train_wc: Teacher训练的工况
        mcid_source_wc: MCID的源域工况
        mcid_target_wcs: MCID的目标域工况列表
        config: PlotConfig配置对象
    """
    if config is None:
        config = PlotConfig()

    # 设置绑图样式
    setup_plot_style(config)

    # 获取类别信息
    num_classes = len(np.unique(np.concatenate(labels_list)))
    palette = config.colors.get_palette(num_classes)

    # 创建图形
    fig, axes = plt.subplots(
        2, len(wcs),
        figsize=(config.figure.width, config.figure.height),
        facecolor=config.figure.background_color
    )

    # 调整子图间距
    plt.subplots_adjust(
        wspace=config.layout.wspace,
        hspace=config.layout.hspace,
        left=config.layout.left_margin,
        right=config.layout.right_margin,
        top=config.layout.top_margin,
        bottom=config.layout.bottom_margin
    )

    # 绑制每个子图
    for col, wc in enumerate(wcs):
        # ===== 确定标题 =====
        if config.title.subplot_titles is not None:
            # 使用自定义标题
            teacher_title = config.title.subplot_titles[0][col]
            mcid_title = config.title.subplot_titles[1][col]
        else:
            # 自动生成标题
            teacher_tag = config.title.train_tag if wc == teacher_train_wc else config.title.unseen_tag

            if wc == mcid_source_wc:
                mcid_tag = config.title.source_tag
            elif mcid_target_wcs and wc in mcid_target_wcs:
                mcid_tag = config.title.target_tag
            else:
                mcid_tag = config.title.unseen_tag

            teacher_title = config.title.auto_title_format.format(wc=wc, tag=teacher_tag)
            mcid_title = config.title.auto_title_format.format(wc=wc, tag=mcid_tag)

        # ===== 第一行: Teacher =====
        ax = axes[0, col]
        embeddings = teacher_embeddings_list[col]
        labels = labels_list[col]

        _plot_scatter(ax, embeddings, labels, palette, config)
        _style_axis(ax, teacher_title, config)

        # 左侧行标签
        if col == 0:
            ax.set_ylabel(
                config.title.row_labels[0],
                fontsize=config.font.label_size,
                fontweight=config.font.weight,
                fontfamily=config.font.family,
                labelpad=config.title.row_label_pad
            )

        # ===== 第二行: MCID =====
        ax = axes[1, col]
        embeddings = mcid_embeddings_list[col]

        _plot_scatter(ax, embeddings, labels, palette, config)
        _style_axis(ax, mcid_title, config)

        # 左侧行标签
        if col == 0:
            ax.set_ylabel(
                config.title.row_labels[1],
                fontsize=config.font.label_size,
                fontweight=config.font.weight,
                fontfamily=config.font.family,
                labelpad=config.title.row_label_pad
            )

    # 添加图例
    if config.legend.show:
        _add_legend(fig, labels_list, palette, config)

    # 添加总标题
    if config.title.show_suptitle:
        fig.suptitle(
            config.title.suptitle_text,
            fontsize=config.font.suptitle_size,
            fontweight=config.font.weight,
            fontfamily=config.font.family,
            y=config.title.suptitle_y
        )

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    base_path = save_path.rsplit('.', 1)[0]

    for fmt in config.figure.save_formats:
        if fmt == 'png':
            out_path = f"{base_path}.png"
            plt.savefig(
                out_path,
                dpi=config.figure.dpi,
                bbox_inches='tight',
                facecolor=config.figure.background_color,
                edgecolor='none',
                pad_inches=0.1
            )
        else:
            out_path = f"{base_path}.{fmt}"
            plt.savefig(
                out_path,
                format=fmt,
                bbox_inches='tight',
                facecolor=config.figure.background_color,
                edgecolor='none',
                pad_inches=0.1
            )
        print(f"Figure saved: {out_path}")

    plt.close()


def _plot_scatter(ax, embeddings, labels, palette, config: PlotConfig):
    """绑制散点图（内部函数）"""
    unique_labels = np.unique(labels)

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl

        # 边缘设置
        edge_color = config.marker.edge_color if config.marker.show_edge else 'none'
        edge_width = config.marker.edge_width if config.marker.show_edge else 0

        ax.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=[palette[int(lbl)]],
            s=config.marker.size,
            alpha=config.marker.alpha,
            marker=config.marker.style,
            edgecolors=edge_color,
            linewidths=edge_width,
            label=f'Class {int(lbl)}',
            zorder=2
        )


def _style_axis(ax, title, config: PlotConfig):
    """设置坐标轴样式（内部函数）"""
    # 设置标题
    ax.set_title(
        title,
        fontsize=config.font.title_size,
        fontweight=config.font.weight,
        fontfamily=config.font.family,
        pad=config.title.title_pad
    )

    # 移除刻度
    ax.set_xticks([])
    ax.set_yticks([])

    # 设置边框
    spine_settings = {
        'top': config.spine.top,
        'bottom': config.spine.bottom,
        'left': config.spine.left,
        'right': config.spine.right
    }

    for spine_name, spine in ax.spines.items():
        if config.spine.visible and spine_settings.get(spine_name, True):
            spine.set_visible(True)
            spine.set_linewidth(config.spine.linewidth)
            spine.set_color(config.spine.color)
        else:
            spine.set_visible(False)

    # 设置背景
    ax.set_facecolor(config.figure.background_color)

    # 网格
    if config.spine.grid_visible:
        ax.grid(
            True,
            linestyle=config.spine.grid_style,
            alpha=config.spine.grid_alpha,
            color=config.spine.grid_color,
            zorder=0
        )
    else:
        ax.grid(False)


def _add_legend(fig, labels_list, palette, config: PlotConfig):
    """添加图例（内部函数）"""
    unique_labels = np.unique(np.concatenate(labels_list))

    # 创建图例元素
    legend_elements = []
    for lbl in unique_labels:
        # 获取标签名称
        if config.labels.class_names and int(lbl) in config.labels.class_names:
            label_name = config.labels.class_names[int(lbl)]
        else:
            label_name = config.labels.default_format.format(int(lbl))

        edge_color = config.marker.edge_color if config.marker.show_edge else 'none'

        legend_elements.append(
            plt.Line2D(
                [0], [0],
                marker=config.marker.style,
                color='w',
                markerfacecolor=palette[int(lbl)],
                markersize=config.legend.marker_size,
                markeredgecolor=edge_color,
                markeredgewidth=0.5,
                label=label_name
            )
        )

    # 确定图例位置
    if config.legend.position == 'right':
        loc = 'center left'
        bbox = (config.legend.bbox_x, config.legend.bbox_y)
        ncol = config.legend.ncol
    elif config.legend.position == 'bottom':
        loc = 'upper center'
        bbox = (0.5, -0.05)
        ncol = len(unique_labels)
    else:
        loc = 'center left'
        bbox = (config.legend.bbox_x, config.legend.bbox_y)
        ncol = config.legend.ncol

    # 添加图例
    legend = fig.legend(
        handles=legend_elements,
        loc=loc,
        bbox_to_anchor=bbox,
        title=config.legend.title,
        frameon=config.legend.frameon,
        fontsize=config.font.legend_size,
        title_fontsize=config.font.legend_title_size,
        framealpha=config.legend.frame_alpha,
        edgecolor=config.legend.frame_edgecolor,
        fancybox=config.legend.fancybox,
        borderpad=config.legend.borderpad,
        labelspacing=config.legend.labelspacing,
        handletextpad=config.legend.handletextpad,
        ncol=ncol
    )

    # 设置图例字体
    legend.get_title().set_fontfamily(config.font.family)
    legend.get_title().set_fontweight(config.legend.title_fontweight)
    for text in legend.get_texts():
        text.set_fontfamily(config.font.family)


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_data_path(root_dir, wc):
    """获取数据路径"""
    # 优先使用test，如果没有则使用train
    test_path = os.path.join(root_dir, wc, 'test')
    train_path = os.path.join(root_dir, wc, 'train')

    if os.path.exists(test_path):
        return test_path
    elif os.path.exists(train_path):
        return train_path
    else:
        raise FileNotFoundError(f"Data not found for {wc}")


def main():
    parser = argparse.ArgumentParser(
        description="PU数据集 Teacher vs MCID t-SNE对比可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Checkpoint 参数
    parser.add_argument(
        "--teacher_ckpt", type=str,
        default="ckpts/teacher/PU/train_1_best_model.pth",
        help="Teacher model checkpoint"
    )
    parser.add_argument(
        "--mcid_ckpt", type=str,
        default="ckpts/mcid/PU/mcid_train_1_meta_2_3_best.pth",
        help="MCID model checkpoint"
    )

    # 配置文件参数
    parser.add_argument(
        "--teacher_config", type=str,
        default="configs/teacher_PU_train_1.yaml",
        help="Teacher config file"
    )
    parser.add_argument(
        "--mcid_config", type=str,
        default="configs/mcid_PU_train_1_meta_2_3.yaml",
        help="MCID config file"
    )

    # 其他参数
    parser.add_argument("--max_per_class", type=int, default=100, help="Max samples per class")
    parser.add_argument("--save_dir", type=str, default="experiments/SNE/PU", help="Output directory")
    parser.add_argument("--save_name", type=str, default="ieee", help="Output filename (without extension)")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 设置设备
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # 工况列表
    wcs = ['WC1', 'WC2', 'WC3', 'WC4']

    # 加载配置
    print("\n[Step 1] Loading configs...")
    teacher_config = load_config(args.teacher_config)
    mcid_config = load_config(args.mcid_config)

    # 获取训练信息
    teacher_train_wc = teacher_config['data'].get('train_wc', 'WC1')
    mcid_source_wc = mcid_config['data'].get('source_wc', ['WC1'])
    if isinstance(mcid_source_wc, list):
        mcid_source_wc = mcid_source_wc[0]
    mcid_target_wcs = mcid_config['data'].get('target_wcs', [])

    print(f"  Teacher trained on: {teacher_train_wc}")
    print(f"  MCID source: {mcid_source_wc}, targets: {mcid_target_wcs}")

    # 初始化特征提取器
    print("\n[Step 2] Initializing models...")
    teacher_extractor = FeatureExtractor(teacher_config, device)
    teacher_extractor.load_checkpoint(args.teacher_ckpt)
    print(f"  Teacher model loaded: {args.teacher_ckpt}")

    mcid_extractor = FeatureExtractor(mcid_config, device)
    mcid_extractor.load_checkpoint(args.mcid_ckpt)
    print(f"  MCID model loaded: {args.mcid_ckpt}")

    # 数据根目录
    data_root = teacher_config['data']['root_dir']
    batch_size = teacher_config['training']['batch_size']

    # 提取特征
    print("\n[Step 3] Extracting features...")
    teacher_embeddings_list = []
    mcid_embeddings_list = []
    labels_list = []

    for wc in tqdm(wcs, desc="Processing WCs"):
        # 获取数据
        data_path = get_data_path(data_root, wc)
        loader = get_dataloader(data_path, batch_size, shuffle=False)

        # Teacher特征提取（同时获取采样索引）
        teacher_features, labels, sampled_idx = teacher_extractor.extract_features(loader, args.max_per_class)
        print(f"  {wc} Teacher features: mean={teacher_features.mean():.4f}, std={teacher_features.std():.4f}")
        teacher_emb = run_tsne(teacher_features, random_state=args.seed)
        teacher_embeddings_list.append(teacher_emb)

        # MCID特征提取 (使用相同的采样索引确保一致性)
        loader = get_dataloader(data_path, batch_size, shuffle=False)
        mcid_features, mcid_labels, _ = mcid_extractor.extract_features(loader, sampled_idx=sampled_idx)
        print(f"  {wc} MCID features: mean={mcid_features.mean():.4f}, std={mcid_features.std():.4f}")
        mcid_emb = run_tsne(mcid_features, random_state=args.seed)
        mcid_embeddings_list.append(mcid_emb)

        labels_list.append(labels)

        print(f"  {wc}: {len(labels)} samples")

    # 绘图
    print("\n[Step 4] Plotting...")
    save_path = os.path.join(args.save_dir, f"{args.save_name}.png")

    # =========================================================================
    # 配置绑图参数（可根据需要修改）
    # =========================================================================
    config = PlotConfig()

    # ----- 字体 -----
    config.font.family = 'Times New Roman'
    config.font.title_size = 14
    config.font.label_size = 16
    config.font.legend_size = 10
    config.font.weight = 'bold'

    # ----- 图形尺寸 -----
    config.figure.width = 16
    config.figure.height = 8
    config.figure.dpi = 300
    config.figure.background_color = 'white'
    config.figure.save_formats = ['png', 'pdf', 'svg']  # 保存格式

    # ----- 散点样式 -----
    config.marker.style = 'o'       # 'o'圆, 's'方, '^'三角, 'D'菱形, '*'星
    config.marker.size = 10
    config.marker.alpha = 0.9
    config.marker.show_edge = False
    config.marker.edge_color = 'white'
    config.marker.edge_width = 0.3

    # ----- 颜色方案 -----
    config.colors.scheme = 'nature'  # 'nature', 'science', 'ieee', 'elegant', 'tableau', 'custom'
    # config.colors.custom_colors = ['#FF0000', '#00FF00', ...]  # 自定义颜色

    # ----- 边框/轴线 -----
    config.spine.visible = True
    config.spine.linewidth = 1.0
    config.spine.color = '#333333'
    config.spine.grid_visible = False

    # ----- 图例 -----
    config.legend.show = True
    config.legend.position = 'right'  # 'right', 'bottom'
    config.legend.title = 'Fault Types'
    config.legend.marker_size = 8

    # ----- 标题 -----
    config.title.show_suptitle = True
    config.title.row_labels = ['Teacher', 'CTSL']
    config.title.auto_title_format = '{wc} ({tag})'  # 自动标题格式
    config.title.train_tag = 'Train'
    config.title.unseen_tag = 'Unseen'
    config.title.source_tag = 'Source'
    config.title.target_tag = 'Target'

    # 自定义每个子图标题（可选，设为None则自动生成）
    # config.title.subplot_titles = [
    #     ['WC1 (Train)', 'WC2 (Unseen)', 'WC3 (Unseen)', 'WC4 (Unseen)'],  # 第一行
    #     ['WC1 (Source)', 'WC2 (Target)', 'WC3 (Target)', 'WC4 (Unseen)']   # 第二行
    # ]

    # ----- 类别标签 -----
    # config.labels.class_names = {
    #     0: 'Normal', 1: 'IF', 2: 'OF', 3: 'BF',
    #     4: 'IF+OF', 5: 'IF+BF', 6: 'OF+BF', 7: 'IF+OF+BF'
    # }
    config.labels.default_format = 'Class {}'

    # =========================================================================

    plot_comparison_grid(
        teacher_embeddings_list,
        mcid_embeddings_list,
        labels_list,
        wcs,
        save_path,
        teacher_train_wc=teacher_train_wc,
        mcid_source_wc=mcid_source_wc,
        mcid_target_wcs=mcid_target_wcs,
        config=config
    )

    print("\n[Done] Visualization complete!")


if __name__ == "__main__":
    main()
