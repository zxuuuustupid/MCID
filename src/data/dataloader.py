import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class NpyDataset(Dataset):
    def __init__(self, folder_path):
        """
        读取指定文件夹下所有的 .npy 文件
        假设文件名即标签 (例如: "0.npy" -> label=0, "1.npy" -> label=1)
        """
        self.data_list = []
        self.label_list = []

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"找不到文件夹: {folder_path}")

        files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        if not files:
            raise RuntimeError(f"文件夹 {folder_path} 里没有 .npy 文件")

        # print(f"正在加载 {folder_path} ...")

        for f in files:
            # === 核心逻辑：文件名转标签 ===
            try:
                # 去掉 .npy 后缀，直接转 int
                label_idx = int(os.path.splitext(f)[0])
            except ValueError:
                print(f"警告: 文件名 {f} 不是数字，无法作为标签，已跳过。")
                continue

            file_path = os.path.join(folder_path, f)

            # === 读取数据 ===
            # data shape: (N, C, L)
            data = np.load(file_path)

            # 强转 float32
            if data.dtype != np.float32:
                data = data.astype(np.float32)

            self.data_list.append(torch.from_numpy(data))

            # 生成对应的标签 Tensor
            # 如果 data 是 (100, 1, 1024), 那么就需要 100 个 label_idx
            num_samples = data.shape[0]
            labels = torch.full((num_samples,), label_idx, dtype=torch.long)
            self.label_list.append(labels)

        # === 拼接所有文件的数据 ===
        if self.data_list:
            self.x = torch.cat(self.data_list, dim=0) # 拼接样本维度
            self.y = torch.cat(self.label_list, dim=0)
            # print(f"加载完成 -> 数据: {self.x.shape}, 标签: {self.y.shape}")
        else:
            raise RuntimeError("没有加载到任何有效数据！")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# === 外部调用的接口 ===
def get_dataloader(path, batch_size=32, shuffle=True):
    dataset = NpyDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True
    )
