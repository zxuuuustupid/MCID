import os
import glob
import re
import pandas as pd
import numpy as np
from scipy.fftpack import fft

# ================= 核心配置区域 =================

# 数据集根目录 (修改为你提供的路径)
RAW_DATA_ROOT = r"F:\Project\TableGPT\data\轴箱轴承"

# 输出保存路径
OUTPUT_ROOT = r"F:\Project\mid\S-MID\data\SWJTU"

# 故障文件夹到标签的映射 (1-H -> 0, 2-IF -> 1, ...)
# 文件夹名: 标签 Label
FOLDER_TO_LABEL = {
    "1-H": 0,   # 健康
    "2-IF": 1,  # 内圈故障
    "3-OF": 2,  # 外圈故障
    "4-BF": 3,  # 滚动体故障
    "5-CF": 4   # 保持架故障 (假设CF是Cage Fault)
}

# 训练配置
TRAIN_NUM = 1000
TEST_NUM = 200
WINDOW_SIZE = 2048
OVERLAP_RATIO = 0.9
STRIDE = int(WINDOW_SIZE * (1 - OVERLAP_RATIO))

# ================= 信号处理模块 (保持不变) =================

def advanced_signal_process(sample):
    """
    输入: (Channels, Window_Size)
    处理: Z-score -> FFT -> Log -> MinMax
    输出: (Channels, Window_Size)
    """
    processed_sample = []

    for ch in range(sample.shape[0]):
        sig = sample[ch, :]

        # 1. 样本内归一化 (Z-score)
        if np.std(sig) == 0:
            sig = np.zeros_like(sig)
        else:
            sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)

        # 2. 傅里叶变换 (取模)
        fft_res = fft(sig)
        mag = np.abs(fft_res)

        # 3. 对数压缩
        mag = np.log1p(mag)

        # 4. 再次标准化 (Min-Max 到 0-1)
        if np.max(mag) - np.min(mag) == 0:
            mag = np.zeros_like(mag)
        else:
            mag = (mag - np.min(mag)) / (np.max(mag) - np.min(mag) + 1e-6)

        processed_sample.append(mag)

    return np.array(processed_sample)

# ================= 切片逻辑 (保持不变) =================

def sliding_window_with_process(data_matrix, window_size, stride):
    n_channels, n_points = data_matrix.shape
    if n_points < window_size:
        return np.array([])

    n_samples = (n_points - window_size) // stride + 1
    samples = []

    for i in range(n_samples):
        start = i * stride
        end = start + window_size
        slice_data = data_matrix[:, start:end]

        # 信号处理
        processed_data = advanced_signal_process(slice_data)
        samples.append(processed_data)

    return np.array(samples)

def process_one_file(file_path):
    """读取 Excel 文件 (只取最后一个 Sheet)"""
    try:
        # --- 修改开始 ---
        # 1. 使用 ExcelFile 加载文件结构，不直接读取数据
        xl = pd.ExcelFile(file_path, engine='openpyxl')

        # 2. 获取最后一个 Sheet 的名称
        last_sheet_name = xl.sheet_names[-1]

        # 3. 指定 sheet_name 读取数据
        df = pd.read_excel(file_path, sheet_name=last_sheet_name, header=0, engine='openpyxl',usecols="A:C")
        # --- 修改结束 ---

        # 转置：变成 (Channels, Time_Steps)
        data = df.values.astype(np.float32).T
        return data
    except Exception as e:
        print(f"    [读取失败] {os.path.basename(file_path)}: {e}")
        return None

def extract_wc_from_filename(filename):
    """
    从文件名解析工况 (WC)
    例如: '20201104-1振动数据.xlsx' -> 提取出 1
    """
    # 正则表达式寻找 '-' 和 '振动数据' 之间的数字
    match = re.search(r'-(\d+)振动数据', filename)
    if match:
        return int(match.group(1))
    return None

# ================= 主逻辑部分 (保持不变) =================

def main():
    print(f"开始处理轴箱轴承数据... 输出路径: {OUTPUT_ROOT}")

    # 1. 遍历每一个故障类型的文件夹 (1-H, 2-IF...)
    for folder_name, label in FOLDER_TO_LABEL.items():
        folder_path = os.path.join(RAW_DATA_ROOT, folder_name)

        # if label != 4:
        #     continue

        if not os.path.exists(folder_path):
            print(f"[警告] 文件夹不存在: {folder_path}")
            continue

        # 获取该文件夹下所有的 xlsx 文件
        xlsx_files = glob.glob(os.path.join(folder_path, "*.xlsx"))

        print(f"\n正在处理文件夹: {folder_name} (Label: {label}), 发现文件数: {len(xlsx_files)}")

        # 2. 遍历该文件夹下的每一个文件
        for file_path in xlsx_files:
            filename = os.path.basename(file_path)

            # 从文件名解析工况 (WC1, WC2...)
            wc_idx = extract_wc_from_filename(filename)

            if wc_idx is None:
                print(f"  [跳过] 无法从文件名解析工况: {filename}")
                continue

            # 3. 准备输出目录: OUTPUT/WC{i}/train/
            wc_name = f"WC{wc_idx}"
            train_dir = os.path.join(OUTPUT_ROOT, wc_name, "train")
            test_dir = os.path.join(OUTPUT_ROOT, wc_name, "test")
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            # 4. 读取数据
            print(f"  -> 读取文件: {filename} (工况: {wc_name})")
            raw_data = process_one_file(file_path)

            if raw_data is None:
                continue

            # 5. 切片与处理
            samples = sliding_window_with_process(raw_data, WINDOW_SIZE, STRIDE)

            total_needed = TRAIN_NUM + TEST_NUM
            if len(samples) < total_needed:
                print(f"     [警告] 样本不足! 产生: {len(samples)}, 需要: {total_needed}")
                # 这种情况下你可以选择跳过，或者有多少存多少
                # continue

            # 打乱数据
            np.random.seed(42)
            np.random.shuffle(samples)

            # 划分训练集和测试集
            # 注意：如果样本不足，这里会自动截断，不会报错，但数量会少
            final_train = samples[:TRAIN_NUM]
            final_test = samples[TRAIN_NUM : TRAIN_NUM + TEST_NUM]

            # 6. 保存为 .npy
            # 文件名直接用标签命名，如 0.npy, 1.npy
            train_save_path = os.path.join(train_dir, f"{label}.npy")
            test_save_path = os.path.join(test_dir, f"{label}.npy")

            np.save(train_save_path, final_train)
            np.save(test_save_path, final_test)

            print(f"     [完成] 保存 Label {label} 到 {wc_name} | Train: {final_train.shape}")

if __name__ == "__main__":
    main()
