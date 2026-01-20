# import os
# import glob
# import pandas as pd
# import numpy as np
# import random

# # ================= 核心配置区域 =================

# # 1. 刚才 Fed 数据转换后的 CSV 根目录
# RAW_DATA_ROOT = r"F:\Project\mid\Fed_all_csv_3ch"
# # 2. 处理后的时域样本保存目录
# OUTPUT_ROOT = r"F:\Project\mid\S-MID\data\Fed"

# # 3. 故障类型映射 (根据你的文件夹名定义)
# FAULT_TYPE_MAP = {
#     "Normal": 0,    # 正常
#     "balance": 1,   # 这种故障
#     "left": 2,      # 这种故障
#     "composite": 3  # 复合故障
# }

# # 4. 工况映射 (根据文件名中的速度数字识别)
# # 250 -> WC1, 300 -> WC2, 350 -> WC3, 400 -> WC4
# WC_IDENTIFIER = {
#     "250": "WC1",
#     "300": "WC2",
#     "350": "WC3",
#     "400": "WC4"
# }

# # 5. 样本参数
# TRAIN_NUM = 1000
# TEST_NUM = 200
# WINDOW_SIZE = 1024
# OVERLAP_RATIO = 0.85
# STRIDE = int(WINDOW_SIZE * (1 - OVERLAP_RATIO))

# # ================= 信号处理 =================

# def z_score_norm(sig):
#     """ 去直流 + 标准化 """
#     sig = sig - np.mean(sig)
#     std = np.std(sig)
#     return sig / std if std > 1e-6 else sig

# # ================= 主逻辑 =================

# def main():
#     print(f"🚀 开始制作 Fed 数据集 (时域版)...")

#     # 遍历故障文件夹 (Normal, balance, left, composite)
#     fault_folders = [d for d in os.listdir(RAW_DATA_ROOT) if os.path.isdir(os.path.join(RAW_DATA_ROOT, d))]

#     for fault_name in fault_folders:
#         if fault_name not in FAULT_TYPE_MAP:
#             print(f"  ⏭️ 跳过未定义故障文件夹: {fault_name}")
#             continue

#         label = FAULT_TYPE_MAP[fault_name]
#         fault_path = os.path.join(RAW_DATA_ROOT, fault_name)

#         # 获取该故障下所有的 CSV 文件
#         csv_files = glob.glob(os.path.join(fault_path, "*.csv"))

#         for csv_file in csv_files:
#             file_name = os.path.basename(csv_file)

#             # --- 识别工况 (WC) ---
#             wc_alias = None
#             for key, val in WC_IDENTIFIER.items():
#                 if key in file_name:
#                     wc_alias = val
#                     break

#             if wc_alias is None:
#                 print(f"  ⚠️ 无法识别工况，跳过文件: {file_name}")
#                 continue

#             print(f"📂 处理: {fault_name} | 工况: {wc_alias} | 文件: {file_name}")

#             # --- 建立输出目录 ---
#             train_path = os.path.join(OUTPUT_ROOT, wc_alias, "train")
#             test_path = os.path.join(OUTPUT_ROOT, wc_alias, "test")
#             os.makedirs(train_path, exist_ok=True)
#             os.makedirs(test_path, exist_ok=True)

#             try:
#                 # 1. 读取数据
#                 df = pd.read_csv(csv_file, usecols=[0])
#                 raw_signal = df.values.flatten().astype(np.float32)

#                 # 2. 滑动窗口切分
#                 samples = []
#                 n_points = len(raw_signal)
#                 for start in range(0, n_points - WINDOW_SIZE, STRIDE):
#                     segment = raw_signal[start : start + WINDOW_SIZE]
#                     processed_seg = z_score_norm(segment)
#                     samples.append(processed_seg.reshape(1, -1))

#                 samples = np.array(samples)

#                 # 3. 随机筛选训练集和测试集
#                 if len(samples) < (TRAIN_NUM + TEST_NUM):
#                     print(f"     ⚠️ 样本不足 ({len(samples)}), 将按比例分配")
#                     actual_train = int(len(samples) * 0.8)
#                     actual_test = len(samples) - actual_train
#                 else:
#                     actual_train = TRAIN_NUM
#                     actual_test = TEST_NUM

#                 # 打乱
#                 np.random.seed(42)
#                 indices = np.arange(len(samples))
#                 np.random.shuffle(indices)

#                 final_train = samples[indices[:actual_train]]
#                 final_test = samples[indices[actual_train : actual_train + actual_test]]

#                 # 4. 保存为 npy (文件名即标签)
#                 np.save(os.path.join(train_path, f"{label}.npy"), final_train)
#                 np.save(os.path.join(test_path, f"{label}.npy"), final_test)

#                 print(f"     ✅ 完成: 提取 {len(samples)} 样本 -> Train:{actual_train}, Test:{actual_test}")

#             except Exception as e:
#                 print(f"     ❌ 处理出错: {e}")

#     print(f"\n✨ Fed 数据集预处理全部完成！")
#     print(f"📍 输出路径: {OUTPUT_ROOT}")

# if __name__ == "__main__":
#     main()




import os
import glob
import pandas as pd
import numpy as np
import random

# ================= 核心配置区域 =================
RAW_DATA_ROOT = r"F:\Project\mid\Fed_all_csv_3ch"
OUTPUT_ROOT = r"F:\Project\mid\S-MID\data\Fed"

FAULT_TYPE_MAP = {
    "Normal": 0,
    "balance": 1,
    "left": 2,
    "composite": 3
}

WC_IDENTIFIER = {
    "250": "WC1",
    "300": "WC2",
    "350": "WC3",
    "400": "WC4"
}

TRAIN_NUM = 1000
TEST_NUM = 200
WINDOW_SIZE = 1024
OVERLAP_RATIO = 0.85
STRIDE = int(WINDOW_SIZE * (1 - OVERLAP_RATIO))

# ================= 信号处理 (支持多通道) =================

def z_score_norm_3ch(sig_3ch):
    """
    对三通道数据分别进行标准化
    输入维度: (1024, 3) -> 输出维度: (3, 1024)
    """
    # 每一列(通道)独立计算均值和标准差
    means = np.mean(sig_3ch, axis=0)
    stds = np.std(sig_3ch, axis=0)

    # 标准化
    normed = (sig_3ch - means) / (stds + 1e-6)

    # 转置成 (通道数, 长度)，即 (3, 1024)，符合深度学习输入习惯
    return normed.T

# ================= 主逻辑 =================

def main():
    print(f"🚀 开始制作 Fed 数据集 (时域 3通道版)...")

    fault_folders = [d for d in os.listdir(RAW_DATA_ROOT) if os.path.isdir(os.path.join(RAW_DATA_ROOT, d))]

    for fault_name in fault_folders:
        if fault_name not in FAULT_TYPE_MAP:
            continue

        label = FAULT_TYPE_MAP[fault_name]
        fault_path = os.path.join(RAW_DATA_ROOT, fault_name)
        csv_files = glob.glob(os.path.join(fault_path, "*.csv"))

        for csv_file in csv_files:
            file_name = os.path.basename(csv_file)

            wc_alias = None
            for key, val in WC_IDENTIFIER.items():
                if key in file_name:
                    wc_alias = val
                    break

            if wc_alias is None: continue

            print(f"📂 处理: {fault_name} | 工况: {wc_alias} | 文件: {file_name}")

            train_path = os.path.join(OUTPUT_ROOT, wc_alias, "train")
            test_path = os.path.join(OUTPUT_ROOT, wc_alias, "test")
            os.makedirs(train_path, exist_ok=True)
            os.makedirs(test_path, exist_ok=True)

            try:
                # --- 修改 1: 读取全部 3 列 ---
                df = pd.read_csv(csv_file) # 不再指定 usecols=[0]
                # 转换为 numpy，形状应该是 (总长度, 3)
                raw_signal = df.values.astype(np.float32)

                # --- 修改 2: 滑动窗口切分 (保持 2D 形状) ---
                samples = []
                n_points = raw_signal.shape[0]
                for start in range(0, n_points - WINDOW_SIZE, STRIDE):
                    # 此时 segment 形状是 (1024, 3)
                    segment = raw_signal[start : start + WINDOW_SIZE, :]

                    # 标准化并转置，得到 (3, 1024)
                    processed_seg = z_score_norm_3ch(segment)
                    samples.append(processed_seg)

                samples = np.array(samples) # 最终形状: (样本数, 3, 1024)

                if len(samples) < (TRAIN_NUM + TEST_NUM):
                    actual_train = int(len(samples) * 0.8)
                    actual_test = len(samples) - actual_train
                else:
                    actual_train = TRAIN_NUM
                    actual_test = TEST_NUM

                np.random.seed(42)
                indices = np.arange(len(samples))
                np.random.shuffle(indices)

                final_train = samples[indices[:actual_train]]
                final_test = samples[indices[actual_train : actual_train + actual_test]]

                # 保存
                np.save(os.path.join(train_path, f"{label}.npy"), final_train)
                np.save(os.path.join(test_path, f"{label}.npy"), final_test)

                print(f"     ✅ 完成: 样本形状 {final_train.shape} (N, Channel, Width)")

            except Exception as e:
                print(f"     ❌ 处理出错: {e}")

    print(f"\n✨ Fed 3通道数据处理完成！")

if __name__ == "__main__":
    main()
