import os
import glob
import pandas as pd
import numpy as np
import random

# ================= 核心配置区域 =================

# CSV 根目录（刚才转换好的结果）
RAW_DATA_ROOT = r"F:\Project\mid\德国数据集\领域泛化\PUdata_1_csv"
# 处理后的时域样本保存目录
OUTPUT_ROOT = r"F:\Project\mid\S-MID\data\PU"

# 工况映射：如果文件夹名是 900_7_1000，程序会把它归类为 WC1，依此类推
WC_DIR_MAP = {
    "900_7_1000": "WC2",
    "1500_7_1000": "WC1",
    "1500_1_1000": "WC3",
    "1500_7_400": "WC4"
}

# 故意弄少的故障类型映射（仅保留这 8 类）
FAULT_TYPE_MAP = {
    "K001": 0,  # 正常状态
    "KA15": 1,  # 内圈故障
    "KA04": 2,  # 内圈故障
    "KI18": 3,  # 外圈故障
    "KI21": 4,  # 外圈故障
    "KB27": 5,  # 滚动体故障
    "KB23": 6,  # 滚动体故障
    "KB24": 7,  # 滚动体故障
}

# 样本参数
TRAIN_NUM = 1000
TEST_NUM = 200
WINDOW_SIZE = 1024
OVERLAP_RATIO = 0.8  # 0.8 的重叠率足以从 25.6万点中切出 >1200 个样本
STRIDE = int(WINDOW_SIZE * (1 - OVERLAP_RATIO))

# ================= 信号处理 =================

def z_score_norm(sig):
    """ 去直流 + 标准化 """
    sig = sig - np.mean(sig)
    std = np.std(sig)
    return sig / std if std > 1e-6 else sig

# ================= 主逻辑 =================

def main():
    print(f"🚀 开始制作 Paderborn 精简版数据集...")

    # 获取 RAW_DATA_ROOT 下所有的工况文件夹
    all_wc_dirs = [d for d in os.listdir(RAW_DATA_ROOT) if os.path.isdir(os.path.join(RAW_DATA_ROOT, d))]

    for wc_dir in all_wc_dirs:
        # 确定输出的工况别名
        wc_alias = WC_DIR_MAP.get(wc_dir, wc_dir)
        wc_path = os.path.join(RAW_DATA_ROOT, wc_dir)

        print(f"\n📂 处理工况: {wc_dir} -> {wc_alias}")

        # 建立输出目录
        train_path = os.path.join(OUTPUT_ROOT, wc_alias, "train")
        test_path = os.path.join(OUTPUT_ROOT, wc_alias, "test")
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        # 只遍历我们在 FAULT_TYPE_MAP 中定义的故障
        for fault_code, label in FAULT_TYPE_MAP.items():
            # 这里的 fault_code 对应文件夹名，如 K001, KA15...
            target_folder = os.path.join(wc_path, fault_code)

            if not os.path.exists(target_folder):
                continue

            # 找到文件夹下唯一的 CSV 文件
            csv_files = glob.glob(os.path.join(target_folder, "*.csv"))
            if not csv_files:
                continue

            csv_file = csv_files[0] # 取第一个（通常也就一个）

            try:
                # 1. 读取数据（第一列是振动信号）
                df = pd.read_csv(csv_file, usecols=[0])
                raw_signal = df.values.flatten().astype(np.float32)

                # 2. 滑动窗口切分
                samples = []
                n_points = len(raw_signal)
                for start in range(0, n_points - WINDOW_SIZE, STRIDE):
                    segment = raw_signal[start : start + WINDOW_SIZE]
                    # 处理信号（去直流+标准化）
                    processed_seg = z_score_norm(segment)
                    # 增加维度变成 (1, 1024) 对应 (Channel, Length)
                    samples.append(processed_seg.reshape(1, -1))

                samples = np.array(samples)

                # 3. 随机筛选训练集和测试集
                if len(samples) < (TRAIN_NUM + TEST_NUM):
                    print(f"   ⚠️ {fault_code}: 样本数不足 ({len(samples)}), 将按比例分配")
                    actual_train = int(len(samples) * 0.8)
                    actual_test = len(samples) - actual_train
                else:
                    actual_train = TRAIN_NUM
                    actual_test = TEST_NUM

                # 打乱
                np.random.seed(42)
                indices = np.arange(len(samples))
                np.random.shuffle(indices)

                train_data = samples[indices[:actual_train]]
                test_data = samples[indices[actual_train : actual_train + actual_test]]

                # 4. 保存为 npy
                np.save(os.path.join(train_path, f"{label}.npy"), train_data)
                np.save(os.path.join(test_path, f"{label}.npy"), test_data)

                print(f"   ✅ {fault_code} (Label {label}): 切分出 {len(samples)} 个样本 -> 提取 Train:{actual_train}, Test:{actual_test}")

            except Exception as e:
                print(f"   ❌ 处理 {csv_file} 出错: {e}")

    print(f"\n✨ 数据预处理全部完成！")
    print(f"📍 输出路径: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
