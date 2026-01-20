# import os
# import numpy as np
# import pandas as pd
# import scipy.io as sio
# from pathlib import Path

# # ================= 配置区域 =================
# # 1. 根目录配置
# # 指向包含 balance 等多个故障文件夹的那个大文件夹
# SOURCE_ROOT = r'F:\Project\mid\Fed'
# TARGET_ROOT = r'F:\Project\mid\Fed_all_csv'

# # 2. 提取参数
# # 14个通道中的第5个（索引为 4）
# # TARGET_CHANNEL_IDX = 4
# # 提取 4, 5, 6 通道（对应索引 3, 4, 5）
# TARGET_CHANNELS = slice(3, 6)


# DATA_KEY = 'Datas'

# # 3. 过滤参数
# MIN_DATA_LENGTH = 100000 # 根据实际情况调整，防止处理空文件
# # ===========================================

# def convert_all_fed_data(source_root, target_root):
#     """
#     全量递归处理 Fed 目录下所有子文件夹中的 MAT 文件
#     """
#     source_path = Path(source_root)
#     target_path = Path(target_root)

#     # 获取所有 .mat 文件（递归搜索所有子目录）
#     mat_files = list(source_path.rglob('*.mat'))

#     print(f"🚀 开始全量处理...")
#     print(f"📂 源根目录: {source_root}")
#     print(f"📂 目标根目录: {target_root}")
#     print(f"📊 待处理文件总数: {len(mat_files)}")
#     # print(f"🎯 提取通道: 第 {TARGET_CHANNEL_IDX + 1} 通道")
#     print("-" * 60)

#     success_count = 0
#     fail_count = 0
#     skip_count = 0

#     for mat_file in mat_files:
#         try:
#             # --- 1. 构建目标路径（保持原有文件夹结构） ---
#             # 例如: Fed/balance/file1.mat -> Fed_all_csv/balance/file1.csv
#             relative_path = mat_file.relative_to(source_path)
#             target_csv = target_path / relative_path.with_suffix('.csv')

#             # 创建目标子文件夹
#             target_csv.parent.mkdir(parents=True, exist_ok=True)

#             # --- 2. 加载数据 ---
#             # 这种格式通常直接 load 即可，不需要 squeeze_me 也可以
#             mat_data = sio.loadmat(str(mat_file))

#             # --- 3. 提取指定通道 ---
#             if DATA_KEY in mat_data:
#                 full_matrix = mat_data[DATA_KEY]

#                 # 确保是二维矩阵且列数足够
#                 if full_matrix.ndim == 2 and full_matrix.shape[1] > TARGET_CHANNEL_IDX:
#                     # signal = full_matrix[:, TARGET_CHANNEL_IDX]
#                     signal = full_matrix[:, TARGET_CHANNELS]


#                     # 长度过滤
#                     if signal.size < MIN_DATA_LENGTH:
#                         print(f"  ⚠️ 跳过: {relative_path} (长度 {signal.size} 过短)")
#                         skip_count += 1
#                         continue

#                     # --- 4. 保存数据 ---
#                     # 修改前：df = pd.DataFrame(signal, columns=['vibration_signal'])
#                     # 修改后：
#                     df = pd.DataFrame(signal, columns=['vibration_ch4', 'vibration_ch5', 'vibration_ch6'])

#                     # 缺失值简单处理（前向填充）
#                     if df['vibration_signal'].isnull().any():
#                         df['vibration_signal'] = df.fillna(method='ffill')

#                     df.to_csv(target_csv, index=False)
#                     print(f"  ✓ 成功: {relative_path} ({signal.size} 点)")
#                     success_count += 1
#                 else:
#                     print(f"  ✗ 失败: {relative_path} (矩阵形状 {full_matrix.shape} 不符)")
#                     fail_count += 1
#             else:
#                 print(f"  ✗ 失败: {relative_path} (未找到 '{DATA_KEY}' 变量)")
#                 fail_count += 1

#         except Exception as e:
#             print(f"  ❌ 严重错误: {mat_file.name} -> {str(e)}")
#             fail_count += 1

#     # --- 总结报告 ---
#     print("\n" + "=" * 60)
#     print("✨ 处理总结报告:")
#     print(f"  - 成功转换文件: {success_count}")
#     print(f"  - 提取失败文件: {fail_count}")
#     print(f"  - 长度不足跳过: {skip_count}")
#     print(f"  - 数据保存在: {target_root}")
#     print("=" * 60)

# if __name__ == "__main__":
#     if not os.path.exists(SOURCE_ROOT):
#         print(f"❌ 错误: 找不到源目录 '{SOURCE_ROOT}'，请检查路径。")
#     else:
#         convert_all_fed_data(SOURCE_ROOT, TARGET_ROOT)



import os
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path

# ================= 配置区域 =================
SOURCE_ROOT = r'F:\Project\mid\Fed'
TARGET_ROOT = r'F:\Project\mid\Fed_all_csv_3ch'

# 提取 Matlab 的 4, 5, 6 通道 -> 对应 Python 索引 [3, 4, 5]
TARGET_CHANNELS = [3, 4, 5]
DATA_KEY = 'Datas'

MIN_DATA_LENGTH = 100000
# ===========================================

def convert_all_fed_data(source_root, target_root):
    source_path = Path(source_root)
    target_path = Path(target_root)

    mat_files = list(source_path.rglob('*.mat'))

    print(f"🚀 开始全量处理 (4-5-6三通道模式)...目标列索引: {TARGET_CHANNELS}")

    success_count = 0
    fail_count = 0

    for mat_file in mat_files:
        try:
            relative_path = mat_file.relative_to(source_path)
            target_csv = target_path / relative_path.with_suffix('.csv')
            target_csv.parent.mkdir(parents=True, exist_ok=True)

            mat_data = sio.loadmat(str(mat_file))

            if DATA_KEY in mat_data:
                full_matrix = mat_data[DATA_KEY]

                # 确保是二维矩阵且列数至少有 6 列 (这样才能取到索引 5)
                if full_matrix.ndim == 2 and full_matrix.shape[1] >= 6:
                    # 核心修改：一次性切出 3 列
                    signal = full_matrix[:, TARGET_CHANNELS]

                    # 保存为三列 CSV
                    df = pd.DataFrame(signal, columns=['ch4', 'ch5', 'ch6'])

                    if df.isnull().any().any():
                        df = df.fillna(method='ffill')

                    df.to_csv(target_csv, index=False)
                    print(f"  ✓ 成功: {relative_path}")
                    success_count += 1
                else:
                    print(f"  ✗ 失败: {relative_path} (列数不足 {full_matrix.shape[1]})")
                    fail_count += 1
            else:
                print(f"  ✗ 失败: {relative_path} (未找到 Datas)")
                fail_count += 1

        except Exception as e:
            print(f"  ❌ 严重错误: {mat_file.name} -> {str(e)}")
            fail_count += 1

    print(f"\n✨ 处理完成！成功: {success_count}, 失败: {fail_count}")

if __name__ == "__main__":
    convert_all_fed_data(SOURCE_ROOT, TARGET_ROOT)
