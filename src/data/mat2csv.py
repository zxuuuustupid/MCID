import os
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
import traceback

# ================= 配置区域 =================
# 核心过滤：只提取 64kHz 的高频信号（约 25.6万点）
# 探测发现你的数据是 256823 点，所以设置 200000 是安全的
MIN_DATA_LENGTH = 200000
# 目标信号名称
TARGET_SIGNAL_NAME = 'vibration_1'
VIBRATION_KEYWORDS = ['vibration', 'vibration_1', 'vib_1', 'acc']
# ===========================================

def mat_to_csv_paderborn(source_root, target_root):
    """
    针对 Paderborn University 轴承数据集优化的 MAT 转 CSV 工具
    """
    source_path = Path(source_root)
    target_path = Path(target_root)
    target_path.mkdir(parents=True, exist_ok=True)

    # 递归获取所有 .mat 文件
    mat_files = list(source_path.rglob('*.mat'))

    print(f"🚀 找到 {len(mat_files)} 个 MAT 文件，准备开始精准提取...")

    success_count = 0
    fail_count = 0
    skip_count = 0

    for mat_file in mat_files:
        try:
            # 保持原始文件夹层级结构
            relative_path = mat_file.relative_to(source_path)
            target_csv = target_path / relative_path.with_suffix('.csv')
            target_csv.parent.mkdir(parents=True, exist_ok=True)

            base_name = mat_file.stem  # 获取文件名（不含后缀）

            # 1. 加载 MAT 文件
            # 使用 struct_as_record=False 方便通过 . 访问属性
            mat_data = sio.loadmat(str(mat_file), struct_as_record=False, squeeze_me=True)

            # 2. 提取信号
            signal_data = None

            # PU 数据集的根变量通常与文件名一致
            if base_name in mat_data:
                root_obj = mat_data[base_name]
                signal_data = extract_vibration_from_pu_struct(root_obj)
            else:
                # 如果文件名不是键名，尝试寻找第一个非系统变量
                for k in mat_data.keys():
                    if not k.startswith('__'):
                        signal_data = extract_vibration_from_pu_struct(mat_data[k])
                        if signal_data is not None: break

            # 3. 校验并转换
            if signal_data is not None:
                # 扁平化处理
                signal_data = signal_data.flatten()

                # 长度过滤：过滤掉 1.6万点的低频信号（force/torque等）
                if signal_data.size < MIN_DATA_LENGTH:
                    print(f"  ⚠️ 跳过 {base_name}: 长度不足 ({signal_data.size} 点，疑似非振动信号)")
                    skip_count += 1
                    continue

                # 转换为 DataFrame 并保存
                df = pd.DataFrame(signal_data, columns=['vibration_signal'])

                # 处理 NaN（如果存在）
                if df['vibration_signal'].isnull().any():
                    df['vibration_signal'] = df['vibration_signal'].ffill()

                df.to_csv(target_csv, index=False)
                print(f"  ✓ 成功: {base_name} (Length: {signal_data.size})")
                success_count += 1
            else:
                print(f"  ✗ 失败: {base_name} 未找到名为 '{TARGET_SIGNAL_NAME}' 的高频信号")
                fail_count += 1

        except Exception as e:
            print(f"  ✗ 严重错误 {mat_file.name}: {str(e)}")
            # traceback.print_exc() # 如果需要详细错误日志可取消注释
            fail_count += 1

    print("\n" + "=" * 60)
    print("✨ 处理总结:")
    print(f"  - 成功转换 (高频振动): {success_count}")
    print(f"  - 长度不足跳过 (低频干扰): {skip_count}")
    print(f"  - 提取失败: {fail_count}")
    print(f"  - 保存根目录: {target_root}")
    print("=" * 60)

def extract_vibration_from_pu_struct(struct_obj):
    """
    针对 PU 数据集 Y 字段数组设计的提取逻辑
    """
    # 策略 1: 遍历 Y 数组（探测发现振动信号在此处）
    if hasattr(struct_obj, 'Y'):
        y_fields = struct_obj.Y
        # 判断 Y 是否为数组（PU 数据集 Y 通常包含 7 个传感器对象）
        if isinstance(y_fields, np.ndarray):
            for sensor in y_fields:
                # 检查 Name 属性是否匹配
                sensor_name = getattr(sensor, 'Name', '').lower()
                if any(k == sensor_name for k in VIBRATION_KEYWORDS):
                    if hasattr(sensor, 'Data'):
                        return sensor.Data
        # 如果 Y 不是数组而是单个数（极少见）
        elif hasattr(y_fields, 'Name') and TARGET_SIGNAL_NAME in y_fields.Name:
            return getattr(y_fields, 'Data', None)

    # 策略 2: 备份方案 - 遍历 X 字段
    if hasattr(struct_obj, 'X'):
        x_fields = struct_obj.X
        if isinstance(x_fields, np.ndarray):
            for item in x_fields:
                if hasattr(item, 'Name') and any(k in str(item.Name).lower() for k in VIBRATION_KEYWORDS):
                    return getattr(item, 'Data', None)

    return None

if __name__ == "__main__":
    # === 路径配置 ===
    SOURCE_DIR = r'F:\Project\mid\德国数据集\领域泛化\PUdata_1'
    TARGET_DIR = r'F:\Project\mid\德国数据集\领域泛化\PUdata_1_csv'

    print("--- Paderborn University (PU) Precision Extraction Tool ---")
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ 错误: 找不到源目录 {SOURCE_DIR}")
    else:
        mat_to_csv_paderborn(SOURCE_DIR, TARGET_DIR)
