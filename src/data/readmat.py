# import scipy.io as sio

# file_path = r'F:\Project\mid\德国数据集\领域泛化\PUdata_1\900_7_1000\K001\N09_M07_F10_K001_1.mat'
# mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
# root_key = [k for k in mat.keys() if not k.startswith('__')][0]
# root_obj = mat[root_key]

# print(f"--- 文件 {root_key} 的所有信号清单 ---")
# for i, sensor in enumerate(root_obj.Y):
#     name = sensor.Name
#     length = sensor.Data.size
#     raster = sensor.Raster
#     print(f"通道 [{i}] | 信号: {name:15} | 点数: {length:8} | 采样率: {raster}")



import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def analyze_mat_detailed(file_path):
    if not isinstance(file_path, str):
        print("❌ 路径格式错误")
        return

    print(f"🔍 正在深度解析文件: {file_path}")
    print("=" * 70)

    try:
        # 加载 MAT 文件
        # struct_as_record=False 使结构体像对象一样访问
        # squeeze_me=True 移除多余的维度
        data = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)

        # 获取所有非系统变量
        keys = [k for k in data.keys() if not k.startswith('__')]
        print(f"📁 根目录下发现变量: {keys}")

        for key in keys:
            print(f"\n🏷️  变量名: [ {key} ]")
            _inspect_object(data[key], indent=1)

        # 尝试可视化（如果变量里包含大型数值数组）
        _plot_potential_signal(data, keys)

    except Exception as e:
        print(f"❌ 解析过程中发生错误: {e}")

def _inspect_object(obj, indent=0):
    spacing = "  " * indent

    # 情况 A: 结构体 (Struct)
    if hasattr(obj, '_fieldnames'):
        print(f"{spacing}📂 类型: Matlab Struct")
        print(f"{spacing}📝 包含字段: {obj._fieldnames}")
        for field in obj._fieldnames:
            val = getattr(obj, field)
            print(f"{spacing}└── 字段: {field}")
            _inspect_object(val, indent + 2)

    # 情况 B: Numpy 数组 (数值数据)
    elif isinstance(obj, np.ndarray):
        if obj.dtype == 'O':
            print(f"{spacing}📦 类型: 对象数组 (Cell/Object Array), 长度: {obj.size}")
            if obj.size > 0:
                _inspect_object(obj.flat[0], indent + 2)
        else:
            print(f"{spacing}📊 类型: 数值矩阵 ({obj.dtype}), 形状: {obj.shape}")
            if obj.size > 0:
                print(f"{spacing}   📈 统计: Max={np.max(obj):.4f}, Min={np.min(obj):.4f}, Mean={np.mean(obj):.4f}")
                if obj.size > 3:
                    print(f"{spacing}   🔢 预览: {obj.flat[:5]} ...")

    # 情况 C: 标量/字符串
    else:
        print(f"{spacing}📄 类型: {type(obj).__name__}, 值: {obj}")

def _plot_potential_signal(data_dict, keys):
    """如果发现大型一维数组，自动绘制前1000个点观察波形"""
    for key in keys:
        val = data_dict[key]
        # 如果是数值型数组且点数较多
        if isinstance(val, np.ndarray) and val.dtype != 'O' and val.size > 500:
            signal = val.flatten()
            plt.figure(figsize=(12, 4))
            plt.plot(signal[:1000])
            plt.title(f"Signal Preview: {key} (First 1000 points)")
            plt.xlabel("Points")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.show()
            break # 只画第一个找到的信号

if __name__ == "__main__":
    TARGET_FILE = r"F:\Project\mid\Fed\balance\250_2_D_BPH_12.mat"
    analyze_mat_detailed(TARGET_FILE)
