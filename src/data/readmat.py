import scipy.io as sio

file_path = r'F:\Project\mid\德国数据集\领域泛化\PUdata_1\900_7_1000\K001\N09_M07_F10_K001_1.mat'
mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
root_key = [k for k in mat.keys() if not k.startswith('__')][0]
root_obj = mat[root_key]

print(f"--- 文件 {root_key} 的所有信号清单 ---")
for i, sensor in enumerate(root_obj.Y):
    name = sensor.Name
    length = sensor.Data.size
    raster = sensor.Raster
    print(f"通道 [{i}] | 信号: {name:15} | 点数: {length:8} | 采样率: {raster}")
