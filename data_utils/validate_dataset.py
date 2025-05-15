import numpy as np
import os

def validate_dataset(data_dir, num_classes=13):
    # 遍历所有数据文件
    for filename in os.listdir(data_dir):
        if filename.endswith(".npy"):
            path = os.path.join(data_dir, filename)
            data = np.load(path)
            
            print(f"\n== 检查文件: {filename} ==")
            print(f"形状: {data.shape}")
            print(f"类型: {data.dtype}")
            
            # 如果是标签文件
            if "label" in filename:
                assert data.dtype in [np.int32, np.int64], "标签必须是整数类型!"
                assert data.min() >=0 and data.max() < num_classes, f"标签范围应为 0-{num_classes-1}，实际为 {data.min()}-{data.max()}"

            # 如果是点云文件
            if "data" in filename:
                assert data.shape[-1] >= 3, "点云至少需要XYZ坐标!"
                if data.shape[-1] > 3:
                    rgb = data[..., 3:6]
                    assert rgb.min() >=0 and rgb.max() <=255, "RGB值应在0-255范围内"

if __name__ == "__main__":
    validate_dataset("D:/pn/Pointnet_ice/data/stanford_indoor3d/")
