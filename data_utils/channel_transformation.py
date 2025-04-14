import os
from tqdm import tqdm

def process_all_txt_files(root_dir, output_root):
    """处理数据集根目录下所有层级的txt文件"""
    # 遍历所有目录（包含子目录）
    for foldername, subfolders, filenames in tqdm(os.walk(root_dir), desc="Scanning directories"):
        # 过滤并处理所有txt文件
        txt_files = [f for f in filenames if f.lower().endswith(".txt")]
        if not txt_files:
            continue

        # 创建对应的输出目录
        relative_path = os.path.relpath(foldername, root_dir)
        output_dir = os.path.join(output_root, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        # 处理当前目录下的所有txt文件
        for filename in tqdm(txt_files, desc=f"Processing {os.path.basename(foldername)}", leave=False):
            input_path = os.path.join(foldername, filename)
            output_path = os.path.join(output_dir, filename)

            # 执行六列转三列操作
            with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
                for line in infile:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        # 保留前三列（XYZ坐标）
                        outfile.write(" ".join(parts[:3]) + "\n")

if __name__ == "__main__":
    # 配置路径（根据实际情况修改）
    dataset_root = r"D:/pn/Pointnet_ice/data/s3dis/Stanford3dDataset_v1.2_Aligned_Version"
    output_root = r"D:/pn/Pointnet_ice/data/s3dis/Stanford3dDataset_Processed"

    # 执行处理
    process_all_txt_files(dataset_root, output_root)
    print(f"\n处理完成！所有文件已保存在：{output_root}")
