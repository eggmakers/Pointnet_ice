import os

def truncate_txt_files(root_dir, save_to_new=False, new_root_dir='output'):
    """
    遍历根目录及其子文件夹，处理所有 .txt 文件，保留每行前四列。
    如果 save_to_new=True，则保存到 new_root_dir 中对应结构，否则覆盖原文件。
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                file_path = os.path.join(dirpath, filename)
                
                # 读取并处理文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                processed_lines = []
                for line in lines:
                    columns = line.strip().split()
                    truncated = columns[:4]
                    processed_lines.append(' '.join(truncated) + '\n')

                # 决定保存位置
                if save_to_new:
                    relative_path = os.path.relpath(dirpath, root_dir)
                    new_dir = os.path.join(new_root_dir, relative_path)
                    os.makedirs(new_dir, exist_ok=True)
                    new_file_path = os.path.join(new_dir, filename)
                else:
                    new_file_path = file_path

                # 写回文件
                with open(new_file_path, 'w', encoding='utf-8') as f:
                    f.writelines(processed_lines)

                print(f"Processed: {new_file_path}")

# 使用示例
# 将 root_directory 替换为你的文件夹路径
root_directory = "D:\pn\Pointnet_ice\data_ice\AREA_ALL"  # ← 修改为你的路径
truncate_txt_files(root_directory, save_to_new=False)