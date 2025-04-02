import numpy as np


def txt_to_pcd(txt_file, pcd_file):
    # 读取txt文件
    data = np.loadtxt(txt_file)  # 假设每行是 [x, y, z, label]

    # 确保数据格式正确
    assert data.shape[1] == 4, "每行数据必须包含4列：x, y, z, label"

    # 提取坐标 (x, y, z) 和标签 (label)
    points = data[:, :3]  # (x, y, z)
    labels = data[:, 3]   # label

    # 打开PCD文件进行写入
    with open(pcd_file, 'w') as f:
        # 写入PCD头部信息
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION .7\n")
        f.write("FIELDS x y z label\n")
        f.write("SIZE 4 4 4 4\n")  # 每个字段占用的字节数
        f.write("TYPE F F F I\n")  # 坐标类型是浮动数，标签是整数
        f.write("COUNT 1 1 1 1\n")  # 每个字段只有1个值
        f.write(f"WIDTH {len(points)}\n")  # 点云数据的宽度（即点数）
        f.write("HEIGHT 1\n")  # 高度是1，表示点云数据是一个单行
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")

        # 写入点云数据
        for i in range(len(points)):
            f.write(f"{points[i][0]} {points[i][1]} {points[i][2]} {int(labels[i])}\n")





# # 读取第一个文件的内容
# with open('input_3.txt', 'r') as file1:
#     lines1 = file1.readlines()

# # 读取第二个文件的内容
# with open('pl_save.txt', 'r') as file2:
#     lines2 = file2.readlines()

# # 打开输出文件
# with open('output_3.txt', 'w') as output_file:
#     for i in range(len(lines1)):
#         line1 = lines1[i]
#         line2 = lines2[i]
#         # 处理第一个文件的行，分割出x, y, z
#         x, y, z = map(float, line1.strip().split())
#         # 处理第二个文件的行，获取数字
#         num = float(line2.strip())
#         # 组合成新的行并写入输出文件
#         new_line = f"{x} {y} {z} {num}\n"
#         output_file.write(new_line)
# print("输出文件 output_3.txt 已生成。")
# # 读取第一个文件的内容

# 用法
txt_file = 'output_4.txt'  # 输入的txt文件
pcd_file = 'output.pcd'  # 输出的pcd文件

txt_to_pcd(txt_file, pcd_file)
print(f"转换完成，保存为 {pcd_file}")
# 使用Open3D可视化