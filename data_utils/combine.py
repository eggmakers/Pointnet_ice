import numpy as np

def combine_txt_files(file1_path, file2_path, output_path):
    try:
        # 读取第一个文件的前三列
        data1 = np.loadtxt(file1_path, usecols=(0, 1, 2))
        # 读取第二个文件的列
        data2 = np.loadtxt(file2_path)

        # 确保两个文件的行数相同
        if data1.shape[0] != data2.shape[0]:
            raise ValueError("两个文件的行数不相同，无法拼接")

        # 拼接数据
        combined_data = np.column_stack((data1, data2))

        # 保存拼接后的数据到新文件
        np.savetxt(output_path, combined_data, fmt='%.6f', delimiter=' ')

        print(f"文件已成功拼接并保存到 {output_path}")
    except FileNotFoundError:
        print(f"文件未找到: {file1_path} 或 {file2_path}")
    except Exception as e:
        print(f"发生错误: {e}")

# 示例使用
file1_path = 'D:/pn/Pointnet_ice/log/conferenceRoom_1.txt'  # 替换为第一个文件的路径
file2_path = 'D:/pn/Pointnet_ice/log/Area_5_conferenceRoom_1.txt'  # 替换为第二个文件的路径
output_path = 'D:/pn/Pointnet_ice/log/data_utilscombined_file.txt'  # 替换为输出文件的路径

combine_txt_files(file1_path, file2_path, output_path)