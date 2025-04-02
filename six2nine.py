import numpy as np


def normalize(data):
    """
    对输入的数据进行归一化处理
    :param data: 输入的一维数组
    :return: 归一化后的一维数组
    """
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)


input_file = 'input.txt'
output_file = 'output.txt'

# 读取数据
data = np.loadtxt(input_file, delimiter=' ')

# 提取前三列
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# 进行归一化
x_norm = normalize(x)
y_norm = normalize(y)
z_norm = normalize(z)

# 将归一化结果添加到原数据的右侧
new_data = np.column_stack((data, x_norm, y_norm, z_norm))

# 写入输出文件，使用空格作为分隔符
with open(output_file, 'w') as f:
    np.savetxt(f, new_data, fmt='%.6f', delimiter=' ')
    