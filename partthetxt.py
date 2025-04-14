import os
input_file = 'output_3.txt'
output_file = 'output_4.txt'
num_lines = 50000
try:
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # 逐行读取输入文件
        for i, line in enumerate(infile):
            if i < num_lines:
                # 写入前 num_lines 行到输出文件
                outfile.write(line)
            else:
                break
    print(f"已成功保留前 {num_lines} 行并保存到 {output_file}。")

except FileNotFoundError:
    print(f"未找到输入文件 {input_file}。")
except PermissionError:
    print(f"没有权限删除文件 {output_file}。")