import os


def process_point_cloud_file(input_file_path, output_base_folder, input_folder):
    try:
        with open(input_file_path, 'r') as file:
            lines = file.readlines()

        output1_lines = []
        output2_lines = []
        output3_lines = []

        for line in lines:
            values = line.strip().split()
            x, y, z, r, g, _ = map(float, values[:6])

            output1_lines.append(f"{x} {y} {z}\n")
            output2_lines.append(f"{x} {y} {z} {r} {r} {r}\n")
            output3_lines.append(f"{x} {y} {z} {g} {g} {g}\n")

        # 获取输入文件相对于输入文件夹的相对路径
        rel_path = os.path.relpath(input_file_path, input_folder)

        for i, output_lines in enumerate([output1_lines, output2_lines, output3_lines], start=1):
            output_folder = os.path.join(output_base_folder, f'output{i}')
            output_file_path = os.path.join(output_folder, rel_path)
            # 创建输出文件所在的目录
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            # 将处理后的数据写入输出文件
            with open(output_file_path, 'w') as output_file:
                output_file.writelines(output_lines)

    except FileNotFoundError:
        print(f"错误：未找到文件 {input_file_path}。")
    except Exception as e:
        print(f"发生未知错误：{e}")


def process_folder(input_folder):
    output_base_folder = os.path.dirname(input_folder)
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                process_point_cloud_file(file_path, output_base_folder, input_folder)


if __name__ == "__main__":
    input_folder = 'train'
    process_folder(input_folder)