# """
# # Description: This is a test file for the joseph.py file
# #本文件用于测试以及检验pointnet++的所有函数的功能
# """
# import sys
# import argparse

# #创建一个新的解释器
# parser =argparse.ArgumentParser()

# #同样实现乘法操作
# parser.add_argument('--a', type=int, default=1, help='First number')
# parser.add_argument('--b', type=int, default=1, help='Second number')
# parser.add_argument('method', type=str, help='Method')
# parser.add_argument('verbose', action="store_true",  help='print verbose output')

# #解析命令行
# args=parser.parse_args()



# print(args)



import argparse
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='图像缩放工具')
    parser.add_argument('input', help='输入图像路径')
    parser.add_argument('output', help='输出图像路径')
    parser.add_argument('--size', required=True, type=int, nargs=2,
                        metavar=('WIDTH', 'HEIGHT'), help='目标尺寸（宽 高）')
    
    args = parser.parse_args()
    width, height = args.size

    try:
        # 打开并缩放图像
        with Image.open(args.input) as img:
            resized_img = img.resize(
                (width, height),
                resample=Image.Resampling.LANCZOS
            )
            resized_img.save(args.output)
        print(f"图像已成功保存至：{args.output}")
        
    except FileNotFoundError:
        print(f"错误：输入文件 {args.input} 不存在")
    except Exception as e:
        print(f"处理图像时发生错误：{str(e)}")

if __name__ == "__main__":
    main()