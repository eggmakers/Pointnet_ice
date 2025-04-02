import argparse
import os
import torch
from pathlib import Path
import numpy as np
import importlib
import sys
from data_utils.indoor3d_util import g_label2color
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {i: cat for i, cat in enumerate(seg_classes.keys())}

def parse_args():
    '''参数设置'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--input_file', type=str, default='input_3.txt', help='点云输入文件路径（.txt 格式，包含 x y z 坐标）')
    parser.add_argument('--output_file', type=str, default='output.txt', help='分类结果输出文件路径（.txt 格式，保存 x y z label）')
    parser.add_argument('--gpu', type=str, default='0', help='指定 GPU 设备')
    parser.add_argument('--model_dir', type=str, default=os.path.join(BASE_DIR, 'log', 'sem_seg', 'pointnet2_sem_seg', 'checkpoints'),
                        help='训练好的模型路径目录（默认路径为 log/sem_seg/pointnet2_sem_seg/checkpoints）')
    return parser.parse_args()

def load_point_cloud(file_path):
    '''加载点云数据'''
    points = np.loadtxt(file_path)  # 假设输入文件格式为 x y z
    return points

def save_output(file_path, points, labels):
    '''保存输出为 x y z label 格式的文件'''
    with open(file_path, 'w') as f:
        for point, label in zip(points, labels):
            f.write(f"{point[0]} {point[1]} {point[2]} {label}\n")

def load_model(model_dir, num_classes=13):
    '''加载模型'''
    model_path = os.path.join(model_dir, 'best_model.pth')  # 直接指定加载 best_model.pth
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 未找到！")
    
    MODEL = importlib.import_module('pointnet_sem_seg')  # 使用模块名直接加载模型
    model = MODEL.get_model(num_classes).cuda()  # 获取模型并转移到GPU
    checkpoint = torch.load(model_path)  # 加载训练好的模型
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def classify_points(model, points, num_point=4096, batch_size=32):
    '''使用模型对点云进行分类（修正维度错误版）'''
    model.eval()
    num_points = points.shape[0]
    predictions = []

    # 新逻辑：将点云切割为多个块（block）
    num_blocks = (num_points + num_point - 1) // num_point  # 计算总块数
    for block_idx in range(0, num_blocks, batch_size):
        # 创建批次容器（修正维度为3通道）
        batch_data = np.zeros((batch_size, num_point, 3))  # [B, N, 3]
        
        # 填充当前批次
        real_batch_size = min(batch_size, num_blocks - block_idx)
        for i in range(real_batch_size):
            # 计算当前块的起止索引
            start_idx = (block_idx + i) * num_point
            end_idx = min(start_idx + num_point, num_points)
            
            # 获取当前块的点云数据（自动处理末尾不足的情况）
            current_block = points[start_idx:end_idx]  # [K, 3]
            
            # 填充到容器（修正维度对齐）
            batch_data[i, :current_block.shape[0], :] = current_block  # [1, N, 3]

        # 转为tensor并预测
        torch_data = torch.Tensor(batch_data).float().cuda().transpose(2, 1)  # [B, 3, N]
        with torch.no_grad():
            seg_pred, _ = model(torch_data)
            batch_pred = seg_pred.contiguous().cpu().data.max(2)[1].numpy()  # [B, N]
            
            # 只保留有效预测（处理末尾块）
            for i in range(real_batch_size):
                valid_pred = batch_pred[i, :end_idx-start_idx]
                predictions.append(valid_pred)

    # 合并所有预测结果
    return np.concatenate(predictions, axis=0)

def main(args):
    '''主函数'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 加载点云数据
    points = load_point_cloud(args.input_file)
    print(f"加载了 {len(points)} 个点云数据.")

    # 加载模型
    model = load_model(args.model_dir)
    
    # 对点云进行分类
    labels = classify_points(model, points)

    # # 保存分类结果
    save_output(args.output_file, points, labels)
    print(f"分类结果已保存到 {args.output_file}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
