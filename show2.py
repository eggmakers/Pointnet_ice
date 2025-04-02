import argparse
import os
import numpy as np
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
from data_utils.S3DISDataLoader import ScannetDatasetWholeScene
from data_utils.indoor3d_util import g_label2color


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ceiling','floor','wall','beam','column','window','door','table','chair','sofa','bookcase','board','clutter']
class2label = {cls: i for i,cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i,cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--log_dir', type=str, default='pointnet2_sem_seg', help='Experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='Whether visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=5, help='Aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()

def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n]:
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool

def load_input_data(file_path, num_points=4096):
    """
    从input.txt文件加载点云数据
    :param file_path: 输入文件路径
    :param num_points: 点云的点数
    :return: 点云数据（num_points x 3的numpy数组）
    """
    # 从文件加载点云数据
    data = np.loadtxt(file_path)
    if data.shape[0] < num_points:
        # 如果点云数量小于设定的点数，填充数据
        padding = np.zeros((num_points - data.shape[0], 3))
        data = np.vstack((data, padding))
    else:
        # 如果点云数量大于设定的点数，裁剪数据
        data = data[:num_points, :]
    return data

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 解析参数
    args = parse_args()

    # 加载模型 
    # 没问题
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(13).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    # 加载点云数据（从 input.txt 读取）
    input_file = "input_3.txt"  # 假设文件与当前脚本同级
    all_points = np.loadtxt(input_file)
    total_points = (all_points.shape[0] + 4095) // 4096 
    #批次数
    

    all_predictions = []
    # 将点云数据分批次并进行预测
    with torch.no_grad():
        for i in range(total_points):

            start_idx = i * 4096
            end_idx = min(start_idx + 4096, all_points.shape[0])
            points = all_points[start_idx:end_idx, :3]
            if i == total_points - 1:
                # 处理最后一批数据，可能不足4096个点
                points = all_points[i * 4096:, :3]
                
                # 判断当前批次的点数是否小于4096
                num_points_in_batch = points.shape[0]
                
                if num_points_in_batch < 4096:
                    # 如果当前批次的点数不足4096，用零填充至4096
                    padding = np.zeros((4096 - num_points_in_batch, 3))  # 生成缺少的点数
                    points = np.vstack((points, padding))  # 填充至4096个点

                # 现在，points 的形状是 (4096, 3)，可以继续进行后续处理 

            batch_data = np.zeros((args.batch_size, args.num_point, 3))
            batch_label = np.zeros((args.batch_size, args.num_point))
            batch_point_index = np.zeros((args.batch_size, args.num_point))
            batch_smpw = np.zeros((args.batch_size, args.num_point))

            batch_data[:, :, 0:3] = points  # 假设输入的点云数据是 (N, 3) 数组
            torch_data = torch.Tensor(batch_data).float().cuda()
            torch_data = torch_data.transpose(2, 1)

            seg_pred, _ = classifier(torch_data)
            batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()
        
            all_predictions.append(batch_pred_label)
        all_predictions = np.concatenate(all_predictions, axis=1)
        # 保存预测标签
        pl_save_path = "pl_save.txt"
        with open(pl_save_path, 'w') as pl_save:
            for i in all_predictions[0]:
                pl_save.write(str(int(i)) + '\n')

        print(f"Predicted labels saved to {pl_save_path}")

if __name__ == '__main__':
    args = parse_args()
    main(args)
