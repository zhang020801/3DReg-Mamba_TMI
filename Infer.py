import os
import glob
import warnings

import torch
import numpy as np
import SimpleITK as sitk

from utils import losses
from utils.config import args
from utils.datagenerators_atlas import Dataset
from Models.STN import SpatialTransformer
from natsort import natsorted

from Models.RegMamba import RegMamba
import pymia.evaluation.metric as metric
import pymia.evaluation.evaluator as eval_
import pymia.evaluation.writer as writer
import time
import csv

import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument")
warnings.filterwarnings("ignore", message="Default grid_sample and affine_grid behavior has changed")

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join('./Result_image/LPBA40/', name))


def compute_label_dice(gt, pred):
    cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
            63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
            163, 164, 165, 166]
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)

def write_csv(file_path, Time, mean_J_DET, std_J_DET):
    with open(file_path, mode='a', newline='') as file: 
        writer = csv.writer(file)
        writer.writerow([Time, mean_J_DET, std_J_DET])


def train():
    make_dirs()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    datasets_name = 'LPBA40'
    result_summary_file = '../3DRegMamba/infer_result/' + datasets_name +'_results_summary.csv'
    result_Time_J_DET_file = '../3DRegMamba/infer_result/' + datasets_name +'_result_Time_J_DET.csv'

    metrics = [metric.DiceCoefficient(), metric.HausdorffDistance(percentile=95, metric='HDRFDST95'),metric.VolumeSimilarity()]
    labels = {21: 'L superior frontal gyrus', 22: 'R superior frontal gyrus', 23: 'L middle frontal gyrus',
              24: 'R middle frontal gyrus', 25: 'L inferior frontal gyrus', 26: 'R inferior frontal gyrus',
              27: 'L precentral gyrus', 28: 'R precentral gyrus', 29: 'L middle orbitofrontal gyrus',
              30: 'R middle orbitofrontal gyrus', 31: 'L lateral orbitofrontal gyrus', 32: 'R lateral orbitofrontal gyrus',
              33: 'L gyrus rectus', 34: 'R gyrus rectus', 41: 'L postcentral gyrus', 42: 'R postcentral gyrus',
              43: 'L superior parietal gyrus', 44: 'R superior parietal gyrus', 45: 'L supramarginal gyrus',
              46: 'R supramarginal gyrus', 47: 'L angular gyrus', 48: 'R angular gyrus', 49: 'L precuneus',
              50: 'R precuneus', 61: 'L superior occipital gyrus', 62: 'R superior occipital gyrus',
              63: 'L middle occipital gyrus', 64: 'R middle occipital gyrus', 65: 'L inferior occipital gyrus',
              66: 'R inferior occipital gyrus', 67: 'L cuneus', 68: 'R cuneus', 81: 'L superior temporal gyrus',
              82: 'R superior temporal gyrus', 83: 'L middle temporal gyrus', 84: 'R middle temporal gyrus',
              85: 'L inferior temporal gyrus', 86: 'R inferior temporal gyrus', 87: 'L parahippocampal gyrus',
              88: 'R parahippocampal gyrus', 89: 'L lingual gyrus', 90: 'R lingual gyrus', 91: 'L fusiform gyrus',
              92: 'R fusiform gyrus', 101: 'L insular cortex', 102: 'R insular cortex', 121: 'L cingulate gyrus',
              122: 'R cingulate gyrus', 161: 'L caudate', 162: 'R caudate', 163: 'L putamen', 164: 'R putamen',
              165: 'L hippocampus', 166: 'R hippocampus'
              }
    evaluator = eval_.SegmentationEvaluator(metrics, labels)
    evaluator_summary = eval_.SegmentationEvaluator(metrics, labels)

    f_img = sitk.ReadImage(args.atlas_file)
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    vol_size = input_fixed.shape[2:]
    input_fixed_eval = torch.from_numpy(input_fixed).to(device).float()

    fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.label_dir, "S01.delineation.structure.label.nii.gz")))[np.newaxis, np.newaxis, ...]
    fixed_label = torch.from_numpy(fixed_label).to(device).float()

    net = RegMamba().to(device)
    best_model = torch.load('./experiments/LPBA40/dsc0.7117epoch101.pth.tar')['state_dict']
    net.load_state_dict(best_model)

    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)
    net.train()

    train_files = glob.glob(os.path.join(args.train_dir, '*.nii.gz'))
    DS = Dataset(files=train_files)
    print("Number of training images: ", len(DS))
    test_file_lst = glob.glob(os.path.join(args.test_dir, "*.nii.gz"))

    net.eval()
    STN_label.eval()

    TIME = []
    J_DET = []
    with torch.no_grad():
        for file in test_file_lst:
            name = os.path.split(file)[1]
            input_moving = sitk.GetArrayFromImage(sitk.ReadImage(file))[np.newaxis, np.newaxis, ...]
            input_moving = torch.from_numpy(input_moving).to(device).float()
            label_file = glob.glob(os.path.join(args.label_dir, name[:3] + "*"))[0]
            input_label = sitk.GetArrayFromImage(sitk.ReadImage(label_file))[np.newaxis, np.newaxis, ...]
            input_label = torch.from_numpy(input_label).to(device).float()

            start = time.time()
            pred_img, pred_flow = net(input_moving, input_fixed_eval)
            TIME.append(time.time() - start)
            pred_label = STN_label(input_label, pred_flow)

            evaluator.evaluate(pred_label[0, 0, ...].cpu().detach().numpy(), fixed_label[0, 0, ...].cpu().detach().numpy(), name)
            evaluator_summary.evaluate(pred_label[0, 0, ...].cpu().detach().numpy(), fixed_label[0, 0, ...].cpu().detach().numpy(), name)

            print('\nSubject-wise results...')
            writer.ConsoleWriter().write(evaluator.results)
            result_every_image_file = '../3DRegMamba/infer_result/LPBA40_every_image_result/' + name[:3] + '.csv'
            writer.CSVStatisticsWriter(result_every_image_file).write(evaluator.results)

            evaluator.clear()
            negative_jacobian_percentage = losses.negative_jacobin(pred_flow[0].permute(1, 2, 3, 0).cpu().numpy())
            J_DET.append(negative_jacobian_percentage)


            tmpName = str(file[54:57]) 
            save_image(pred_img, f_img, tmpName + '_warpped.nii.gz')
            save_image(pred_flow.permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img, tmpName + "_flow.nii.gz")
            save_image(pred_label, f_img, tmpName + "_label.nii.gz")
            del input_moving, input_label
            print('ok')

        functions = {'MEAN': np.mean, 'STD': np.std}
        writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator_summary.results)
        print('\nAggregated statistic results...')
        writer.ConsoleStatisticsWriter(functions=functions).write(evaluator_summary.results)
        print("Time:",np.mean(TIME))
        print("np.mean(J_DET):", np.mean(J_DET))
        print("np.std(J_DET):", np.std(J_DET))
        write_csv(result_Time_J_DET_file, np.mean(TIME), np.mean(J_DET), np.std(J_DET))

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train()
