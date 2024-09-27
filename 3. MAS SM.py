import torch
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
import cv2
import math
import gc
from skimage import color
from skimage import segmentation
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from saliency_map_functions import *
from classification_functions import *
from scipy import ndimage


image_path = "D:\\ChestX-Det-Dataset"
mask_path = "D:\\ChestX-Det10-Dataset-Mask\\Mass"
cons_path = "D:\\ChestX-Det-Dataset-Lung-Top2"

test_csv = pd.read_csv("Dataset\\mul_test.csv")
test_csv = test_csv.iloc[np.where(test_csv.loc[:, 'mas'] == 1)]

test_cases_bsl = Lung_seg(csv_file=test_csv, img_dir=image_path, mask_dir=mask_path)
test_cases_loader_bsl = DataLoader(test_cases_bsl, batch_size=1, shuffle=False)
test_cases_cons = Lung_seg_cons(csv_file=test_csv, img_dir=image_path, mask_dir=mask_path, cons_dir=cons_path)
test_cases_loader_cons = DataLoader(test_cases_cons, batch_size=1, shuffle=False)
# Instantiating CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
VGG11_optimal = torch.load("Model\\mas-VGG11.pt")

results_df = []
#
# Alex_optimal = torch.load("model\\mas-AlexNet.pt")
#
# iou_list_test_case, dice_list_test_case, iou_case, dice_case = \
#     explain_cases_baseline(test_loader_seg=test_cases_loader_bsl, device=device,
#                            cls_model=Alex_optimal, threshold=95)
#
# iou_std = bootstrap_sample(value_list=np.array(iou_list_test_case), times=100)
# dice_std = bootstrap_sample(value_list=np.array(dice_list_test_case), times=100)
# results_df.append([f"AlexNet", f"Baseline",
#                    f"{format(float(iou_case)*100, '.2f')} ({iou_std})",
#                    f"{format(float(dice_case)*100, '.2f')} ({dice_std})"])
#
# iou_list_test_case, dice_list_test_case, iou_case, dice_case = \
#     explain_cases_cons(test_loader_seg=test_cases_loader_cons, device=device,
#                        cls_model=Alex_optimal, threshold=95)
#
# iou_std = bootstrap_sample(value_list=np.array(iou_list_test_case), times=100)
# dice_std = bootstrap_sample(value_list=np.array(dice_list_test_case), times=100)
# results_df.append([f"AlexNet", f"Masked",
#                    f"{format(float(iou_case)*100, '.2f')} ({iou_std})",
#                    f"{format(float(dice_case)*100, '.2f')} ({dice_std})"])

iou_list_test_case, dice_list_test_case, iou_case, dice_case = \
    explain_cases_baseline(test_loader_seg=test_cases_loader_bsl, device=device,
                           cls_model=VGG11_optimal, threshold=95)

iou_std = bootstrap_sample(value_list=np.array(iou_list_test_case), times=100)
dice_std = bootstrap_sample(value_list=np.array(dice_list_test_case), times=100)
results_df.append([f"VGG-11", f"Baseline",
                   f"{format(float(iou_case)*100, '.2f')} ({iou_std})",
                   f"{format(float(dice_case)*100, '.2f')} ({dice_std})"])

iou_list_test_case_msk, dice_list_test_case, iou_case, dice_case = \
    explain_cases_cons(test_loader_seg=test_cases_loader_cons, device=device,
                       cls_model=VGG11_optimal, threshold=95)

iou_std = bootstrap_sample(value_list=np.array(iou_list_test_case_msk), times=100)
dice_std = bootstrap_sample(value_list=np.array(dice_list_test_case), times=100)
results_df.append([f"VGG-11", f"Masked",
                   f"{format(float(iou_case)*100, '.2f')} ({iou_std})",
                   f"{format(float(dice_case)*100, '.2f')} ({dice_std})"])

Res18_optimal = torch.load("model\\mas-Res18.pt")

iou_list_test_case, dice_list_test_case, iou_case, dice_case = \
    explain_cases_baseline(test_loader_seg=test_cases_loader_bsl, device=device,
                           cls_model=Res18_optimal, threshold=95)

iou_std = bootstrap_sample(value_list=np.array(iou_list_test_case), times=100)
dice_std = bootstrap_sample(value_list=np.array(dice_list_test_case), times=100)
results_df.append([f"ResNet-18", f"Baseline",
                   f"{format(float(iou_case)*100, '.2f')} ({iou_std})",
                   f"{format(float(dice_case)*100, '.2f')} ({dice_std})"])

iou_list_test_case, dice_list_test_case, iou_case, dice_case = \
    explain_cases_cons(test_loader_seg=test_cases_loader_cons, device=device,
                       cls_model=Res18_optimal, threshold=95)

iou_std = bootstrap_sample(value_list=np.array(iou_list_test_case), times=100)
dice_std = bootstrap_sample(value_list=np.array(dice_list_test_case), times=100)
results_df.append([f"ResNet-18", f"Masked",
                   f"{format(float(iou_case)*100, '.2f')} ({iou_std})",
                   f"{format(float(dice_case)*100, '.2f')} ({dice_std})"])

pd.DataFrame(results_df).to_csv("Result\\mas_sm.csv", index=False, encoding="cp1252")
