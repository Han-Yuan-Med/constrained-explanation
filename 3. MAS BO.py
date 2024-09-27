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
from boundary_functions import *
from classification_functions import *
from scipy import ndimage


image_path = "D:\\ChestX-Det-Dataset"
mask_path = "D:\\ChestX-Det10-Dataset-Mask\\Mass"
cons_path = "D:\\ChestX-Det-Dataset-Lung-Top2"


test_csv = pd.read_csv("Dataset\\mul_test.csv")
test_csv = test_csv.iloc[np.where(test_csv.loc[:, 'mas'] == 1)]

test_cases_cons = Lung_seg_cons(csv_file=test_csv, img_dir=image_path, mask_dir=mask_path, cons_dir=cons_path)
test_cases_loader_cons = DataLoader(test_cases_cons, batch_size=1, shuffle=False)
# Instantiating CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

results_df = []

iou_list_test_case, dice_list_test_case, iou_case, dice_case = \
    explain_cases_bound(test_loader_seg=test_cases_loader_cons, device=device, threshold=95)

iou_std = bootstrap_sample(value_list=np.array(iou_list_test_case), times=100)
dice_std = bootstrap_sample(value_list=np.array(dice_list_test_case), times=100)
results_df.append([f"", f"Boundary",
                   f"{format(float(iou_case)*100, '.2f')} ({iou_std})",
                   f"{format(float(dice_case)*100, '.2f')} ({dice_std})"])

pd.DataFrame(results_df).to_csv("Result\\mas_bound.csv", index=False, encoding="cp1252")
