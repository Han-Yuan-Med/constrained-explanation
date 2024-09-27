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

iou_list_test_case, dice_list_test_case, iou_case, dice_case = \
    explain_cases_baseline(test_loader_seg=test_cases_loader_bsl, device=device,
                           cls_model=VGG11_optimal, threshold=95)

iou_list_test_case_msk, dice_list_test_case, iou_case, dice_case = \
    explain_cases_cons(test_loader_seg=test_cases_loader_cons, device=device,
                       cls_model=VGG11_optimal, threshold=95)

idx = np.argmax(np.asarray(iou_list_test_case_msk).astype(np.float32)-
          np.asarray(iou_list_test_case).astype(np.float32))

image = cv2.imread(image_path + "\\" + test_csv.iloc[idx, 0], cv2.IMREAD_UNCHANGED)
mask = cv2.imread(mask_path + "\\" + test_csv.iloc[idx, 0], cv2.IMREAD_GRAYSCALE)
image_mask = color.label2rgb(mask, image, colors=['fuchsia'])
image_mask = cv2.resize(image_mask, (224, 224))
cv2.imwrite('Figure\\SM\\mas-SM-Image.png', cv2.resize(image, (224, 224)))
Image.fromarray((image_mask * 255).astype(np.uint8)).save('Figure\\SM\\mas-SM-Mask.png')
cons = cv2.imread('D:\\ChestX-Det-Dataset-Lung-Top2' + "\\" + test_csv.iloc[idx, 0], cv2.IMREAD_UNCHANGED)
cv2.imwrite('Figure\\SM\\mas-SM-cons.png', cons)

test_cases_cons = Lung_seg_cons(csv_file=test_csv.iloc[np.array([idx]), :], img_dir=image_path, mask_dir=mask_path,
                                cons_dir='D:\\ChestX-Det-Dataset-Lung-Top2')
test_cases_loader_cons = DataLoader(test_cases_cons, batch_size=1, shuffle=False)
VGG11_optimal.eval()
# fixed_mask = torch.tensor(fixed_mask)
for data_test in tqdm(test_cases_loader_cons):
    images_test, labels_test, cons_test = data_test[0].to(device).float(), \
                                          np.array(data_test[1]).flatten(), \
                                          np.array(data_test[2]).flatten()
    binary_list_iou = saliency_map_org(model=VGG11_optimal, image=images_test)
    binary_list_iou_cons = binary_list_iou * cons_test

    threshold_value = np.percentile(binary_list_iou, 95)
    binary_list_iou[np.where(binary_list_iou >= threshold_value)] = 1
    binary_list_iou[np.where(binary_list_iou != 1)] = 0
    binary_list_iou.astype("uint8")
    binary_list_iou = np.resize(binary_list_iou * 255, (224, 224))
    image = cv2.resize(image, (224, 224))
    image_mask = color.label2rgb(binary_list_iou, image, colors=['darkorange'])
    Image.fromarray((image_mask * 255).astype(np.uint8)).save('Figure\\SM\\mas-SM-BSL.png')

    threshold_value_cons = np.percentile(binary_list_iou_cons, 95)
    binary_list_iou_cons[np.where(binary_list_iou_cons >= threshold_value_cons)] = 1
    binary_list_iou_cons[np.where(binary_list_iou_cons != 1)] = 0
    binary_list_iou_cons.astype("uint8")
    binary_list_iou_cons = np.resize(binary_list_iou_cons * 255, (224, 224))
    image_mask = color.label2rgb(binary_list_iou_cons, image, colors=['cyan'])
    Image.fromarray((image_mask * 255).astype(np.uint8)).save('Figure\\SM\\mas-SM-PRO.png')
