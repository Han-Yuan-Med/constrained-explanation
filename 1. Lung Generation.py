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
lung_seg_model = torch.load("Model\\Lung-Seg-UNet-VGG11.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

for i in os.listdir(image_path):
    img_path = os.path.join(image_path, i)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    image_fused = np.zeros((3, 224, 224))
    image_fused[0] = image / 255
    image_fused[1] = image / 255
    image_fused[2] = image / 255
    image_fused = torch.from_numpy(image_fused).to(device).float().unsqueeze(0)

    outputs_test = lung_seg_model(image_fused)
    outputs_test = torch.sigmoid(outputs_test).detach().cpu().numpy().flatten()
    binary_list = copy.deepcopy(outputs_test)
    binary_list[np.where(binary_list >= 0.5)] = 1
    binary_list[np.where(binary_list != 1)] = 0
    binary_list = binary_list.astype("uint8")
    binary_list = np.resize(binary_list * 255, (224, 224))
    cv2.imwrite(os.path.join("D:\\ChestX-Det-Dataset-Lung-Initial", f"{i}"), binary_list)

    contours, hierarchy = cv2.findContours(binary_list, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    lung = np.zeros((224, 224))
    cv2.drawContours(lung, [contours[0]], 0, 1, -1)
    if len(contours) > 1:
        cv2.drawContours(lung, [contours[1]], 0, 1, -1)
    cv2.imwrite(os.path.join("D:\\ChestX-Det-Dataset-Lung-Top2", f"{i}"), lung * 255)

    closing_type = cv2.MORPH_ELLIPSE
    closing_element = cv2.getStructuringElement(closing_type, (19, 19))
    lung_closing = cv2.morphologyEx(lung, cv2.MORPH_CLOSE, closing_element)
    cv2.imwrite(os.path.join("D:\\ChestX-Det-Dataset-Lung-Closing", f"{i}"), lung_closing * 255)

    dilatation_type = cv2.MORPH_ELLIPSE
    element = cv2.getStructuringElement(dilatation_type, (15, 15))
    lung_dilate = cv2.dilate(lung_closing, element)

    cv2.imwrite(os.path.join("D:\\ChestX-Det-Dataset-Lung-Dilation", f"{i}"), lung_dilate * 255)
