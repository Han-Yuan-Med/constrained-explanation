import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage import io
import matplotlib.image as img
import numpy as np
from sklearn import metrics
from tqdm import tqdm
# from bootstrap_functions import *
from classification_functions import *
import torch.optim.lr_scheduler as lr_scheduler


image_path = "D:\\ChestX-Det-Dataset"
train_csv = pd.read_csv("Dataset\\mul_train.csv")
val_csv = pd.read_csv("Dataset\\mul_valid.csv")
test_csv = pd.read_csv("Dataset\\mul_test.csv")

train_csv = pd.concat([train_csv.iloc[np.where(train_csv.loc[:, 'ptx'] == 1)], train_csv.iloc[-367:, :]])
val_csv = pd.concat([val_csv.iloc[np.where(val_csv.loc[:, 'ptx'] == 1)], val_csv.iloc[-122:, :]])
test_csv = pd.concat([test_csv.iloc[np.where(test_csv.loc[:, 'ptx'] == 1)], test_csv.iloc[-122:, :]])

train_set = Lung_cls(csv_file=train_csv.loc[:, ['File', 'ptx']], img_dir=image_path)
val_set = Lung_cls(csv_file=val_csv.loc[:, ['File', 'ptx']], img_dir=image_path)
test_set = Lung_cls(csv_file=test_csv.loc[:, ['File', 'ptx']], img_dir=image_path)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

# Now using the VGG11
VGG11_model = torchvision.models.vgg11(weights=torchvision.models.VGG11_Weights.DEFAULT)

# Model description
VGG11_model.classifier[6] = nn.Linear(4096, 2)
print(VGG11_model.eval())

for child in VGG11_model.children():
    print(child)
    for param in child.parameters():
        print(param.requires_grad)

# Instantiating CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Move the input and VGG11_model to GPU if it is available
VGG11_model.to(device)

# Weighted Loss
weights = [round((1 - round(len(np.where(train_csv.loc[:, 'ptx'] == 0)[0]) / len(train_csv), 1)), 2),
           round(len(np.where(train_csv.loc[:, 'ptx'] == 0)[0]) / len(train_csv), 2)]

class_weights = torch.FloatTensor(weights).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer SGD
optimizer = optim.SGD(filter(lambda p: p.requires_grad, VGG11_model.parameters()), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
epoch_num = 50
auroc_value = np.zeros(epoch_num)
acc_value = np.zeros(epoch_num)
thres_value = np.zeros(epoch_num)
early_stopping = 0
auroc_value_optimal = 0
patience = 5
for epoch in range(epoch_num):  # loop over the dataset multiple times
    train_image(cls_model=VGG11_model, train_loader=train_loader, criterion=criterion,
                optimizer=optimizer, device=device)
    # VGG11_model.eval()
    # calculate the fpr and tpr for all thresholds of the classification
    print(f"Finish training at epoch {epoch}; Start validation")
    # correct = 0
    # total = 0
    prob_list = []
    label_list = []
    with torch.no_grad():
        for data in val_loader:
            images, labels = data[0].float().to(device), data[1].to(device)
            outputs = VGG11_model(images)
            label_list = np.concatenate((label_list, labels.cpu().numpy()), axis=None)
            prob_list = np.concatenate((prob_list, torch.sigmoid(outputs)[:, 1].detach().cpu().numpy()), axis=None)

    fpr, tpr, thresholds = metrics.roc_curve(label_list, prob_list)
    auroc_value[epoch] = metrics.auc(fpr, tpr)
    thres_value[epoch] = thresholds[np.argmax(tpr - fpr)]
    print(f"AUROC on the validation set at epoch {epoch} is {auroc_value[epoch]}")
    print(f'Optimal threshold is {thres_value[epoch]}')

    if auroc_value[epoch] > auroc_value_optimal:
        PATH = f"Model\\ptx-VGG11.pt"
        torch.save(VGG11_model, PATH)
        auroc_value_optimal = auroc_value[epoch]
        early_stopping = 0
        print(f'Reset stopping index to {early_stopping}')

    else:
        early_stopping = early_stopping + 1
        print(f'current early stopping index is {early_stopping}')

    if early_stopping == patience:
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1} Adam lr {before_lr} -> {after_lr} \n")
        early_stopping = 0

print("Finished Training")

print(f'Optimal AUROC value on validation set is {round(auroc_value[np.argmax(auroc_value)], 3)}, '
      f'(threshold is {round(thres_value[np.argmax(auroc_value)], 3)}), '
      f'Meanwhile, the accuracy is {round(acc_value[np.argmax(auroc_value)], 3)}')

# Evaluate model performance on the test set
VGG11_optimal = torch.load("Model\\ptx-VGG11.pt")
prob_list = []
label_list = []
with torch.no_grad():
    for data in tqdm(val_loader):
        images, labels = data[0].float().to(device), data[1].cpu().numpy()
        outputs = VGG11_optimal(images)
        label_list = np.concatenate((label_list, labels), axis=None)
        prob_list = np.concatenate((prob_list, torch.sigmoid(outputs)[:, 1].detach().cpu().numpy()), axis=None)

# Calculate accuracy, auroc, sensitivity, specificity
fpr, tpr, thresholds = metrics.roc_curve(label_list, prob_list)
thres_val = round(thresholds[np.argmax(tpr - fpr)], 3)
prob_list[np.where(prob_list >= thres_val)] = 1
prob_list[np.where(prob_list != 1)] = 0
tn, fp, fn, tp = metrics.confusion_matrix(label_list, prob_list).ravel()
print(f'AUROC on the validation set is {round(metrics.auc(fpr, tpr), 3)}')
print(f'Threshold determined by Youden index is {thres_val}')
print(f'Accuracy on the validation set is {round((tp+tn) / (tp+tn+fp+fn), 3)}')
print(f'Sensitivity on the validation set is {round(tp / (tp+fn), 3)}')
print(f'Specificity on the validation set is {round(tn / (tn+fp), 3)}')

prob_list = []
label_list = []
with torch.no_grad():
    for data in tqdm(test_loader):
        images, labels = data[0].float().to(device), data[1].cpu().numpy()
        outputs = VGG11_optimal(images)
        label_list = np.concatenate((label_list, labels), axis=None)
        prob_list = np.concatenate((prob_list, torch.sigmoid(outputs)[:, 1].detach().cpu().numpy()), axis=None)

# Calculate accuracy, auroc, sensitivity, specificity
fpr, tpr, thresholds = metrics.roc_curve(label_list, prob_list)
precision, recall, thresholds = metrics.precision_recall_curve(label_list, prob_list)
auc_std, prc_std, acc_std, sen_std, spe_std, ppv_std, npv_std = \
    bootstrap_cls(prob_list=prob_list, label_list=label_list, threshold=thres_val, times=100)

prob_list[np.where(prob_list >= thres_val)] = 1
prob_list[np.where(prob_list != 1)] = 0
tn, fp, fn, tp = metrics.confusion_matrix(label_list, prob_list).ravel()

print(f'AUROC on the test set is {round(metrics.auc(fpr, tpr), 3)}')
print(f'AUPRC on the test set is {round(metrics.auc(recall, precision), 3)}')
print(f'Threshold determined by Youden index is {thres_val}')
print(f'Accuracy on the test set is {round((tp+tn) / (tp+tn+fp+fn), 3)}')
print(f'Sensitivity on the test set is {round(tp / (tp+fn), 3)}')
print(f'Specificity on the test set is {round(tn / (tn+fp), 3)}')
print(f'PPV on the test set is {round(tp / (tp+fp), 3)}')
print(f'NPV on the test set is {round(tn / (tn+fn), 3)}')

results_df = [[f"VGG-11", f"{thres_val}",
               f"{format(metrics.auc(fpr, tpr), '.3f')} ({auc_std})",
               f"{format(metrics.auc(recall, precision), '.3f')} ({prc_std})",
               f"{format((tp + tn) / (tp + tn + fp + fn), '.3f')} ({acc_std})",
               f"{format(tp / (tp + fn), '.3f')} ({sen_std})",
               f"{format(tn / (tn + fp), '.3f')} ({spe_std})",
               f"{format(tp / (tp + fp), '.3f')} ({ppv_std})",
               f"{format(tn / (tn + fn), '.3f')} ({npv_std})",
               ]]

results_df = pd.DataFrame(results_df)
results_df.columns = ['Model', 'Threshold', 'AUROC', 'AUPRC', 'Accuracy', 'Sensitivity', 'Specificity',
                      'PPV', 'NPV']
results_df.to_csv("Result\\ptx_cls_vgg_11.csv", index=False, encoding="cp1252")

