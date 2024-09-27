import torch
import numpy as np
from sklearn import metrics
import cv2
import copy
from tqdm import tqdm


def saliency_map(model, image, threshold):
    image.requires_grad_()
    model(image)[:, 1].backward()
    outputs, _ = torch.max(image.grad.data.abs(), dim=1)
    outputs = outputs.flatten()
    outputs = np.array(outputs.detach().cpu())
    # if np.max(outputs) != 0:
    #     outputs = (outputs - np.min(outputs)) / np.max(outputs)
        # outputs = (outputs - np.min(outputs)) / np.max(outputs)
    threshold_value = np.percentile(outputs, threshold)
    outputs = outputs.flatten()
    outputs[np.where(outputs >= threshold_value)] = 1
    outputs[np.where(outputs != 1)] = 0

    return outputs


def saliency_map_org(model, image):
    image.requires_grad_()
    model(image)[:, 1].backward()
    outputs, _ = torch.max(image.grad.data.abs(), dim=1)
    outputs = outputs.flatten()
    outputs = np.array(outputs.detach().cpu())
    if np.max(outputs) != 0:
        outputs = (outputs - np.min(outputs)) / np.max(outputs)
    outputs = outputs.flatten()

    return outputs


def explain_cases_baseline(test_loader_seg, device, cls_model, threshold):
    cls_model.eval()
    iou_list_test_case = []
    dice_list_test_case = []

    for data_test in tqdm(test_loader_seg):
        images_test, labels_test = data_test[0].to(device).float(), np.array(data_test[1]).flatten()
        if labels_test.max() == 0:
            continue
        else:
            binary_list_iou = saliency_map_org(model=cls_model, image=images_test)

            threshold_value = np.percentile(binary_list_iou, threshold)
            binary_list_iou[np.where(binary_list_iou >= threshold_value)] = 1
            binary_list_iou[np.where(binary_list_iou != 1)] = 0
            binary_list_iou = binary_list_iou.astype("uint8")
            iou = metrics.jaccard_score(labels_test, binary_list_iou)
            iou = format(iou, '.3f')

            dice = metrics.f1_score(labels_test, binary_list_iou)
            dice = format(dice, '.3f')

            iou_list_test_case.append(iou)
            dice_list_test_case.append(dice)

    iou_case = format(np.array(iou_list_test_case).astype('float').mean(), '.4f')
    dice_case = format(np.array(dice_list_test_case).astype('float').mean(), '.4f')

    print(f'IoU on sick test set is {iou_case}')
    print(f'Dice on sick test set is {dice_case}')

    return iou_list_test_case, dice_list_test_case, iou_case, dice_case


def explain_cases_cons(test_loader_seg, device, cls_model, threshold):
    cls_model.eval()
    iou_list_test_case = []
    dice_list_test_case = []

    for data_test in tqdm(test_loader_seg):
        images_test, labels_test, cons_test = data_test[0].to(device).float(), \
                                              np.array(data_test[1]).flatten(), \
                                              np.array(data_test[2]).flatten()

        if labels_test.max() == 0:
            continue
        else:
            binary_list_iou = saliency_map_org(model=cls_model, image=images_test)
            binary_list_iou = binary_list_iou * cons_test

            threshold_cons = 100 - round(((224 * 224 * (1-threshold/100)) / sum(cons_test)) * 100)
            if len(np.where(binary_list_iou != 0)[0]) > 0:
                threshold_value = np.percentile(binary_list_iou[np.where(binary_list_iou != 0)[0]], threshold_cons)
            else:
                threshold_value = np.percentile(binary_list_iou, threshold)

            binary_list_iou[np.where(binary_list_iou >= threshold_value)] = 1
            binary_list_iou[np.where(binary_list_iou != 1)] = 0
            binary_list_iou = binary_list_iou.astype("uint8")
            iou = metrics.jaccard_score(labels_test, binary_list_iou)
            iou = format(iou, '.3f')

            dice = metrics.f1_score(labels_test, binary_list_iou)
            dice = format(dice, '.3f')

            iou_list_test_case.append(iou)
            dice_list_test_case.append(dice)

    iou_case = format(np.array(iou_list_test_case).astype('float').mean(), '.4f')
    dice_case = format(np.array(dice_list_test_case).astype('float').mean(), '.4f')

    print(f'IoU on sick test set is {iou_case}')
    print(f'Dice on sick test set is {dice_case}')

    return iou_list_test_case, dice_list_test_case, iou_case, dice_case


def saliency_map_mul(model, image, cat_id):
    image.requires_grad_()
    model(image)[0][cat_id].backward()
    outputs, _ = torch.max(image.grad.data.abs(), dim=1)
    outputs = outputs.flatten()
    outputs = np.array(outputs.detach().cpu())
    if np.max(outputs) != 0:
        outputs = (outputs - np.min(outputs)) / np.max(outputs)
    outputs = outputs.flatten()

    return outputs


def explain_cases_mul_baseline(test_loader_seg, device, cls_model, threshold, cat_id):
    cls_model.eval()
    iou_list_test_case = []
    dice_list_test_case = []

    for data_test in tqdm(test_loader_seg):
        images_test, labels_test = data_test[0].to(device).float(), np.array(data_test[1]).flatten()
        if labels_test.max() == 0:
            continue
        else:
            binary_list_iou = saliency_map_mul(model=cls_model, image=images_test, cat_id=cat_id)

            threshold_value = np.percentile(binary_list_iou, threshold)
            binary_list_iou[np.where(binary_list_iou >= threshold_value)] = 1
            binary_list_iou[np.where(binary_list_iou != 1)] = 0
            binary_list_iou = binary_list_iou.astype("uint8")
            iou = metrics.jaccard_score(labels_test, binary_list_iou)
            iou = format(iou, '.3f')

            dice = metrics.f1_score(labels_test, binary_list_iou)
            dice = format(dice, '.3f')

            iou_list_test_case.append(iou)
            dice_list_test_case.append(dice)

    iou_case = format(np.array(iou_list_test_case).astype('float').mean(), '.4f')
    dice_case = format(np.array(dice_list_test_case).astype('float').mean(), '.4f')

    print(f'IoU on sick test set is {iou_case}')
    print(f'Dice on sick test set is {dice_case}')

    return iou_list_test_case, dice_list_test_case, iou_case, dice_case


def explain_cases_mul_cons(test_loader_seg, device, cls_model, threshold, cat_id):
    cls_model.eval()
    iou_list_test_case = []
    dice_list_test_case = []

    for data_test in tqdm(test_loader_seg):
        images_test, labels_test, cons_test = data_test[0].to(device).float(), \
                                              np.array(data_test[1]).flatten(), \
                                              np.array(data_test[2]).flatten()

        if labels_test.max() == 0:
            continue
        else:
            binary_list_iou = saliency_map_mul(model=cls_model, image=images_test, cat_id=cat_id)
            binary_list_iou = binary_list_iou * cons_test

            threshold_cons = 100 - round(((224 * 224 * (1-threshold/100)) / sum(cons_test)) * 100)
            if len(np.where(binary_list_iou != 0)[0]) > 0:
                threshold_value = np.percentile(binary_list_iou[np.where(binary_list_iou != 0)[0]], threshold_cons)
            else:
                threshold_value = np.percentile(binary_list_iou, threshold)

            binary_list_iou[np.where(binary_list_iou >= threshold_value)] = 1
            binary_list_iou[np.where(binary_list_iou != 1)] = 0
            binary_list_iou = binary_list_iou.astype("uint8")
            iou = metrics.jaccard_score(labels_test, binary_list_iou)
            iou = format(iou, '.3f')

            dice = metrics.f1_score(labels_test, binary_list_iou)
            dice = format(dice, '.3f')

            iou_list_test_case.append(iou)
            dice_list_test_case.append(dice)

    iou_case = format(np.array(iou_list_test_case).astype('float').mean(), '.4f')
    dice_case = format(np.array(dice_list_test_case).astype('float').mean(), '.4f')

    print(f'IoU on sick test set is {iou_case}')
    print(f'Dice on sick test set is {dice_case}')

    return iou_list_test_case, dice_list_test_case, iou_case, dice_case
