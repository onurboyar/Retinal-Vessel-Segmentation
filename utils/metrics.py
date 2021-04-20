import os, cv2
from keras import backend as keras
import skimage.io as io
import numpy as np

def dice_coef(y_true, y_pred):
    smooth = 0.0
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)

def jacard(y_true, y_pred):

    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum ( y_true_f * y_pred_f)
    union = keras.sum ( y_true_f + y_pred_f - y_true_f * y_pred_f)

    return intersection/union


def dice(true_mask, pred_mask, non_seg_score=1.0):

    assert true_mask.shape == pred_mask.shape

    true_mask = np.asarray(true_mask).astype(np.bool)
    pred_mask = np.asarray(pred_mask).astype(np.bool)

    im_sum = true_mask.sum() + pred_mask.sum()
    if im_sum == 0:
        return non_seg_score

    intersection = np.logical_and(true_mask, pred_mask)
    return 2. * intersection.sum() / im_sum

def mean_dice(true_path, pred_path):
    sum = 0
    for img in sorted(os.listdir(pred_path)):
        true_tmp = cv2.imread(true_path + '/' + img, 0)
        pred_tmp = cv2.imread(pred_path + '/' + img, 0)
        a = dice(true_tmp, pred_tmp)
        print(a)
        sum += a
    return sum/len(os.listdir(true_path))
