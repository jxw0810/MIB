import numpy as np
import tensorflow as tf
import keras.backend as K

#自定义loss及评估函数
from keras.losses import binary_crossentropy
from sklearn.metrics import roc_curve, auc


def AUC(y_true, y_pred):
    test_mask = K.flatten(y_true)
    pred_mask = K.flatten(y_pred)
    fpr, tpr, thresholds = roc_curve(test_mask, pred_mask, pos_label=1)
    roc_auc = auc(fpr, tpr)

    return roc_auc

def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss
    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed

def class_weighted_pixelwise_crossentropy(target, output):
    output = tf.clip_by_value(output, 10e-8, 1.-10e-8)
    weights = [0.8, 0.2]
    return -tf.reduce_sum(target * weights * tf.log(output))

# smooth 参数防止分母为0
def dice_coef(y_true, y_pred, smooth=1):
    smooth = 0.0005
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def dice_ratio(y_true, y_pred):
    """
    define the dice ratio
    :param y_pred: segmentation result
    :param y_true: ground truth
    :return:
    """
    y_pred = y_pred.flatten()
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0

    y_true = y_true.flatten()
    y_true[y_true > 0.5] = 1
    y_true[y_true <= 0.5] = 0

    same = (y_pred * y_true).sum()

    dice = 2*float(same)/float(y_true.sum() + y_pred.sum())

    return dice

def focal_loss(y_true, y_pred, gamma=2., alpha=.25):

    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    # return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1 - pt_0))

## intersection over union
def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean( (intersection + eps) / (union + eps), axis=0)

def dice_p_bce(y_true, y_pred):

    return 1e-3 * binary_crossentropy(y_true, y_pred) + dice_coef(y_true, y_pred)

def dice_p_focal(y_true, y_pred):

    return binary_crossentropy(y_true, y_pred) + focal_loss(y_true, y_pred)


# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
# -> the score is computed for each class separately and then summed
# alpha=beta=0.5 : dice coefficient
# alpha=beta=1   : tanimoto coefficient (also known as jaccard)
# alpha+beta=1   : produces set of F*-scores

def binary_crossentropy(y_true, y_pred):
    e=1.0
    return K.mean(-(y_true*K.log(y_pred+K.epsilon())+
                    e*(1-y_true)*K.log(1-y_pred+K.epsilon())),axis=-1)

def tversky(y_true, y_pred, smooth=1):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.5
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)



def focal_tversky(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)

# Generalized Dice loss
def generalized_dice_coeff(y_true, y_pred):
    Ncl = y_pred.shape[-1]
    w = K.zeros(shape=(Ncl,))
    w = K.sum(y_true, axis=(0,1,2))
    w = 1/(w**2+0.000001)
    # Compute gen dice coef:
    numerator = y_true*y_pred
    numerator = w * K.sum(numerator, (0, 1, 2, 3))
    numerator = K.sum(numerator)
    denominator = y_true+y_pred
    denominator = w*K.sum(denominator, (0, 1, 2, 3))
    denominator = K.sum(denominator)
    gen_dice_coef = 2*numerator/denominator
    return gen_dice_coef

def generalized_dice_loss(y_true, y_pred):
    return 1 - generalized_dice_coeff(y_true, y_pred)

import cv2 as cv
def mask2_gray(mask, input_type=None):

        if input_type is 'pred':
            mask0 = mask[:, :, 0]
            mask1 = mask[:, :, 1]
            mask = mask1 - mask0

            # print("mask.shape: ", mask.shape)
            # mask = np.argmax(mask, axis=-1)
            # print( mask[mask > 0.5] )
            mask[mask > 0] = 255
            mask[mask <= 0] = 0
            # mask[mask < 0] = 255
            # mask[mask >= 0] = 0
        mask = mask.astype(dtype=np.uint8)
        rst = mask.copy()
        cv.normalize(mask, rst, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        return rst

def p_r_f1_iou(test_s_y, preds_s):

    pred_s = np.zeros((preds_s.shape[0], preds_s.shape[1], preds_s.shape[2]))
    for i in range(preds_s.shape[0]):
        pred = preds_s[i]
        pred = mask2_gray(pred, input_type='pred')
        pred_s[i] = pred

    # 二值分割图是一个波段的黑白图，正样本值为1，负样本值为0
    # 通过矩阵的逻辑运算分别计算出tp,tn,fp,fn
    test_s = np.argmax(test_s_y, axis=-1)
    seg_inv, gt_inv = np.logical_not(pred_s), np.logical_not(test_s)
    true_pos = float(np.logical_and(pred_s, test_s).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(pred_s, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, test_s).sum()

    # 然后根据公式分别计算出这几种指标
    Precision = true_pos / (true_pos + false_pos + 1e-6)
    Recall = true_pos / (true_pos + false_neg + 1e-6)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg + 1e-6)
    F1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    IoU = true_pos / (true_pos + false_neg + false_pos + 1e-6)

    return Precision, Recall, accuracy, F1, IoU


