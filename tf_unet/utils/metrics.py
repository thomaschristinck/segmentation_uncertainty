"""
Tensorflow implementation of metrics and utils
"""
import tensorflow as tf
import numpy as np
import numpy as np
import datetime
import time
from collections import defaultdict
import copy
from scipy import ndimage


EPSILON = 1e-5
_smooth = 1


def dice_coef(y_tru, y_prd):
    y_tru = tf.reshape(y_tru, [2, -1])
    y_prd = tf.reshape(y_prd, [2, -1])
    y_prd_pr = tf.sigmoid(y_prd)
    intersection = tf.reduce_sum(y_prd_pr * y_tru, 0)
    union = tf.reduce_sum(y_prd_pr, 0) + tf.reduce_sum(y_tru, 0)
    dice = (2. * intersection + _smooth) / (union + _smooth)
    return tf.reduce_mean(dice)


def dice_coef_loss(y_tru, y_prd):
    return -dice_coef(y_tru, y_prd)


def hard_dice(y_tru, y_prd):
    y_prd = tf.sigmoid(y_prd)
    y_prd = tf.round(y_prd)
    intersection = tf.reduce_sum(y_prd * y_tru)
    union = tf.reduce_sum(y_prd) + tf.reduce_sum(y_tru)
    dice = (2. * intersection + _smooth) / (union + _smooth)
    return dice


def generalized_dice_loss(y_tru, y_prd):
    y_prd_pr = tf.sigmoid(y_prd)

    # w1 = 1. / tf.square((tf.reduce_sum(y_tru)))
    # w0 = 1. / tf.square((tf.reduce_sum(1. - y_tru)))
    w1 = 8
    w0 = 1

    numerator = w1 * tf.reduce_sum(y_tru * y_prd_pr) + \
                w0 * tf.reduce_sum((1. - y_tru) * (1. - y_prd_pr))

    denominator = w1 * tf.reduce_sum(y_tru * y_prd_pr) + \
                  w0 * tf.reduce_sum((1. - y_tru) + (1. - y_prd_pr))

    gdl = 1 - 2 * numerator / denominator
    return gdl


def weighted_mc_xentropy(y_tru, mu, log_var, nb_mc, weight, batch_size):
    mu_mc = tf.tile(mu, [1, 1, 1, 1, nb_mc])
    std = tf.exp(log_var)
    noise = tf.random_normal((batch_size, 192, 192, 64, nb_mc)) * std
    prd = mu_mc + noise

    y_tru = tf.tile(y_tru, [1, 1, 1, 1, nb_mc])
    mc_x = tf.nn.weighted_cross_entropy_with_logits(targets=y_tru, logits=prd, pos_weight=weight)
    # mean across mc samples
    mc_x = tf.reduce_mean(mc_x, -1)
    # mean across every thing else
    return tf.reduce_mean(mc_x)


def cross_entropy(y_tru, y_prd, class_weight):
    flat_prd = tf.reshape(y_prd, [-1, 1])
    flat_labels = tf.reshape(y_tru, [-1, 1])
    _epsilon = tf.convert_to_tensor(EPSILON, tf.float32)

    if class_weight is not None:
        # transform to multi class
        flat_prd_0 = tf.constant(1., dtype=tf.float32) - flat_prd
        flat_multi_prd = tf.concat([flat_prd_0, flat_prd], axis=1)

        flat_labels_0 = tf.constant(1., dtype=tf.float32) - flat_labels
        flat_multi_label = tf.concat([flat_labels_0, flat_labels], axis=1)

        # transform back to logits
        flat_multi_prd = tf.clip_by_value(flat_multi_prd, _epsilon, 1 - _epsilon)
        flat_multi_prd = tf.log(flat_multi_prd / (1 - flat_multi_prd))

        # add class weighting
        class_weights = tf.constant(np.array(class_weight, np.float32))
        weighted_labels = tf.multiply(flat_multi_label, class_weights)
        # weight_map = tf.reduce_sum(weight_map, axis=1)
        weighted_loss = tf.nn.softmax_cross_entropy_with_logits(logits=flat_multi_prd, labels=weighted_labels)
        # weighted_loss = tf.multiply(loss_map, weight_map)
        loss = tf.reduce_mean(weighted_loss)

    else:
        # convert to logits
        flat_prd_logits = tf.clip_by_value(flat_prd, _epsilon, 1 - _epsilon)
        flat_prd_logits = tf.log(flat_prd_logits / (1 - flat_prd_logits))

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_prd_logits, labels=flat_labels))

    return loss

def count_lesions(netseg, target, thresh):
    """
    Comparing segmentations volumetrically
    Connected component analysis of between prediction `netseg` and ground truth `target` across lesion bin sizes.
    :param netseg: network output on range [0,1], shape=(NxMxO)
    :type netseg: float16, float32, float64
    :param target: ground truth labels, shape=(NxMxO)
    :type target: int16
    :param thresh: threshold to binarize prediction `h`
    :type thresh: float16, float32, float64
    :return: dict

    **************** Courtesy of Tanya Nair ********************
    """

    netseg[netseg >= thresh] = 1
    netseg[netseg < thresh] = 0
    
    # To Test netseg = gt_mask (should get ROC as tpr = 1 and fdr = 0 everywhere)
    target, _ = remove_tiny_les(target, nvox=2)
    netseg0 = netseg.copy()
    netseg = ndimage.binary_dilation(netseg, structure=ndimage.generate_binary_structure(3, 2))
    labels = {}
    nles = {}
    labels['target'], nles['target'] = ndimage.label(target)
    labels['netseg'], nles['netseg'] = ndimage.label(netseg)
    found_h = np.ones(nles['netseg'], np.int16)
    ntp = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nfp = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nfn = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nb_les = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nles_gt = {'all': nles['target'], 'small': 0, 'med': 0, 'large': 0}

    # Go through ground truth segmentation masks and count true positives/false negatives
    for i in range(1, nles['target'] + 1):
        gt_lesion_size = np.sum(target[labels['target'] == i])
        nles_gt[get_lesion_bin(gt_lesion_size)] += 1
        # List of detected lesions in this area
        h_lesions = np.unique(labels['netseg'][labels['target'] == i])
        # All the voxels in this area contribute to detecting the lesion
        nb_overlap = netseg[labels['target'] == i].sum()
        if nb_overlap >= 3 or nb_overlap >= 0.5 * gt_lesion_size:
            nb_les[get_lesion_bin(gt_lesion_size)] += 1
            ntp[get_lesion_bin(gt_lesion_size)] += 1
            for h_lesion in h_lesions:
                if h_lesion != 0:
                    found_h[h_lesion - 1] = 0
        else:
            nfn[get_lesion_bin(gt_lesion_size)] += 1

    for i in range(1, nles['netseg'] + 1):
        nb_vox = np.sum(netseg0[labels['netseg'] == i])
        if found_h[i - 1] == 1:
            nfp[get_lesion_bin(nb_vox)] += 1

    nb_les['all'] = nb_les['small'] + nb_les['med'] + nb_les['large']
    ntp['all'] = ntp['small'] + ntp['med'] + ntp['large']
    nfp['all'] = nfp['small'] + nfp['med'] + nfp['large']
    nfn['all'] = nfn['small'] + nfn['med'] + nfn['large']
    #print('tp', ntp)
    #print('fp', nfp)
    #print('nb of les', nb_les)
    tpr = {}
    fdr = {}
    for s in ntp.keys():
        # tpr (sensitivity)
        if nb_les[s] != 0:
            tpr[s] = ntp[s] / nles_gt[s]
        elif nles_gt[s] == 0 and ntp[s] == 0:
            tpr[s] = 1
        else:
            tpr[s] = 0
        # ppv (1-fdr)
        if ntp[s] + nfp[s] != 0:
            ppv = ntp[s] / (ntp[s] + nfp[s])
        elif ntp[s] == 0:
            ppv = 1
        else:
            ppv = 0
        fdr[s] = 1 - ppv
 
    return {'ntp': ntp, 'nfp': nfp, 'nfn': nfn, 'fdr': fdr, 'tpr': tpr, 'nles': nb_les, 'nles_gt': nles_gt}


def get_lesion_bin(nvox):
    # Lesion bin - 0 for small lesions, 1 for medium, 2 for large
    if 3 <= nvox <= 10:
        return 'small'
    elif 11 <= nvox <= 50:
        return 'med'
    elif nvox >= 51:
        return 'large'
    else:
        return 'small'


def remove_tiny_les(lesion_image, nvox=2):
    labels, nles = ndimage.label(lesion_image)
    class_ids = np.zeros([nles, 1], dtype=np.int32)

    for i in range(1, nles + 1):
        nb_vox = np.sum(lesion_image[labels == i])
        if nb_vox <= nvox:
            lesion_image[labels == i] = 0
        
        if nb_vox > 0:
            # Classify as lesion. There is a bug here where if we set lesions less than two voxels big
            # to the background class (nb_vox > nvox), then we crash
            class_ids[i-1] = 1

    class_ids = np.asarray(class_ids)

    if class_ids.size == 0:
        class_ids = np.zeros([1, 1], dtype=np.int32)
        class_ids[0] = 0

    return lesion_image, class_ids
