
import numpy as np
from scipy import ndimage


def remove_tiny_les(lesion_image, nvox=2):
    labels, nles = ndimage.label(lesion_image)
    for i in range(1, nles + 1):
        nb_vox = np.sum(lesion_image[labels == i])
        if nb_vox <= nvox:
            lesion_image[labels == i] = 0
    return lesion_image


def get_lesion_bin(nvox):
    if 3 <= nvox <= 10:
        return 'small'
    elif 11 <= nvox <= 50:
        return 'med'
    elif nvox >= 51:
        return 'large'
    else:
        return "small"


def count_lesions(h, t, th):
    """
    Connected component analysis of between prediction `h` and ground truth `t` across lesion bin sizes.
    Lesion counting algorithm:
        for lesion in ground_truth:
            See if there is overlap with a detected lesion
            nb_overlap = number of overlapping voxels with prediction
            if nb_overlap >= 3 or nb_overlap >= 0.5 * lesion_size:
                 lesion is a true positive
            else:
                 lesion is a false negative
        for lesion in prediction:
            if lesion isn't associated with a true positive:
                lesion is a false positive
    Important!!
        h and t should be in their image shape, ie. (256, 256, 64)
            and NOT in a flattened shape, ie. (4194304, 1, 1)
    :param h: network output on range [0,1], shape=(NxMxO)
    :type h: float16, float32, float64
    :param t: ground truth labels, shape=(NxMxO)
    :type t: int16
    :param th: threshold to binarize prediction `h`
    :type th: float16, float32, float64
    :return: dict
    """
    h[h >= th] = 1
    h[h < th] = 0
    h = h.astype(np.int16)

    t = remove_tiny_les(t, nvox=2)
    h0 = h.copy()
    h = ndimage.binary_dilation(h, structure=ndimage.generate_binary_structure(3, 2))

    labels = {}
    nles = {}
    labels['h'], nles['h'] = ndimage.label(h)
    labels['t'], nles['t'] = ndimage.label(t)
    found_h = np.ones(nles['h'], np.int16)
    ntp = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nfp = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nfn = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nb_les = {'all': 0, 'small': 0, 'med': 0, 'large': 0}
    nles_gt = {'all': nles['t'], 'small': 0, 'med': 0, 'large': 0}
    for i in range(1, nles['t'] + 1):
        lesion_size = np.sum(t[labels['t'] == i])
        nles_gt[get_lesion_bin(lesion_size)] += 1
        # list of detected lesions in this area
        h_lesions = np.unique(labels['h'][labels['t'] == i])
        # all the voxels in this area contribute to detecting the lesion
        nb_overlap = h[labels['t'] == i].sum()
        if nb_overlap >= 3 or nb_overlap >= 0.5 * lesion_size:
            nb_les[get_lesion_bin(lesion_size)] += 1
            ntp[get_lesion_bin(lesion_size)] += 1
            for h_lesion in h_lesions:
                if h_lesion != 0:
                    found_h[h_lesion - 1] = 0
        else:
            nfn[get_lesion_bin(lesion_size)] += 1

    for i in range(1, nles['h'] + 1):
        if found_h[i - 1] == 1:
            nb_vox = np.sum(h0[labels['h'] == i])
            nfp[get_lesion_bin(nb_vox)] += 1

    nb_les['all'] = nb_les['small'] + nb_les['med'] + nb_les['large']
    ntp['all'] = ntp['small'] + ntp['med'] + ntp['large']
    nfp['all'] = nfp['small'] + nfp['med'] + nfp['large']
    nfn['all'] = nfn['small'] + nfn['med'] + nfn['large']

    tpr = {}
    fdr = {}
    for s in ntp.keys():
        # tpr (sensitivity)
        if nles_gt[s] != 0:
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