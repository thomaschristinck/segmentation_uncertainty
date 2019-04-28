from tf_unet.analyze.cca import cca_img_no_unc as cca_img
from tf_unet.analyze.cca import global_dice
from tf_unet.utils.metrics import count_lesions
import tensorflow as tf
from timeit import default_timer as timer
import csv
import numpy as np
import nrrd
from tf_unet.utils.np_utils import sigmoid
from tf_unet.utils.unc_metrics import mi_uncertainty, entropy, prd_variance, prd_uncertainty
import os
from os.path import join
import matplotlib.pyplot as plt
from pprint import pprint

_OPTS = {'space': 'RAS', 'space directions': [(1, 0, 0), (0, 1, 0), (0, 0, 3)]}


class BunetAnalyzer:
    def __init__(self, model, model_checkpoint, data_gen, out_dir, nb_mc):
        self.__model = model
        self.__model_checkpoint = model_checkpoint
        self.__data_gen = data_gen
        self.__out_dir = out_dir
        self.__nb_mc = nb_mc

    @staticmethod
    def _get_unc_img(mu_mcs, log_var_mcs):
        bald = bald_uncertainty(sigmoid(mu_mcs))
        ent = entropy(sigmoid(mu_mcs))
        prd_var = prd_variance(log_var_mcs)
        prd_unc = prd_uncertainty(mu_mcs, prd_var)
        return {'bald': bald, 'ent': ent, 'prd_var': prd_var, 'prd_unc': prd_unc}

    @staticmethod
    def _get_prd_stats(y, h):
        stats = cca_img(h, y, 0.5)
        dice = global_dice(h, y)
        stats.update({'dice': dice})
        return stats

    @staticmethod
    def _clip_at_thresh(x, a, thresh):
        x[a >= thresh] = 1
        x[a < thresh] = 0
        return x

    @staticmethod
    def _keep_below_thresh(x, a, thresh):
        x[a >= thresh] = 0
        return x

    def cca(self, out_file, thresh):
        with tf.Session() as sess:

            self.__model.restore(sess, tf.train.latest_checkpoint(self.__model_checkpoint))
            if not os.path.isdir(self.__out_dir):
                os.makedir(self.__out_dir)
            with open(join(self.__out_dir, out_file), 'w', newline='') as csvfile:
                stats_writer = csv.writer(csvfile, delimiter=',')
                stats_writer.writerow(['subj', 'tp', 'mean_fdr', 'mean_tpr', 'mean_dice'])
                start = timer()
                nb_subj = 0
                ustats = {'fdr': 0, 'tpr': 0, 'dice': 0}
                for subj, tp, x, y in self.__data_gen:
                    x_mc = np.repeat(x, self.__nb_mc, 0)
                    mu_mcs = sess.run(self.__model.predictor,
                                      feed_dict={self.__model.x: x_mc, self.__model.keep_prob: 0.5})
                    mu_mcs = np.asarray(mu_mcs, np.float32)[..., 0]
                    h = sigmoid(np.mean(mu_mcs, 0))
                    y = y[0, ..., 0]
                    # Sigmoid thresholding
                    h_unc_thresh = self._clip_at_thresh(h, h, thresh)
                    stats = self._get_prd_stats(y, h_unc_thresh)
                    ustats['fdr'] += stats['fdr']['all']
                    ustats['tpr'] += stats['tpr']['all']
                    ustats['dice'] += stats['dice']
                    nb_subj += 1
                    print('completed subject {}     {:.2f}m'.format(nb_subj, (timer() - start) / 60))
                    stats_writer.writerow([subj[0], tp[0], stats['fdr']['all'], stats['tpr']['all'], stats['dice']])
                stats_writer.writerow(
                    ['mean_subj', '_', ustats['fdr'] / nb_subj, ustats['tpr'] / nb_subj, ustats['dice'] / nb_subj])
        print("completed in {:.2f}m".format((timer() - start) / 60))

    def cca_img_no_unc(h, t, th):
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


    def roc(self, out_file, thresh, nb_subjs):
        

        #subj, tp, x, y = self.__data_gen.get_next()
        with tf.Session() as sess:
            #self.__model.restore(sess, tf.train.latest_checkpoint(self.__model_checkpoint))
            tf.global_variables_initializer()
            element = self.__data_gen.get_next()
            self.__model.restore(sess, tf.train.latest_checkpoint(self.__model_checkpoint))
            if not os.path.isdir(self.__out_dir):
                os.mkdir(self.__out_dir)
            with open(join(self.__out_dir, out_file), 'w', newline='') as csvfile:
                stats_writer = csv.writer(csvfile, delimiter=',')
                stats_writer.writerow(['unc_thresh', 'mean_fdr', 'mean_tpr', 'mean_dice'])
                start = timer()
                nb_subj = 0
                ustats = {}
                [ustats.update({t: {'fdr': 0, 'tpr': 0, 'dice': 0}}) for t in thresh]

                nb_thrs = len(thresh)
                fpr = np.empty((nb_subjs, nb_thrs))
                tpr = np.empty((nb_subjs, nb_thrs))
                fdr = np.empty((nb_subjs, nb_thrs))
                fdr_lesions = np.empty((nb_subjs, nb_thrs))
                tpr_lesions = np.empty((nb_subjs, nb_thrs))
                fdr_lesions_s = np.empty((nb_subjs, nb_thrs))
                tpr_lesions_s = np.empty((nb_subjs, nb_thrs))
                fdr_lesions_m = np.empty((nb_subjs, nb_thrs))
                tpr_lesions_m = np.empty((nb_subjs, nb_thrs))
                fdr_lesions_l = np.empty((nb_subjs, nb_thrs))
                tpr_lesions_l = np.empty((nb_subjs, nb_thrs))

                i = 0
                for sub in range(nb_subjs): # or try:
                    subj, tp, img, y = sess.run(element)

                    x_mc = np.repeat(img, self.__nb_mc, 0)
                    
                    mu_mcs = sess.run(self.__model.predictor,
                                        feed_dict={self.__model.x: x_mc, self.__model.keep_prob: 0.5})
                    mu_mcs = np.asarray(mu_mcs, np.float32)[..., 0]

                    #ent = entropy(sigmoid(mu_mcs))
                    h = sigmoid(np.mean(mu_mcs, 0))
                    y = y[0, ..., 0]
                    # Got prediction; now iterate through thresholds
                    for j, t in enumerate(thresh):
                        #h_unc_thresh = self._keep_below_thresh(h, ent, t)
                        #stats = self._get_prd_stats(y, h_unc_thresh)
                            
                        import copy
                
                        a = copy.copy(h)
                        b = copy.copy(y)

                        lesion_stats = count_lesions(netseg=a.astype(np.float32), target=b.astype(np.int16), thresh=t)

                        tpr_lesions[i, j] = lesion_stats['tpr']['all']
                        fdr_lesions[i, j] = lesion_stats['fdr']['all']
                
                        tpr_lesions_s[i, j] = lesion_stats['tpr']['small']
                        fdr_lesions_s[i, j] = lesion_stats['fdr']['small']
                            
                        tpr_lesions_m[i, j] = lesion_stats['tpr']['med']
                        fdr_lesions_m[i, j] = lesion_stats['fdr']['med']
                            
                        tpr_lesions_l[i, j] = lesion_stats['tpr']['large']
                        fdr_lesions_l[i, j] = lesion_stats['fdr']['large']

                    i += 1
                    if nb_subj % 5 == 0:
                        print('completed subject {}     {:.2f}m'.format(i, (timer() - start) / 60))

           
            fdr_lesions_mean = np.mean(fdr_lesions, axis=0)
            tpr_lesions_mean = np.mean(tpr_lesions, axis=0)
            np.save('/usr/local/data/thomasc/outputs/fdr_lesions_mean.npy', fdr_lesions_mean)
            np.save('/usr/local/data/thomasc/outputs/tpr_lesions_mean.npy', tpr_lesions_mean)
                    
            fdr_lesions_mean_s = np.mean(fdr_lesions_s, axis=0)
            tpr_lesions_mean_s = np.mean(tpr_lesions_s, axis=0)
            np.save('/usr/local/data/thomasc/outputs/fdr_lesions_mean_s.npy', fdr_lesions_mean_s)
            np.save('/usr/local/data/thomasc/outputs/tpr_lesions_mean_s.npy', tpr_lesions_mean_s)

            fdr_lesions_mean_m = np.mean(fdr_lesions_m, axis=0)
            tpr_lesions_mean_m = np.mean(tpr_lesions_m, axis=0)
            np.save('/usr/local/data/thomasc/outputs/fdr_lesions_mean_m.npy', fdr_lesions_mean_m)
            np.save('/usr/local/data/thomasc/outputs/tpr_lesions_mean_m.npy', tpr_lesions_mean_m)

            fdr_lesions_mean_l = np.mean(fdr_lesions_l, axis=0)
            tpr_lesions_mean_l = np.mean(tpr_lesions_l, axis=0)
            np.save('/usr/local/data/thomasc/outputs/fdr_lesions_mean_l.npy', fdr_lesions_mean_l)
            np.save('/usr/local/data/thomasc/outputs/tpr_lesions_mean_l.npy', tpr_lesions_mean_l)

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.plot(fdr_lesions_mean, tpr_lesions_mean, label='lesion level-all')
            plt.plot(fdr_lesions_mean_s, tpr_lesions_mean_s, label='lesion level-small')
            plt.plot(fdr_lesions_mean_m, tpr_lesions_mean_m, label='lesion level-med')
            plt.plot(fdr_lesions_mean_l, tpr_lesions_mean_l, label='lesion level-large') 
            plt.legend(loc="lower right")
            plt.xlabel('fdr')
            plt.ylabel('tpr')
            plt.title('Baseline U-Net Segmentation')
            major_ticks = np.arange(0, 1, 0.1)
            minor_ticks = np.arange(0, 1, 0.02)
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_yticks(major_ticks)
            ax.set_yticks(minor_ticks, minor=True)
            ax.grid(which='both')
            ax.grid(which='minor', alpha=0.2)
            ax.grid(which='major', alpha=0.5)
            fig.savefig(os.path.join('/usr/local/data/thomasc/outputs', "roc_curve_segmentation.png")) 

    def roc_sigmoid(self, out_file, thresh_start, thresh_stop, thresh_step):
        with tf.Session() as sess:
            self.__model.restore(sess, tf.train.latest_checkpoint(self.__model_checkpoint))
            with open(join(self.__out_dir, out_file), 'w', newline='') as csvfile:
                stats_writer = csv.writer(csvfile, delimiter=',')
                stats_writer.writerow(['unc_thresh', 'mean_fdr', 'mean_tpr', 'mean_dice'])
                start = timer()
                nb_subj = 0
                thresh = np.arange(thresh_start, thresh_stop, thresh_step)
                ustats = {}
                [ustats.update({t: {'fdr': 0, 'tpr': 0, 'dice': 0}}) for t in thresh]
                for subj, tp, x, y in self.__data_gen:
                    x_mc = np.repeat(x, self.__nb_mc, 0)
                    mu_mcs = sess.run(self.__model.predictor,
                                      feed_dict={self.__model.x: x_mc, self.__model.keep_prob: 0.5})
                    mu_mcs = np.asarray(mu_mcs, np.float32)[..., 0]
                    h = sigmoid(np.mean(mu_mcs, 0))
                    y = y[0, ..., 0]
                    for t in thresh:
                        h_unc_thresh = self._clip_at_thresh(h, h, t)
                        stats = self._get_prd_stats(y, h_unc_thresh)
                        ustats[t]['fdr'] += stats['fdr']['all']
                        ustats[t]['tpr'] += stats['tpr']['all']
                        ustats[t]['dice'] += stats['dice']
                    nb_subj += 1
                    print('completed subject {}     {:.2f}m'.format(nb_subj, (timer() - start) / 60))
                for t in thresh:
                    stats_writer.writerow(
                        [t, ustats[t]['fdr'] / nb_subj, ustats[t]['tpr'] / nb_subj, ustats[t]['dice'] / nb_subj])
        print("completed in {:.2f}m".format((timer() - start) / 60))

    def write_to_nrrd(self, nb_subjs):
        with tf.Session() as sess:
            print('-----------Restoring from checkpoint------------')
            tf.global_variables_initializer()
            element = self.__data_gen.get_next()
            self.__model.restore(sess, tf.train.latest_checkpoint(self.__model_checkpoint))
            if not os.path.isdir(self.__out_dir):
                os.mkdir(self.__out_dir)
            out_dir = self.__out_dir
            start = timer()
            nb_subj = 0
            print('-----------Starting predictions for each subject------------')

            for sub in range(nb_subjs):
        
                subj, tp, x, y = sess.run(element)

                x_mc = np.repeat(x, self.__nb_mc, 0)

                mu_mcs, log_var_mcs = sess.run([self.__model.predictor, self.__model.log_variance],
                                             feed_dict={self.__model.x: x_mc, self.__model.keep_prob: 0.5})

                mu_mcs = []
                log_var_mcs = []

                for i in range(0, self.__nb_mc):
                   mu_temp, log_temp = sess.run([self.__model.predictor, self.__model.log_variance],
                                               feed_dict={self.__model.x: x, self.__model.keep_prob: 0.5})
                   mu_mcs.append(mu_temp)
                   log_var_mcs.append(log_temp)

                mu_mcs = np.asarray(mu_mcs)
                log_var_mcs = np.asarray(log_var_mcs)


                mu_mcs = mu_mcs[..., 0]
                mu_mcs = np.squeeze(mu_mcs, axis=1)    
                var_mcs = np.var(sigmoid(mu_mcs), 0)
                
                log_var_mcs = log_var_mcs[..., 0]
                log_var_mcs = np.squeeze(log_var_mcs, axis=1)
                mi = mi_uncertainty(sigmoid(mu_mcs))
                ent = entropy(sigmoid(mu_mcs))
                prd_var = prd_variance(log_var_mcs)
               
                h = sigmoid(np.mean(mu_mcs, 0))

                h = h.astype(np.float32)
                h[h < 0.00001] = 0
                var_mcs = var_mcs.astype(np.float32)


                # Here we clip at the threshold - it might be better not to clip and leave everything as the original sigmoid outputs
                # Remove this line for no thresholding

                #h = self._clip_at_thresh(h, h, thresh=0.5)
                #y = y[0, ..., 0]
                t2 = x[0, ..., 1]
                t1 = x[0, ..., 0]
                flair = x[0, ..., 2]
                pd = x[0, ..., 3]

                subj = str(subj[0])
                print(subj)
                tp = str(tp[0])
                print(tp)

                #subj = subj.eval(session=sess).decode('utf-8')
                #tp = tp.eval().decode('utf-8')

                nrrd.write(join(out_dir, subj + '_' + tp+ '_uncmcvar.nrrd'), var_mcs, options=_OPTS) # unc measure, variance of mc samples
                nrrd.write(join(out_dir, subj + '_' + tp + '_t2.nrrd'), t2, options=_OPTS) # t2 mri on its own
                nrrd.write(join(out_dir, subj + '_' + tp + '_t1.nrrd'), t1, options=_OPTS) # t2 mri on its own
                nrrd.write(join(out_dir, subj + '_' + tp + '_flair.nrrd'), flair, options=_OPTS) # t2 mri on its own
                nrrd.write(join(out_dir, subj + '_' + tp + '_pdw.nrrd'), pd, options=_OPTS) # t2 mri on its own
                # nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_h.nrrd'), h, options=_OPTS) # don't use this
                nrrd.write(join(out_dir, subj + '_' + tp + '_target.nrrd'), y, options=_OPTS) # 'target' ground truth lesion labels
                #nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_uncmi.nrrd'), mi, options=_OPTS) # mutual information unc. measure
                #nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_uncent.nrrd'), ent, options=_OPTS) # entropy uncertainty measure
                #nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_uncprvar.nrrd'), prd_var, options=_OPTS) # predicted variance unc. measure (2nd output of the model)

                #h_mu_mcs = np.mean(sigmoid(mu_mcs), 0)
                #h_mu_mcs = h_mu_mcs.astype(np.float32)
                nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_netseg.nrrd'), h, options=_OPTS) # network's segmentation ( y_hat). Use this one!!
                # don't use this
                # mu_nomcs = sess.run(self.__model.predictor, feed_dict={self.__model.x: x, self.__model.keep_prob: 1.0})
                # h_nomcs = sigmoid(mu_nomcs)[0, ..., 0]
                # nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_h_nomcs.nrrd'), h_nomcs, options=_OPTS)

                nb_subj += 1
                if nb_subj % 20 == 0:
                    print('completed subject {}     {:.2f}m'.format(nb_subj, (timer() - start) / 60))
        print("completed in {:.2f}m".format((timer() - start) / 60))
