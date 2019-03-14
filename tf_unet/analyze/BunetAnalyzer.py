from tf_unet.analyze.cca import cca_img_no_unc as cca_img
from tf_unet.analyze.cca import global_dice
import tensorflow as tf
from timeit import default_timer as timer
import csv
import numpy as np
import nrrd
from tf_unet.utils.np_utils import sigmoid
from tf_unet.utils.unc_metrics import mi_uncertainty, entropy, prd_variance, prd_uncertainty
from os.path import join

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

    def roc(self, out_file, thresh_start, thresh_stop, thresh_step):
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
                    ent = entropy(sigmoid(mu_mcs))
                    h = sigmoid(np.mean(mu_mcs, 0))
                    y = y[0, ..., 0]
                    for t in thresh:
                        h_unc_thresh = self._keep_below_thresh(h, ent, t)
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

    def write_to_nrrd(self, out_dir, sess):
        with tf.Session() as sess:
            print('-----------Restoring from checkpoint------------')
            self.__model.restore(sess, tf.train.latest_checkpoint(self.__model_checkpoint))
            start = timer()
            nb_subj = 0
            print('-----------Starting predictions for each subject------------')

            for subj, tp, x, y in self.__data_gen:
                
                x_mc = np.repeat(x, self.__nb_mc, 0)

                mu_mcs, log_var_mcs = sess.run([self.__model.predictor, self.__model.log_variance],
                                             feed_dict={self.__model.x: x_mc, self.__model.keep_prob: 0.5})

                mu_mcs = []
                log_var_mcs = []

                #for i in range(0, self.__nb_mc):
                #    mu_temp, log_temp = sess.run([self.__model.predictor, self.__model.log_variance],
                #                                feed_dict={self.__model.x: x, self.__model.keep_prob: 0.5})
                #    mu_mcs.append(mu_temp)
                #    log_var_mcs.append(log_temp)

                #mu_mcs = np.asarray(mu_mcs)
                #log_var_mcs = np.asarray(log_var_mcs)

                """
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
                """

                # Here we clip at the threshold - it might be better not to clip and leave everything as the original sigmoid outputs
                # Remove this line for no thresholding

                #h = self._clip_at_thresh(h, h, thresh=0.5)
                #y = y[0, ..., 0]
                t2 = x[0, ..., 1]
                t1 = x[0, ..., 0]
                flair = x[0, ..., 2]
                pd = x[0, ..., 3]

                subj = subj.eval(session=sess).decode('utf-8')
                tp = tp.eval().decode('utf-8')

                #nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_uncmcvar.nrrd'), var_mcs, options=_OPTS) # unc measure, variance of mc samples
                #nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_t2.nrrd'), t2, options=_OPTS) # t2 mri on its own
                nrrd.write(join(out_dir, subj + '_' + tp + '_t1.nrrd'), t1, options=_OPTS) # t2 mri on its own
                nrrd.write(join(out_dir, subj + '_' + tp + '_flair.nrrd'), flair, options=_OPTS) # t2 mri on its own
                nrrd.write(join(out_dir, subj + '_' + tp + '_pdw.nrrd'), pd, options=_OPTS) # t2 mri on its own
                # nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_h.nrrd'), h, options=_OPTS) # don't use this
                #nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_target.nrrd'), y, options=_OPTS) # 'target' ground truth lesion labels
                #nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_uncmi.nrrd'), mi, options=_OPTS) # mutual information unc. measure
                #nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_uncent.nrrd'), ent, options=_OPTS) # entropy uncertainty measure
                #nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_uncprvar.nrrd'), prd_var, options=_OPTS) # predicted variance unc. measure (2nd output of the model)

                #h_mu_mcs = np.mean(sigmoid(mu_mcs), 0)
                #h_mu_mcs = h_mu_mcs.astype(np.float32)
                #nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_netseg.nrrd'), h, options=_OPTS) # network's segmentation ( y_hat). Use this one!!
                # don't use this
                # mu_nomcs = sess.run(self.__model.predictor, feed_dict={self.__model.x: x, self.__model.keep_prob: 1.0})
                # h_nomcs = sigmoid(mu_nomcs)[0, ..., 0]
                # nrrd.write(join(out_dir, subj[0] + '_' + tp[0] + '_h_nomcs.nrrd'), h_nomcs, options=_OPTS)

                nb_subj += 1
                if nb_subj % 20 == 0:
                    print('completed subject {}     {:.2f}m'.format(nb_subj, (timer() - start) / 60))
        print("completed in {:.2f}m".format((timer() - start) / 60))
