import numpy as np
import tensorflow as tf
from os.path import join, exists, isdir
from os import listdir
from timeit import default_timer as timer
import nibabel as nib
import sys

DEFAULT_DATA_DIR = '/usr/local/data/thomasc/paul/ASCEND' #'/cim/data/neurorx/101MS326/imaging_data' # 
IMG_TAG = "_ISPC-stx152lsq6.mnc.gz"
IMG_NII_TAG = "_norm_ANAT-brain_ISPC-stx152lsq6.nii"
LES_TAG = "_ct2f_ISPC-stx152lsq6.mnc.gz"
LES_NII_TAG = "_ct2f_ISPC-stx152lsq6.nii"
GAD_TAG = "_gvf_ISPC-stx152lsq6.mnc.gz"
DATASET_TAG = "101MS326_"
DEFAULT_TPS = ['screening', 'w24', 'w48', 'w72', 'w96', 'w108']
MODALITIES = ['t1p', 't2w', 'flr', 'pdw']

class Converter:
    def __init__(self, tfrecord, data_dir=DEFAULT_DATA_DIR, img_dtype=np.float32, label_dtype=np.int16):
        self._tfrecord = tfrecord
        self._data_dir = data_dir
        self._img_dtype = img_dtype
        self._label_dtype = label_dtype
        self._data_shape = [60, 256, 256]

    @staticmethod
    def _bytes_feature(x):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))

    @staticmethod
    def _int64_feature(x):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[x]))

    def run(self):
        shape = {0: self._int64_feature(self._data_shape[0]),
                 1: self._int64_feature(self._data_shape[1]),
                 2: self._int64_feature(self._data_shape[2])}

        writer = tf.python_io.TFRecordWriter(self._tfrecord)#,
                                             #options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP))

        sites = [i for i in listdir(self._data_dir)]
        subjs = [[site,subj] for site in sites for subj in listdir(join(self._data_dir,site)) if '.scannerdb' not in subj]
        count = 0
        start = timer()
        subj_list = [] 
        for site,subj in subjs:
            first_time = True 
            for tp in DEFAULT_TPS:
                if isdir(join(self._data_dir,site,subj,tp)):
                    ct2f_pth = join(self._data_dir,site,subj,tp,DATASET_TAG+site+"_"+subj+"_"+tp+LES_TAG)
                    flr_pth = join(self._data_dir,site,subj,tp,DATASET_TAG+site+"_"+subj+"_"+tp+"_flr"+IMG_TAG)
                    pdw_pth = join(self._data_dir,site,subj,tp,DATASET_TAG+site+"_"+subj+"_"+tp+"_pdw"+IMG_TAG)
                    t1c_pth = join(self._data_dir,site,subj,tp,DATASET_TAG+site+"_"+subj+"_"+tp+"_t1p"+IMG_TAG)
                    t2w_pth = join(self._data_dir,site,subj,tp,DATASET_TAG+site+"_"+subj+"_"+tp+"_t2w"+IMG_TAG)
                    
                    if np.all([exists(ct2f_pth), exists(flr_pth), exists(pdw_pth), exists(t1c_pth), exists(t2w_pth)]) and (not(site=='152-OHS-1' and subj=='subject_152-001') and not(subj.endswith('MR05')) and not(subj.startswith('Dummy'))):
                        if first_time:
                            subj_list.append(subj)
                            first_time = False
                        data = []
                        if subj.startswith('Dummy') or subj.endswith('MR05'):
                            print("ERROR - Kept dummy subject")
                        if subj == 'nii_9590710':
                            for m in MODALITIES:
                                img = nib.load(join(self._data_dir,site,subj,tp,DATASET_TAG+site+"_"+subj+"_"+tp+"_"+m+IMG_NII_TAG)).get_data().astype(np.float32)
                                img = (img-img.min()) / (img.max()-img.min())
                                img = np.clip(img,0.0,1.0)
                                data.append(img)
                            data = np.asarray(data, self._img_dtype).transpose(0,3,2,1)
                            label = np.expand_dims(nib.load(join(self._data_dir,site,subj,tp,DATASET_TAG+site+"_"+subj+"_"+tp+LES_NII_TAG)).get_data().astype(self._label_dtype),0)
                            label = label.transpose(0,3,2,1)

                        else:
                            for m in MODALITIES:
                                try:
                                    img = nib.load(join(self._data_dir,site,subj,tp,DATASET_TAG+site+"_"+subj+"_"+tp+"_"+m+IMG_TAG)).get_data().astype(np.float32)
                                    img = (img-img.min()) / (img.max()-img.min())
                                    img = np.clip(img,0.0,1.0)
                                except:
                                    print("Site {} and subject {} not properly read. You might want to check this out", site, subj)
                                data.append(img)
                            data = np.asarray(data, self._img_dtype)
                            label = np.expand_dims(nib.load(join(self._data_dir,site,subj,tp,DATASET_TAG+site+"_"+subj+"_"+tp+LES_TAG)).get_data().astype(self._label_dtype),0)

                        assert data.shape == (4,60,256,256), print("{} is not the right data shape".format(data.shape))
                        assert label.shape == (1,60,256,256), print("{} is not the right data shape".format(label.shape))
                        if (count+1) % 100 ==  0:
                            print('{} images complete {:.2f}m'.format(count+1,(timer()-start)/60) )
                        count += 1

                        data_raw = data.tostring()
                        label_raw = label.tostring()

                        if tp == 'screening':
                            tp = 'w0'

                        example = tf.train.Example(features=tf.train.Features(feature={
                            'dim0': shape[0],
                            'dim1': shape[1],
                            'dim2': shape[2],
                            'subj': self._int64_feature(int(str(subj[8:11]) + str(subj[12:15]))),
                            'time_point': self._int64_feature(int(tp[1:])),
                            'img': self._bytes_feature(data_raw),
                            'label': self._bytes_feature(label_raw)}))

        print("wrote {} images to {}".format(count,self._tfrecord))
        writer.close()
        print('closed writer')
        sys.stdout.flush()
        print('cleared stdout')


        subj_array = np.asarray(subj_list)
        #np.save('/cim/data/mslaq_raw/tf_ascend/ascend.npy', subj_array)
        np.save('/usr/local/data/thomasc/paul/test/ascend/ascend.npy', subj_array)



if __name__=="__main__":
    c = Converter('/usr/local/data/thomasc/paul/test/ascend/ascend.tfrecord') #'/cim/data/mslaq_raw/tf_ascend/ascend.tfrecord')    #/cim/tnair/mslaq5.tfrecord) 
    c.run()
    print('Done')
