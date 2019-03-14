import numpy as np
import tensorflow as tf
from os.path import join, exists, isdir
from os import listdir
from timeit import default_timer as timer
import nibabel as nib
import sys

DEFAULT_DATA_DIR = 'usr/local/data/thomasc/mni-resample/MNI-resampled'   #'/cim/data/neurorx/MS-LAQ-302-STX/imaging_data' '/cim/data/neurorx/MS-LAQ-302-STX/imaging_data' 
IMG_TAG = "_icbm.mnc"
LES_TAG = "_ct2f_icbm.mnc"
DEFAULT_TPS = ['m0', 'm12', 'm24', 'baseline', 'screening', 'w48', 'w96']
MODALITIES = ['t1p', 't2w', 'flr', 'pdw']

NORM_CONSTANTS = {'t1p':{'max': 1020.0050537880522},'t2w':{'max': 1016.9968841077286},
                  'flr':{'max': 1014.9998077363241},'pdw':{'max': 1016.9996490424962}}

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

        # List all subjects except those from ascend
        subjs = [i for i in listdir(self._data_dir) if not i.startswith('101')]
        tps = [[subj,tp] for subj in subjs for tp in listdir(join(self._data_dir,subj)) if '.scannerdb' not in tp]
        count = 0
        start = timer()
        subj_list = [] #np.array(len(subjs), dtype=np.dtype('a11'))
        for idx,subj in enumerate(subjs):
            first_time = True
            for tp in DEFAULT_TPS:
                if isdir(join(self._data_dir,subj,tp)):
                    ct2f_pth = join(self._data_dir,subj,tp, subj+'_'+tp+LES_TAG)
                    flr_pth = join(self._data_dir,subj,tp,subj+"_"+tp+"_flr"+IMG_TAG)
                    pdw_pth = join(self._data_dir,subj,tp,subj+"_"+tp+"_pdw"+IMG_TAG)
                    t1c_pth = join(self._data_dir,subj,tp,subj+"_"+tp+"_t1p"+IMG_TAG)
                    t2w_pth = join(self._data_dir,subj,tp,subj+"_"+tp+"_t2w"+IMG_TAG)

                    # Format subject name for saving ('109' denotes Confirm/Define (marked by 2); 'MSLAQ' denotes MSLAQ (marked by 1);
                    # 'MBP' denotes Maestro3 (marked by 3))
                    if subj.startswith('MS-LAQ'):
                        save_subj = '1' + subj[15:21]
                    elif subj.startswith('109'):
                        save_subj = '2' + subj[13:16] + subj[17:20]
                    elif subj.startswith('MBP'):
                        save_subj = '3' + subj[18:21] + subj[22:25]
                    else
                        raise Error('Unknown subject type (unknown trial): ' + subj)

                    if np.all([exists(ct2f_pth), exists(flr_pth), exists(pdw_pth), exists(t1c_pth), exists(t2w_pth)]):
                        if first_time:
                            subj_list.append(save_subj)
                            first_time = False
                        data = []
                        for m in MODALITIES:
                            img = nib.load(join(self._data_dir,subj,tp,subj+"_"+tp+"_"+m+IMG_TAG)).get_data().astype(np.float32)
                            img = (img-img.min()) / (NORM_CONSTANTS[m]['max']-img.min())
                            img = np.clip(img,0.0,1.0)
                            data.append(img)
                        data = np.asarray(data, self._img_dtype)
                        label = np.expand_dims(nib.load(join(self._data_dir,subj,tp,subj+"_"+tp+LES_TAG)).get_data().astype(self._label_dtype),0)

                        assert data.shape == (4,60,256,256), print("{} is not the right data shape".format(data.shape))
                        assert label.shape == (1,60,256,256), print("{} is not the right data shape".format(label.shape))
                        if (count+1) % 100 ==  0:
                            print('{} images complete {:.2f}m'.format(count+1,(timer()-start)/60) )
                        count += 1

                        # Format uniformly (timepoints)
                        if tp == 'screening':
                            save_tp = '0'
                        elif tp == 'baseline':
                            save_tp = '0'
                        else:
                            save_tp = tp[1:]

                        data_raw = data.tostring()
                        label_raw = label.tostring()
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'dim0': shape[0],
                            'dim1': shape[1],
                            'dim2': shape[2],
                            'subj': self._int64_feature(int(save_subj)),
                            'time_point': self._int64_feature(int(save_tp)),
                            'img': self._bytes_feature(data_raw),
                            'label': self._bytes_feature(label_raw)}))

                        writer.write(example.SerializeToString())
  
        subj_array = np.asarray(subj_list)
        np.save('/usr/local/data/thomasc/mni-resample/outputs/mni_resample_list.npy', subj_array) #'/cim/data/mslaq_raw/tf_mslaq/mslaq.npy', subj_array)
 
        print("wrote {} images to {}".format(count,self._tfrecord))
        writer.close()
        print('closed writer')
        sys.stdout.flush()
        print('cleared stdout')


if __name__=="__main__":
    c = Converter('/usr/local/data/thomasc/mni-resample/outputs/mni_resample.tfrecord')              #'/cim/data/mslaq_raw/tf_mslaq/mslaq.tfrecord')  #'/usr/local/data/thomasc/paul/test/mslaq/mslaq.tfrecord')     #/cim/tnair/mslaq5.tfrecord)
    c.run()
    print('Done')

# Important data paths:
# Tanya's original MSLAQ tfrecord '/home/rain/tnair/tnair/thesis/data/mslaq/mslaq2.tfrecord'
# My full MSLAQ tfrecord (DGX) '/cim/data/mslaq_raw/tf_mslaq/mslaq.tfrecord'
# My mini, modified tfrecord for MSLAQ '/usr/local/data/thomasc/paul/test/mslaq/mslaq.tfrecord'