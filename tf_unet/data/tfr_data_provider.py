from abc import abstractmethod
import numpy as np
import random
import tensorflow as tf

_SEED = 5


class BrainDataProvider:

    def __init__(self, tfrecord_path,config):
        """
        Data provider for multimodal brain MRI.
        Usage:
        :param tfrecord_path: path to tfrecord
        :param config: dict to config options
        """
        self._tfrecord = tfrecord_path
        
        self._mode = config.get('mode')
        self._shuffle = config.get('shuffle', True)
        self._augment = config.get('augment', False)

        # Tanya's npy file '/usr/local/data/tnair/thesis/data/mslaq/mslaq_subj_list.npy'
        # My mini npy '/usr/local/data/thomasc/paul/test/mslaq/mslaq.npy'

        # Load numpy files containing list of subject names
        self._subjects = np.load(config.get('mni_resample_list', '/usr/local/data/thomasc/mni-resample/outputs/mni_resample_list.npy'))#'/cim/data/mslaq_raw/tf_mslaq/mslaq.npy')) #
        #self._subjects = list(map(int, self._subjects))
        random.shuffle(self._subjects)

        nb_train = round(len(self._subjects) * 0.4)
        nb_valid = round(len(self._subjects) * 0.1)
       
      
        if self._mode == 'train':
            train_array = np.asarray(self._subjects[:nb_train])
            self._subjects = self._subjects[:nb_train]
            np.save('/usr/local/data/thomasc/mni-resample/outputs/train.npy', train_array)
        elif self._mode == 'valid':
            non_train_array = np.asarray(self._subjects[nb_train:nb_train + nb_valid])
        elif self._mode == 'test':
            self._subjects = self._subjects[nb_train + nb_valid:]
        self._subjects = tf.constant(np.asarray(self._subjects, np.int32))
    def __len__(self):
        return len(self._subjects)

    def _parse_tfrecord(self, tfrecord):
        features = tf.parse_single_example(tfrecord,
                                           features={'img': tf.FixedLenFeature([], tf.string),
                                                     'label': tf.FixedLenFeature([], tf.string),
                                                     'subj': tf.FixedLenFeature([], tf.int64),  #int64
                                                     'time_point': tf.FixedLenFeature([], tf.int64)})
        print("Subj: ", features['subj'])
        subj = tf.cast(features['subj'], tf.int32)
        print("Subj: ", subj)
        return subj, features['time_point'], features['img'], features['label']
        #return features['subj'], features['time_point'], features['img'], features['label']

    @staticmethod
    def _process_data(x):
        x = tf.transpose(x, [3, 2, 1, 0])  # (bs, z, y, x, modality)
        print('x shape :', x.shape)
        x = x[:192, 16:208, :, :]
        # x = tf.clip_by_value(x, 0., 1.)
        x = tf.pad(x, paddings=tf.constant([[0, 0], [0, 0], [0, 0], [0, 0]]),
                   mode='CONSTANT', constant_values=0)
        print('x shape (post crop):', x.shape)
        return x

    def _subj_filter(self, subj, tp, img, label):
        return tf.reduce_any(tf.equal(subj, self._subjects))

    def _remove_subj_tp(self, subj, tp, img, label):
        img = tf.decode_raw(img, tf.float32)
        label = tf.cast(tf.decode_raw(label, tf.int16), tf.float32)
        print("Image shape (pre): ", img.shape)
        img = tf.reshape(img, tf.stack([4, 64, 229, 193]))
        label = tf.reshape(label, tf.stack([1, 64, 229, 193]))
        #img = tf.reshape(img, tf.stack([4, 60, 256, 256]))
        #label = tf.reshape(label, tf.stack([1, 60, 256, 256]))
        # label = tf.expand_dims(tf.reshape(label, tf.stack([60, 256, 256])), 0)
        img = self._process_data(img)  # [..., :-1]
        label = self._process_data(label)
        return img, label


    @abstractmethod
    def get_generator(self, batch_size, nb_epochs):
        raise NotImplementedError


class BrainVolumeDataProvider(BrainDataProvider):
    def get_generator(self, batch_size, nb_epochs):
        """
        Generator that yields examples (subj, time_point, x_batch, y_batch) for use during model training
        :param batch_size: batch size of examples to yield
        :param nb_epochs: number of epochs to generate
        """
        dataset = tf.data.TFRecordDataset([self._tfrecord])
        #dataset = tf.data.TFRecordDataset(self._mslaq_tfrecord)
        dataset = dataset.map(self._parse_tfrecord, num_parallel_calls=10)
        dataset = dataset.filter(self._subj_filter)
        dataset = dataset.map(self._remove_subj_tp, num_parallel_calls=10)
        if self._shuffle:
            dataset = dataset.shuffle(buffer_size=2, seed=_SEED) # From 20
        dataset = dataset.repeat(nb_epochs)
        dataset = dataset.batch(1)  # changes from batch_size
        dataset = dataset.prefetch(2) # From 50
        iterator = dataset.make_initializable_iterator()
        return iterator

    def get_test_generator(self, batch_size=1, nb_epochs=1):
        """
        Generator that yields examples (subj, time_point, x_batch, y_batch) for use during model evaluation
        :param batch_size: batch size of examples to yield
        :param nb_epochs: number of epochs to generate
        """
        dataset = tf.data.TFRecordDataset([self._tfrecord])
        dataset = dataset.map(self._parse_tfrecord).filter(self._subj_filter)
        dataset = dataset.repeat(nb_epochs)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(50)
        iterator = dataset.make_initializable_iterator()
        return iterator