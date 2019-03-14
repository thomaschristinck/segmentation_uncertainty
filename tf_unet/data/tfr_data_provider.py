from abc import abstractmethod
import numpy as np
import random
import tensorflow as tf

_SEED = 5


class BrainDataProvider:

    def __init__(self, mslaq_tfrecord_path, ascend_tfrecord_path, config):
        """
        Data provider for multimodal brain MRI.
        Usage:
        :param tfrecord_path: path to tfrecord
        :param config: dict to config options
        """
        self._mslaq_tfrecord = mslaq_tfrecord_path
        self._ascend_tfrecord = ascend_tfrecord_path
        self._mode = config.get('mode')
        self._shuffle = config.get('shuffle', True)
        self._augment = config.get('augment', False)

        # Tanya's npy file '/usr/local/data/tnair/thesis/data/mslaq/mslaq_subj_list.npy'
        # My mini npy '/usr/local/data/thomasc/paul/test/mslaq/mslaq.npy'

        '''
        self._subjects = np.load(config.get('mslaq_subj_list', '/usr/local/data/thomasc/paul/test/mslaq/mslaq.npy'))
        self._subjects = [i[4:] for i in self._subjects]

        '''

        # Load numpy files containing list of subject names
        self._mslaq_subjects = np.load(config.get('mslaq_subj_list', '/usr/local/data/thomasc/paul/test/mslaq/mslaq.npy'))#'/cim/data/mslaq_raw/tf_mslaq/mslaq.npy')) #
        self._ascend_subjects = np.load(config.get('ascend_subj_list', '/usr/local/data/thomasc/paul/test/ascend/ascend.npy'))#'/cim/data/mslaq_raw/tf_ascend/ascend.npy')) #
        # Properly format the names (such that subjects are represented by integers)
        self._mslaq_subjects = [i[4:] for i in self._mslaq_subjects]
        self._ascend_subjects = [i[8:] for i in self._ascend_subjects]
        self._ascend_subjects = [i.replace('-','') for i in self._ascend_subjects]
       
        self._subjects = self._mslaq_subjects
        self._subjects.extend(self._ascend_subjects)
        random.shuffle(self._subjects)


        self._nb_folds = config.get('nb-folds', 2)  # Changed from 10 folds
        fold_length = len(self._subjects) // self._nb_folds
        self._subjects = self._rotate(self._subjects, config.get('fold', 0) * fold_length)
        train_idx =12 #(self._nb_folds - 5) * fold_length
        valid_idx = 26   #(self._nb_folds - 1) * fold_length
        print("subj length: ", len(self._subjects))
        print("train and valid idx: ", train_idx, valid_idx)

        if self._mode == 'train':
            self._subjects = self._subjects[:train_idx]
        elif self._mode == 'valid':
            # Examples not in training set
            non_train_array = np.asarray(self._subjects[train_idx:])
            # Examples in validation set
            valid_array = np.asarray(self._subjects[train_idx:valid_idx])
            self._subjects = self._subjects[train_idx:valid_idx]
            np.save('/usr/local/data/thomasc/paul/test/validation.npy', valid_array) 
            #np.save('/cim/data/mslaq_raw/validation.npy', non_train_array) #'/usr/local/data/thomasc/paul/test/val.npy', valid_subj_array) #
        elif self._mode == 'test':
            self._subjects = self._subjects[valid_idx:]
            test_subj_array = np.asarray(self._subjects)
            np.save('/usr/local/data/thomasc/paul/test/test.npy', test_subj_array)  #'/cim/data/mslaq_raw/test.npy', test_subj_array)  #
        # Use this if we're trying to use never before seen data for validation/saving
        elif self._mode == 'nontrain':
            nontrain_array = np.load('/cim/data/mslaq_raw/validation.npy')
            self._subjects = nontrain_array.tolist()
        self._subjects = tf.constant(np.asarray(self._subjects, np.int32))

    def __len__(self):
        return len(self._subjects)

    def _parse_tfrecord(self, tfrecord):
        features = tf.parse_single_example(tfrecord,
                                           features={'img': tf.FixedLenFeature([], tf.string),
                                                     'label': tf.FixedLenFeature([], tf.string),
                                                     'subj': tf.FixedLenFeature([], tf.string),  #int64
                                                     'time_point': tf.FixedLenFeature([], tf.string)})
        #subj = tf.cast(features['subj'], tf.int32)
        #return subj, features['time_point'], features['img'], features['label']
        return features['subj'], features['time_point'], features['img'], features['label']

    @staticmethod
    def _process_data(x):
        x = tf.transpose(x, [3, 2, 1, 0])  # (bs, z, y, x, modality)
        x = x[12:204, 12:204, :, :]
        # x = tf.clip_by_value(x, 0., 1.)
        x = tf.pad(x, paddings=tf.constant([[0, 0], [0, 0], [2, 2], [0, 0]]),
                   mode='CONSTANT', constant_values=0)
        return x

    @staticmethod
    def _rotate(l, n):
        return l[-n:] + l[:-n]

    def _subj_filter(self, subj, tp, img, label):
        return tf.reduce_any(tf.equal(subj, self._subjects))

    def _remove_subj_tp(self, subj, tp, img, label):
        img = tf.decode_raw(img, tf.float32)
        label = tf.cast(tf.decode_raw(label, tf.int16), tf.float32)
        img = tf.reshape(img, tf.stack([4, 60, 256, 256]))
        label = tf.reshape(label, tf.stack([1, 60, 256, 256]))
        # label = tf.expand_dims(tf.reshape(label, tf.stack([60, 256, 256])), 0)
        img = self._process_data(img)  # [..., :-1]
        label = self._process_data(label)
        return img, label

    def _prep_subj_tp(self, subj, tp, img, label):
        img = tf.decode_raw(img, tf.float32)
        label = tf.cast(tf.decode_raw(label, tf.int16), tf.float32)
        img = tf.reshape(img, tf.stack([4, 60, 256, 256]))
        label = tf.reshape(label, tf.stack([1, 60, 256, 256]))
        #img = tf.decode_raw(img, tf.float32)
        #tp = tf.decode_raw(img, tf.float32)
        # label = tf.expand_dims(tf.reshape(label, tf.stack([60, 256, 256])), 0)
        img = self._process_data(img)  # [..., :-1]
        label = self._process_data(label)
        print(subj)
        return img, label, subj, tp

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
        dataset = tf.data.TFRecordDataset([self._mslaq_tfrecord, self._ascend_tfrecord])
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
        dataset = tf.data.TFRecordDataset([self._mslaq_tfrecord, self._ascend_tfrecord])
        dataset = dataset.map(self._parse_tfrecord)
        #dataset = dataset.filter(self._subj_filter)
        #dataset = dataset.map(self._remove_subj_tp)
        dataset = dataset.map(self._prep_subj_tp)
        dataset = dataset.repeat(nb_epochs)
        dataset = dataset.batch(1)
        dataset = dataset.prefetch(2)
        iterator = dataset.make_initializable_iterator()
        return iterator
