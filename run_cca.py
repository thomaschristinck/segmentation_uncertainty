from tf_unet.analyze.BunetAnalyzer import BunetAnalyzer as Analyzer
#from tf_unet.data.data_provider import BrainVolumeDataProvider as DataProvider
from tf_unet.data.tfr_data_provider import BrainVolumeDataProvider as DataProvider
from tf_unet.models.bunet import BUnet
from os import makedirs
from os.path import join
from argparse import ArgumentParser
import json
from shutil import copy
import tensorflow as tf

def main(args):
    #sess= tf.Session()
    '''
    h5_path = args.data
    for i in ['train_img', 'val_img', 'test_img', 'all_img']:
        d = join(args.output, i)
        makedirs(d, exist_ok=True)

    '''
    out_dir = args.output

    with open(args.config, 'r') as f:
        cfg = json.loads(f.read())
    makedirs(out_dir, exist_ok=True)
    copy(args.config, out_dir)
    expt_cfg = cfg['expt']
    model_cfg = cfg['model']




    #train_ds = DataProvider(h5_path, {'mode': 'train', 'shuffle': False, 'modalities': [0, 1, 2, 3]})
    #valid_ds = DataProvider(h5_path, {'mode': 'valid', 'shuffle': False, 'modalities': [0, 1, 2, 3]})
    #test_ds = DataProvider(h5_path, {'mode': 'test', 'shuffle': False, 'modalities': [0, 1, 2, 3]})
    #full_ds = DataProvider(h5_path, {'mode': 'all', 'shuffle': False, 'modalities': [0,1,2,3]})
    #test_ds = DataProvider(expt_cfg['mslaq_data_path'],expt_cfg['ascend_data_path'], {'mode': 'nontrain', 'shuffle': False})
    test_ds = DataProvider(expt_cfg['mslaq_data_path'], {'mode': 'test', 'shuffle': False})

    # Generate examples for model evaluation
    #train_gen = train_ds.get_test_generator(1)
    #valid_gen = valid_ds.get_test_generator(1)
    #test_gen = test_ds.get_test_generator(1)
    #full_gen = full_ds.get_test_generator(1)
    test_gen = test_ds.get_test_generator()
    #nb_subjs = test_ds._length
    nb_subjs = 100

    print('-----------Initializing BUnet------------')

    net = BUnet(nb_ch=4,
                nb_kers=32,
                nb_mc=5,
                depth=4,
                weight_decay=0.0001,
                loss_fn='adam',
                batch_size=1)

    #sigmoid_thresh = 0.00001

    #train_analyzer = Analyzer(net, args.checkpoint_path, train_gen, join(out_dir, 'train_img'), nb_mc=10)
    #valid_analyzer = Analyzer(net, args.checkpoint_path, valid_gen, join(out_dir, 'valid_img'), nb_mc=10)
    test_analyzer = Analyzer(net, args.checkpoint_path, test_gen, join(out_dir, 'test_img'), nb_mc=2)
    #full_analyzer = Analyzer(net, args.checkpoint_path, full_gen, join(out_dir, '3d_all_img_small'), nb_mc=10)
    #full_analyzer = Analyzer(net, args.checkpoint_path, test_gen, out_dir, nb_mc=10)

    #train_analyzer.cca('train_stats_thresh{}.csv'.format(sigmoid_thresh), sigmoid_thresh)
    #valid_analyzer.cca('valid_stats_thresh{}.csv'.format(sigmoid_thresh), sigmoid_thresh)
    #test_analyzer.cca('test_stats_thresh{}.csv'.format(sigmoid_thresh), sigmoid_thresh)
    thresh = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 1.0, 1.1]
    #out_file = 'roc.csv'

    #test_analyzer.roc(out_file, thresh, nb_subjs)
    test_analyzer.write_to_nrrd(nb_subjs)


def _parser():
    usage = 'python run_cca.py -i /cim/data/mslaq.hdf5 -c /path/to/checkpoint/ -o /output/directory'
    parser = ArgumentParser(prog='train_unet', usage=usage)
    #parser.add_argument('-i', '--data', help='Training HDF5', required=True)
    parser.add_argument('-p', '--checkpoint_path', help='Model Checkpoint', required=True)
    parser.add_argument('-o', '--output', help='Model Output', required=True)
    parser.add_argument('-c', '--config', help='Config file (.json)', required=True)
    return parser


if __name__ == '__main__':
    main(_parser().parse_args())

'''
With hdf5 loader on local machine:
python run_cca.py -i /usr/local/data/tnair/thesis/mslaq_brains_edss_gz.hdf5 -p /usr/local/data/thomasc/checkpoints/bunet_checkpoint -o /usr/local/data/thomasc/unet_out/
With tfrecord on local machine:
python run_cca.py -p /usr/local/data/thomasc/checkpoints/bunet_checkpoint -o /usr/local/data/thomasc/test_ops -c /usr/local/data/thomasc/checkpoints/train_bunet.json
python run_cca.py -p /usr/local/data/thomasc/checkpoint -o /usr/local/data/thomasc/checkpoint/roc -c ~/Desktop/segmentation_uncertainty/tf_unet/configs/train_bunet.json

(checkpoint can be found on dgx)
DGX Usage:
python3 run_cca.py -p /cim/data/mslaq_raw/checkpoint -o /cim/data/mslaq_raw/outputs -c /cim/data/mslaq_raw/tf_unet/tf_unet/configs/train_bunet.json 
'''