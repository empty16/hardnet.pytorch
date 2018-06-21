# config.py
# Description: configuration module.

import argparse
from Utils import str2bool
import json
import os


# ----------------------------------------
# Model options
parser = argparse.ArgumentParser(description='PyTorch HardNet')

# ----------------------------------------
# Experiment configs
parser.add_argument('--w1bsroot', type=str,
                    default='../wxbs-descriptors-benchmark/code',
                    help='path to dataset')
parser.add_argument('--dataroot', type=str,
                    default='../data/',
                    help='path to dataset')
parser.add_argument('--log-dir', default='../logs',
                    help='folder to output log')
parser.add_argument('--model-dir', default='../logs/',
                    help='folder to output model checkpoints')
parser.add_argument('--experiment-name', default= 'notredame_train',
                    help='experiment path')


# ----------------------------------------
# Train settings
parser.add_argument('--imageSize', type=int, default=32,
                    help='the height / width of the input image to network')
parser.add_argument('--resume', default='', type=str, metavar='checkpoint name',
                    help='name to latest checkpoint (default: none)')
parser.add_argument('--training-set', default= 'notredame',
                    help='Other options: notredame, yosemite')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--batch-size', type=int, default=1024, metavar='BS',
                    help='input batch size for training (default: 1024)')
# batch-reduce is the sampling strategy of the data samples
parser.add_argument('--batch-reduce', default= 'min',
                    help='Other options: average, random, random_global, L2Net')
parser.add_argument('--lr', type=float, default=10.0, metavar='LR',
                    help='learning rate (default: 10.0. Yes, ten is not typo)')
parser.add_argument('--lr-decay', default=1e-6, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-6')
parser.add_argument('--loss', default= 'triplet_margin',
                    help='Other options: softmax, contrastive')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: SGD)')
parser.add_argument('--n-triplets', type=int, default=5000000, metavar='N',
                    help='how many triplets will generate from the dataset')
parser.add_argument('--mean-image', type=float, default=0.443728476019,
                    help='mean of train dataset for normalization')
parser.add_argument('--std-image', type=float, default=0.20197947209,
                    help='std of train dataset for normalization')
parser.add_argument('--fliprot', type=str2bool, default=True,
                    help='turns on flip and 90deg rotation augmentation')
parser.add_argument('--margin', type=float, default=1.0, metavar='MARGIN',
                    help='the margin value for the triplet loss and contrastive function (default: 1.0')
parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                    help='gor parameter')
parser.add_argument('--act-decay', type=float, default=0,
                    help='activity L2 decay, default 0')
parser.add_argument('--num-workers', default= 0,
                    help='Number of workers to be created')

# Sift Init config
parser.add_argument('--hardnegatives', type=int, default=7,
                    help='the height/width of the input image to network')

# ARF-based Network config
parser.add_argument('--use-arf', type=str2bool, default=False,
                    help='upgrading Conv to ARF')
parser.add_argument('--orientation', type=int, default=8, metavar='O',
                    help='nOrientation for ARFs (default: 8)')

# SRN-based Network config
parser.add_argument('--use-srn', type=str2bool, default=False,
                    help='upgrading network to SRN structure')
parser.add_argument('--use-smooth', type=str2bool, default=False,
                    help='3*3 smooth conv layer of SRN structure')

# ----------------------------------------
# Test settings
parser.add_argument('--test-batch-size', type=int, default=1024, metavar='BST',
                    help='input batch size for testing (default: 1024)')

# ----------------------------------------
# config on-off
parser.add_argument('--enable-logging',type=str2bool, default=True,
                    help='output to tensorlogger')
parser.add_argument('--pin-memory',type=bool, default= False,
                    help='')
parser.add_argument('--decor',type=str2bool, default = False,
                    help='L2Net decorrelation penalty')
parser.add_argument('--anchorave', type=str2bool, default=False,
                    help='anchorave no use here')
parser.add_argument('--anchorswap', type=str2bool, default=True,
                    help='turns on anchor swap(anchor vs. positive)')
parser.add_argument('--gor',type=str2bool, default=False,
                    help='use global orthogonal regularization')

# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')


def get_config(argv):
    config, unparsed = parser.parse_known_args()
    # only parse a few of the parameters
    return config, unparsed


def save_config(config):
    param_path = os.path.join(config.log_dir, config.experiment_name, "params.json")

    print("[*] MODEL dir: %s" % config.log_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, "w") as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


#
# config.py ends here