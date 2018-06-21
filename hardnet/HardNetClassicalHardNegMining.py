#!/usr/bin/python2 -utt
#-*- coding: utf-8 -*-

"""
This is HardNet local patch descriptor. The training hardnet is based on PyTorch TFeat implementation
https://github.com/edgarriba/examples/tree/master/triplet
by Edgar Riba.

If you use this hardnet, please cite
@article{HardNet2017,
 author = {Anastasiya Mishchuk, Dmytro Mishkin, Filip Radenovic, Jiri Matas},
    title = "{Working hard to know your neighbor's margins:Local descriptor learning loss}",
     year = 2017}
(c) 2017 by Anastasiia Mishchuk, Dmytro Mishkin 
"""

from __future__ import division, print_function
import sys
from copy import deepcopy
import argparse
import torch
import torch.nn.init
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
import numpy as np
import random
import cv2
import copy
from EvalMetrics import ErrorRateAt95Recall
from Loggers import Logger, FileLogger
from W1BS import w1bs_extract_descs_and_save
from Utils import L2Norm, cv2_scale, np_reshape, adjust_learning_rate, create_optimizer
from Utils import str2bool
import torch.utils.data as data
import torch.utils.data as data_utils
import torch.nn.functional as F
from config import get_config, save_config
from models.TFNet import TFNet
from Utils import resume_pretrained_model

# using the Fair AI Similarity Search package
import faiss

# Training settings
args = None


class TripletPhotoTour(dset.PhotoTour):
    """From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, train=True, transform=None, batch_size = None, *arg, **kw):
        super(TripletPhotoTour, self).__init__(*arg, **kw)
        self.transform = transform
        self.train = train
        self.n_triplets = args.n_triplets
        self.batch_size = batch_size

        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            self.triplets = self.generate_triplets(self.labels, self.n_triplets)

    @staticmethod
    def generate_triplets(labels, num_triplets):
        def create_indices(_labels):
            inds = dict()
            for idx, ind in enumerate(_labels):
                if ind not in inds:
                    inds[ind] = []
                inds[ind].append(idx)
            return inds

        triplets = []
        indices = create_indices(labels)
        unique_labels = np.unique(labels.numpy())
        n_classes = unique_labels.shape[0]
        # add only unique indices in batch
        already_idxs = set()

        for x in tqdm(range(num_triplets)):
            if len(already_idxs) >= args.batch_size:
                already_idxs = set()
            c1 = np.random.randint(0, n_classes - 1)
            while c1 in already_idxs:
                c1 = np.random.randint(0, n_classes - 1)
            already_idxs.add(c1)
            c2 = np.random.randint(0, n_classes - 1)
            while c1 == c2:
                c2 = np.random.randint(0, n_classes - 1)
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]) - 1)
                n2 = np.random.randint(0, len(indices[c1]) - 1)
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]) - 1)
            n3 = np.random.randint(0, len(indices[c2]) - 1)
            triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
        return torch.LongTensor(np.array(triplets))

    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        if not self.train:
            m = self.matches[index]
            img1 = transform_img(self.data[m[0]])
            img2 = transform_img(self.data[m[1]])
            return img1, img2, m[2]

        t = self.triplets[index]
        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]

        img_a = transform_img(a)
        img_p = transform_img(p)
        img_n = transform_img(n)

        # transform images if required
        # if args.fliprot:
        #     do_flip = random.random() > 0.5
        #     do_rot = random.random() > 0.5
        #
        #     if do_rot:
        #         img_a = img_a.permute(0,2,1)
        #         img_p = img_p.permute(0,2,1)
        #
        #     if do_flip:
        #         img_a = torch.from_numpy(deepcopy(img_a.numpy()[:,:,::-1]))
        #         img_p = torch.from_numpy(deepcopy(img_p.numpy()[:,:,::-1]))

        return img_a, img_p, img_n

    def __len__(self):
        if self.train:
            return self.triplets.size(0)
        else:
            return self.matches.size(0)

class TripletPhotoTourHardNegatives(dset.PhotoTour):
    """From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """
    def __init__(self, negative_indices, train=True, transform=None, batch_size = None, *arg, **kw):
        super(TripletPhotoTourHardNegatives, self).__init__(*arg, **kw)
        self.transform = transform

        self.train = train
        self.n_triplets = args.n_triplets
        self.negative_indices = negative_indices
        self.batch_size = batch_size

        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            self.triplets = self.generate_triplets(self.labels, self.n_triplets, self.negative_indices)


    @staticmethod
    def generate_triplets(labels, num_triplets, negative_indices):
        def create_indices(_labels):
            inds = dict()
            for idx, ind in enumerate(_labels):
                if ind not in inds:
                    inds[ind] = []
                inds[ind].append(idx)
            return inds

        triplets = []
        indices = create_indices(labels)
        unique_labels = np.unique(labels.numpy())
        n_classes = unique_labels.shape[0]

        # add only unique indices in batch
        already_idxs = set()
        count  = 0
        for x in tqdm(range(num_triplets)):
            if len(already_idxs) >= args.batch_size:
                already_idxs = set()
            c1 = np.random.randint(0, n_classes - 1)
            while c1 in already_idxs:
                c1 = np.random.randint(0, n_classes - 1)
            already_idxs.add(c1)
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]) - 1)
                n2 = np.random.randint(0, len(indices[c1]) - 1)
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]) - 1)
            indx = indices[c1][n1]
            if(len(negative_indices[indx])>0):
                negative_indx = random.choice(negative_indices[indx])
            else:
                count+=1
                c2 = np.random.randint(0, n_classes - 1)
                while c1 == c2:
                    c2 = np.random.randint(0, n_classes - 1)
                n3 = np.random.randint(0, len(indices[c2]) - 1)
                negative_indx = indices[c2][n3]

            already_idxs.add(c1)

            triplets.append([indices[c1][n1], indices[c1][n2], negative_indx])

        print(count)
        print('triplets are generated. amount of triplets: {}'.format(len(triplets)))
        return torch.LongTensor(np.array(triplets))



    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        if not self.train:
            m = self.matches[index]
            img1 = transform_img(self.data[m[0]])
            img2 = transform_img(self.data[m[1]])
            return img1, img2, m[2]

        t = self.triplets[index]
        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]

        img_a = transform_img(a)
        img_p = transform_img(p)
        img_n = transform_img(n)

        return img_a, img_p, img_n

    def __len__(self):
        if self.train:
            return self.triplets.size(0)
        else:
            return self.matches.size(0)

class PhototourTrainingData(data.Dataset):

    def __init__(self, data):
        self.data_files = data

    def __getitem__(self, item):
        res = self.data_files[item]
        return res

    def __len__(self):
        return len(self.data_files)


def create_loaders(test_dataset_names):

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}

    transform = transforms.Compose([
            transforms.Lambda(cv2_scale),
            transforms.Lambda(np_reshape),
            transforms.ToTensor(),
            transforms.Normalize((args.mean_image,), (args.std_image,))])

    trainPhotoTourDataset =  TripletPhotoTour(train=True,
                             batch_size=args.batch_size,
                             root=args.dataroot,
                             name=args.training_set,
                             download=True,
                             transform=transform)

    test_loaders = [{'name': name,
                     'dataloader': torch.utils.data.DataLoader(
                         TripletPhotoTour(train=False,
                                          batch_size=args.test_batch_size,
                                          root=args.dataroot,
                                          name=name,
                                          download=True,
                                          transform=transform),
                         batch_size=args.test_batch_size,
                         shuffle=False, **kwargs)}
                    for name in test_dataset_names]

    return trainPhotoTourDataset, test_loaders

def train(train_loader, model, optimizer, epoch, logger):
    # switch to train mode
    model.train()
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, (data_a, data_p, data_n) in pbar:
        if args.cuda:
            data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()

        data_a, data_p, data_n = Variable(data_a), Variable(data_p), Variable(data_n)

        out_a, out_p, out_n = model(data_a), model(data_p), model(data_n)

        #hardnet loss
        loss = F.triplet_margin_loss(out_p, out_a, out_n, margin=args.margin, swap=args.anchorswap)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adjust_learning_rate(optimizer, args)
        if(logger!=None):
         logger.log_value('loss', loss.data[0]).step()

        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data_a), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    loss.data[0]))

    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))

def test(test_loader, model, epoch, logger, logger_test_name):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:

        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()

        data_a, data_p, label = Variable(data_a, volatile=True), \
                                Variable(data_p, volatile=True), Variable(label)

        out_a, out_p = model(data_a), model(data_p)
        dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy())
        ll = label.data.cpu().numpy().reshape(-1, 1)
        labels.append(ll)

        if batch_idx % args.log_interval == 0:
            pbar.set_description(logger_test_name+' Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(test_loader.dataset),
                       100. * batch_idx / len(test_loader)))

    num_tests = test_loader.dataset.matches.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    distances = np.vstack(distances).reshape(num_tests)

    fpr95 = ErrorRateAt95Recall(labels, 1.0 / (distances + 1e-8))
    print('\33[91mTest set: Accuracy(FPR95): {:.8f}\n\33[0m'.format(fpr95))

    if (args.enable_logging):
        if(logger!=None):
            logger.log_value(logger_test_name+' fpr95', fpr95)
    return


def main(trainPhotoTourDataset, test_loaders, model, logger, file_logger):
    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))

    if (args.enable_logging):
        file_logger.log_string('logs.txt', '\nparsed options:\n{}\n'.format(vars(args)))

    if args.cuda:
        model.cuda()

    optimizer1 = create_optimizer(model.features, args)

    # optionally resume from a checkpoint

    if args.resume:
        model = resume_pretrained_model(model, args)


    start = args.start_epoch
    end = start + args.epochs

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}

    transform = transforms.Compose([
            transforms.Lambda(cv2_scale),
            transforms.Lambda(np_reshape),
            transforms.ToTensor(),
            transforms.Normalize((args.mean_image,), (args.std_image,))])

    for epoch in xrange(start, end):

        model.eval()
        # #
        descriptors = get_descriptors_for_dataset(model, trainPhotoTourDataset)
        #
        np.save('descriptors.npy', descriptors)
        descriptors = np.load('descriptors.npy')
        #
        hard_negatives = get_hard_negatives(trainPhotoTourDataset, descriptors)
        np.save('descriptors_min_dist.npy', hard_negatives)
        hard_negatives = np.load('descriptors_min_dist.npy')
        print(hard_negatives[0])

        trainPhotoTourDatasetWithHardNegatives = TripletPhotoTourHardNegatives(train=True,
                                                                               negative_indices=hard_negatives,
                                                                               batch_size=args.batch_size,
                                                                               root=args.dataroot,
                                                                               name=args.training_set,
                                                                               download=True,
                                                                               transform=transform)

        train_loader = torch.utils.data.DataLoader(trainPhotoTourDatasetWithHardNegatives,
                                                   batch_size=args.batch_size,
                                                   shuffle=False, **kwargs)

        train(train_loader, model, optimizer1, epoch, logger)

        # iterate over test loaders and test results
        for test_loader in test_loaders:
            test(test_loader['dataloader'], model, epoch, logger, test_loader['name'])

        if TEST_ON_W1BS :
            # print(weights_path)
            patch_images = w1bs.get_list_of_patch_images(
                DATASET_DIR=args.w1bsroot.replace('/code', '/data/W1BS'))
            desc_name = 'curr_desc'

            for img_fname in patch_images:
                w1bs_extract_descs_and_save(img_fname, model, desc_name, cuda = args.cuda,
                                            mean_img=args.mean_image,
                                            std_img=args.std_image)

            DESCS_DIR = args.w1bsroot.replace('/code', "/data/out_descriptors")
            OUT_DIR = args.w1bsroot.replace('/code', "/data/out_graphs")

            force_rewrite_list = [desc_name]
            w1bs.match_descriptors_and_save_results(DESC_DIR=DESCS_DIR, do_rewrite=True,
                                                    dist_dict={},
                                                    force_rewrite_list=force_rewrite_list)
            if(args.enable_logging):
                w1bs.draw_and_save_plots_with_loggers(DESC_DIR=DESCS_DIR, OUT_DIR=OUT_DIR,
                                         methods=["SNN_ratio"],
                                         descs_to_draw=[desc_name],
                                         logger=file_logger,
                                         tensor_logger = None)
            else:
                w1bs.draw_and_save_plots(DESC_DIR=DESCS_DIR, OUT_DIR=OUT_DIR,
                                         methods=["SNN_ratio"],
                                         descs_to_draw=[desc_name],
                                         really_draw = False)



# build KNN graph using FAISS gpu devices
def BuildKNNGraphByFAISS_GPU(descriptor, k):
    dbsize, dim = descriptor.shape
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    res = faiss.StandardGpuResources()
    nn = faiss.GpuIndexFlatL2(res, dim, flat_config)
    nn.add(descriptor)
    dists,idx = nn.search(descriptor, k+1)
    return idx[:,1:], dists[:,1:]


def get_descriptors_for_dataset(model, trainPhotoTourDataset):

    transformed = []

    for img in trainPhotoTourDataset.data:
        transformed.append(trainPhotoTourDataset.transform(img.numpy()))
    print(len(transformed))
    phototour_loader = data_utils.DataLoader(PhototourTrainingData(transformed), batch_size=128, shuffle=False)
    descriptors = []
    pbar = tqdm(enumerate(phototour_loader))
    for batch_idx, data_a in pbar:

        if args.cuda:
            model.cuda()
            data_a = data_a.cuda()

        data_a = Variable(data_a, volatile=True),
        out_a = model(data_a[0])
        descriptors.extend(out_a.data.cpu().numpy())

    return descriptors

def remove_descriptors_with_same_index(min_dist_indices, indices, labels, descriptors):

    res_min_dist_indices = []

    for current_index in range(0, len(min_dist_indices)):
        # get indices of the same 3d points
        point3d_indices = labels[indices[current_index]]
        indices_to_remove = []
        for indx in min_dist_indices[current_index]:
            # add to removal list indices of the same 3d point and same images in other 3d point
            if(indx in point3d_indices or (descriptors[indx] == descriptors[current_index]).all()):
                indices_to_remove.append(indx)

        curr_desc = [x for x in min_dist_indices[current_index] if x not in indices_to_remove]
        res_min_dist_indices.append(curr_desc)

    return res_min_dist_indices

# using the hard negative sampling strategy to get the hard samples to build the triplet
def get_hard_negatives(trainPhotoTourDataset, descriptors):

    def create_indices(_labels):
        inds = dict()
        for idx, ind in enumerate(_labels):
            if ind not in inds:
                inds[ind] = []
            inds[ind].append(idx)
        return inds

    labels = create_indices(trainPhotoTourDataset.labels)
    indices = {}
    for key, value in labels.iteritems():
        for ind in value:
            indices[ind] = key

    # the hard samples is selected using the similarity serch on the distance space at batch_size unit
    print('getting closest indices .... ')
    descriptors_min_dist, inidices = BuildKNNGraphByFAISS_GPU(descriptors, 12)

    print('removing descriptors with same indices .... ')
    descriptors_min_dist = remove_descriptors_with_same_index(descriptors_min_dist, indices, labels, descriptors)

    return descriptors_min_dist


if __name__ == '__main__':

    args, unparsed = get_config(sys.argv)
    if len(unparsed) > 0:
        raise RuntimeError("Unknown arguments were given! Check the command line!")

    dataset_names = ['liberty', 'notredame', 'yosemite']

    # check if path to w1bs dataset testing module exists
    if os.path.isdir(args.w1bsroot):
        sys.path.insert(0, args.w1bsroot)
        import utils.w1bs as w1bs
        TEST_ON_W1BS = True

    # set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
    # order to prevent any memory allocation on unused GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)

    LOG_DIR = args.log_dir + args.experiment_name
    # create loggin directory
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    # set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger, file_logger = None, None
    model = TFNet()

    if(args.enable_logging):
        #logger = Logger(LOG_DIR)
        file_logger = FileLogger(LOG_DIR)

    test_dataset_names = copy.copy(dataset_names)
    test_dataset_names.remove(args.training_set)

    trainPhotoTourDataset, test_loaders = create_loaders(test_dataset_names)
    main(trainPhotoTourDataset, test_loaders, model, logger, file_logger)
