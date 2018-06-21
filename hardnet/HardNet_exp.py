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

import copy
import os
import random
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from tqdm import tqdm

from EvalMetrics import ErrorRateAt95Recall
from Losses import loss_HardNet, loss_random_sampling, loss_L2Net, global_orthogonal_regularization, CorrelationPenaltyLoss
from Utils import cv2_scale, np_reshape, create_optimizer, adjust_learning_rate, resume_pretrained_model
from W1BS import w1bs_extract_descs_and_save
from config import get_config, save_config
from dataset import TripletDataLoader

# using the global variable here
args = None

def create_loaders(load_random_triplets = False, train=True):

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}
    transform = transforms.Compose([
            transforms.Lambda(cv2_scale),
            transforms.Lambda(np_reshape),
            transforms.ToTensor(),
            transforms.Normalize((args.mean_image,), (args.std_image,))])

    if train:
        data_loader = TripletDataLoader(
            #(train=True, transform=None, batch_size=None, fliprot=True, load_random_triplets=False, n_triplets=None)
            dataset_params = (True, transform, args.batch_size, args.fliprot, load_random_triplets, args.n_triplets),
            dataset_kws = {
             'root': args.dataroot,
             'name': args.training_set,
             'download': True
             },
            batch_size=args.batch_size,
            shuffle=False, **kwargs)

    else:
        test_dataset_names = copy.copy(dataset_names)
        test_dataset_names.remove(args.training_set)

        # using the for to generate the dict-list
        data_loader = [{'name': name,
                        'dataloader': TripletDataLoader(
                            # (train=True, transform=None, batch_size=None, fliprot=True, load_random_triplets=False, n_triplets=None)
                            dataset_params = (False, transform, args.test_batch_size, args.fliprot, load_random_triplets),
                            dataset_kws = {'root': args.dataroot,
                                           'name': name,
                                           'download': True},
                            batch_size=args.test_batch_size,
                            shuffle=False, **kwargs)}
                        for name in test_dataset_names]

    return data_loader


def train(train_loader, model, optimizer, epoch, logger, load_triplets=False):

    # switch to train mode
    model.train()
    pbar = tqdm(enumerate(train_loader))
    for batch_idx, data in pbar:
        if load_triplets:
            data_a, data_p, data_n = data
        else:
            data_a, data_p = data

        if args.cuda:
            data_a, data_p  = data_a.cuda(), data_p.cuda()
            data_a, data_p = Variable(data_a), Variable(data_p)
            out_a, out_p = model(data_a), model(data_p)

        # load_triplets=Flase for the L2Net and HardNet, these two generate the positive patch based on the batch data
        if load_triplets:
            data_n  = data_n.cuda()
            data_n = Variable(data_n)
            out_n = model(data_n)

        # for the comparision with L2Net, and random_global
        if args.batch_reduce == 'L2Net':
            loss = loss_L2Net(out_a, out_p, anchor_swap = args.anchorswap,
                    margin = args.margin, loss_type = args.loss)
        elif args.batch_reduce == 'random_global':
            # using the random nagative patch samples from the dataset
            loss = loss_random_sampling(out_a, out_p, out_n,
                margin=args.margin,
                anchor_swap=args.anchorswap,
                loss_type = args.loss)
        else:
            loss = loss_HardNet(out_a, out_p,
                            margin=args.margin,
                            anchor_swap=args.anchorswap,
                            anchor_ave=args.anchorave,
                            batch_reduce = args.batch_reduce,
                            loss_type = args.loss)

        # E2 loss in L2Net for descriptor componet correlation
        if args.decor:
            loss += CorrelationPenaltyLoss()(out_a)

        # gor for HardNet
        if args.gor:
            loss += args.alpha * global_orthogonal_regularization(out_a, out_n)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adjust_learning_rate(optimizer, args)
        if batch_idx % args.log_interval == 0:
            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data_a), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                    loss.data[0]))
            if (args.enable_logging):
                logger.log_string('logs', 'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data_a), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.data[0]))

    try:
        os.stat('{}{}'.format(args.model_dir, suffix))
    except:
        os.makedirs('{}{}'.format(args.model_dir, suffix))

    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}{}/checkpoint_{}.pth'.format(args.model_dir, suffix, epoch))

def test(test_loader, model, epoch, logger, logger_test_name):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    # data_anchor(img_idx), data_positive(img_idx), match_or_not(0/1)
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:

        if args.cuda:
            data_a, data_p = data_a.cuda(), data_p.cuda()

        data_a, data_p, label = Variable(data_a, volatile=True), \
                                Variable(data_p, volatile=True), Variable(label)

        out_a, out_p = model(data_a), model(data_p)
        dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy().reshape(-1,1))
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
    print('Test set: Accuracy(FPR95): {:.8f}\n'.format(fpr95))

    if (args.enable_logging):
        logger.log_string('logs', 'Test Epoch {}/{}: Accuracy(FPR95)@{}: {:.8f}'.format(
            epoch, args.start_epoch+args.epochs, logger_test_name, fpr95))
    return


def main(train_loader, test_loaders, model, logger, file_logger):
    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))

    if (args.enable_logging):
       file_logger.log_string('logs', '\nparsed options:\n{}\n'.format(vars(args)))

    if args.cuda:
        model.cuda()

    optimizer1 = create_optimizer(model.features, args)

    # optionally resume from a checkpoint
    if args.resume:
        model = resume_pretrained_model(model, args, suffix)
    
    start = args.start_epoch
    end = start + args.epochs

    for epoch in xrange(start, end):
        # iterate over test loaders and test results
        train(train_loader, model, optimizer1, epoch, file_logger, triplet_flag)

        for test_loader in test_loaders:
            test(test_loader['dataloader'], model, epoch, file_logger, test_loader['name'])
        
        if TEST_ON_W1BS:
            # print(weights_path)
            patch_images = w1bs.get_list_of_patch_images(
                DATASET_DIR = args.w1bsroot.replace('/code', '/data/W1BS'))
            desc_name = 'curr_desc'# + str(random.randint(0,100))
            
            DESCS_DIR = log_dir + '/temp_descs/' #args.w1bsroot.replace('/hardnet', "/data/out_descriptors")
            OUT_DIR = DESCS_DIR.replace('/temp_descs/', "/out_graphs/")

            for img_fname in patch_images:
                # a bug occur @func
                # save (patch*feat_dim) desc matrix to the desc file
                w1bs_extract_descs_and_save(img_fname, model, desc_name, cuda = args.cuda,
                                            mean_img=args.mean_image,
                                            std_img=args.std_image, out_dir = DESCS_DIR)


            force_rewrite_list = [desc_name]
            w1bs.match_descriptors_and_save_results(DESC_DIR=DESCS_DIR, do_rewrite=True,
                                                    dist_dict={},
                                                    force_rewrite_list=force_rewrite_list)
            if(args.enable_logging):
                # DESC_DIR, OUT_DIR, methods, colors, lines,
                # descs_to_draw, really_draw, logger, tensor_logger = None
                w1bs.draw_and_save_plots_with_loggers(DESC_DIR=DESCS_DIR, OUT_DIR=OUT_DIR,
                                         methods=["SNN_ratio"],
                                         descs_to_draw=[desc_name],
                                         really_draw=True,
                                         logger=file_logger,
                                         tensor_logger = logger)
            else:
                w1bs.draw_and_save_plots(DESC_DIR=DESCS_DIR, OUT_DIR=OUT_DIR,
                                         methods=["SNN_ratio"],
                                         descs_to_draw=[desc_name],
                                         really_draw=True)
        # re-generate the triplets slices
        if epoch + 1 < end:
            train_loader.resample_dataset_triplets()

    print('HardNet train done!')

if __name__ == '__main__':
    args, unparsed = get_config(sys.argv)
    if len(unparsed) > 0:
        raise RuntimeError("Unknown arguments were given! Check the command line!")

    # experiment settings
    suffix = '{}_{}_{}'.format(args.experiment_name, args.training_set, args.batch_reduce)

    if args.gor:
        suffix = suffix + '_gor_alpha{:1.1f}'.format(args.alpha)
    if args.anchorswap:
        suffix = suffix + '_as'
    if args.anchorave:
        suffix = suffix + '_av'
    if args.fliprot:
        suffix = suffix + '_fliprot'
    if args.use_arf:
        suffix = suffix + '_arf'
    if args.use_srn:
        suffix = suffix + '_srn'
    suffix = suffix + '_' + str(int(args.n_triplets/1000)) +'k'

    # Here only the random_global hard negative mining and gor need to generate the triplets at the dataset calling
    # L2Net and HardNet generate the negative patch on the fly, using the output of the calculation
    triplet_flag = (args.batch_reduce == 'random_global') or args.gor
    dataset_names = ['liberty', 'notredame', 'yosemite']
    TEST_ON_W1BS = False

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

    # create loggin directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    log_dir = os.path.join(args.log_dir, suffix)
    DESCS_DIR = os.path.join(log_dir,'temp_descs')
    if TEST_ON_W1BS:
        if not os.path.isdir(DESCS_DIR):
            os.makedirs(DESCS_DIR)
    logger, file_logger = None, None
    if args.use_srn:
        from models.HardNet_SRN import HardNet_SRN
        model = HardNet_SRN(args.use_smooth)
    else:
        from models.HardNet import HardNet
        model = HardNet(args.use_arf, args.orientation)

    if(args.enable_logging):
        from Loggers import Logger, FileLogger
        logger = Logger(log_dir) # remove all the file in LOG_DIR and begin log
        file_logger = FileLogger(os.path.join(args.log_dir, suffix))

    # load_random_triplets=False, the return data is (anchor, positive)
    # no negative patch, the negative patch is generated on-the-fly
    train_loader = create_loaders(load_random_triplets=triplet_flag)
    test_loaders = create_loaders(load_random_triplets=triplet_flag, train=False)
    main(train_loader, test_loaders, model, logger, file_logger)
