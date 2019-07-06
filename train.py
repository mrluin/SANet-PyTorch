import argparse
import datetime
import torch
import random
import numpy as np
import os
from Data.Dataset import PotsdamDataset
from torch.utils.data import DataLoader
from Configs.config import Configurations
from visdom import Visdom
from BaseTrainer import BaseTrainer
from BaseTester import BaseTester
from Models.FCNs import FCN8s

def launcher_train(model, configs, args, loader_train, loader_valid, begin_time, resume_file, loss_weight, visdom):

    Trainer = BaseTrainer(model = model,
                          configs = configs,
                          args = args,
                          loader_train = loader_train,
                          loader_valid = loader_valid,
                          begin_time = begin_time,
                          resume_file = resume_file,
                          loss_weight = loss_weight,
                          visdom = visdom)
    Trainer.train()
    print("Training phase Done !")


def launcher_test(model, configs, args, loader_test, begin_time, resume_file, loss_weight):

    Tester = BaseTester(model = model,
                        configs = configs,
                        args = args,
                        loader_test = loader_test,
                        begin_time = begin_time,
                        resume_file = resume_file,
                        loss_weight = loss_weight)

    Tester.eval_and_predict()
    print("Evaluation Done !")

def Launcher(configs, args, train, valid, test):

    # unbalanced weight
    loss_weight = torch.tensor([2.942786693572998,
                                3.0553994178771973,
                                3.2896230220794678,
                                4.235183238983154,
                                7.793163776397705,
                                6.382354259490967])

    model = FCN8s(configs=configs)
    print(model.name)

    # visdom
    viz = Visdom(server = args.server, port = args.port, env = model.name)
    assert viz.check_connection(timeout_seconds = 3), \
        'No connection could be formed quickly'

    begin_time = datetime.datetime.now().strftime('%m%d_%H%M%S')

    # TODO dataset P and V exchange

    loader_train=None
    loader_valid=None
    loader_test=None

    # TODO worker init, the last batch should be paid more attention to !!!
    if train == True:
        dataset_train = PotsdamDataset(configs=configs, subset='train')
        loader_train = DataLoader(dataset = dataset_train,
                                  batch_size = configs.batch_size,
                                  shuffle = True,
                                  num_workers = args.threads,
                                  pin_memory = False,
                                  drop_last = False)
    if valid == True:
        dataset_valid = PotsdamDataset(configs=configs, subset='valid')
        loader_valid = DataLoader(dataset = dataset_valid,
                                  batch_size = configs.batch_size,
                                  shuffle = False,
                                  num_workers = args.threads,
                                  pin_memory = False,
                                  drop_last = False)
    launcher_train(model=model,
                   configs=configs,
                   args=args,
                   loader_train=loader_train,
                   loader_valid=loader_valid,
                   begin_time=begin_time,
                   resume_file=args.resume_file,
                   loss_weight=loss_weight,
                   visdom=viz)

    if test == True:
        dataset_test = PotsdamDataset(configs=configs, subset='test')
        loader_test = DataLoader(dataset = dataset_test,
                                 batch_size = configs.batch_size,
                                 shuffle = False,
                                 pin_memory = False,
                                 num_workers = args.threads,
                                 drop_last = True)
        # TODO here, in test phase no need for resume_file manually added, automatically using `checkpoint-best.pth`
        # TODO independent lancher_test

        launcher_test(model=model,
                      configs=configs,
                      args=args,
                      loader_test=loader_test,
                      begin_time=begin_time,
                      resume_file=args.resume_file,
                      loss_weight=loss_weight,
                      )

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='configurations of training environment setting')

    # for visdom
    DEFAULT_PORT = 8097
    DEFAULT_HOSTNAME = 'http://localhost'

    parser.add_argument('-port', metavar='port', type=int, default=DEFAULT_PORT,
                        help='port the visdom server is running on.')
    parser.add_argument('-server', metavar='server', type=str, default=DEFAULT_HOSTNAME,
                        help='Server address of the target to run the demo on.')
    parser.add_argument('-input', metavar='input', type=str, default= None,
                        help='root path to directory containing input images, including train & valid & test')
    parser.add_argument('-output', metavar='output', type=str, default= None,
                        help='root path to directory containing all the output, including predictions, logs and ckpt')
    parser.add_argument('-resume_file', metavar='resume_file', type=str, default=None,
                        help='path to ckpt which will be loaded')
    parser.add_argument('-threads', metavar='threads', type=int, default=8,
                        help='number of thread used for DataLoader')
    parser.add_argument('-gpu', metavar='gpu', type=int, default=0,
                        help='gpu id to be used for prediction')
    parser.add_argument('-train', metavar='train', type=bool, default=True,
                        help='control the launcher')
    # TODO control validation phase
    parser.add_argument('-valid', metavar='valid', type=bool, default=True,
                        help='control the launcher')
    parser.add_argument('-test', metavar='test', type=bool, default=True,
                        help='control the launcher')
    parser.add_argument('-config_path', metavar='config_path', type=str, default='./Configs/config.cfg',
                        help='path to config file path')

    args = parser.parse_args()
    assert os.path.exists(args.config_path), \
        'config file path cannot be found'

    configs = Configurations(args.config_path)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # set random seeds
    random.seed(configs.random_seed)
    np.random.seed(configs.random_seed)
    torch.manual_seed(configs.random_seed)


    Launcher(configs = configs, args = args, train=args.train, valid=args.valid, test=args.test)