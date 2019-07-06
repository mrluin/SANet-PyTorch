import torch
import numpy as np
import os
import time
import tqdm
import torch.optim as optim
import torch.nn as nn
from Utils.tools import AverageMeter, ensure_dir
from metrics import Metrics
from PIL import Image


class BaseTester(object):
    def __init__(self,
                 model,
                 configs,
                 args,
                 loader_test,
                 begin_time,
                 resume_file,
                 loss_weight):
        super(BaseTester, self).__init__()

        # for general
        self.configs = configs
        self.args = args
        self.device = torch.device('cpu') if self.args.gpu == -1 else torch.device('cuda:{}'.format(self.args.gpu))

        # for training
        self.model = model.to(self.device)
        self.loss_weight = loss_weight.to(self.device)
        self.loss = self._loss(loss_function = self.configs.loss_fn).to(self.device)
        #self.optimizer = self._optimizer(lr_algorithm = self.configs.optimizer)
        #self.lr_scheduler = self._lr_scheduler()

        # time
        self.begin_time = begin_time

        # data
        self.loader_test = loader_test

        # for resume/save path
        self.history = {
            'eval': {
                'loss': [],
                'accuracy': [],
                'miou': [],
                'time': [],
                'f1score': [],
            },
        }
        self.path_logs = os.path.join(self.configs.path_output, 'test_logs', self.model.name, self.begin_time)
        self.path_predict = os.path.join(self.configs.path_output, 'predict', self.model.name, self.begin_time)

        self.resume_file = resume_file if resume_file is not None else \
            os.path.join(self.configs.path_output, 'checkpoints', self.model.name, self.begin_time, 'checkpoint-best.pth')

        ensure_dir(self.path_logs)
        ensure_dir(self.path_predict)


    def eval_and_predict(self):

        self._resume_ckpt(self.resume_file)

        self.model.eval()

        inference_time = AverageMeter()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        ave_loss = AverageMeter()
        ave_acc = AverageMeter()
        ave_iou = AverageMeter()
        ave_iou_pc = AverageMeter()
        ave_f1 = AverageMeter()
        ave_f1_pc = AverageMeter()

        with torch.no_grad():
            tic = time.time()
            for step, (data, target, filename) in enumerate(self.loader_test, start = 1):

                # data
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                data_time.update(time.time() - tic)

                inf_tic = time.time()
                logits = self.model(data)
                inference_time.update(time.time() - inf_tic)
                self._save_pred(logits, filename)

                loss = self.loss(logits, target)
                # TODO return cpu version:test
                metrics = Metrics(logits, target, self.configs.nb_classes)
                acc = metrics.acc
                f1_score_per_class, f1_score_overall = metrics.f1_score, metrics.mean_f1_score
                iou_per_class, iou_overall = metrics.iou, metrics.mean_iou

                # time and metrics
                batch_time.update(time.time() - tic)
                ave_loss.update(loss.data.item())
                # TODO return cpu version:test
                ave_acc.update(acc.data.item())
                ave_f1.update(f1_score_overall.data.item())
                ave_f1_pc.update(f1_score_per_class.data)
                ave_iou_pc.update(iou_per_class.data)
                ave_iou.update(iou_overall.data.item())

                # TODO nan detector
                assert ave_acc.average() != float('nan') and ave_f1.average() != float('nan') and \
                       (ave_f1_pc.average() != float('nan')).all(), 'Appears nan value in {}step of testing phase!'.format(step)

            # display evaluation result at the end
            print('Evaluation phase !\n'
                  'Time: {:.2f},  Data: {:.2f},\n'
                  'F1_Score: {:6.4f},  IoU:{:6.4f}\n'
                  'class0: {:6.4f},        {:6.4f}\n'
                  'class1: {:6.4f},        {:6.4f}\n'
                  'class2: {:6.4f},        {:6.4f}\n'
                  'class3: {:6.4f},        {:6.4f}\n'
                  'class4: {:6.4f},        {:6.4f}\n'
                  'class5: {:6.4f},        {:6.4f}\n'
                  'Accuracy: {:6.4f},      Loss: {:.6f}'
                  .format(batch_time.average(), data_time.average(),
                          ave_f1.average(),       ave_iou.average(),
                          ave_f1_pc.average()[0], ave_iou_pc.average()[0],
                          ave_f1_pc.average()[1], ave_iou_pc.average()[1],
                          ave_f1_pc.average()[2], ave_iou_pc.average()[2],
                          ave_f1_pc.average()[3], ave_iou_pc.average()[3],
                          ave_f1_pc.average()[4], ave_iou_pc.average()[4],
                          ave_f1_pc.average()[5], ave_iou_pc.average()[5],
                          ave_acc.average(), ave_loss.average()))

            print('For inference !\n'
                  'Total Time cost: {}s\n'
                  'Average Time cost per batch: {}s!'
                  .format(inference_time._get_sum(), inference_time.average()))

        self.history['eval']['loss'].append(ave_loss.average())
        self.history['eval']['accuracy'].append(ave_acc.average())
        self.history['eval']['miou'].append(ave_iou.average())
        self.history['eval']['f1score'].append(ave_f1.average())
        self.history['eval']['time'].append(inference_time.average())

        # test phase history
        print("     + Saved history of evaluation phase !")
        hist_path = os.path.join(self.path_logs, "history_eval.txt")
        with open(hist_path, 'w') as f:
            f.write(str(self.history))

    def _save_pred(self, logits, filenames):

        # here need to extend from 1-dim to 3-dim in channel dimension

        invert_mask_mapping = {
            0: (255, 255, 255),   # impervious surfaces
            1: (0, 0, 255),       # Buildings
            2: (0, 255, 255),     # Low Vegetation
            3: (0, 255, 0),       # Tree
            4: (255, 255, 0),     # Car
            5: (255, 0, 0),       # background/Clutter
        }
        for index, score_map in enumerate(logits):

            label_map_1 = torch.argmax(score_map, dim = 0).unsqueeze(0).cpu()

            # torch.expand share memory, so we choose cat operation
            label_map_3 = torch.cat([label_map_1, label_map_1, label_map_1], dim=0)
            #print(label_map_3.shape)
            label_map_3 = label_map_3.permute(1,2,0)

            for k in invert_mask_mapping:
                label_map_3[(label_map_3 == torch.tensor([k,k,k])).all(dim=2)] = torch.tensor(invert_mask_mapping[k])

            label_map_3 = Image.fromarray(np.asarray(label_map_3, dtype = np.uint8))

            # filename of the image like top_potsdam_2_10_RGB_x.tif
            filename = filenames[index].split('/')[-1].split('.')
            save_filename = filename[0] + '_pred.' + filename[1]
            save_path = os.path.join(self.path_predict, save_filename)

            label_map_3.save(save_path)


    def _resume_ckpt(self, resume_file):

        # resume function for testing phase, it just need model.state_dict()
        # TODO whether needs the optimizer in the testing phase !!!

        resume_path = os.path.join(resume_file)
        print("     + Loading Checkpoint: {} ... ".format(resume_path))
        checkpoint = torch.load(resume_path)
        assert str(self.model) == checkpoint['arch'], \
            'The model architecture of the checkpoint is not matched to the current model architecture'
        self.model.load_state_dict(checkpoint['state_dict'])
        #assert str(self.optimizer) == checkpoint['optimizer'], \
        #   'The optimizer of the checkpoint is not matched to the current optimizer'

        #print("     + Optimizer State Loaded ! :D ")
        print("     + Checkpoint file: '{}' , Loaded ! \n"
              "     + Prepare to test ! ! !"
              .format(self.resume_file))

    def _loss(self, loss_function):
        """
         add the loss function that you need
        :param loss_function: cross_entropy
        :return:
        """
        if loss_function == 'crossentropy':
            loss = nn.CrossEntropyLoss(weight=self.loss_weight)
            return loss

    def _optimizer(self, lr_algorithm):

        if lr_algorithm == 'adam':
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=self.configs.init_lr,
                                   betas=(0.9, 0.999),
                                   eps=self.configs.epsilon,
                                   weight_decay=self.configs.weight_decay,
                                   amsgrad=False)
            return optimizer
        if lr_algorithm == 'sgd':
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=self.configs.init_lr,
                                  momentum=self.configs.momentum,
                                  dampening=0,
                                  weight_decay=self.configs.weight_decay,
                                  nesterov=True)
            return optimizer

    def _lr_scheduler(self):

        # poly learning scheduler

        lambda1 = lambda epoch: pow((1-((epoch-1)/self.configs.epochs)), 0.9)
        lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        return lr_scheduler