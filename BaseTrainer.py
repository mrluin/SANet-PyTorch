import os
import math
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
import time
import numpy as np

from Utils.tools import AverageMeter, ensure_dir, match_image_label
from metrics import Metrics
from opts import visdom_init, visdom_update

class BaseTrainer(object):
    def __init__(self,
                 model,
                 configs,
                 args,
                 loader_train,
                 loader_valid,
                 begin_time,
                 resume_file,
                 loss_weight,
                 visdom):
        super(BaseTrainer, self).__init__()
        print("     + Training Start ... ...")

        # for general
        self.configs = configs
        self.args = args
        self.device = (self._device(self.args.gpu))
        self.model = model.to(self.device)
        self.loader_train = loader_train
        # TODO validation phase controller
        self.loader_valid = loader_valid

        # for time
        self.begin_time = begin_time               # part of ckpt name
        self.save_period = self.configs.save_period # for save ckpt
        self.dis_period = self.configs.dis_period   # for display

        # directory
        self.path_checkpoints = os.path.join(self.configs.path_output, 'checkpoints', model.name, self.begin_time)
        self.path_logs = os.path.join(self.configs.path_output, 'logs', model.name, self.begin_time)
        ensure_dir(self.path_checkpoints)
        ensure_dir(self.path_logs)

        if self.loader_train is not None and self.loader_valid is not None:
            self.history = {
                'train': {
                    'epoch': [],
                    'loss': [],
                    'accuracy': [],
                    'miou': [],
                    'f1score': [],
                },
                'valid': {
                    'epoch': [],
                    'loss': [],
                    'accuracy': [],
                    'miou': [],
                    'f1score': [],
                }
            }
        else:
            self.history = {
                'train': {
                    'epoch': [],
                    'loss': [],
                    'accuracy': [],
                    'miou': [],
                    'f1score': [],
                }
            }

        # TODO
        # for resume update curve
        self.windows_name = {
            'miou': [],
            'loss': [],
            'accuracy': [],
            'lr': [],
            'f1score': [],
        }

        # for optimizer
        self.loss_weight = loss_weight.to(self.device)
        self.loss = self._loss(loss_function=self.configs.loss_fn).to(self.device)
        self.optimizer = self._optimizer(lr_algorithm=self.configs.optimizer)
        self.lr_scheduler = self._lr_scheduler()
        self.weight_init_algorithm = self.configs.weight_init_algorithm
        self.current_lr = self.configs.init_lr

        print(self.optimizer)
        print(self.loss)

        # for training
        self.start_epoch = 1
        self.early_stop = self.configs.early_stop # early stop steps
        self.monitor_mode = self.configs.monitor.split('/')[0]
        self.monitor_metric = self.configs.monitor.split('/')[1]
        self.monitor_best = -math.inf
        self.best_epoch = -1

        # monitor
        if self.monitor_mode != 'off':
            assert self.monitor_mode in ['min', 'max']
            self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf

        # for resuming
        self.resume_file = resume_file
        self.resume = True if self.resume_file is not None else False

        if self.resume:
            self._resume_ckpt(resume_file = self.resume_file)

        self.visdom = visdom


    def train(self):

        if self.visdom is not None:
            if self.resume == False:
                # create panes for training phase for loss metrics learning_rate
                #print("     + Visualization init ... ...")

                visdom_windows = visdom_init(self.visdom,
                                             ['train', 'valid'] if self.loader_valid is not None else ['train', 'test'],
                                             ['loss','accuracy','f1score','lr','miou'],configs=self.configs)

                self.windows_name['loss'].append(str(visdom_windows[0]))
                self.windows_name['accuracy'].append(str(visdom_windows[1]))
                self.windows_name['f1score'].append(str(visdom_windows[2]))
                self.windows_name['lr'].append(str(visdom_windows[3]))
                self.windows_name['miou'].append(str(visdom_windows[4]))

            else:
                # resume condition here already loaded the resume_file in the init phase of the class
                print("     + Loading visdom file ... ... Done!")
                print("     + Visdom Loaded, Training !")
        else:
            print("     + Visdom unabled, Training !")

        total_epochs = self.configs.epochs
        for epoch in range(self.start_epoch, total_epochs + 1):

            train_log = self._train_epoch(epoch)

            if self.loader_valid is not None:
                eval_log = self._eval_epoch(epoch)

                # if self.loader_valid is None, choose loader_test to get best ckpt
                visdom_update(self.visdom, ['loss', 'accuracy', 'f1score', 'lr', 'iou'],
                              epoch, self.windows_name, self.current_lr, train_log=train_log, eval_log=eval_log)

                # save ckpt and best ckpt
                best = False
                not_improved_count = 0
                if self.monitor_mode != 'off':
                    improved = (self.monitor_mode == 'min' and eval_log['val_'+self.monitor_metric] < self.monitor_best) or \
                               (self.monitor_mode == 'max' and eval_log['val_'+self.monitor_metric] > self.monitor_best)
                    if improved:
                        # TODO need to confirm
                        self.monitor_best = eval_log['val_'+self.monitor_metric]
                        best = True
                        self.best_epoch = eval_log['epoch']

                    else:
                        not_improved_count += 1

                    if not_improved_count > self.early_stop:
                        print("     + Validation Performance didn\'t improve for {} epochs."
                              "     + Training stop :/"
                              .format(not_improved_count))
                        break
            
                if epoch % self.save_period == 0 or best == True:
                    self._save_ckpt(epoch, is_best = best)

        # saving the history when training is done
        print("     + Saving History ... ... ")
        hist_path = os.path.join(self.path_logs, 'history_train_valid.txt')
        with open(hist_path, 'w') as f:
            f.write(str(self.history))


    def _train_epoch(self, epoch):

        # lr update
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch)
            for param_group in self.optimizer.param_groups:
                self.current_lr = param_group['lr']

        batch_time = AverageMeter()
        data_time = AverageMeter()
        ave_loss = AverageMeter()
        ave_acc = AverageMeter()
        ave_iou = AverageMeter()
        ave_iou_pc = AverageMeter() # per-class
        ave_f1 = AverageMeter()
        ave_f1_pc = AverageMeter() # per-class


        self.model.train()
        tic = time.time()

        for step, (data, target) in enumerate(self.loader_train, start = 1):

            data = data.to(self.device, non_blocking = True)
            target = target.to(self.device, non_blocking = True)

            data_time.update(time.time() - tic)

            # forward
            logits = self.model(data)
            loss = self.loss(logits, target) # CELoss includes softmax

            # metrics
            metrics = Metrics(logits, target, self.configs.nb_classes)

            # TODO return cpu version: test
            acc = metrics.acc
            f1_score_pc, f1_score_overall = metrics.f1_score, metrics.mean_f1_score
            iou_pc, iou_overall = metrics.iou, metrics.mean_iou

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # update average metrics
            # multi-class item() can not perform well
            batch_time.update(time.time() - tic)
            ave_loss.update(loss.data.item())
            # TODO return cpu version: test
            ave_acc.update(acc.data.item())
            ave_f1_pc.update(f1_score_pc.data)
            ave_f1.update(f1_score_overall.data.item())
            ave_iou_pc.update(iou_pc.data)
            ave_iou.update(iou_overall.data.item())

            # TODO nan detector
            assert ave_acc.average() != float('nan') and ave_f1.average() != float('nan') and \
                   (ave_f1_pc.average() != float('nan')).all(), \
                'Appears nan value in {}epoch {}step in training phase!'.format(epoch, step)

            if step % self.dis_period == 0:
                print('Epoch: [{}][{}/{}],\n'
                      'Learning_Rate: {:.6f},\n'
                      'Time: {:.4f},       Data:     {:.4f},\n'
                      'F1_Score: {:6.4f},  IoU:{:6.4f}\n'
                      'class0: {:6.4f},        {:6.4f}\n'
                      'class1: {:6.4f},        {:6.4f}\n'
                      'class2: {:6.4f},        {:6.4f}\n'
                      'class3: {:6.4f},        {:6.4f}\n'
                      'class4: {:6.4f},        {:6.4f}\n'
                      'class5: {:6.4f},        {:6.4f}\n'
                      'Accuracy: {:6.4f},      Loss: {:.6f}'
                      .format(epoch, step, len(self.loader_train),
                              self.current_lr,
                              batch_time.average(), data_time.average(),
                              ave_f1.average(),       ave_iou.average(),
                              ave_f1_pc.average()[0], ave_iou_pc.average()[0],
                              ave_f1_pc.average()[1], ave_iou_pc.average()[1],
                              ave_f1_pc.average()[2], ave_iou_pc.average()[2],
                              ave_f1_pc.average()[3], ave_iou_pc.average()[3],
                              ave_f1_pc.average()[4], ave_iou_pc.average()[4],
                              ave_f1_pc.average()[5], ave_iou_pc.average()[5],
                              ave_acc.average(), ave_loss.average()))


            tic = time.time()

        #  train log and return
        self.history['train']['epoch'].append(epoch)
        self.history['train']['loss'].append(ave_loss.average())
        self.history['train']['accuracy'].append(ave_acc.average())
        self.history['train']['f1score'].append(ave_f1.average())
        self.history['train']['miou'].append(ave_iou.average())
        return {
            'epoch': epoch,
            'loss': ave_loss.average(),
            'accuracy': ave_acc.average(),
            'miou': ave_iou.average(),
            'f1score': ave_f1.average(),
        }


    def _eval_epoch(self, epoch):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        ave_loss = AverageMeter()
        ave_acc = AverageMeter()
        ave_iou = AverageMeter()
        ave_iou_pc = AverageMeter()
        ave_f1 = AverageMeter()
        ave_f1_pc = AverageMeter()

        self.model.eval()

        with torch.no_grad():
            tic = time.time()
            for steps, (data, target) in enumerate(self.loader_valid, start=1):

                # processing no blocking
                # non_blocking tries to convert asynchronously with respect to the host if possible
                # converting CPU tensor with pinned memory to CUDA tensor
                # overlap transfer if pinned memory
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                data_time.update(time.time() - tic)
                logits = self.model(data)
                loss = self.loss(logits, target)

                # TODO return cpu version :test
                metrics = Metrics(logits, target, self.configs.nb_classes)
                acc = metrics.acc
                f1_score_per_class, f1_score_overall = metrics.f1_score, metrics.mean_f1_score
                iou_per_class, iou_overall = metrics.iou, metrics.mean_iou
                # update ave metrics
                batch_time.update(time.time()-tic)
                ave_loss.update(loss.data.item())

                # TODO return cpu version : test
                ave_acc.update(acc.data.item()) # overall
                ave_f1.update(f1_score_overall.data.item()) # overall
                ave_f1_pc.update(f1_score_per_class.data) # per_class
                ave_iou_pc.update(iou_per_class.data)
                ave_iou.update(iou_overall.data.item())

                tic = time.time()

                # TODO nan detector
                assert ave_acc.average() != float('nan') and ave_f1.average() != float('nan') and \
                       (ave_f1_pc.average() != float('nan')).all(), \
                    'Appears nan value in {}epoch {}step of valid phase!'.format(epoch ,steps)

            # display validation at the end
            print('Epoch {} validation done !'.format(epoch))
            print('Time: {:.4f},       Data:     {:.4f},\n'
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

        self.history['valid']['epoch'].append(epoch)
        self.history['valid']['loss'].append(ave_loss.average())
        self.history['valid']['f1score'].append(ave_f1.average())
        self.history['valid']['accuracy'].append(ave_acc.average())
        self.history['valid']['miou'].append(ave_iou.average())

        #  validation log and return
        return {
            'epoch': epoch,
            'val_loss': ave_loss.average(),
            'val_accuracy': ave_acc.average(),
            'val_miou': ave_iou.average(),
            'val_f1score': ave_f1.average()
        }

    def _device(self, gpu):

        if gpu == -1:
            device = torch.device('cpu')
            return device
        else:
            device = torch.device('cuda:{}'.format(gpu))
            return device

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

    def _weight_init(self, module):

        # no bias use
        #classname = module.__class__.__name__
        #if classname.find('Conv') != -1:
        if isinstance(module, nn.Conv2d):
            if self.weight_init_algorithm == 'kaiming':
                init.kaiming_normal_(module.weight.data)
            else:
                init.xavier_normal_(module.weight.data)
        #elif classname.find('BatchNorm') != -1:
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)

    def _save_ckpt(self, epoch, is_best):

        state = {
            'epoch': epoch + 1,
            'arch': str(self.model),
            'state_dict': self.model.state_dict(),
            'optimizer': str(self.optimizer),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'monitor_metric': self.monitor_metric,
            'monitor_best': self.monitor_best,
            'history': self.history,
            'windows_name': self.windows_name,
        }

        filename = os.path.join(self.path_checkpoints, 'checkpoint-epoch{}.pth'.format(epoch))
        if is_best:
            best_filename = os.path.join(self.path_checkpoints, 'checkpoint-best.pth')
            print("     + Saving Best Checkpoint : Epoch {}  path: {} ...  ".format(epoch, best_filename))
            torch.save(state, best_filename)
        else:
            print("     + Saving Checkpoint per {} epochs, path: {} ... ".format(self.save_period, filename))
            torch.save(state, filename)

    def _resume_ckpt(self, resume_file):

        resume_path = os.path.join(resume_file)
        print("     + Loading Checkpoint: {} ... ".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch']
        assert str(self.model) == checkpoint['arch'], \
            'The model architecture of the checkpoint is not matched to the current model architecture'
        self.model.load_state_dict(checkpoint['state_dict'])
        assert str(self.optimizer) == checkpoint['optimizer'], \
            'The optimizer of the checkpoint is not matched to the current optimizer'
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        assert self.monitor_metric == checkpoint['monitor_metric'], \
            'The monitor metric is not matched the current monitor metric'
        self.monitor_best = checkpoint['monitor_best']
        self.history = checkpoint['history']
        self.windows_name = checkpoint['windows_name']

        print("     + Checkpoint file: '{}' , Start epoch {} Loaded !\n"
              "     + Prepare to run ! ! !"
              .format(resume_path, self.start_epoch))
