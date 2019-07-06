import argparse
import torch
import random
import numpy as np
import os
import time
from Data.Dataset import PotsdamDataset
from torch.utils.data import DataLoader
from Configs.config import Configurations
from BaseTester import BaseTester
from Utils.tools import AverageMeter
from metrics import Metrics
from PIL import Image
from Models.FCNs import FCN8s



class Predictor(object):
    def __init__(self, configs, args, model, dataloader_predict):
        super(Predictor, self).__init__()

        self.configs = configs

        self.args = args
        self.device = torch.device('cpu' if self.args.gpu == -1 else 'cuda')
        self.model = model.to(self.device)
        self.dataloader_predict = dataloader_predict
        assert args.resume_file is not None, \
            'The path of checkpoint-best.pth can not be None'
        self.resume_ckpt_path = args.resume_file
        self.predict_path = args.save_path

    def predict(self):

        self._resume_ckpt(self.resume_ckpt_path)
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
                # TODO return cpu version
                metrics = Metrics(logits, target, self.configs.nb_classes)
                acc = metrics.acc
                f1_score_per_class, f1_score_overall = metrics.f1_score, metrics.mean_f1_score
                iou_per_class, iou_overall = metrics.iou, metrics.mean_iou
                # time and metrics
                batch_time.update(time.time() - tic)
                ave_loss.update(loss.data.item())
                # TODO return cpu version
                ave_acc.update(acc.data.item())
                ave_iou_pc.update(iou_per_class.data)
                ave_iou.update(iou_overall.data.item())
                ave_f1.update(f1_score_overall.data.item())
                ave_f1_pc.update(f1_score_per_class.data)

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
                          ave_f1.average(), ave_iou.average(),
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
            save_path = os.path.join(self.predict_path, save_filename)

            label_map_3.save(save_path)

    def _resume_ckpt(self, resume_file):

        resume_path = os.path.join(resume_file)
        print("     + Loading Checkpoint: {} ... ".format(resume_path))
        checkpoint = torch.load(resume_path)
        assert str(self.model) == checkpoint['arch'], \
            'The model architecture of the checkpoint is not matched to the current model architecture'
        self.model.load_state_dict(checkpoint['state_dict'])

        print("     + Checkpoint file: '{}'\n"
              "     + Prepare to run ! ! !"
              .format(resume_path))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='configurations of training environment setting')

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
    parser.add_argument('-config_path', metavar='config_path', type=str, default='./Configs/config.cfg',
                        help='path to config file path')
    parser.add_argument('-save_path', metavar='save_path', type=str, default=None,
                        help='path to directory that save output images')



    args = parser.parse_args()
    assert os.path.exists(args.save_path), \
        'the path to directory that saves output images cannot be found.'
    assert os.path.exists(args.config_path), \
        'config file path cannot be found'

    configs = Configurations(args.config_path)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # set random seeds
    random.seed(configs.random_seed)
    np.random.seed(configs.random_seed)
    torch.manual_seed(configs.random_seed)


    loss_weight = torch.tensor([2.942786693572998,
                                3.0553994178771973,
                                3.2896230220794678,
                                4.235183238983154,
                                7.793163776397705,
                                6.382354259490967])

    model = FCN8s(configs=configs)
    print(model.name)
    dataset_test = PotsdamDataset(configs=configs, subset='test')
    loader_test = DataLoader(dataset=dataset_test,
                             batch_size=configs.batch_size,
                             shuffle=False,
                             pin_memory=False,
                             num_workers=args.threads,
                             drop_last=True)

    predictor = Predictor(configs=configs, args=args, model=model, dataloader_predict=loader_test)
    predictor.predict()

