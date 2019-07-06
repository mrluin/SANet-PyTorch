import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import sys
import os
import numpy as np
import argparse

from Configs.config import Configurations
import torchvision.transforms.functional as TF
from torch.autograd import Function
from torchvision import models
from torchvision import utils
from PIL import Image
from Models.Proposed_Method.Spatial_Transformer_FCN_based import Spatial_Transformer_FCN_v2_2
from Models.FCNs import FCN8s

class FeatureExtractor():
    """
    Extracting activations and registering gradients from target
    intermediate layers
    """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
        self.outputs = []
    def save_gradient_hook(self, model, grad_input, grad_output):

        self.gradients.append(grad_output)

    def save_output_hook(self, model, input, output):

        self.outputs.append(output)

    def __call__(self, x):

        self.gradients = []
        # _modules means what defined in the __init__
        # TODO the original code only support the sequential model
        for name, module in self.model._modules.items():
            if name in self.target_layers:
                module.register_forward_hook(self.save_output_hook)
                module.register_backward_hook(self.save_gradient_hook)
            '''
            x = module(x)
            # if the module is in set of target layers save its gradient and output
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                #x.register_hook(self.save_output)
                outputs += [x]
                '''
        network_output = self.model(x)

        # here x is output of the network, outputs is the list of output of target_layers
        return self.outputs, network_output


class ModelOutputs():

    def __init__(self, model, target_layers):

        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):

        # gradients of target_layers
        return self.feature_extractor.gradients

    def __call__(self, x):

        # output of target layers, output of the network
        target_activations, output = self.feature_extractor(x)
        return target_activations, output

def preprocessing_images(path_to_image, gpu):

    means = [0.3366, 0.3599, 0.3333]
    stds = [0.1030, 0.1031, 0.1066]

    image = Image.open(path_to_image)
    image = TF.to_tensor(image)
    image = TF.normalize(image, means, stds)
    # unsqueeze from [C, H, W] to [1, C, H, W]
    image.unsqueeze_(0)
    image = np.asarray(image)

    # TODO here input init leaf tensor
    #print(torch.tensor(image, device = torch.device('cuda:{}'.format(gpu)), requires_grad=True).is_leaf)
    #print(image.requires_grad_(True).to('cuda:{}'.format(gpu)).is_leaf)
    return torch.tensor(image, device = torch.device('cuda:{}'.format(gpu)), requires_grad=True)

def show_cam_on_image(image, mask):

    # [H, W, 3]
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    cv2.imwrite('heatmap.jpg', heatmap)

    # tmp_image is normalized array
    tmp_image = np.transpose(image.detach().cpu().numpy().squeeze(0), axes=(1,2,0))
    cam = np.float32(heatmap / 255.)*0.9  + np.float32(tmp_image)*0.1

    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    cv2.imwrite('cam.jpg', np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, target_layer_names, gpu):

        self.model = model
        for module in self.model.modules():
            if isinstance(module, nn.Dropout2d):
                module.eval()
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

        self.cuda = gpu
        #if self.cuda != -1:
        #    self.model = model.to('cuda:{}'.format(self.cuda))

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):

        return self.model(input)

    def __call__(self, input, index=None):

        # output of intermediate layer, output of network / in cuda version
        features, output = self.extractor(input)
        one_hot = torch.zeros((1, output.shape[1], output.shape[2], output.shape[3]),
                              dtype=torch.float32).to('cuda:{}'.format(self.cuda))
        if index == None:
            # [1, H, W]
            print('none index')
            index = torch.argmax(output, dim=1)
            print(index)
            one_hot = one_hot.scatter_(1, index.unsqueeze(1), 1)

        else:
            one_hot[0, index, :, :] = 1

        # output [1, C, H, W]
        # one_hot [1, C, H, W]
        one_hot = torch.sum(one_hot * output)

        # TODO
        self.model.zero_grad()
        # perform backward on gpu
        one_hot.backward()
        # [B, C, H, W]
        #print(self.extractor.get_gradients())
        grads_val = self.extractor.get_gradients()[-1][0]
        # [C, H, W]
        target = features[-1][0, :].cpu()
        # [C]
        weights = torch.mean(grads_val, dim=(2, 3))[0, :].cpu()
        # [H, W]
        cam = torch.zeros(target.shape[1:], dtype=torch.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        # clip > 0
        cam = torch.where(cam > 0, cam, torch.tensor([0.]))
        # to the original resolution
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=input.size()[2:],mode='bilinear',
                            align_corners=True)
        #print(cam.shape)
        cam = cam.squeeze()
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        # normalized cam
        cam = cam.detach().numpy()
        print(cam)
        return cam

class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(ctx, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        ctx.save_for_backward(input, output)

        return output
    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input

class GuidedBackpropReLUModel:
    def __init__(self, model, gpu):

        # change model init scheme

        self.model = model
        self.model.eval()
        self.cuda = gpu
        if self.cuda != -1:
            self.model.to('cuda:{}'.format(self.cuda))

    def forward(self, input):

        return self.model(input)

    def __call__(self, input, index=None):

        output = self.forward(input)

        one_hot = torch.zeros((1, output.size()[1], output.size()[2],
                               output.size()[3]), dtype=torch.float32).to('cuda:{}'.format(self.cuda))
        if index == None:
            index = torch.argmax(output, dim=1)
            one_hot = one_hot.scatter_(1, index.unsqueeze(1), 1)
        else:
            one_hot[0, index, :, :] = 1

        one_hot = torch.sum(one_hot * output)

        # TODO zero_grad
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.data
        output = output[0, :, :, :].cpu().numpy()

        return output

if __name__ == '__main__':



    parser = argparse.ArgumentParser(description='grad_cam for segmentation')
    parser.add_argument('-gpu', metavar='gpu', type=int, default=0,
                        help='which gpu to use, when -1 means cpu')
    parser.add_argument('-image_path', metavar='image_path', type=str,
                        default='./expamples/image.png', help='path to input image')
    parser.add_argument('-ckpt_path', metavar='ckpt_path', type=str,
                        default='./expamples/ckpt.pth', help='path to checkpoint')
    parser.add_argument('-config_path', metavar='config_path', type=str,
                        default='./Configs/config/cfg', help='path to config file')
    args = parser.parse_args()

    configs = Configurations(args.config_path)

    assert os.path.exists(args.image_path), \
        'path to input images does not exist'

    assert os.path.exists(args.ckpt_path), \
        'path ot checkpoint does not exist'

    # For network and its weight
    #model = Spatial_Transformer_FCN_v2_2(configs=configs)
    model = FCN8s(6).to('cuda:{}'.format(args.gpu))
    checkpoint = torch.load(args.ckpt_path, map_location={'cuda:0' :'cuda:1'})
    #print(checkpoint.device)
    weight = checkpoint['state_dict']

    model.load_state_dict(weight)


    grad_cam = GradCam(model=model, target_layer_names=['score_upup'], gpu=args.gpu)

    # TODO read and preprocessing by PIL.Image and process by torchvision
    input = preprocessing_images(args.image_path, args.gpu)

    # If None, returns the map for the highest scoreing category
    # Otherwise, targets the requested index
    target_index = 1

    mask = grad_cam(input, index=target_index)

    show_cam_on_image(input, mask)

    '''
    # So far, we cannot change the nn.Module to function in Sequential
    # The guided back does not work well
    model_gbrelu = Spatial_Transformer_FCN_based_counter(pretrained=None, relu_layer=GuidedBackpropReLU.apply)
    checkpoint = torch.load(args.ckpt_path)
    weight = checkpoint['state_dict']
    model.load_state_dict(weight,strict=False)

    # TODO The guided back relu doesn't work
    guidedback_model = GuidedBackpropReLUModel(model=model_gbrelu, gpu=args.gpu)

    # gb in cuda version, the same shape with the input [C, H, W]
    guided_back = guidedback_model(input, index=target_index)

    # guided_back_image
    utils.save_image(torch.from_numpy(guided_back), 'gb.jpg')

    cam_mask = np.zeros(guided_back.shape)
    for i in range(0, guided_back.shape[0]):
        cam_mask[i, :, :] = mask

    cam_gb = cam_mask * guided_back
    utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')
    '''
