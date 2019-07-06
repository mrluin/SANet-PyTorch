import torch
import torch.nn as nn
import torch.nn.functional as F


def visdom_init(visdom, phase_list, element_list, configs):
    """
    :param visdom: server
    #:param resume: boolean argument, whether resume from ckpt
    :param phase_list: ['train', 'valid'] or ['train', 'test'], don't support train phase only, because it is not worth
    :param element_list: subset of ['loss', 'accuracy', 'miou', 'f1score', 'lr']
    :return list of visdom_windows of each element in element_list in order
    """
    print("     + Visualization init ... ...")

    windows = []

    for dis_element in element_list:
        if dis_element in ['loss', 'accuracy', 'miou', 'f1score']:
            window = visdom.line(
                X=torch.stack((torch.ones(1), torch.ones(1)), 1),
                Y=torch.stack((torch.ones(1), torch.ones(1)), 1),
                opts=dict(title='{}_{}_{}'.format(phase_list[0], phase_list[1], dis_element),
                          showlegend=True,
                          legend=['{}_{}'.format(phase_list[0], dis_element),
                                  '{}_{}'.format(phase_list[1], dis_element)],
                          xtype='linear',
                          label='epoch',
                          xtickmin=0,
                          xtick=True,
                          xtickstep=10,
                          ytype='linear',
                          ylabel='{}'.format(dis_element),
                          ytickmin=0,
                          ytick=True,
                          )
                )
            windows.append(window)
        elif dis_element == 'lr':
            window = visdom.line(
                X = torch.ones(1),
                Y = torch.tensor([configs.init_lr]),
                opts = dict(title = '{}'.format(dis_element),
                            showlegend=True,
                            legend=['{}'.format(dis_element)],
                            xtype='linear',
                            xlabel='epoch',
                            xtickmin=0,
                            xtick=True,
                            xtickstep=10,
                            ytype='linear',
                            ytickmin=0,
                            ylabel='{}'.format(dis_element),
                            ytick=True)
                )
            windows.append(window)
    return windows

def visdom_update(visdom, element_list, epoch, windows, current_lr, train_log=None, eval_log=None):
    """
    :param train_log: training log of each epoch
    :param valid_log: validation log of each epoch
    :param element_list:
    :return:
    """
    for update_element in element_list:
        if update_element in ['loss', 'accuracy', 'miou', 'f1score']:
            visdom.line(
                X = torch.stack((torch.ones(1) * epoch, torch.ones(1) * epoch), 1),
                Y = torch.stack((torch.tensor([train_log['{}'.format(update_element)]]), torch.tensor([eval_log['val_{}'.format(update_element)]])), 1),
                win = windows['{}'.format(update_element)][0],
                update='append' if epoch != 1 else 'insert',
            )
        elif update_element == 'lr':
            visdom.line(
                X = torch.ones(1) * epoch,
                Y = torch.tensor([current_lr]),
                win = windows['lr'][0],
                update='append' if epoch != 1 else 'insert',
            )

def _nostride2dilation(m, dilation):

    if isinstance(m, nn.Conv2d):
        if m.stride == (2, 2):
            m.stride = (1, 1)
            if m.kernel_size == (3, 3):
                m.dilation = (dilation // 2, dilation // 2)
                m.padding = (dilation // 2, dilation // 2)
        else:
            if m.kernel_size == (3, 3):
                m.dilation = (dilation, dilation)
                m.padding = (dilation, dilation)



