import os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class AverageMeter(object):
    """
        # Computes and stores the average and current value
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def _get_sum(self):
        return self.sum


def get_name(filename):

    # for specific dataset
    return filename.split('/')[-1].split('.')[0]

def match_image_label(image_files, label_files):

    for image, label in zip(image_files, label_files):

        if get_name(image).replace('RGB', 'label') != get_name(label):
            return False
    return True

