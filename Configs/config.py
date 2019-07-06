from configparser import ConfigParser
import configparser

class Configurations():
    def __init__(self, config_file=None):
        super(Configurations, self).__init__()

        if config_file is None:
            config_file = './Configs/config.cfg'

        config = ConfigParser()

        config._interpolation = configparser.ExtendedInterpolation()
        config.read(filenames = config_file)

        self.config = config
        self.config_path = config_file
        #self.add_section = 'Additional'
        print('Loaded Config File Successfully ...')

        # print configurations
        for section in self.config.sections():
            for k, v in self.config.items(section):
                print(k, ":", v)

    @property
    def root(self):
        return self.config.get('Directory', 'root')

    @property
    def path_source_images(self):
        return self.config.get('Directory', 'path_source_images')

    @property
    def path_source_labels(self):
        return self.config.get('Directory', 'path_source_labels')

    @property
    def path_cropped_images(self):
        return self.config.get('Directory', 'path_cropped_images')

    @property
    def path_cropped_labels(self):
        return self.config.get('Directory', 'path_cropped_labels')

    @property
    def path_output(self):
        return self.config.get('Directory', 'path_output')

    @property
    def size_cropped_images_h(self):
        return self.config.getint('DataProcessing', 'size_cropped_images_h')

    @property
    def size_cropped_images_w(self):
        return self.config.getint('DataProcessing', 'size_cropped_images_w')

    @property
    def size_overlap(self):
        return self.config.getint('DataProcessing', 'size_overlap')

    @property
    def nb_classes(self):
        return self.config.getint('Data', 'nb_classes')

    @property
    def batch_size(self):
        return self.config.getint('Data', 'batch_size')

    @property
    def random_seed(self):
        return self.config.getint('General', 'random_seed')

    @property
    def init_lr(self):
        return self.config.getfloat('Optimizer', 'init_lr')

    @property
    def lr_decay(self):
        return self.config.getfloat('Optimizer', 'lr_decay')

    @property
    def momentum(self):
        return self.config.getfloat('Optimizer', 'momentum')

    @property
    def weight_decay(self):
        return self.config.getfloat('Optimizer', 'weight_decay')

    @property
    def epsilon(self):
        return self.config.getfloat('Optimizer', 'epsilon')

    @property
    def optimizer(self):
        return self.config.get('Optimizer', 'optimizer')

    @property
    def monitor(self):
        return self.config.get('Training', 'monitor')

    @property
    def weight_init_algorithm(self):
        return self.config.get('Training', 'weight_init_algorithm')

    @property
    def loss_fn(self):
        return self.config.get('Training', 'loss_fn')

    @property
    def verbosity(self):
        return self.config.getint('Training', 'verbosity')

    @property
    def early_stop(self):
        return self.config.getint('Training', 'early_stop')

    @property
    def save_period(self):
        return self.config.getint('Training', 'save_period')

    @property
    def dis_period(self):
        return self.config.getint('Training', 'dis_period')

    @property
    def epochs(self):
        return self.config.getint('Training', 'epochs')


