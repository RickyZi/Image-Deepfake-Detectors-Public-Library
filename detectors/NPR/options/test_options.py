from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # parser.add_argument('--dataroot')
        parser.add_argument('--model_path')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--earlystop_epoch', type=int, default=15)
        parser.add_argument('--lr', type=float, default=0.00002, help='initial learning rate for adam')
        parser.add_argument('--niter', type=int, default=0, help='# of iter at starting learning rate')

        # ----------------------------------- #
        # add ft flag
        parser.add_argument('--ft', action='store_true', help='Path to pretrained model to load')

        # add tf2k flag
        parser.add_argument('--tf2k', type = bool, default = False, help = 'Use 2k dataset and splits for training and testing')
        
        # add dataset flag
        parser.add_argument('--dataset', type = str, default = 'dataset', help = 'Which dataset to use (default: dataset)') # add custom dataset for demo
        # ----------------------------------- #

        self.isTrain = False
        return parser
