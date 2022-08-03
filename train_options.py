import argparse

parser = argparse.ArgumentParser(description='WTALC')

# basic setting
parser.add_argument('--gpus', type=int, default=[2], nargs='+', help='used gpu')
parser.add_argument('--run-type', type=int, default=0, help='train (0) or test (1) or plot (2)')
parser.add_argument('--code-type', type=int, default=1, help='clone code type (1-5)')
parser.add_argument('--model-id', type=int, default=13, help='model id for saving model')

# 1 128-128
# 2 128-64
# 3 128-32
# 4 64-64
# 5 64-32
# 6 32-32
# 7 none

# sparse
# 8 max
# 9 mean
# 10 concat

# att
# 11 max
# 12 mean
# 13 concat

# loading model
parser.add_argument('--pretrained', default=False, help='is pretrained model')
parser.add_argument('--load-epoch', type=int, default=None, help='epoch of loaded model')

# storing parameters
parser.add_argument('--save-interval', type=int, default=1, help='interval for storing model (default: 2000)')

# language settings
parser.add_argument('--root', default='data/', help='dataset to train on')
parser.add_argument('--lang', default='c', help='dataset to train on')

# input paramaters
parser.add_argument('--batch-size', type=int, default=32, help='number of instances in a batch of data (default: 10)')
parser.add_argument('--input-dim', type=int, default=128, help='size of feature (default: 1024)')
parser.add_argument('--encode-dim1', type=int, default=128, help='size of feature (default: 1024)')
parser.add_argument('--encode-dim2', type=int, default=128, help='size of feature (default: 1024)')

# training paramaters
parser.add_argument('--optimizer', type=str, default='Adam', help='used optimizer')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate (default: 0.0001)')
parser.add_argument('--weight-decay', type=float, default=0.001, help='learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.9, help='learning rate (default: 0.0001)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--max-epoch', type=int, default=8, help='maximum iteration to train (default: 50000)')

# Learning Rate Decay
parser.add_argument('--warmup', default=False, action='store_true', help='whether use warm up')
parser.add_argument('--gradual-warm-up', default=False, action='store_true',
                    help='whether use gradual warm up (valid when warm up is true)')
parser.add_argument('--warmup-lr', type=float, default=0.00001, help='warm up learning rate')
parser.add_argument('--warmup-epoch', type=int, default=100, help='warm up iteration (default: 10000)')
parser.add_argument('--decay-type', type=int, default=0,
                    help='weight decay type (0 for None, 1 for step decay, 2 for cosine decay)')
parser.add_argument('--changeLR_list', type=int, default=[5, 1000], help='change lr step')
