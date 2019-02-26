from __future__ import print_function

import argparse
import gc
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.externals import joblib
from torch import nn

from dataloader import TextClassDataLoader
from util import AverageMeter, accuracy

np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--save-freq', '-sf', default=10, type=int, metavar='N', help='model save frequency(epoch)')
parser.add_argument('--embedding-size', default=50, type=int, metavar='N', help='embedding size')
parser.add_argument('--hidden-size', default=128, type=int, metavar='N', help='rnn hidden size')
parser.add_argument('--layers', default=2, type=int, metavar='N', help='number of rnn layers')
parser.add_argument('--classes', default=8, type=int, metavar='N', help='number of output classes')
parser.add_argument('--min-samples', default=5, type=int, metavar='N', help='min number of tokens')
parser.add_argument('--cuda', default=False, action='store_true', help='use cuda')
parser.add_argument('--glove', default='glove/glove.6B.300d.txt', help='path to glove txt')
parser.add_argument('--gen', default='gen/glove_', help='path to glove txt')
parser.add_argument('--rnn', default='LSTM', choices=['LSTM', 'GRU'], help='rnn module type')
parser.add_argument('--mean_seq', default=False, action='store_true', help='use mean of rnn output')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
args = parser.parse_args()

gen = args.gen + str(args.embedding_size) + 'v'
# load vocab
d_word_index, model = None, None
if os.path.exists(gen + '/d_word_index.pkl'):
    d_word_index = joblib.load(gen + '/d_word_index.pkl')

# create tester
print("===> creating dataloaders ...")
val_loader = TextClassDataLoader('data/test_pdtb.tsv', d_word_index, batch_size=args.batch_size)

# load model
if os.path.exists(gen + '/rnn_50.pkl'):
    model = joblib.load(gen + '/rnn_50.pkl')

# optimizer and loss
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                             weight_decay=args.weight_decay)

criterion = nn.CrossEntropyLoss()
print(optimizer)
print(criterion)

if args.cuda:
    torch.backends.cudnn.enabled = True
    cudnn.benchmark = True
    model.cuda()
    criterion = criterion.cuda()


def test(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target, seq_lengths) in enumerate(val_loader):

        if args.cuda:
            input = input.cuda()
            target = target.cuda()

        # compute output
        output = model(input, seq_lengths)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0][0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i != 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]  Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})  Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))
            gc.collect()

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


test(val_loader, model, criterion)
