import argparse
from torchvision import datasets, transforms
import torch.optim as optim
from model import *
from model_glcn import GLCN
from utils import *
import os
import torch

num_labeled = 1000
num_valid = 1000
eval_freq = 10
lr = 0.005
cuda_device = "0"


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | svhn')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--num_epochs', type=int, default=5000)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--epsilon', type=float, default=2.5)
parser.add_argument('--top_bn', type=bool, default=True)
parser.add_argument('--method', default='vat')
parser.add_argument('--lr', type=float, default=0.1)

parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--out_channels', type=int, default=7)
parser.add_argument('--ngcn_layers', type=int, default=30)
parser.add_argument('--nclass', type=int, default=10)
parser.add_argument('--gamma_reg', type=float, default=0.01)
parser.add_argument('--lamda_reg', type=float, default=0.00001)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--cuda', dest='cuda', default='0', type=str)
parser.add_argument('--mode', default='gpu', help='cpu/gpu')

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

# set up gpu
if opt.mode == 'gpu':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda)
    print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']), flush= True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print('Using CPU', flush= True)
opt.device = torch.device('cuda:0' if opt.mode == 'gpu' else 'cpu')

def tocuda(x):
    if opt.use_cuda:
        return x.cuda()
    return x


def train(model, x, y, optimizer, lamda_reg=0.0):

    model.train()
    ce = nn.CrossEntropyLoss()

    semi_outputs, loss_GL, S = model(x)
    print("The learned S is ", torch.sum(S, dim=-1))
    ce_loss = ce(semi_outputs[:num_labeled], y)
    loss = ce_loss + lamda_reg * loss_GL
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return semi_outputs, loss, ce_loss

def eval(y_pred, y):

    # print(semi_outputs.shape)
    # y_pred = semi_outputs[num_labeled:(num_labeled+num_valid)]
    prob, idx = torch.max(y_pred, dim=1)
    return torch.eq(idx, y).float().mean()

# Several Ways to initialize the weights
# 1. initialize different weights using different initialization
def weights_init(m):
    """

    Usage: model.apply(weights_init)
    :param m:
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)

# 2. weight different weights using different torch.nn methods
def init_all(model, init_funcs):
    """
    Usage: init_all(model, init_funcs)

    :param model:
    :param init_funcs:
    :return:
    """
    for p in model.parameters():
        init_func = init_funcs.get(len(p.shape), init_funcs["default"])
        init_func(p)

init_funcs = {
    1: lambda x: torch.nn.init.normal_(x, mean=0., std=1.), # can be bias
    2: lambda x: torch.nn.init.xavier_normal_(x, gain=1.), # can be weight
    3: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.), # can be conv1D filter
    4: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.), # can be conv2D filter
    "default": lambda x: torch.nn.init.xavier_uniform_(x, gain=1.), # everything else
}

if opt.dataset == 'svhn':
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=opt.dataroot, split='train', download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ])),
        batch_size=100, shuffle=True)

elif opt.dataset == 'cifar10':
    num_labeled = 1000
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=opt.dataroot, train=True, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ])),
        batch_size=100, shuffle=True)

else:
    raise NotImplementedError

train_data = []
train_target = []

for (data, target) in train_loader:
    train_data.append(data)
    train_target.append(target)

train_data = torch.cat(train_data, dim=0)
train_target = torch.cat(train_target, dim=0)

print(opt.dataset)
print("Total number of training data is ", train_data.shape)

# all_data = tocuda(train_data[:10000])
# all_target = tocuda(train_target[:10000])

all_data = train_data[:10000].to(opt.device)
all_target = train_target[:10000].to(opt.device)


train_data, valid_data, test_data = all_data[:num_labeled, ], \
                                    all_data[num_labeled:num_valid+num_labeled, ], \
                                    all_data[num_valid+num_labeled:, ]
train_target, valid_target, test_target = all_target[:num_labeled], \
                                          all_target[num_labeled:num_valid + num_labeled, ], \
                                          all_target[num_valid + num_labeled:, ]

model = GLCN(opt.in_channels, opt.out_channels, opt.ngcn_layers,
             opt.nclass, opt.gamma_reg, opt.dropout).to(opt.device)
# model = tocuda(model)

# model.apply(weights_init)
init_all(model, init_funcs)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

min_valid_acc = 0.0
no_increase_step = 0

final_output = None
# train the network
for epoch in range(opt.num_epochs):

    if epoch > opt.epoch_decay_start:
        decayed_lr = (opt.num_epochs - epoch) * opt.lr / (opt.num_epochs - opt.epoch_decay_start)
        optimizer.lr = decayed_lr
        optimizer.betas = (0.5, 0.999)

    # training
    semi_outputs, v_loss, ce_loss = train(model, all_data, train_target, optimizer, opt.lamda_reg)

    print("Epoch :", epoch, "GLCN Loss :", v_loss.item(), "CE Loss :", ce_loss.item())

    # evaluating
    if epoch % eval_freq == 0 or epoch + 1 == opt.num_epochs:
        train_preds = semi_outputs[:num_labeled]
        train_accuracy = eval(train_preds, train_target)
        print("Train accuracy :", train_accuracy.item())

        val_preds = semi_outputs[num_labeled:num_valid+num_labeled]
        val_accuracy = eval(val_preds, valid_target)
        print("Valid accuracy :", val_accuracy.item())

        if val_accuracy > min_valid_acc:
            min_valid_acc = val_accuracy
            no_increase_step = 0
        else:
            no_increase_step += 1

        if no_increase_step == 100:
            final_output = semi_outputs
            break

    final_output = semi_outputs

test_preds = final_output[num_valid+num_labeled:]
test_accuracy = eval(test_preds, test_target)
print("Test accuracy :", test_accuracy.item())