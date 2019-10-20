import argparse
from torchvision import datasets, transforms
import torch.optim as optim
from model import *
from utils import *
import os

batch_size = 32
eval_batch_size = 100
unlabeled_batch_size = 128
num_labeled = 1000
num_valid = 1000
num_iter_per_epoch = 400
eval_freq = 5
lr = 0.001
cuda_device = "0"


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | svhn')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--num_epochs', type=int, default=120)
parser.add_argument('--epoch_decay_start', type=int, default=80)
parser.add_argument('--epsilon', type=float, default=2.5)
parser.add_argument('--top_bn', type=bool, default=True)
parser.add_argument('--method', default='vat')
parser.add_argument('--train', default=True, action='store_false')

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

def tocuda(x):
    if opt.use_cuda:
        return x.cuda()
    return x

def train(model, x, y, ul_x, optimizer):

    ce = nn.CrossEntropyLoss()
    y_pred = model(x)
    ce_loss = ce(y_pred, y)

    ul_y = model(ul_x)
    v_loss = vat_loss(model, ul_x, ul_y, eps=opt.epsilon)
    loss = v_loss + ce_loss
    if opt.method == 'vatent':
        loss += entropy_loss(ul_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return v_loss, ce_loss


def eval(model, x, y):

    y_pred = model(x)
    prob, idx = torch.max(y_pred, dim=1)
    return torch.eq(idx, y).float().mean()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)


if opt.dataset == 'svhn':
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=opt.dataroot, split='train', download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=opt.dataroot, split='test', download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ])),
        batch_size=eval_batch_size, shuffle=True)

elif opt.dataset == 'cifar10':
    num_labeled = 4000
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=opt.dataroot, train=True, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=opt.dataroot, train=False, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                      ])),
        batch_size=eval_batch_size, shuffle=True)
elif opt.dataset == 'mnist':
    # num_labeled = 1000
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=opt.dataroot, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
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

unique_labels = np.unique(train_target)
print("Unique Labels: ", unique_labels)
n_class = len(unique_labels)
nSamples_per_class_train = 100
nSamples_per_class_val = 100
nSamples_per_unlabel = 1000 - nSamples_per_class_train - nSamples_per_class_val

select_train_data = []
select_train_label = []
select_val_data = []
select_val_label = []

unlabeled_train_data = []
unlabeled_train_label = []

for label in unique_labels:
    label_mask = (train_target == label)
    current_label_X = train_data[label_mask]
    current_label_y = train_target[label_mask]
    select_train_data.append(current_label_X[:nSamples_per_class_train])
    select_train_label.append(current_label_y[:nSamples_per_class_train])
    select_val_data.append(current_label_X[nSamples_per_class_train:nSamples_per_class_train+nSamples_per_class_val])
    select_val_label.append(current_label_y[nSamples_per_class_train:nSamples_per_class_train + nSamples_per_class_val])

    unlabeled_train_data.append(current_label_X[nSamples_per_class_train + nSamples_per_class_val:1000])
    unlabeled_train_label.append(current_label_y[nSamples_per_class_train + nSamples_per_class_val:1000])

train_data = torch.cat(select_train_data, dim=0)
labeled_train = torch.cat(select_train_label, dim=0)
valid_data = torch.cat(select_val_data, dim=0)
labeled_val = torch.cat(select_val_label, dim=0)

unlabeled_train_data = torch.cat(unlabeled_train_data, dim=0)
unlabeled_train_label = torch.cat(unlabeled_train_label, dim=0)

# valid_data, train_data = train_data[:num_valid, ], train_data[num_valid:, ]
# valid_target, train_target = train_target[:num_valid], train_target[num_valid:, ]
# 
# labeled_train, labeled_target = train_data[:num_labeled, ], train_target[:num_labeled, ]
# unlabeled_train = train_data[num_labeled:, ]

in_channels = 3
if opt.dataset == 'mnist':
    in_channels = 1
model = tocuda(VAT(opt.top_bn, in_channels))
model.apply(weights_init)
optimizer = optim.Adam(model.parameters(), lr=lr)
max_val_acc = 0.0
patience = 0
max_patience = 10

path_best_model = f'./saved_models/{opt.dataset}/test_model'
if not os.path.exists(os.path.dirname(path_best_model)):
    os.mkdir(os.path.dirname(path_best_model))

if opt.train:
    # train the network
    for epoch in range(opt.num_epochs):

        if epoch > opt.epoch_decay_start:
            decayed_lr = (opt.num_epochs - epoch) * lr / (opt.num_epochs - opt.epoch_decay_start)
            optimizer.lr = decayed_lr
            optimizer.betas = (0.5, 0.999)

        for i in range(num_iter_per_epoch):
            batch_indices = torch.LongTensor(np.random.choice(labeled_train.size()[0], batch_size, replace=False))
            x = train_data[batch_indices]
            y = labeled_train[batch_indices]
            batch_indices_unlabeled = torch.LongTensor(np.random.choice(
                unlabeled_train_data.size()[0], unlabeled_batch_size, replace=False))

            ul_x = unlabeled_train_data[batch_indices_unlabeled]
            v_loss, ce_loss = train(model.train(), Variable(tocuda(x)), Variable(tocuda(y)), Variable(tocuda(ul_x)),
                                    optimizer)
            if i % 100 == 0:
                print("Epoch :", epoch, "Iter :", i, "VAT Loss :", v_loss.item(), "CE Loss :", ce_loss.item(), flush=True)

        if epoch % eval_freq == 0 or epoch + 1 == opt.num_epochs:
            # batch_indices = torch.LongTensor(np.random.choice(labeled_val.size()[0], batch_size, replace=False))
            # x = valid_data[batch_indices]
            # y = labeled_val[batch_indices]
            # train_accuracy = eval(model.eval(), Variable(tocuda(x)), Variable(tocuda(y)))
            # print("Val accuracy :", train_accuracy.item())

            val_accuracy = 0.0
            counter = 0
            for i in range(0, valid_data.shape[0], eval_batch_size):
                data = valid_data[i:i + eval_batch_size]
                target = labeled_val[i:i + eval_batch_size]
                acc = eval(model.eval(), Variable(tocuda(data)), Variable(tocuda(target)))
                val_accuracy += eval_batch_size * acc
                counter += eval_batch_size
            print("Val accuracy :", val_accuracy.item()/counter, flush=True)

            if max_val_acc < val_accuracy:
                max_val_acc = val_accuracy
                patience = 0
                torch.save(model.state_dict(), path_best_model)
            else:
                patience += 0
                if patience >= max_patience:
                    break

model.train()
if os.path.exists(path_best_model):
    # original saved file with DataParallel
    state_dict = torch.load(path_best_model)
    model.load_state_dict(state_dict)

test_accuracy = 0.0
counter = 0
# for (data, target) in test_loader:
#     n = data.size()[0]
#     acc = eval(model.eval(), Variable(tocuda(data)), Variable(tocuda(target)))
#     test_accuracy += n*acc
#     counter += n

test_pred = []
for i in range(0, unlabeled_train_data.shape[0], eval_batch_size):
    data = unlabeled_train_data[i:i+eval_batch_size]
    target = unlabeled_train_label[i:i+eval_batch_size]
    test_pred.append(target)
    acc = eval(model.eval(), Variable(tocuda(data)), Variable(tocuda(target)))
    test_accuracy += eval_batch_size * acc
    counter += eval_batch_size

print("Full test accuracy :", test_accuracy.item()/counter, flush=True)

# write the resulted data
# compose all the data and label together
test_pred = torch.cat(test_pred, dim=0)
all_data = torch.cat([train_data, valid_data, unlabeled_train_data], dim=0)
all_target = torch.cat([labeled_train, labeled_val, unlabeled_train_label], dim=0)
construct_graph_label = torch.cat([labeled_train, labeled_val, test_pred], dim=0)
all_data = all_data.cpu().numpy()
all_target = all_target.cpu().numpy()
construct_graph_label = construct_graph_label.cpu().numpy()
N = all_data.shape[0]
num_labeled = train_data.shape[0]
num_valid = valid_data.shape[0]

col_list = []
row_list = []
correct_connect_sum = 0
connect_sum = 0
for i in range(N):
    label_i = construct_graph_label[i]
    same_label_ind = np.arange(N)[construct_graph_label==label_i]
    col_list += same_label_ind.tolist()
    row_list += [i] * len(same_label_ind)

    connected_labels = construct_graph_label[same_label_ind]
    connected_gt_labels = all_target[same_label_ind]
    correct_connect_sum += np.sum(connected_gt_labels == connected_labels)
    connect_sum += len(same_label_ind)

print("The ratio of correctly connected nodes is ", correct_connect_sum / connect_sum)

dist_list = [1] * len(col_list)
W = scipy.sparse.coo_matrix((dist_list, (row_list, col_list)), shape=(N, N))
# No self-connections.
W.setdiag(0)
# Non-directed graph.
bigger = W.T > W
W = W - W.multiply(bigger) + W.T.multiply(bigger)
assert W.nnz % 2 == 0
assert np.abs(W - W.T).mean() < 1e-10

data_path = f'./data/{opt.dataset}/'
if not os.path.exists(os.path.dirname(data_path)):
    os.mkdir(os.path.dirname(data_path))
np.save(data_path + 'all_input.npy', all_data)
np.save(data_path + 'all_target.npy', all_target)
np.save(data_path + 'train_ind.npy', np.arange(num_labeled))
np.save(data_path + 'val_ind.npy', np.arange(num_labeled, num_valid+num_labeled))
np.save(data_path + 'test_ind.npy', np.arange(num_valid+num_labeled, N))
scipy.sparse.save_npz(data_path + 'adj.npz', W)

