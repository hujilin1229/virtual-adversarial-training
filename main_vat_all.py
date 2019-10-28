import argparse
from torchvision import datasets, transforms
import torch.optim as optim
from model import *
from utils import *
import os
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
np.random.seed(42)

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

parser.add_argument('--num_labeled', type=int, default=1000, help='Number of Labeled data.')
parser.add_argument('--num_valid', type=int, default=1000, help='Number of Validation data.')

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

def eval_featmap(model, x):

    y_pred = model(x, featmap_only=True)
    return y_pred

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

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=opt.dataroot, train=False, download=True,
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

valid_data, train_data = train_data[:opt.num_valid, ], train_data[opt.num_valid:, ]
valid_target, train_target = train_target[:opt.num_valid], train_target[opt.num_valid:, ]

labeled_train, labeled_target = train_data[:opt.num_labeled, ], train_target[:opt.num_labeled, ]
unlabeled_train = train_data[opt.num_labeled:, ]

max_val_acc = 0.0
patience = 0
max_patience = 50
in_channels = 3
if opt.dataset == 'mnist':
    in_channels = 1

model = tocuda(VAT(opt.top_bn, in_channels))
model.apply(weights_init)
optimizer = optim.Adam(model.parameters(), lr=lr)

path_best_model = f'./saved_models/{opt.dataset}/test_model_{opt.method}_all_{opt.num_labeled}'
if not os.path.exists(os.path.dirname(path_best_model)):
    os.mkdir(os.path.dirname(path_best_model))

if os.path.exists(path_best_model):
    # original saved file with DataParallel
    state_dict = torch.load(path_best_model)
    model.load_state_dict(state_dict)

if opt.train:
    # train the network
    for epoch in range(opt.num_epochs):
        if epoch > opt.epoch_decay_start:
            decayed_lr = (opt.num_epochs - epoch) * lr / (opt.num_epochs - opt.epoch_decay_start)
            optimizer.lr = decayed_lr
            optimizer.betas = (0.5, 0.999)

        for i in range(num_iter_per_epoch):

            batch_indices = torch.LongTensor(np.random.choice(labeled_train.size()[0], batch_size, replace=False))
            x = labeled_train[batch_indices]
            y = labeled_target[batch_indices]
            batch_indices_unlabeled = torch.LongTensor(np.random.choice(unlabeled_train.size()[0], unlabeled_batch_size, replace=False))
            ul_x = unlabeled_train[batch_indices_unlabeled]

            v_loss, ce_loss = train(model.train(), Variable(tocuda(x)), Variable(tocuda(y)), Variable(tocuda(ul_x)),
                                    optimizer)

            if i % 100 == 0:
                print("Epoch :", epoch, "Iter :", i, "VAT Loss :", v_loss.item(), "CE Loss :", ce_loss.item(), flush=True)

        if epoch % eval_freq == 0 or epoch + 1 == opt.num_epochs:

            batch_indices = torch.LongTensor(np.random.choice(labeled_train.size()[0], batch_size, replace=False))
            x = labeled_train[batch_indices]
            y = labeled_target[batch_indices]
            train_accuracy = eval(model.eval(), Variable(tocuda(x)), Variable(tocuda(y)))
            print("Train accuracy :", train_accuracy.item(), flush=True)

            val_accuracy = 0.0
            counter = 0

            for (data, target) in test_loader:
                test_accuracy = eval(model.eval(), Variable(tocuda(data)), Variable(tocuda(target)))
                batch_i = data.shape[0]
                val_accuracy += test_accuracy  * batch_i
                counter += batch_i

            avg_val_accuracy = val_accuracy / counter
            print("Test accuracy :", avg_val_accuracy, flush=True)

            if max_val_acc < avg_val_accuracy:
                max_val_acc = avg_val_accuracy
                patience = 0
                torch.save(model.state_dict(), path_best_model)
            else:
                patience += 0
                if patience >= max_patience:
                    break

if os.path.exists(path_best_model):
    # original saved file with DataParallel
    state_dict = torch.load(path_best_model)
    model.load_state_dict(state_dict)

test_accuracy = 0.0
counter = 0
for (data, target) in test_loader:
    n = data.size()[0]
    acc = eval(model.eval(), Variable(tocuda(data)), Variable(tocuda(target)))
    test_accuracy += n*acc
    counter += n

print("Full test accuracy :", test_accuracy.item()/counter, flush=True)


# obtain the feature map
valid_data, train_data = train_data[:opt.num_valid, ], train_data[opt.num_valid:, ]
valid_target, train_target = train_target[:opt.num_valid], train_target[opt.num_valid:, ]

labeled_train, labeled_target = train_data[:opt.num_labeled, ], train_target[:opt.num_labeled, ]
unlabeled_train, unlabeled_target = train_data[opt.num_labeled:, ], train_target[opt.num_labeled:, ]

all_training_data = torch.cat([labeled_train, valid_data, unlabeled_train], dim=0)
all_training_label = torch.cat([labeled_target, unlabeled_target, train_target], dim=0)

print("Total Number of Training Data is ", all_training_data.shape[0], flush=True)
test_accuracy = 0.0
counter = 0
K = 10
model.eval()
feature_maps = []
targets = []
all_data = [all_training_data]
with torch.no_grad():
    for i in range(0, all_training_data.shape[0], eval_batch_size):
        data = all_training_data[i:i+eval_batch_size]
        target = all_training_label[i:i+eval_batch_size]
        # target = unlabeled_train_label[i:i+eval_batch_size]
        pred_featmaps = eval_featmap(model, Variable(tocuda(data)))
        # print(i, pred_featmaps.shape)
        feature_maps.append(pred_featmaps.cpu())
        targets.append(target)

    for (data, target) in test_loader:
        all_data.append(data)
        tmp_accuracy = eval(model.eval(), Variable(tocuda(data)), Variable(tocuda(target)))
        batch_i = data.shape[0]
        test_accuracy += tmp_accuracy * batch_i
        counter += batch_i

        pred_featmaps = eval_featmap(model, Variable(tocuda(data)))
        # print(i, pred_featmaps.shape)
        feature_maps.append(pred_featmaps.cpu())
        targets.append(target)


all_data = torch.cat(all_data, dim=0)
feature_maps = torch.cat(feature_maps, dim=0)
all_target = torch.cat(targets, dim=0)
N = feature_maps.shape[0]
feature_maps = feature_maps.view(N, -1)
# all_target = torch.cat([train_target, valid_target, test_target], dim=0)
feature_maps = feature_maps.numpy()
all_target = all_target.cpu().numpy()
all_data = all_data.numpy()

nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(feature_maps)
distances, indices = nbrs.kneighbors(feature_maps)

col_indices = np.reshape(indices, (-1))
row_indices = np.repeat(np.arange(N), K)

dist_list = [1] * len(col_indices)
W = scipy.sparse.coo_matrix((dist_list, (row_indices, col_indices)), shape=(N, N))
# No self-connections.
W.setdiag(0)
# Non-directed graph.
bigger = W.T > W
W = W - W.multiply(bigger) + W.T.multiply(bigger)
assert W.nnz % 2 == 0
assert np.abs(W - W.T).mean() < 1e-10

data_path = f'../data/vat_feat_nn_all/{opt.dataset}/'
data_path_P = Path(data_path)
data_path_P.mkdir(parents=True, exist_ok=True)
# if not os.path.exists(os.path.dirname(data_path)):
#     os.mkdir(os.path.dirname(data_path))
np.save(data_path + 'all_input.npy', all_data)
np.save(data_path + 'all_featmap.npy', feature_maps)
np.save(data_path + 'all_target.npy', all_target)
np.save(data_path + 'train_ind.npy', np.arange(num_labeled))
np.save(data_path + 'val_ind.npy', np.arange(num_labeled, num_valid+num_labeled))
np.save(data_path + 'test_ind.npy', np.arange(num_valid+num_labeled, N))
scipy.sparse.save_npz(data_path + 'adj.npz', W)

correct_connect_sum = 0
connect_sum = 0
contain_correct_label_num = 0
for i in range(N):
    label_i = all_target[i]
    nn_labels = all_target[indices[i]]
    correct_connect_sum += np.sum(nn_labels == label_i)
    if np.sum(nn_labels == label_i) > 0:
        contain_correct_label_num += 1

    connect_sum += len(indices[i])

print("The ratio of correctly connected nodes is ", correct_connect_sum / connect_sum)
print("Num of contain correct label is ", contain_correct_label_num)