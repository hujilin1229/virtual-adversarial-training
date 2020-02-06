import argparse
from torchvision import datasets, transforms
import torch.optim as optim
from model import *
from utils import *
import os
import data
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
np.random.seed(42)

batch_size = 32
eval_batch_size = 50
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
parser.add_argument('--save_data', default=False, action='store_true')

parser.add_argument('--num_train', type=int, default=100)
parser.add_argument('--num_val', type=int, default=100)
parser.add_argument('--num_total', type=int, default=1000)
parser.add_argument('--num_class', type=int, default=100)

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

def tocuda(x):
    if opt.use_cuda:
        return x.cuda()
    return x

def construct_adjacent_matrix(K, feature_maps, N):

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

    return W

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
    return torch.eq(idx, y).float().mean(), idx

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

# elif opt.dataset == 'imagenet12':
#     num_labeled = 4000
#     train_loader = torch.utils.data.DataLoader(
#         datasets.ImageNet(root=opt.dataroot, train=True, download=True,
#                       transform=transforms.Compose([
#                           transforms.ToTensor(),
#                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                       ])),
#         batch_size=batch_size, shuffle=True)
#
#     test_loader = torch.utils.data.DataLoader(
#         datasets.ImageNet(root=opt.dataroot, train=False, download=True,
#                       transform=transforms.Compose([
#                           transforms.ToTensor(),
#                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                       ])),
#         batch_size=eval_batch_size, shuffle=True)
# elif opt.dataset == 'stl10':
#     # num_labeled = 4000
#     # splits = ('train', 'train+unlabeled', 'unlabeled', 'test')
#     train_loader = torch.utils.data.DataLoader(
#         datasets.STL10(root=opt.dataroot, split='train+unlabeled', download=True,
#                       transform=transforms.Compose([
#                           transforms.ToTensor(),
#                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                       ])),
#         batch_size=batch_size, shuffle=True)
#
#     test_loader = torch.utils.data.DataLoader(
#         datasets.STL10(root=opt.dataroot, split='test', download=True,
#                       transform=transforms.Compose([
#                           transforms.ToTensor(),
#                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                       ])),
#         batch_size=eval_batch_size, shuffle=True)
# elif opt.dataset == 'tiny-imagenet-200' or opt.dataset == 'mini-imagenet':
#     if opt.dataset == 'tiny-imagenet-200':
#         folder = 'train'
#     else:
#         folder = 'all'
#     train_data = data.prepare_imagenet(data_dir=opt.dataroot, dataset=opt.dataset, folder=folder)
#     # print('Preparing data loaders ...')
#     kwargs = {} if opt.use_cuda else {'num_workers': 1, 'pin_memory': True}
#
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
#                                                 shuffle=True, **kwargs)

    # test_loader = torch.utils.data.DataLoader(val_data, batch_size=eval_batch_size,
    #                                               shuffle=True, **kwargs)

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
unique_labels = np.sort(unique_labels)
unique_labels = unique_labels[:opt.num_class]
print("Unique Labels: ", unique_labels, flush=True)
print("Number of labels: ", len(unique_labels))
n_class = len(unique_labels)
# parser.add_argument('--num_train', type=int, default=100)
# parser.add_argument('--num_val', type=int, default=100)
# parser.add_argument('--num_total', type=int, default=1000)

nSamples_per_class_train = opt.num_train
nSamples_per_class_val = opt.num_val


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

    unlabeled_train_data.append(current_label_X[nSamples_per_class_train + nSamples_per_class_val:opt.num_total])
    unlabeled_train_label.append(current_label_y[nSamples_per_class_train + nSamples_per_class_val:opt.num_total])

nSamples_unlabel = sum([unlabel_data.shape[0] for unlabel_data in unlabeled_train_data])

train_data = torch.cat(select_train_data, dim=0)
train_target = torch.cat(select_train_label, dim=0)
valid_data = torch.cat(select_val_data, dim=0)
valid_target = torch.cat(select_val_label, dim=0)
test_data = torch.cat(unlabeled_train_data, dim=0)
test_target = torch.cat(unlabeled_train_label, dim=0)
# random shuffle the data
train_random_ind = np.arange(nSamples_per_class_train * n_class)
val_random_ind = np.arange(nSamples_per_class_val * n_class)
test_random_ind = np.arange(nSamples_unlabel)
np.random.shuffle(train_random_ind)
np.random.shuffle(val_random_ind)
# np.random.shuffle(test_random_ind)

train_data = train_data[train_random_ind]
train_target = train_target[train_random_ind]

valid_data = valid_data[val_random_ind]
valid_target = valid_target[val_random_ind]

unlabeled_train_data = test_data[test_random_ind]
unlabeled_train_label = test_target[test_random_ind]

all_data = torch.cat([train_data, valid_data, test_data], dim=0)
print("All Data Shape is ", all_data.shape)
print("Target Shape is ", train_target.shape)
# valid_data, train_data = train_data[:num_valid, ], train_data[num_valid:, ]
# valid_target, train_target = train_target[:num_valid], train_target[num_valid:, ]
# 
# labeled_train, labeled_target = train_data[:num_labeled, ], train_target[:num_labeled, ]
# unlabeled_train = train_data[num_labeled:, ]

in_channels = 3
if opt.dataset == 'mnist':
    in_channels = 1
model = tocuda(VAT(opt.top_bn, in_channels, n_class=n_class))
model.apply(weights_init)
optimizer = optim.Adam(model.parameters(), lr=lr)
max_val_acc = 0.0
patience = 0
max_patience = 10

path_best_model = f'./saved_models/{opt.dataset}/test_model_{n_class}'
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
            batch_indices = torch.LongTensor(np.random.choice(train_target.size()[0], batch_size, replace=False))
            x = train_data[batch_indices]
            y = train_target[batch_indices]
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
                target = valid_target[i:i + eval_batch_size]
                acc, _ = eval(model.eval(), Variable(tocuda(data)), Variable(tocuda(target)))
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
    acc, pred = eval(model.eval(), Variable(tocuda(data)), Variable(tocuda(target)))

    test_pred.append(pred)
    test_accuracy += eval_batch_size * acc
    counter += eval_batch_size

print("Full test accuracy :", test_accuracy.item()/counter, flush=True)

if opt.save_data:
    test_accuracy = 0.0
    counter = 0
    K = 10
    model.eval()

    feature_maps = []
    # evaluate
    with torch.no_grad():
        for i in range(0, all_data.shape[0], eval_batch_size):
            data = all_data[i:i + eval_batch_size]
            # target = unlabeled_train_label[i:i+eval_batch_size]
            pred_featmaps = eval_featmap(model, Variable(tocuda(data)))
            # print(i, pred_featmaps.shape)
            feature_maps.append(pred_featmaps.cpu())

    feature_maps = torch.cat(feature_maps, dim=0)
    N = feature_maps.shape[0]
    feature_maps = feature_maps.view(N, -1)
    all_target = torch.cat([train_target, valid_target, test_target], dim=0)
    feature_maps = feature_maps.numpy()
    all_target = all_target.cpu().numpy()

    # W = construct_adjacent_matrix(K, feature_maps, N)

    # # Construct Adjacent Matrix
    # nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(feature_maps)
    # distances, indices = nbrs.kneighbors(feature_maps)
    #
    # col_indices = np.reshape(indices, (-1))
    # row_indices = np.repeat(np.arange(N), K)
    #
    # dist_list = [1] * len(col_indices)
    # W = scipy.sparse.coo_matrix((dist_list, (row_indices, col_indices)), shape=(N, N))
    # # No self-connections.
    # W.setdiag(0)
    # # Non-directed graph.
    # bigger = W.T > W
    # W = W - W.multiply(bigger) + W.T.multiply(bigger)
    # assert W.nnz % 2 == 0
    # assert np.abs(W - W.T).mean() < 1e-10
    #

    data_path = f'./data/vat_feat_nn_split_all/{opt.dataset}/'
    # data_path = f'../data/vat_feat_nn/{opt.dataset}_{n_class}/'

    print("Total number of data is ", N)

    data_path_P = Path(data_path)
    data_path_P.mkdir(parents=True, exist_ok=True)
    np.save(data_path + 'all_input.npy', all_data)
    np.save(data_path + 'all_featmap.npy', feature_maps)
    np.save(data_path + 'all_target.npy', all_target)
    np.save(data_path + 'train_ind.npy', np.arange(num_labeled))
    np.save(data_path + 'val_ind.npy', np.arange(num_labeled, num_valid + num_labeled))
    np.save(data_path + 'test_ind.npy', np.arange(num_valid + num_labeled, N))
    # scipy.sparse.save_npz(data_path + 'adj.npz', W)


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


    # # write the resulted data
    # # compose all the data and label together
    # test_pred = torch.cat(test_pred, dim=0).cpu()
    # all_data = torch.cat([train_data, valid_data, unlabeled_train_data], dim=0)
    # all_target = torch.cat([train_target, valid_target, unlabeled_train_label], dim=0)
    # construct_graph_label = torch.cat([train_target, valid_target, test_pred], dim=0)
    # all_data = all_data.cpu().numpy()
    # all_target = all_target.cpu().numpy()
    # construct_graph_label = construct_graph_label.cpu().numpy()
    # N = all_data.shape[0]
    # num_labeled = train_data.shape[0]
    # num_valid = valid_data.shape[0]
    #
    # col_list = []
    # row_list = []
    # correct_connect_sum = 0
    # connect_sum = 0
    # K = 10
    # for i in range(N):
    #     label_i = construct_graph_label[i]
    #     same_label_ind = np.arange(N)[construct_graph_label==label_i]
    #     same_label_ind = np.random.choice(same_label_ind, K, replace=False)
    #     # print(i, same_label_ind)
    #
    #     col_list += same_label_ind.tolist()
    #     row_list += [i] * len(same_label_ind)
    #     connected_labels = construct_graph_label[same_label_ind]
    #     connected_gt_labels = all_target[same_label_ind]
    #     correct_connect_sum += np.sum(connected_gt_labels == connected_labels)
    #     connect_sum += len(same_label_ind)
    #
    # print("The ratio of correctly connected nodes is ", correct_connect_sum / connect_sum)
    #
    # dist_list = [1] * len(col_list)
    # W = scipy.sparse.coo_matrix((dist_list, (row_list, col_list)), shape=(N, N))
    # # No self-connections.
    # W.setdiag(0)
    # # Non-directed graph.
    # bigger = W.T > W
    # W = W - W.multiply(bigger) + W.T.multiply(bigger)
    # assert W.nnz % 2 == 0
    # assert np.abs(W - W.T).mean() < 1e-10
    #
    # data_path = f'./data/{opt.dataset}_{n_class}/'
    # if not os.path.exists(os.path.dirname(data_path)):
    #     os.mkdir(os.path.dirname(data_path))
    # np.save(data_path + 'all_input.npy', all_data)
    # np.save(data_path + 'all_target.npy', all_target)
    # np.save(data_path + 'train_ind.npy', np.arange(num_labeled))
    # np.save(data_path + 'val_ind.npy', np.arange(num_labeled, num_valid+num_labeled))
    # np.save(data_path + 'test_ind.npy', np.arange(num_valid+num_labeled, N))
    # scipy.sparse.save_npz(data_path + 'adj.npz', W)

