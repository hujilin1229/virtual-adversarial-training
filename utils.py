import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import scipy
import sklearn.metrics
import sklearn.neighbors
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial.distance

def kl_div_with_logit(q_logit, p_logit):

    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = ( q *logq).sum(dim=1).mean(dim=0)
    qlogp = ( q *logp).sum(dim=1).mean(dim=0)

    return qlogq - qlogp


def _l2_normalize(d):

    d = d.numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)


def vat_loss(model, ul_x, ul_y, xi=1e-6, eps=2.5, num_iters=1):

    # find r_adv

    d = torch.Tensor(ul_x.size()).normal_()
    for i in range(num_iters):
        d = xi *_l2_normalize(d)
        d = Variable(d.cuda(), requires_grad=True)
        y_hat = model(ul_x + d)
        delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
        delta_kl.backward()

        d = d.grad.data.clone().cpu()
        model.zero_grad()

    d = _l2_normalize(d)
    d = Variable(d.cuda())
    r_adv = eps *d
    # compute lds
    y_hat = model(ul_x + r_adv.detach())
    delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
    return delta_kl


def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p*F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''

    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)

def adjacency(edge_pairs, select_position, idx_dict, sigma2=1.0, directed=False):
    """
    Return the adjacency matrix of a kNN graph.
    """

    num_rows, num_cols = select_position.shape
    N, k = edge_pairs.shape
    row_list = []
    col_list = []
    dist_list = []
    for i in range(N):
        node_i = idx_dict[i]
        i_row, i_col = node_i // num_cols, node_i % num_cols
        for j in edge_pairs[i]:
            if j == -1:
                continue
            row_list.append(i)
            col_list.append(j)
            node_j = idx_dict[j]
            j_row, j_col = node_j // num_cols, node_j % num_cols
            dist_i_j = 1.0 - np.abs(select_position[i_row, i_col] - select_position[j_row, j_col]) / \
                       max(select_position[i_row, i_col], select_position[j_row, j_col])

            # dist_i_j = 1.0
            dist_list.append(dist_i_j)

    W = scipy.sparse.coo_matrix((dist_list, (row_list, col_list)), shape=(N, N))

    # No self-connections.
    W.setdiag(0)

    if not directed:
        # Non-directed graph.
        bigger = W.T > W
        W = W - W.multiply(bigger) + W.T.multiply(bigger)
        assert W.nnz % 2 == 0
        assert np.abs(W - W.T).mean() < 1e-10

    # assert type(W) is scipy.sparse.csr.csr_matrix
    return W

def distance_sklearn_metrics(z, k=4, metric='euclidean'):
    """Compute exact pairwise distances."""
    d = sklearn.metrics.pairwise.pairwise_distances(
            z, metric=metric, n_jobs=-2)
    # k-NN graph.
    idx = np.argsort(d)[:, 1:k+1]
    d.sort()
    d = d[:, 1:k+1]
    # set idx of non-neighbour nodes to -1
    non_neighbour = d > np.sqrt(2)
    idx[non_neighbour] = -1

    return d, idx

def distance_scipy_spatial(z, k=4, metric='euclidean'):
    """Compute exact pairwise distances."""
    d = scipy.spatial.distance.pdist(z, metric)
    d = scipy.spatial.distance.squareform(d)
    # k-NN graph.
    idx = np.argsort(d)[:, 1:k+1]
    d.sort()
    d = d[:, 1:k+1]

    return d, idx