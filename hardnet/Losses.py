import torch
import torch.nn as nn
import sys

def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and positive descriptors
    calculate euclidean distance matrix"""
    
    # here anchor*anchor is equal torch.mul(anchor, anchor)
    # the element-wise value multiplication is returned
    d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
    d2_sq = torch.sum(positive * positive, dim=1).unsqueeze(-1)

    eps = 1e-6
    # tensor.repeat(): repeat at each dims, and dims from right to left
    return torch.sqrt((d1_sq.repeat(1, anchor.size(0)) + torch.t(d2_sq.repeat(1, positive.size(0)))
                      - 2.0 * torch.bmm(anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0))
                      +eps)

def distance_vectors_pairwise(anchor, positive, negative):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix
    The distance metric is Euclidean distance L2-dist
    """

    a_sq = torch.sum(anchor * anchor, dim=1)
    p_sq = torch.sum(positive * positive, dim=1)
    n_sq = torch.sum(negative * negative, dim=1)

    eps = 1e-8
    d_a_p = torch.sqrt(a_sq + p_sq - 2*torch.sum(anchor * positive, dim = 1) + eps)
    d_a_n = torch.sqrt(a_sq + n_sq - 2*torch.sum(anchor * negative, dim = 1) + eps)
    d_p_n = torch.sqrt(p_sq + n_sq - 2*torch.sum(positive * negative, dim = 1) + eps)
    return d_a_p, d_a_n, d_p_n

# random triplets sampling loss fucntion
def loss_random_sampling(anchor, positive, negative, anchor_swap = False, margin = 1.0, loss_type = "triplet_margin"):
    """Loss with random sampling (no hard in batch).
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.size() == negative.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    (pos, d_a_n, d_p_n) = distance_vectors_pairwise(anchor, positive, negative)

    # distance based anchor, if anchor swap, get the min(anchor, positive)
    if anchor_swap:
       min_neg = torch.min(d_a_n, d_p_n)
    else:
       min_neg = d_a_n

    if loss_type == "triplet_margin":
        # the func is (m + D_p - D_n)
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
    elif loss_type == 'softmax':
        # here the output is 2-class log-softmax loss(1/0) from L2Net
        exp_pos = torch.exp(2.0 - pos)
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps
        loss = - torch.log( exp_pos / exp_den )
    elif loss_type == 'contrastive':
        loss = torch.clamp(margin - min_neg, min=0.0) + pos
    else: 
        print ('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss

def loss_L2Net(anchor, positive, anchor_swap = False,  margin = 1.0, loss_type = "triplet_margin"):
    """L2Net losses: using whole batch as negatives, not only hardest.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    # batch-size distance matrix
    dist_matrix = distance_matrix_vector(anchor, positive)

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix) # positive sample distance

    #L2Net loss=E1+E2+E3
    if loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos1)
        exp_den = torch.sum(torch.exp(2.0 - dist_matrix), 1) + eps
        loss = -torch.log( exp_pos / exp_den )

        # anchor_swap=True for L2Net, because the element in the diagnal should be
        # the min in each row/column respectively
        if anchor_swap:
            exp_den1 = torch.sum(torch.exp(2.0 - dist_matrix), 0) + eps
            loss += -torch.log( exp_pos / exp_den1 )
    else: 
        print ('Only softmax loss works with L2Net sampling')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss

def loss_HardNet(anchor, positive, anchor_swap = False, anchor_ave = False,
        margin = 1.0, batch_reduce = 'min', loss_type = "triplet_margin"):
    """HardNet margin loss - calculates loss based on distance matrix based on positive distance and closest negative distance.
    """

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive) +eps # D = A_t*P
    eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix+eye*10

    # get all the indices which value<0.008
    mask = (dist_without_min_on_diag.ge(0.008).float()-1.0)*(-1)
    mask = mask.type_as(dist_without_min_on_diag)*10
    dist_without_min_on_diag = dist_without_min_on_diag+mask

    # sampling strategy of Hardest in batch
    if batch_reduce == 'min':
        # mining the value < 0.008(without mining on the diagonal)
        min_neg = torch.min(dist_without_min_on_diag, 1)[0]
        if anchor_swap:
            min_neg2 = torch.min(dist_without_min_on_diag, 0)[0]
            min_neg = torch.min(min_neg, min_neg2)

            """ print for debug
            dist_matrix_a = distance_matrix_vector(anchor, anchor)+ eps
            dist_matrix_p = distance_matrix_vector(positive,positive)+eps
            dist_without_min_on_diag_a = dist_matrix_a+eye*10
            dist_without_min_on_diag_p = dist_matrix_p+eye*10
            min_neg_a = torch.min(dist_without_min_on_diag_a,1)[0]
            min_neg_p = torch.t(torch.min(dist_without_min_on_diag_p,0)[0])
            min_neg_3 = torch.min(min_neg_p,min_neg_a)
            min_neg = torch.min(min_neg,min_neg_3)
            print (min_neg_a)
            print (min_neg_p)
            print (min_neg_3)
            print (min_neg)
            """
        min_neg = min_neg
        pos = pos1
    elif batch_reduce == 'average':
        # why repeat pos value here?
        pos = pos1.repeat(anchor.size(0)).view(-1,1).squeeze(0)
        min_neg = dist_without_min_on_diag.view(-1,1)
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).contiguous().view(-1,1)
            # compare anchor-pos vs. pos-anchor value
            min_neg = torch.min(min_neg, min_neg2)
        min_neg = min_neg.squeeze(0)
    elif batch_reduce == 'random':
        idxs = torch.autograd.Variable(torch.randperm(anchor.size()[0]).long()).cuda()
        min_neg = dist_without_min_on_diag.gather(1, idxs.view(-1,1))# dim=1, col-idx
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).gather(1,idxs.view(-1,1)) 
            min_neg = torch.min(min_neg, min_neg2)
        min_neg = torch.t(min_neg).squeeze(0)
        pos = pos1
    else: 
        print ('Unknown batch reduce mode. Try min, average or random')
        sys.exit(1)

    # calculate the loss depends on the loss_type
    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
    elif loss_type == 'softmax':
        # Softmin used here: (-x) instead of x as the input
        # log-likelihood cost function instead of cross-entropy cost function
        exp_pos = torch.exp(2.0 - pos)
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps
        loss = - torch.log( exp_pos / exp_den )
    elif loss_type == 'contrastive':
        loss = torch.clamp(margin - min_neg, min=0.0) + pos
    else: 
        print ('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)

    loss = torch.mean(loss)
    return loss

# global orthogonal regulariztion for the anchor and the negative
def global_orthogonal_regularization(anchor, negative):

    # get the (1024, 1) dim distance
    neg_dis = torch.sum(torch.mul(anchor, negative), 1)
    fea_dim = anchor.size(1)
    gor = torch.pow(torch.mean(neg_dis), 2) + torch.clamp(torch.mean(torch.pow(neg_dis,2))-1.0/fea_dim, min=0.0)
    
    return gor


# E3 Loss for L2Net intermediate feature map regularization
def intermediate_featuremap_regularization(anchor, positive, anchor_swap = False):
    # anchor/positive size:(batch_size, width, height)
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    eps = 1e-8
    anchor = anchor.view(-1, anchor.size(1)*anchor.size(1))
    positive = positive.view(-1, positive.size(1) * positive.size(1))

    distance_batch = torch.bmm(anchor, positive).squeeze(0)
    pos = torch.diag(distance_batch)
    exp_pos = torch.exp(pos)
    exp_den = torch.sum(torch.exp(distance_batch), 1) + eps
    loss = torch.log(exp_pos / exp_den)
    if anchor_swap:
        exp_den1 = torch.sum(torch.exp(distance_batch), 0) + eps
        loss += torch.log(exp_pos / exp_den1)

    loss = torch.mean(loss)
    return loss

# CPR: correlation penalty loss for descriptor in L2Net(E2_loss)
class CorrelationPenaltyLoss(nn.Module):
    def __init__(self):
        super(CorrelationPenaltyLoss, self).__init__()

    def forward(self, input):
        # get the mean value of all the batch-size data
        mean1 = torch.mean(input, dim=0)
        zeroed = torch.add(input, -mean1.expand_as(input))

        # get the feat-dim=128 (128, 128) correlation matrix
        cor_mat = torch.bmm(torch.t(zeroed).unsqueeze(0), zeroed.unsqueeze(0)).squeeze(0)
        d = torch.diag(torch.diag(cor_mat)) #get the diag matrix
        no_diag = cor_mat - d
        d_sq = no_diag * no_diag
        return torch.sqrt(d_sq.sum())/input.size(0)


# Loss.py end of here