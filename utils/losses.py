"""
*Preliminary* pytorch implementation.

Losses for VoxelMorph
"""

import math
import torch
import numpy as np
from utils.config import args
import torch.nn.functional as F

import SimpleITK as sitk
from torch.distributions.dirichlet import Dirichlet as Dir
from torch import lgamma, digamma
from utils.kl_inv import klInvFunction


def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

    if (penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0


def mse_loss(x, y):
    return torch.mean((x - y) ** 2)



def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)



def ncc_loss(I, J, win=None):

    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    if win is None:
        win = [9] * ndims
    sum_filt = torch.ones([1, 1, *win]).to("cuda:{}".format(args.gpu))
    pad_no = math.floor(win[0] / 2)
    stride = [1] * ndims
    padding = [pad_no] * ndims
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)
    cc = cross * cross / (I_var * J_var + 1e-5)
    return -1 * torch.mean(cc)


def compute_local_sums(I, J, filt, stride, padding, win):
    I2, J2, IJ = I * I, J * J, I * J
    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)
    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    return I_var, J_var, cross



def cc_loss(x, y):
    dim = [2, 3, 4]
    mean_x = torch.mean(x, dim, keepdim=True)
    mean_y = torch.mean(y, dim, keepdim=True)
    mean_x2 = torch.mean(x ** 2, dim, keepdim=True)
    mean_y2 = torch.mean(y ** 2, dim, keepdim=True)
    stddev_x = torch.sum(torch.sqrt(mean_x2 - mean_x ** 2), dim, keepdim=True)
    stddev_y = torch.sum(torch.sqrt(mean_y2 - mean_y ** 2), dim, keepdim=True)
    return -torch.mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))



def Get_Ja(flow):
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3
    '''
    D_y = (flow[:, 1:, :-1, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_x = (flow[:, :-1, 1:, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_z = (flow[:, :-1, :-1, 1:, :] - flow[:, :-1, :-1, :-1, :])
    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])
    return D1 - D2 + D3



def NJ_loss(ypred):
    '''
    Penalizing locations where Jacobian has negative determinants
    '''
    Neg_Jac = 0.5 * (torch.abs(Get_Ja(ypred)) - Get_Ja(ypred))
    return torch.sum(Neg_Jac)



def mutual_information(img1, img2, num_bins=256, eps=1e-10):

    B, C, D, H, W = img1.shape

    img1 = img1.reshape(img1.size(0), -1)
    img2 = img2.reshape(img2.size(0), -1)


    img1_scaled = ((img1 - img1.min(dim=1, keepdim=True)[0]) /
                   (img1.max(dim=1, keepdim=True)[0] - img1.min(dim=1, keepdim=True)[0] + eps) * (num_bins - 1)).long()

    img2_scaled = ((img2 - img2.min(dim=1, keepdim=True)[0]) /
                   (img2.max(dim=1, keepdim=True)[0] - img2.min(dim=1, keepdim=True)[0] + eps) * (num_bins - 1)).long()

    joint_hist = torch.zeros((B, num_bins, num_bins), dtype=torch.float32, device=img1.device)


    for b in range(B):
        indices = img1_scaled[b] * num_bins + img2_scaled[b]
        joint_hist[b].view(-1).scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float32))


    p_x = joint_hist.sum(dim=2) 
    p_y = joint_hist.sum(dim=1)  


    p_xy = joint_hist / joint_hist.sum(dim=(1, 2), keepdim=True) + eps


    p_x = p_x / p_x.sum(dim=1, keepdim=True) + eps
    p_y = p_y / p_y.sum(dim=1, keepdim=True) + eps

    mi = (p_xy * (torch.log(p_xy) - torch.log(p_x.unsqueeze(2)) - torch.log(p_y.unsqueeze(1)))).sum(dim=(1, 2))

    return mi


def mask_computer(outputs, output_true):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    mask = []
    MI = []
    for i in range(len(outputs)):
        MI.append(mutual_information(outputs[i], output_true))
    MI = torch.tensor(MI).to(device)
    for i in range(len(MI)):
        if MI[i] < torch.mean(MI):
            mask.append(1)
        else:
            mask.append(0)

    return torch.tensor(mask).bool()


def KL_loss(model):
    exp_post = torch.exp(model.post)
    res = lgamma(exp_post.sum()) - lgamma(exp_post).sum()
    res -= lgamma(model.prior.sum()) - lgamma(model.prior).sum()
    res += torch.sum((exp_post - model.prior) * (digamma(exp_post) - digamma(exp_post.sum())))

    return res


def sigmoid_loss(mask, thetas, c=12):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    mask = mask.to(device)
    w_theta = torch.stack([torch.where(mask, t, torch.zeros_like(t)).sum(0) for t in thetas])
    return torch.sigmoid(c * (w_theta - 0.5))


def risk_loss(mask, model, mc_draws=1):
    thetas = Dir(torch.exp(model.post)).rsample(torch.Size([mc_draws]))
    return sigmoid_loss(mask, thetas)


def seeger_bound_computer(mask, model, delta=0.05, mc_draws=1):
    kl = KL_loss(model)
    const = np.log(2 * (10 ** 0.5) / delta)
    risk = risk_loss(mask, model, mc_draws=mc_draws).mean(0)
    bound = klInvFunction.apply(risk, (kl + const) / 10)

    return bound


def seeger_bound_loss(outputs, output_true, model, mc_draws=10, delta=0.05):
    mask = mask_computer(outputs, output_true)
    return seeger_bound_computer(mask, model, delta=delta, mc_draws=mc_draws)


def negative_jacobin(flow):

    w, h, l, c = np.shape(flow)
    flow_image = sitk.GetImageFromArray(flow.astype('float64'), isVector=True)
    determinant = sitk.DisplacementFieldJacobianDeterminant(flow_image)
    neg_jacobin = (sitk.GetArrayFromImage(determinant)) < 0
    cnt = np.sum(neg_jacobin)
    norm_cnt = cnt / (h * w * l)
    return norm_cnt * 100
