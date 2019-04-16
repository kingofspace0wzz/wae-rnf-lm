import torch
import torch.nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import numpy as np

EPS = 1e-08

def recon_loss(outputs, targets, pad_id, id='entropy'):
    if id == 'entropy':
        loss = F.cross_entropy(outputs.view(-1, outputs.size(2)), targets.view(-1),
                               reduction='sum', ignore_index=pad_id)
    elif id == 'l2':
        loss =F.mse_loss(outputs.view(-1, outputs.size(2)), targets.view(-1),
                               reduction='sum')
        loss = 0.2 * (loss.sqrt() + EPS).mean()
    elif id == 'l2sq':
        loss = 0.05 * F.mse_loss(outputs, targets,
                               reduction='mean')
    elif id == 'l1':
        loss = 0.02 * F.l1_loss(outputs.view(-1, outputs.size(2)), targets.view(-1),
                               reduction='mean')
    return loss

def total_kld(q_z, p_z):
    return torch.sum(kl_divergence(q_z, p_z))

def flow_kld(q_z, p_z, z, z0, sum_log_j):
    batch_size = z.size(0)
    e_log_pz = -torch.sum(p_z.entropy()) / batch_size
    return total_kld(q_z, p_z) / batch_size - torch.mean(sum_log_j)

def compute_nll(q_z, p_z, z, z0, sum_log_j, re_loss):
    batch_size = z.size(0)
    z_dim = z.size(1)
    e_log_pz = p_z.log_prob(z).sum(-1).mean()
    e_log_px_z = -re_loss / batch_size
    e_log_qz = -torch.sum(q_z.entropy()) / batch_size
    return -(e_log_px_z + e_log_pz - e_log_qz + torch.mean(sum_log_j))

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
    return m + torch.log(sum_exp)

def mutual_info(q_z, p_z, z):
    batch_size = z.size(0)
    z_dim = z.size(1)
    # log_qz = q_z.log_prob(
        # z.unsqueeze(1).expand(-1, batch_size, -1))
    log_qz = q_z.log_prob(z).sum(-1).unsqueeze(1).expand(batch_size, -1)
    e_log_q_zx = -torch.sum(q_z.entropy()) / batch_size 
    e_log_qz = (log_sum_exp(log_qz, dim=1) - np.log(batch_size)).mean()

    return e_log_q_zx - e_log_qz

def mutual_info_flow(q_z, p_z, z, z0, sum_log_j):
    batch_size = z.size(0)
    z_dim = z.size(1)
    # log_flow_qz = q_z.log_prob(z).sum(-1) + sum_log_j 
    log_flow_qz = q_z.log_prob(z0).sum(-1)
    # compute E_xE_{q(z'|x)}log(q(z'|x))
    e_log_q_flow_zx = torch.sum(log_flow_qz) / (batch_size)
    e_log_flow_qz = (log_sum_exp(log_flow_qz.unsqueeze(1).expand(batch_size, -1), dim=1) - np.log(batch_size)).mean()
    
    return e_log_q_flow_zx - e_log_flow_qz + sum_log_j.mean()

def gaussian_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-10*kernel)

def compute_mmd(p, q, kernel='g'):
    if kernel == 'g':
        # use gaussian kernel
        x = q.rsample()
        y = p.sample()
        x_kernel = gaussian_kernel(x, x)
        y_kernel = gaussian_kernel(y, y)
        xy_kernel = gaussian_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    elif kernel == 'im':
        # use im kernel
        mmd = im_kernel(q, p)
    else:
        raise NotImplementedError
    return mmd

def im_kernel(q_z, p_z, z_var=1):
    sample_qz = q_z.rsample()
    sample_pz = p_z.sample()
    batch_size = sample_pz.size(0)
    z_dim = sample_qz.size(1)
    Cbase = 2 * z_dim * z_var

    norms_pz = torch.sum(sample_pz.pow(2), dim=1, keepdim=True)
    dotprobs_pz = torch.matmul(sample_pz, sample_pz.t())
    distances_pz = norms_pz + norms_pz.t() - 2. * dotprobs_pz

    norms_qz = torch.sum(sample_qz.pow(2), dim=1, keepdim=True)
    dotprobs_qz = torch.matmul(sample_qz, sample_qz.t())
    distances_qz = norms_qz + norms_qz.t() - 2. * dotprobs_qz

    dotprobs = torch.matmul(sample_qz, sample_pz.t())
    distances = norms_qz + norms_pz.t() - 2. * dotprobs

    stat = 0.
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = Cbase * scale
        res1 = C / (C + distances_qz)
        res1 += C / (C + distances_pz)
        res1 = res1 * (1 - torch.eye(batch_size, device=sample_pz.device))
        res1 = torch.sum(res1) / (batch_size * batch_size - batch_size)
        res2 = C / (C + distances)
        res2 = torch.sum(res2) * 2. / (batch_size * batch_size)
        stat += res1 - res2
    return stat