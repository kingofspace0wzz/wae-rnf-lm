import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

import numpy as np


def seq_recon_loss(outputs, targets, pad_id):
    return F.cross_entropy(
        outputs.view(-1, outputs.size(2)),
        targets.view(-1),
        size_average=False, ignore_index=pad_id)


def bow_recon_loss(outputs, targets):
    """
    Note that outputs is the bag-of-words log likelihood predictions. 
    targets is the target counts. 

    """
    return - torch.sum(targets * outputs)


def total_kld(posterior, prior):
    """
    Use pytorch built-in distributions to calculate kl divergence.

    """
    return torch.sum(kl_divergence(posterior, prior))

def kl_approx():
    pass

def e_log_p(posterior, prior):
    """
    Analytically calculate the expected value of prior under the posterior. 
    D(q||p) = - H(q) - e_log_p(q, p)
    """
    return - torch.sum(kl_divergence(posterior, prior) + posterior.entropy())


def logsumexp(inputs, dim):
    s, _ = inputs.max(dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    return outputs

            
def kld_decomp(posterior, prior, z):
    batch_size = z.size(0)
    code_size = z.size(1)
    log_probs = posterior.log_prob(
        z.unsqueeze(1).expand(-1, batch_size, -1))
    e_log_qzx = - torch.sum(posterior.entropy()) / batch_size
    e_log_qz  = torch.sum(
        logsumexp(log_probs.sum(2), dim=1)
    ) / batch_size - np.log(batch_size)
    e_log_qzi = torch.sum(
        logsumexp(log_probs, dim=1)
    ) / batch_size - code_size * np.log(batch_size)
    e_log_pz  = e_log_p(posterior, prior) / batch_size
    mutual_info = e_log_qzx - e_log_qz
    total_corr = e_log_qz - e_log_qzi
    dimwise_kl = e_log_qzi - e_log_pz
    return mutual_info, total_corr, dimwise_kl

def gaussian_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel)

def compute_mmd(x, y):
    x_kernel = gaussian_kernel(x, x)
    y_kernel = gaussian_kernel(y, y)
    xy_kernel = gaussian_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd
