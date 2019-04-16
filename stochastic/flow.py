import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from clustering.kmeans import lloyd

eps = 1e-8

class PlanarNormalizingFlow(nn.Module):
    """
    Planar normalizing flow [Rezende & Mohamed 2015].
    Provides a tighter bound on the ELBO by giving more expressive
    power to the approximate distribution, such as by introducing
    covariance between terms.
    """
    def __init__(self, in_features):
        super(PlanarNormalizingFlow, self).__init__()
        self.u = nn.Parameter(torch.randn(in_features))
        self.w = nn.Parameter(torch.randn(in_features))
        self.b = nn.Parameter(torch.ones(1))

        #self.A = nn.Parameter(torch.rand(1), requires_grad=True)
        #self.register_parameter("A", self.A)

        #self.band = nn.Parameter(torch.rand(1), requires_grad=True)
        #self.register_parameter("band", self.band)

    def beta(self, z, centers, band=0.5):
        kernel, d = metric(z, centers, band)
        return kernel + 1, d

    def forward(self, z, centers=None, reg='g', band=0.5):
        if centers is not None:
            if reg == 'g':
                beta, d = self.beta(z, centers, band)
            elif reg == 'im':
                beta = im_metric(z, centers)
        #     u = self.u / self.beta(z, centers)

        # Create uhat such that it is parallel to w
        uw = torch.dot(self.u, self.w)
        muw = -1 + F.softplus(uw)
        uhat = self.u + (muw - uw) * torch.transpose(self.w, 0, -1) / torch.sum(self.w ** 2)

        # Equation 21 - Transform z
        zwb = torch.mv(z, self.w) + self.b

        # if centers is not None:
        #     f_z = z + (uhat.view(1, -1) * torch.tanh(zwb).view(-1, 1)) / beta.view(-1, 1)
        # else:
        #     f_z = z + (uhat.view(1, -1) * torch.tanh(zwb).view(-1, 1))

        # # Compute the Jacobian using the fact that
        # # tanh(x) dx = 1 - tanh(x)**2
        # if centers is not None:
        #     psi = (1 - torch.tanh(zwb)**2).view(-1, 1) * self.w.view(1, -1) / beta.view(-1, 1)
        # else:
        #     psi = (1 - torch.tanh(zwb)**2).view(-1, 1) * self.w.view(1, -1)
        
        f_z = z + (uhat.view(1, -1) * torch.tanh(zwb).view(-1, 1))
        psi = (1 - torch.tanh(zwb)**2).view(-1, 1) * self.w.view(1, -1)
        psi_u = torch.mv(psi, uhat)

        # Return the transformed output along
        # with log determninant of J
        logdet_jacobian = torch.log(torch.abs(1 + psi_u) + 1e-8)

        # XXX: current version of RNF
        if centers is not None:
            if reg == 'g':
                penalty = -torch.log(beta + eps)
            else:
                penalty = band * torch.log(torch.abs(beta) + eps)
        else:
            penalty = torch.zeros(1).to(z.device)

        return f_z, logdet_jacobian, penalty


class NormalizingFlows(nn.Module):
    """
    Presents a sequence of normalizing flows as a torch.nn.Module.
    """
    def __init__(self, in_features, flow_type=PlanarNormalizingFlow, n_flows=1, reg='g', band=0.5):
        super(NormalizingFlows, self).__init__()
        self.flows = nn.ModuleList([flow_type(in_features) for _ in range(n_flows)])
        self.reg = reg
        self.band = band

    def forward(self, z, centers=False):
        log_det_jacobian = []
        penalty = []
        # centers = None
        # if kmeans:
        #     _, centers = lloyd(z, 3)

        for flow in self.flows:
            z, j, d = flow(z, centers, reg=self.reg, band=self.band)
            log_det_jacobian.append(j)
            penalty.append(d)

        return z, sum(log_det_jacobian), sum(penalty)

def square_norm(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    norm = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return norm

def metric(z, centers, band=0.5):
    batch_size = z.size(0)
    n = centers.size(0) # number of centers
    d = z.unsqueeze(1).expand(-1, n, -1) - centers.unsqueeze(0).expand(batch_size, n, -1)
    # take distance norm over batch
    norm = torch.norm(d, p=2, dim=2, keepdim=True).squeeze(2).transpose(0, 1)
    measure, _ = norm.min(dim=0)
    # compute kernel
    kernel = torch.exp(-band * measure**2)
    return kernel, measure

def im_metric(z, centers):
    batch_size = z.size(0)
    n = centers.size(0) # number of centers
    z_dim = z.size(1)
    d = z.unsqueeze(1).expand(-1, n, -1) - centers.unsqueeze(0).expand(batch_size, n, -1)
    norm = torch.norm(d, p=2, dim=2, keepdim=True).squeeze(2).transpose(0, 1)
    measure, _ = norm.min(dim=0)
    Cbase = 2 * z_dim
    stat = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = Cbase * scale
        stat += C / (C + measure.pow(2))
    return stat