import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (Transform, TransformedDistribution,
                                AffineTransform, Distribution)
from torch.distributions import constraints
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform
from torch.distributions.utils import lazy_property
from utils import _batch_diag, _batch_mahalanobis, _batch_mv, _standard_normal

import math

class NormalInvCDF(Transform):
    domain = constraints.real
    codomain = constraints.interval(0, 1)
    bijective = True
    
    def __init__(self):
        super(NormalInvCDF, self).__init__()
        self.normal_dist = Normal(0., 1.)

    def __eq__(self, other):
        return isinstance(other, NormalInvCDF)

    def _call(self, x):
        return self.normal_dist.cdf(x)

    def _inverse(self, y):
        return self.normal_dist.icdf(y)

    def log_abs_det_jacobian(self, x, y):
        # return self.normal_dist.log_prob(x)
        return torch.log(y)

class Concat(Transform):

    def __init__(self):
        super(Concat, self).__init__()

    def _call(self, x):
        split_xs = torch.chunk(x, )

class GaussianCopulaDistribution(MultivariateNormal):
    arg_constraints = {'loc': constraints.real_vector,
                       'covariance_matrix': constraints.positive_definite,
                       'precision_matrix': constraints.positive_definite,
                       'scale_tril': constraints.lower_cholesky}
    support = constraints.real
    has_rsample = True
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None):
        super(GaussianCopulaDistribution, self).__init__()


class GaussianCopula(Distribution):
    arg_constraints = {'loc': constraints.real, 
                       'scale_tril': constraints.lower_cholesky,
                       'covariance_matrx': constraints.positive_definite}
    # support = constraints.positive
    def __init__(self, loc, scale_tril=None, covariance_matrix=None, validate_args=None):
        if scale_tril is not None:
            base_dist = MultivariateNormal(loc, scale_tril=scale_tril)
        elif covariance_matrix is not None:
            base_dist = MultivariateNormal(loc, covariance_matrix=covariance_matrix)
        # super(GaussianCopula, self).__init__(base_dist, NormalInvCDF(), validate_args=validate_args)
        
        # self.loc = loc
        # self.scale_tril = scale_tril
        # self.covariance_matrix = covariance_matrix
        # self.scale = torch.diagonal(covariance_matrix, 0)
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        
        loc_ = loc.unsqueeze(-1)  # temporarily add dim on right
        self.diag = None
        if scale_tril is not None:
            if scale_tril.dim() < 2:
                raise ValueError("scale_tril matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self.scale_tril, loc_ = torch.broadcast_tensors(scale_tril, loc_)
            self.covariance_matrix = torch.matmul(scale_tril, torch.transpose(scale_tril, dim0=-2, dim1=-1))
            # self.diag = torch.sum(scale_tril * scale_tril, dim=-1)
        elif covariance_matrix is not None:
            if covariance_matrix.dim() < 2:
                raise ValueError("covariance_matrix must be at least two-dimensional, "
                                 "with optional leading batch dimensions")
            self.covariance_matrix, loc_ = torch.broadcast_tensors(covariance_matrix, loc_)
       
        self.loc = loc_[..., 0]  # drop rightmost dim
        batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
        super(GaussianCopula, self).__init__(batch_shape, event_shape, validate_args=validate_args)

        if scale_tril is not None:
            self._unbroadcasted_scale_tril = scale_tril
        else:    
            self._unbroadcasted_scale_tril = torch.cholesky(self.covariance_matrix)

        self.multinormal = base_dist

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GaussianCopula, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self.event_shape
        cov_shape = batch_shape + self.event_shape + self.event_shape
        new.loc = self.loc.expand(loc_shape)
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril
        if 'covariance_matrix' in self.__dict__:
            new.covariance_matrix = self.covariance_matrix.expand(cov_shape)
        if 'scale_tril' in self.__dict__:
            new.scale_tril = self.scale_tril.expand(cov_shape)
        if 'precision_matrix' in self.__dict__:
            new.precision_matrix = self.precision_matrix.expand(cov_shape)
        super(GaussianCopula, new).__init__(batch_shape,
                                                self.event_shape,
                                                validate_args=False)
        new._validate_args = self._validate_args
        return new

    # def expand(self, batch_shape, _instance=None):
    #     new = self._get_checked_instance(GaussianCopula, _instance)
    #     return super(GaussianCopula, self).expand(batch_shape, _instance=new)

    def prob(self, x):
        return torch.exp(self.log_prob(x))

    # @lazy_property
    # def scale_tril(self):
    #     return self.base_dist.scale_tril

    # @lazy_property
    # def covariance_matrix(self):
    #     return self.base_dist.covariance_matrix

    # @property
    # def loc(self):
    #     return self.base_dist.loc

    @property
    def scale(self):
        return self.covariance_matrix.diag()

    @property
    def mean(self):
        return self.base_dist.mean

    def covariance(self):
        return self.covariance_matrix

    def update_scale_tril(self, scale_tril):
        self._unbroadcasted_scale_tril = scale_tril

    def rsample(self, scale_tril=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        # TODO : sample Q (quantile) or U (uniform)
        # We will sample Q for now as it gives the log copula density
        # uniform = Uniform(torch.zeros(shape), torch.ones(shape))
        
        # TODO : rsample a multivariate normal by z = u + L * eps
        if scale_tril is not None:
            eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
            return self.loc + _batch_mv(scale_tril, eps)
        else:
            eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
            return self.loc + _batch_mv(self._unbroadcasted_scale_tril, eps)    

    def log_prob(self, x):
        # x : [batch_size, dim]
        # sigma
        if self.diag is not None:
            _diag = self.diag.sqrt()
        else:
            _diag = _batch_diag(self.covariance_matrix).sqrt()
        diag = torch.stack([torch.diag(v.squeeze()) for v in _diag.chunk(_diag.size(0))], dim=0)

        M_sigma = _batch_mahalanobis(self._unbroadcasted_scale_tril, x)
        
        # M_i = _batch_mahalanobis(torch.eye(x.size(-1), dtype=x.dtype, device=x.device), x)        
        M_i = _batch_mahalanobis(diag, x)
        # print(diag)
        half_log_det = _batch_diag(self._unbroadcasted_scale_tril).log().sum(-1)
        # return 0.5 * (M_i - M_sigma) - half_log_det + _diag.log().sum(-1)
        return 0.5 * (M_i - M_sigma) - half_log_det + _diag.log().sum(-1)

class GaussianCopulaVariable(nn.Module):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    
    def __init__(self, loc, scale, covariance_matrix=None, validate_args=None):
        super(GaussianCopulaVariable, self).__init__()
        # affine = AffineTransform(-loc, 1/scale)
        self.multinormal = MultivariateNormal(loc, covariance_matrix)
        self.normal = Normal(loc, scale)
        self.standard_normal = Normal(0, 1)
        self.loc = loc
        self.scale = scale
        self.covariance_matrix = covariance_matrix

    def forward(self, x):
        
        pass

    def sample(self):
        r'''
        Sample from Gaussian Copula
            q ~ N(0, \Sigma)
            u = [cdf(q_1),...,cdf(q_n)]^T
        '''
        q = self.multinormal.rsample()
        u = torch.stack([self.standard_normal.cdf(q_i/s_i) for q_i, s_i in zip(q, self.scale)]).squeeze()

        return u

# def log_marginals(z, mu, logvar):
#     """log marginals of independent posterior q(z|x)"""
#     dim = z.size(-1)
#     batch_sie = z.size(0)
#     M = (z - mu)/logvar.exp()
#     return -0.5 * (dim*batch_sie * (math.log(2 * math.pi)) + torch.sum(logvar))

def test():
    uniform = Uniform(torch.tensor([0., 0.]), torch.tensor([1., 1.]))
    # mean = torch.Tensor([[0, 0], [0, 0]])
    # cov = torch.Tensor([[[1, 0.4], [0.6, 1]], [[1, 0.5], [0.5, 1]]])
    # copula = GaussianCopula(mean, covariance_matrix=cov)
    # sample = copula.rsample()
    # print(sample)
    # q = copula.rsample()
    # print(copula.log_prob(q))

    mean = torch.Tensor([[0, 0]])
    cov2 = torch.Tensor([[[1, 0.5], [0.5, 1]]])
    copula = GaussianCopula(mean, covariance_matrix=cov2)
    sample = copula.rsample()
    print(sample)
    q = copula.rsample()
    
    print(copula.log_prob(q))

    # normal = Normal(0., 1.)
    # multinormal = MultivariateNormal(torch.tensor([0.,0.]), cov)
    # x = torch.tensor([normal.icdf(mean[0]), normal.icdf(mean[1])])
    # print(multinormal.cdf(x))
if __name__ == "__main__":
    test()
