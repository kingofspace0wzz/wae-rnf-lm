import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from copula import GaussianCopula
from utils import _mask_diag
from loss import seq_recon_loss
import math

class NgramLM(nn.Module):
    """Ngram Language model"""
    def __init__(self, vocab_size, embed_size, context_size, dropout=0.5):
        super(NgramLM, self).__init__()
        self.encoder = nn.Embedding(vocab_size, embed_size)
        self.linear1 = nn.Linear(context_size * embed_size, 128)
        self.linear2 = nn.Linear(128, vocab_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        embeds = self.drop(self.encoder(x).view((1, -1)))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1) 
        return log_probs

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, copula="gaussian"):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        self.copula = copula

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        log_prob = F.log_softmax(decoded, dim=1)
        if self.copula.lower() is "gaussian":
            #log_cd = 
            pass
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden, log_prob

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

class LM(nn.Module):
    """Language model"""
    def __init__(self):
        super(LM, self).__init__()

    def forward(self, x):

        return x

class SeqToBow(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, inputs, ignore_index):
        # inputs dim: batch_size x max_len
        bow = torch.zeros(
            (inputs.size(0), self.vocab_size),
            dtype=torch.float,
            device=inputs.device
        )
        ones = torch.ones_like(
            inputs, dtype=torch.float,
        )
        bow.scatter_add_(1, inputs, ones)
        bow[:, ignore_index] = 0
        return bow


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            input_size, hidden_size, num_layers=1,
            batch_first=True
        )

    def forward(self, inputs, lengths):
        inputs = self.drop(inputs)
        inputs = pack(inputs, lengths, batch_first=True)
        _, hn = self.rnn(inputs)
        return hn


class HiddenToNormal(nn.Module):
    def __init__(self, hidden_size, code_size):
        super().__init__()
        self.fcmu = nn.Linear(hidden_size * 2, code_size)
        self.fclv = nn.Linear(hidden_size * 2, code_size)
        self.bnmu = nn.BatchNorm1d(code_size)
        self.bnlv = nn.BatchNorm1d(code_size)        

    def forward(self, hidden):
        h = torch.cat(hidden, dim=2).squeeze(0)
        mu = self.bnmu(self.fcmu(h))
        lv = self.bnlv(self.fclv(h))
        return Normal(mu, (0.5 * lv).exp())

class HiddenToGaussianCopula(nn.Module):
    # TODO: parameterize it as "copula + normal" or as "copula"
    def __init__(self, hidden_size, code_size):
        super().__init__()
        self.code_size = code_size
        self.fcmu = nn.Linear(hidden_size * 2, code_size)
        self.fclv = nn.Linear(hidden_size * 2, code_size)
        self.bnmu = nn.BatchNorm1d(code_size)
        self.bnlv = nn.BatchNorm1d(code_size)        

    def forward(self, hidden, scale_tril):
        # TODO: need to parameterize cholesky factor for gaussian copula
        batch_size = hidden.size(0)

        h = torch.cat(hidden, dim=2).squeeze(0)
        mu = self.bnmu(self.fcmu(h))
        lv = self.bnlv(self.fclv(h))

        loc = torch.zeros(batch_size, self.code_size)
        copula = GaussianCopula(loc, scale_tril)
        return copula

class HiddenToScaleTril(nn.Module):
    # TODO : parameterize the cholesky factor in Gaussian Copula by a NN
    # XXX: In this case, the parameters are in NN not as cholesky itself
    def __init__(self, hidden_size, code_size):
        super().__init__()
        self.code_size = code_size
        # self.batch_size = batch_size
        self.fclt = nn.Linear(hidden_size * 2, code_size * code_size)
        self.fcdg = nn.Linear(hidden_size * 2, code_size)
        self.fca = nn.Linear(hidden_size * 2, code_size)
        self.fcw = nn.Linear(hidden_size * 2, code_size)
        self.bnlt = nn.BatchNorm1d(code_size * code_size)
        self.bndg = nn.BatchNorm1d(code_size)
        self.bna = nn.BatchNorm1d(code_size)

    def forward(self, hidden, diag=None, decomp="ldl"):
        code_size = self.code_size
        # XXX : we may or may not need batch norm for scale_tril
        
        # {XXX : LDL' decompostition
        # h = torch.cat(hidden, dim=2).squeeze(0)
        # tril_v = F.tanh(self.fclt(h)).unsqueeze(1).reshape(-1, code_size, code_size)
        # tril_v = _mask_diag(tril_v, torch.ones(code_size, device=tril_v.device)).tril()
        # I = torch.eye(code_size, device=h.device).unsqueeze(0).repeat(h.size(0), 1, 1)
        # w = F.tanh(self.fcw(h)).unsqueeze(-1) 
        # diag = F.tanh(I * w)
        # # print(w)
        # # diag = self.fcdg(h) + 1
        # # diag = diag.unsqueeze(2).expand(*diag.size(), diag.size(1)) * torch.eye(diag.size(1), device=tril_v.device)
        # scale = torch.matmul(tril_v, diag)
        # } 

        if decomp is "ldl":
            # {XXX : LDL' with I + aa' method as scale_tril
            h = torch.cat(hidden, dim=2).squeeze(0)
            I = torch.eye(code_size, device=h.device).unsqueeze(0).repeat(h.size(0), 1, 1)
            w = F.relu(self.fcw(h)).unsqueeze(-1) + 1
            a = F.tanh(self.bna(self.fca(h))).unsqueeze(-1)
            # scale = (I * w + torch.matmul(a, torch.transpose(a, dim0=-2, dim1=-1))).tril()
            scale = torch.matmul(a, torch.transpose(a, dim0=-2, dim1=-1)).tril()
            # }

        # { XXX : Cholesky decomposition
        # # cat two hidden vectors
        # h = torch.cat(hidden, dim=2).squeeze(0)
        # # print(self.bnlt(self.fclt(h)))
        # # tril_v : [batch_size, code_size, code_size]
        # tril_v = self.bnlt(self.fclt(h)).unsqueeze(1).reshape(-1, code_size, code_size).tril()
        # # tril_v : [batch_size, code_size, code_size]
        # # XXX : the next line may be unnecessary, but we need to ensure diagonal to be nonzero
        # tril_v = _mask_diag(tril_v, torch.ones(code_size, device=tril_v.device))
        # scale = tril_v
        # }
        else:
            # { XXX : Sigma = wI + aa'
            h = torch.cat(hidden, dim=2).squeeze(0)
            I = torch.eye(code_size, device=h.device).unsqueeze(0).repeat(h.size(0), 1, 1)
            w = F.relu(self.fcw(h)).unsqueeze(-1) + 0.5
            # w = self.fcw(h).unsqueeze(-1).exp()
            a = F.tanh(self.bna(self.fca(h))).unsqueeze(-1)
            scale = I * w + torch.matmul(a, torch.transpose(a, dim0=-2, dim1=-1))
            # }

        # # XXX : non standard scale
        # # if diag is None:
        # #     # diag : [batch_size, code_size]
        # #     diag = F.softplus(self.fcdg(h))
        # # # diag : [batch_size, code_size, code_size]
        # # diag = diag.unsqueeze(2).expand(*diag.size(), diag.size(1)) * torch.eye(diag.size(1))
        
        # # XXX : standard scale, with 1 on diagonals
        # # diag = torch.eye(code_size, dtype=tril_v.dtype, device=tril_v.device)

        # # scale_tril = tril_v + diag}
        return scale

class CodeToHidden(nn.Module):
    def __init__(self, hidden_size, code_size):
        super().__init__()
        self.fc = nn.Linear(code_size, hidden_size * 2)

    def forward(self, z):
        hidden = torch.tanh(self.fc(z)).unsqueeze(0)
        return [x.contiguous() for x in torch.chunk(hidden, 2, 2)]

class SeqDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            input_size, hidden_size, num_layers=1,
            batch_first=True
        )

    def forward(self, inputs, lengths=None, init_hidden=None):
        inputs = self.drop(inputs)
        if lengths is not None:
            inputs = pack(inputs, lengths, batch_first=True)
        outputs, hidden = self.rnn(inputs, init_hidden)
        if lengths is not None:
            outputs, _ = unpack(outputs, batch_first=True)
        outputs = self.drop(outputs)
        return outputs, hidden


class BowDecoder(nn.Module):
    def __init__(self, code_size, vocab_size, dropout):
        super().__init__()
        self.fc = nn.Linear(code_size, vocab_size)
        self.bn = nn.BatchNorm1d(vocab_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        """Inputs: latent code """
        inputs = self.drop(inputs)
        return F.log_softmax(self.bn(self.fc(inputs)), dim=1)


class LstmVAE(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, code_size,
                 dropout, batch_size=32, fix_prior=True, flow_prior=False, decomp='ldl', copula=True):
        super().__init__()
        self.batch_size = batch_size
        self.code_size = code_size
        self.lookup = nn.Embedding(vocab_size, embed_size)
        self.encode = Encoder(embed_size, hidden_size, dropout)
        self.h2z = HiddenToNormal(hidden_size, code_size)
        self.z2h = CodeToHidden(hidden_size, code_size)
        self.decode_seq = SeqDecoder(embed_size, hidden_size, dropout)
        self.fcout = nn.Linear(hidden_size, vocab_size)
        self.seq2bow = SeqToBow(vocab_size)
        self.decode_bow = BowDecoder(code_size, vocab_size, dropout)
        # for priors
        requires_grad = not fix_prior
        self.mus = nn.Parameter(torch.zeros((1, code_size)), requires_grad=requires_grad)
        self.lvs = nn.Parameter(torch.zeros((1, code_size)), requires_grad=requires_grad)
        self.register_parameter('mus', self.mus)
        self.register_parameter('logvars', self.lvs)

        # Register a lower cholesky factor for learning copula
        # TODO : scale_tril has to be strictly POSTITIVE DEFINITE (not semidefinite)
        # diag = torch.ones((batch_size, code_size), dtype=torch.long, requires_grad=True)
        # tril = torch.tril(torch.randn((code_size-1, code_size-1), 
        #                     dtype=torch.long, requires_grad=True))
        # self.tril = nn.Parameter(tril, requires_grad=True)
        # self.diag = nn.Parameter(diag, requires_grad=True)
        # scale_tril = identity + torch.cat((torch.cat((torch.zeros(code_size-1), self.tril), dim=-2), torch.zeros(code_size).reshape(-1, 1)), dim=-1)
        # self.register_parameter('tril', self.tril)
        # self.register_parameter('diag', self.diag)
        
        # XXX : Let us not worry about the contraint first
        # self.scale_tril = nn.Parameter(torch.randn((code_size, code_size)).unsqueeze(0).expand(batch_size, -1, -1).tril(), requires_grad=True)
        # self.register_parameter('cholesky', self.scale_tril)

        # gaussian copula layer
        self.h2c = HiddenToGaussianCopula(hidden_size, code_size)
        # module that parameterize scale_tril
        self.hidden_to_tril = HiddenToScaleTril(hidden_size, code_size)

        self.decomp = decomp
        self.copula = copula

    def prior(self, batch_size):
        return Normal(self.mus.expand(batch_size, -1),
                      (0.5 * self.lvs.expand(batch_size, -1)).exp())

    def _encode(self, inputs, lengths):
        enc_emb = self.lookup(inputs)
        hn = self.encode(enc_emb, lengths)
        q_z = self.h2z(hn)
        return q_z, hn
    
    def _copula(self, hidden):
        scale = self.hidden_to_tril(hidden, decomp=self.decomp)
        loc = torch.zeros((scale.size(0), self.code_size), dtype=scale.dtype, device=scale.device)
        # use covariance matrix
        if self.decomp == "ldl":
            return GaussianCopula(loc, scale_tril=scale)
        elif self.decomp == "cho":
            return GaussianCopula(loc, covariance_matrix=scale)
        else:
            raise NotImplementedError

    def _copula_g(self, z):
        pass

    def forward(self, inputs, lengths, pad_id):
    
        p_z = self.prior(inputs.size(0))    # gaussian prior
        q_z, hn = self._encode(inputs, lengths) # posterior
        if self.training:
            z = q_z.rsample()
        else:
            z = q_z.mean
        
        if self.copula is True:
            # get gaussian copula model from hidden states hn
            # hn = detach(hn)
            copula = self._copula(hn)
            # compute log copula density
            quantile = copula.rsample()
            log_copula = copula.log_prob(quantile)
            log_marginals = q_z.log_prob(z)
        else:
            log_copula = torch.zeros(1)
            log_marginals = torch.zeros(1)

        bow_targets = self.seq2bow(inputs, pad_id)
        bow_outputs = self.decode_bow(z)
        
        init_hidden = self.z2h(z)
        dec_emb = self.lookup(inputs)
        outputs, _ = self.decode_seq(dec_emb, lengths=lengths,
                                     init_hidden=init_hidden)
        seq_outputs = self.fcout(outputs)

        # self.q_z, self.p_z = q_z, p_z
        return q_z, p_z, z, seq_outputs, bow_outputs, bow_targets, log_copula, log_marginals
        # return z, seq_outputs, bow_outputs, bow_targets, log_copula, log_marginals

    def get_dist(self):
        return self.q_z, self.p_z

    def generate(self, z, max_length, sos_id):
        batch_size = z.size(0)
        generated = torch.zeros((batch_size, max_length), dtype=torch.long, device=z.device)
        dec_inputs = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=z.device)
        hidden = self.z2h(z)
        for k in range(max_length):
            dec_emb = self.lookup(dec_inputs)
            outputs, hidden = self.decode_seq(dec_emb, init_hidden=hidden)
            outputs = self.fcout(outputs)
            dec_inputs = outputs.max(2)[1]
            generated[:, k] = dec_inputs[:, 0].clone()
        return generated

    def reconstruct(self, inputs, lengths, max_length,
                    sos_id, sample=True):
        posterior, _ = self._encode(inputs, lengths)
        z = posterior.sample() if sample else posterior.mean
        return self.generate(z, max_length, sos_id)

    def sample(self, num_samples, max_length, sos_id):   
        prior = self.prior(num_samples)
        z = prior.sample()
        return self.generate(z, max_length, sos_id)
    
    def interpolate(self, max_length, sos_id):
        prior = self.prior(1)
        z0 = prior.sample()
        z1 = prior.sample()
        z0_r, _ = self.flow(z0, self.centers)
        z1_r, _ = self.flow(z1, self.centers)
        generated = []
        generated_r = []
        for i in range(11):
            z_int = z1 * i/10 + z0 * (10 - i)/10
            z_int_r = z1_r * i/10 + z0_r * (10 - i)/10
            generate = self.generate(z_int, max_length, sos_id)
            generate_r = self.generate(z_int_r, max_length, sos_id)
            generated.append(generate)
            generated_r.append(generate_r)
        return generated, generate_r

    def iw_nll(self, q_z, p_z, x, targets, lengths, pad_id, nsamples=500):
        batch = []
        for _ in range(nsamples):
            z = q_z.rsample()
        
            log_infer = q_z.log_prob(z).sum(dim=-1) 
            log_prior = p_z.log_prob(z).sum(dim=-1)
            log_gen = self.cond_ll(x, targets, lengths, z, pad_id)

            batch.append(log_prior.view(-1, 1) + log_gen.view(-1, 1) - log_infer.view(-1, 1))

        iw_ll = log_sum_exp(torch.cat(batch, dim=-1), dim=-1) - math.log(nsamples)
        # for _ in range(nsamples):
        #     z = q_z.rsample()

        return -torch.mean(iw_ll)

    def cond_ll(self, inputs, targets, lengths, z, pad_id):
        init_hidden = self.z2h(z)
        dec_embeds = self.lookup(inputs)
        outputs, _ = self.decode_seq(dec_embeds, lengths, init_hidden=init_hidden)
        outputs = self.fcout(outputs)
        loss = seq_recon_loss(outputs, targets, pad_id).expand(z.size(0), 1) / z.size(0)
        return -loss

class HiddenToMultiNormal(nn.Module):
    def __init__(self, hidden_size, code_size):
        super().__init__()
        self.fcmu = nn.Linear(hidden_size * 2, code_size)
        # self.fclv = nn.Linear(hidden_size * 2, code_size)
        self.bnmu = nn.BatchNorm1d(code_size)
        # self.bnlv = nn.BatchNorm1d(code_size) 
        self.hidden_to_tril = HiddenToScaleTril(hidden_size, code_size)       

    def forward(self, hidden):
        h = torch.cat(hidden, dim=2).squeeze(0)
        mu = self.bnmu(self.fcmu(h))
        cov = self.hidden_to_tril(h, 'cho')
        return MultivariateNormal(mu, covariance_matrix=cov)

class MultiNormalVAE(nn.Module):
    """Some Information about MultiNormalVAE"""
    def __init__(self, vocab_size, embed_size, hidden_size, code_size,
                 dropout, batch_size=32, fix_prior=True, flow_prior=False, decomp='ldl', copula=True):
        super().__init__()
        self.batch_size = batch_size
        self.code_size = code_size
        self.lookup = nn.Embedding(vocab_size, embed_size)
        self.encode = Encoder(embed_size, hidden_size, dropout)
        self.h2z = HiddenToMultiNormal(hidden_size, code_size)
        self.z2h = CodeToHidden(hidden_size, code_size)
        self.decode_seq = SeqDecoder(embed_size, hidden_size, dropout)
        self.fcout = nn.Linear(hidden_size, vocab_size)
        self.seq2bow = SeqToBow(vocab_size)
        self.decode_bow = BowDecoder(code_size, vocab_size, dropout)
        # for priors
        requires_grad = not fix_prior
        self.mus = nn.Parameter(torch.zeros((1, code_size)), requires_grad=requires_grad)
        self.lvs = nn.Parameter(torch.zeros((1, code_size)), requires_grad=requires_grad)
        self.register_parameter('mus', self.mus)
        self.register_parameter('logvars', self.lvs)
        # gaussian copula layer
        self.h2c = HiddenToGaussianCopula(hidden_size, code_size)
        # module that parameterize scale_tril
        self.hidden_to_tril = HiddenToScaleTril(hidden_size, code_size)

        self.decomp = decomp
        self.copula = copula

    def prior(self, batch_size):
        # return Normal(self.mus.expand(batch_size, -1),
        #               (0.5 * self.lvs.expand(batch_size, -1)).exp())
        return MultivariateNormal(self.mus.expand(batch_size, -1),
                                    torch.eye(self.code_size, dtype=self.mus.dtype, device=self.mus.device).expand(batch_size, -1, -1))

    def _encode(self, inputs, lengths):
        enc_emb = self.lookup(inputs)
        hn = self.encode(enc_emb, lengths)
        q_z = self.h2z(hn)
        return q_z, hn

    def forward(self, inputs, lengths, pad_id):
    
        p_z = self.prior(inputs.size(0))    # gaussian prior
        q_z, hn = self._encode(inputs, lengths) # posterior
        if self.training:
            z = q_z.rsample()
        else:
            z = q_z.mean
      
        log_copula = torch.zeros(1)
        log_marginals = torch.zeros(1)

        bow_targets = self.seq2bow(inputs, pad_id)
        bow_outputs = self.decode_bow(z)
        
        init_hidden = self.z2h(z)
        dec_emb = self.lookup(inputs)
        outputs, _ = self.decode_seq(dec_emb, lengths=lengths,
                                     init_hidden=init_hidden)
        seq_outputs = self.fcout(outputs)

        # self.q_z, self.p_z = q_z, p_z
        return q_z, p_z, z, seq_outputs, bow_outputs, bow_targets, log_copula, log_marginals
        # return z, seq_outputs, bow_outputs, bow_targets, log_copula, log_marginals

    def get_dist(self):
        return self.q_z, self.p_z

    def generate(self, z, max_length, sos_id):
        batch_size = z.size(0)
        generated = torch.zeros((batch_size, max_length), dtype=torch.long, device=z.device)
        dec_inputs = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=z.device)
        hidden = self.z2h(z)
        for k in range(max_length):
            dec_emb = self.lookup(dec_inputs)
            outputs, hidden = self.decode_seq(dec_emb, init_hidden=hidden)
            outputs = self.fcout(outputs)
            dec_inputs = outputs.max(2)[1]
            generated[:, k] = dec_inputs[:, 0].clone()
        return generated

    def reconstruct(self, inputs, lengths, max_length,
                    sos_id, sample=True):
        posterior, _ = self._encode(inputs, lengths)
        z = posterior.sample() if sample else posterior.mean
        return self.generate(z, max_length, sos_id)

    def sample(self, num_samples, max_length, sos_id):   
        prior = self.prior(num_samples)
        z = prior.sample()
        return self.generate(z, max_length, sos_id)

    def iw_nll(self, q_z, p_z, x, targets, lengths, pad_id, nsamples=500):
        batch = []
        for _ in range(nsamples):
            z = q_z.rsample()
        
            log_infer = q_z.log_prob(z).sum(dim=-1) 
            log_prior = p_z.log_prob(z).sum(dim=-1)
            log_gen = self.cond_ll(x, targets, lengths, z, pad_id)

            batch.append(log_prior.view(-1, 1) + log_gen.view(-1, 1) - log_infer.view(-1, 1))

        iw_ll = log_sum_exp(torch.cat(batch, dim=-1), dim=-1) - math.log(nsamples)
        # for _ in range(nsamples):
        #     z = q_z.rsample()

        return -torch.mean(iw_ll)

    def cond_ll(self, inputs, targets, lengths, z, pad_id):
        init_hidden = self.z2h(z)
        dec_embeds = self.lookup(inputs)
        outputs, _ = self.decode_seq(dec_embeds, lengths, init_hidden=init_hidden)
        outputs = self.fcout(outputs)
        loss = seq_recon_loss(outputs, targets, pad_id).expand(z.size(0), 1) / z.size(0)
        return -loss

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

def detach(states):
    return [state.detach() for state in states]

def test():
    pass

if __name__ == "__main__":
    test()
