import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.distributions.normal import Normal
from loss import recon_loss
import math

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, code_dim, 
                 dropout, 
                 enc_type='lstm', batch_norm=True):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        if enc_type is 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        elif enc_type is 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        else:
            raise NotImplementedError 
        self.fcmu = nn.Linear(hidden_dim * 2, code_dim)
        self.fclv = nn.Linear(hidden_dim * 2, code_dim)
        self.fcvar = nn.Linear(hidden_dim * 2, 1)
        self.bnmu = nn.BatchNorm1d(code_dim)
        self.bnlv = nn.BatchNorm1d(code_dim)
        self.bn = batch_norm
        self.code_dim = code_dim

    def forward(self, inputs, lengths, dist='normal', fix=True):
        inputs = pack(self.drop(inputs), lengths, batch_first=True)
        _, hn = self.rnn(inputs)
        h = torch.cat(hn, dim=2).squeeze(0)
        if dist == 'normal':
            p_z = Normal(torch.zeros((h.size(0), self.code_dim), device=h.device),
                      (0.5 * torch.zeros((h.size(0), self.code_dim), device=h.device)).exp())
            mu, lv = self.fcmu(h), self.fclv(h)
            if self.bn:
                mu, lv = self.bnmu(mu), self.bnlv(lv)
            return hn, Normal(mu, (0.5 * lv).exp()), p_z

        elif dist == 'vmf':
            mu = self.fcmu(h)
            mu = mu / mu.norm(dim=-1, keepdim=True)
            var = F.softplus(self.fcvar(h)) + 1
            if fix:
                var = torch.ones_like(var) * 80
            return hn, VonMisesFisher(mu, var), HypersphericalUniform(self.code_dim - 1, device=mu.device)
        else:
            raise NotImplementedError


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, code_dim, 
                 dropout, 
                 de_type='lstm'):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        if de_type is 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        elif de_type is 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        else:
            raise NotImplementedError

    def forward(self, inputs, lengths=None, init_hidden=None):
        inputs = self.drop(inputs)
        if lengths is not None:
            inputs = pack(inputs, lengths, batch_first=True)
        outputs, hidden = self.rnn(inputs, init_hidden)
        if lengths is not None:
            outputs, _ = unpack(outputs, batch_first=True)
        outputs = self.drop(outputs)
        return outputs, hidden
    
class DecoderCNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, code_dim, dropout):
        super().__init__()

    def forward(self, inputs, lengths=None, init_hidden=None):
        pass

class LstmVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, code_dim,
                 dropout,
                 flow_prior=False, centers=None, enc_type='lstm', de_type='lstm', dist='normal', fix=True, device=None):
        super().__init__()
        self.device = device
        # apply flows over prior
        self.flow_prior = flow_prior
        # centers
        self.centers = centers
        self.dist = dist
        self.fix = fix
        self.de_type = de_type
        self.code_dim = code_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.encoder = Encoder(embed_dim, hidden_dim, code_dim, dropout, enc_type)
        self.decoder = Decoder(embed_dim, hidden_dim, code_dim, dropout, de_type)
        self.decoder_cnn = DecoderCNN(embed_dim, hidden_dim, code_dim, dropout)
        self.fc = nn.Linear(code_dim, hidden_dim * 2)   # used to map latent space to hidden
        self.fcout = nn.Linear(hidden_dim, vocab_size)  # output layer

        self.flow = None

    def add_flow(self, flow):
        self.flow = flow
    
    def standard_normal(self, size, device):
        p_z = Normal(torch.zeros((size, self.code_dim), device=self.device),
                      (0.5 * torch.zeros((size, self.code_dim), device=self.device)).exp())
        return p_z
    
    def forward(self, inputs, lengths, pad_id):
        batch_size = inputs.size(0)
        enc_embeds = self.embed(inputs)
        # setup prior
        # p_z = Normal(torch.zeros((batch_size, self.code_dim), device=inputs.device),
        #               (0.5 * torch.zeros((batch_size, self.code_dim), device=inputs.device)).exp())
        hn, q_z, p_z = self.encoder(enc_embeds, lengths, self.dist, self.fix)
        if self.training:
            z = q_z.rsample()
        else:
            z = q_z.mean
        
        # set up the flow
        sum_log_jacobian = torch.zeros_like(z)
        # initialize latent codes
        z0 = z
        sum_penalty = torch.zeros(1).to(z.device)
        if self.flow is not None:
            z, sum_log_jacobian, sum_penalty = self.flow(z, self.centers)

        init_hidden = torch.tanh(self.fc(z)).unsqueeze(0)
        init_hidden = [hn.contiguous() for hn in torch.chunk(init_hidden, 2, 2)]
        dec_embeds = self.embed(inputs)
        outputs, _ = self.decoder(dec_embeds, lengths, init_hidden=init_hidden)
        outputs = self.fcout(outputs)

        if self.flow_prior:
            z_0 = p_z.sample()
            z_k, _, _ = self.flow_prior(z_0)

        return q_z, p_z, z, outputs, sum_log_jacobian, sum_penalty, z0

    def generate(self, z, max_length, sos_id):
        batch_size = z.size(0)
        generated = torch.zeros((batch_size, max_length), dtype=torch.long, device=z.device)
        dec_inputs = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=z.device)
        hidden = self.z2h(z)
        for k in range(max_length):
            dec_emb = self.lookup(dec_inputs)
            outputs, hidden = self.decoder(dec_emb, init_hidden=hidden)
            outputs = self.fcout(outputs)
            dec_inputs = outputs.max(2)[1]
            generated[:, k] = dec_inputs[:, 0].clone()
        return generated

    def reconstruct(self, inputs, lengths, max_length,
                    sos_id, sample=True):
        enc_embeds = self.embed(inputs)
        _, q_z = self.encode(enc_embeds, lengths)
        z = q_z.sample() if sample else q_z.mean
        return self.generate(z, max_length, sos_id)

    def sample(self, num_samples, max_length, sos_id):
        prior = self.standard_normal(num_samples)
        z = prior.sample()
        return self.generate(z, max_length, sos_id)
    
    def interpolate(self, max_length, sos_id):
        prior = self.standard_normal(1)
        z0 = prior.sample()
        z1 = prior.sample()
        z0_r, _, _ = self.flow(z0, self.centers)
        z1_r, _, _ = self.flow(z1, self.centers)
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
            sum_log_jacobian = torch.zeros(1).to(z.device)
            z0 = z
            if self.flow is not None:
                z, sum_log_jacobian, _ = self.flow(z, self.centers)

            log_infer = q_z.log_prob(z0).sum(dim=-1) + sum_log_jacobian
            log_prior = p_z.log_prob(z0).sum(dim=-1)
            log_gen = self.cond_ll(x, targets, lengths, z, pad_id)

            batch.append(log_prior.view(-1, 1) + log_gen.view(-1, 1) - log_infer.view(-1, 1))

        iw_ll = log_sum_exp(torch.cat(batch, dim=-1), dim=-1) - math.log(nsamples)
        return -torch.mean(iw_ll)

    def cond_ll(self, inputs, targets, lengths, z, pad_id):
        init_hidden = torch.tanh(self.fc(z)).unsqueeze(0)
        init_hidden = [hn.contiguous() for hn in torch.chunk(init_hidden, 2, 2)]
        dec_embeds = self.embed(inputs)
        outputs, _ = self.decoder(dec_embeds, lengths, init_hidden=init_hidden)
        outputs = self.fcout(outputs)
        loss = recon_loss(outputs, targets, pad_id).expand(z.size(0), 1) / z.size(0)
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