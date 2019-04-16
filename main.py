import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import LstmVAE
from loss import (recon_loss, total_kld, flow_kld, compute_mmd, 
                    mutual_info, mutual_info_flow,
                    compute_nll)
from stochastic.flow import NormalizingFlows
from data import Corpus, get_iterator, PAD_TOKEN, SOS_TOKEN

parser = argparse.ArgumentParser(description='RNF WAE')
parser.add_argument('--data', type=str, default='~/data/ptb',
                    help="location of the data folder")
parser.add_argument('--max_vocab', type=int, default=20000,
                    help="maximum vocabulary size for the input")
parser.add_argument('--max_length', type=int, default=200,
                    help="maximum sequence length for the input")
parser.add_argument('--embed_dim', type=int, default=200,
                    help="dim of the word embedding")
parser.add_argument('--hidden_dim', type=int, default=200,
                    help="number of hidden units for RNN")
parser.add_argument('--code_dim', type=int, default=32,
                    help="latent code dimension")
parser.add_argument('--dist', type=str, default='normal', help="choice of distribution")
parser.add_argument('--fix', action='store_true', help='use fixed temp in vmf')
parser.add_argument('--enc_type', type=str, default='lstm',
                    help="type of encoder")
parser.add_argument('--de_type', type=str, default='lstm',
                    help="type of decoder")
parser.add_argument('--epochs', type=int, default=48,
                    help="maximum training epochs")
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help="batch size")
parser.add_argument('--dropout', type=float, default=0.2,
                    help="dropout applied to layers (0 = no dropout)")
parser.add_argument('--lr', type=float, default=1e-3,
                    help="learning rate")
parser.add_argument('--beta', type=float, default=0.9)
parser.add_argument('--epoch_size', type=int, default=2000,
                    help="number of training steps in an epoch")
parser.add_argument('--kla', action='store_true',
                    help="use kl annealing")
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--t', type=float, default=0.8)
parser.add_argument('--loss_type', type=str, default='entropy')
parser.add_argument('--save_name', type=str, default="trained_model.pt")
parser.add_argument('--load', action='store_true', 
                    help="load pre-trained model")
parser.add_argument('--flow', action='store_true', help="use NF")
parser.add_argument('--center', action='store_true', help="use RNF")
parser.add_argument('--n_flows', type=int, default=1, help="number of flows")
parser.add_argument('--mmd', action='store_true', help="use mmd instead of kl")
parser.add_argument('--mmd_w', type=float, default=2.5, help="mmd ceiling")
parser.add_argument('--nokld', action='store_true')
parser.add_argument('--sent_length', type=int, default=30, help="length of generated samples")
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--nsamples', type=int, default=500)
parser.add_argument('--kernel', type=str, default='g')
parser.add_argument('--reg', type=str, default='g')
parser.add_argument('--band', type=float, default=0.5)
parser.add_argument('--iw', action='store_true')
parser.add_argument('--test_log_name', type=str, default='log.txt')
parser.add_argument('--save', action='store_true')
args = parser.parse_args()

torch.manual_seed(123)
random.seed(123)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class PositiveClipper(object):
    def __init__(self, frequency=1):
        self.frequency = frequency
    
    def __call__(self, module):
        if hasattr(module, 'band'):
            module.band.data.clamp_(0)

def idx_to_word(idx, idx2word, pad_idx):
    sent_str = [str()]*len(idx)
    for i, sent in enumerate(idx):
        for word_idx in sent:
            if word_idx == pad_idx:
                break
            sent_str[i] += idx2word[word_idx] + " "
        sent_str[i] = sent_str[i].strip()
    
    return sent_str

def save_checkpoint(model, optimizer, epoch, filename):
    state = {"model_state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "epoch": epoch}
    
    torch.save(state, filename)

def load_checkpoint(model, optimizer, device, filename):
    checkpoint = torch.load(filename)
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    return start_epoch, model, optimizer

def weight_schedule(t, start=6000, end=40000):
    return max(min((t - 6000) / 40000, 1), 0)

def run(args, data_iter, model, pad_id, optimizer, epoch, train=True):
    if train is True:
        model.train()
    else:
        model.eval()
    data_iter.init_epoch()
    batch_time = AverageMeter()
    size =  min(len(data_iter.data()), args.epoch_size * args.batch_size)
    re_loss = 0
    kl_divergence = 0
    flow_kl_divergence = 0
    mutual_information1, mutual_information2 = 0, 0
    seq_words = 0
    mmd_loss = 0
    negative_ll = 0
    iw_negative_ll = 0
    sum_log_j = 0
    start = time.time()
    end = time.time()
    for i, batch in enumerate(data_iter):
        if i == args.epoch_size:
            break
        texts, lengths = batch.text
        batch_size = texts.size(0)
        inputs = texts[:, :-1].clone()
        targets = texts[:, 1:].clone()
        q_z, p_z, z, outputs, sum_log_jacobian, penalty, z0 = model(inputs, lengths-1, pad_id)
        if args.loss_type == 'entropy':
            reloss = recon_loss(outputs, targets, pad_id, id=args.loss_type)
        else:
            reloss = recon_loss(inputs, outputs, pad_id, id=args.loss_type)
        
        kld = total_kld(q_z, p_z)
        
        if args.flow:
            f_kld = flow_kld(q_z, p_z, z, z0, sum_log_jacobian)
        else:
            f_kld = torch.zeros(1)

        mi_z= mutual_info(q_z, p_z, z0)
        nll = compute_nll(q_z, p_z, z, z0, sum_log_jacobian, reloss)
        
        if args.iw:
            iw_nll = model.iw_nll(q_z, p_z, inputs, targets, lengths-1, pad_id, args.nsamples)
        else:
            iw_nll = torch.zeros(1)
        
        if args.flow:
            mi_flow = mutual_info_flow(q_z, p_z, z, z0, sum_log_jacobian)
        else:
            mi_flow = torch.zeros(1).to(z.device)

        mmd = torch.zeros(1).to(z.device)
        kld_weight = weight_schedule(args.epoch_size * (epoch - 1) + i) if args.kla else 1.
        if args.mmd:
            # prior_samples = torch.randn(z.size(0), z.size(-1)).to(z.device)
            mmd = compute_mmd(p_z, q_z, args.kernel)
        if kld_weight > args.t:
            kld_weight = args.t
        if args.nokld:
            kld_weight = 0
        
        if train is True:
            optimizer.zero_grad()
            if args.flow:
                # loss = reloss / batch_size + kld_weight * (kld - torch.sum(sum_log_jacobian) + torch.sum(penalty)) / batch_size + (args.mmd_w - kld_weight) * mmd
                loss = reloss / batch_size + kld_weight * (q_z.log_prob(z0).sum() - p_z.log_prob(z).sum()) / batch_size - (torch.sum(sum_log_jacobian) - torch.sum(penalty)) / batch_size + (args.mmd_w - kld_weight) * mmd
            else:
                loss = (reloss + kld_weight * kld) / batch_size + (args.mmd_w - kld_weight) * mmd

            loss.backward()
            optimizer.step()
    
        re_loss += reloss.item() / size
        kl_divergence += kld.item() / size
        flow_kl_divergence += f_kld.item() * batch_size / size
        mutual_information1 += mi_z.item() * batch_size / size
        mutual_information2 += mi_flow.item() * batch_size / size
        seq_words += torch.sum(lengths-1).item()
        mmd_loss += mmd.item() * batch_size / size
        negative_ll += nll.item() * batch_size / size
        iw_negative_ll += iw_nll.item() * batch_size / size
        sum_log_j += torch.sum(sum_log_jacobian).item() / size
        batch_time.update(time.time() - end)
    
    if kl_divergence > 100:
        kl_divergence = 100
        flow_kl_divergence = 100
    if args.iw:
        nll_ppl = math.exp(iw_negative_ll * size / seq_words)
    else:
        nll_ppl = math.exp(negative_ll * size / seq_words)
    
    return re_loss, kl_divergence, flow_kl_divergence, mutual_information1, mutual_information2, mmd_loss, nll_ppl, negative_ll, iw_negative_ll, sum_log_j, start, batch_time

def main(args):
    print("Loading data")
    dataset = args.data.rstrip('/').split('/')[-1]
    corpus = Corpus(
        args.data, max_vocab_size=args.max_vocab,
        max_length=args.max_length
    )
    pad_id = corpus.word2idx[PAD_TOKEN]
    sos_id = corpus.word2idx[SOS_TOKEN]
    vocab_size = len(corpus.word2idx)
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device_id)
    cent_name = dataset + "_centers.pt"
    centers = None
    if args.center:
        centers = torch.load(cent_name).to(device)
        # centers.requires_grad = False
        centers = centers.detach()
    model = LstmVAE(vocab_size, args.embed_dim, args.hidden_dim, args.code_dim, args.dropout, 
                    centers=centers, enc_type=args.enc_type, de_type=args.de_type, dist=args.dist, fix=args.fix, device=device).to(device)
    if args.flow:
        flow = NormalizingFlows(args.code_dim, n_flows=args.n_flows, reg=args.reg, band=args.band).to(device)
        model.add_flow(flow)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta, 0.999), weight_decay=args.wd)

    train_iter = get_iterator(corpus.train, args.batch_size, True,  device)
    valid_iter = get_iterator(corpus.valid, args.batch_size, False, device)
    test_iter  = get_iterator(corpus.test,  args.batch_size, False, device)

    start_epoch = 1
    if args.load is True:
        start_epoch, model, optimizer= load_checkpoint(model, optimizer, device, args.save_name)

    print("\nStart training")
    try:
        for epoch in range(start_epoch, args.epochs+1):
            (re_loss, kl_divergence, flow_kld, mi1, mi2, mmd_loss, nll_ppl, nll, iw_nll, sum_log_j, start, batch_time) = run(args, train_iter, model, pad_id, optimizer, epoch,
                                                                        train=True)
            if args.save:
                save_checkpoint(model, optimizer, epoch, args.save_name)
            print('-' * 90)
            meta = "| epoch {:2d} ".format(epoch)
            print(meta + "| train loss {:5.2f} ({:5.2f}) ({:5.2f}) | train ppl {:5.2f} ({:5.2f} {:5.2f}) | mmd {:5.2f} | mi E {:5.2f} | mi R {:5.2f} | log J {:5.2f}"
                    "| Time {batch_time.val:5.2f} ({batch_time.avg:5.2f})\t".format(re_loss, kl_divergence, flow_kld,
                        nll_ppl, nll, iw_nll, mmd_loss, mi1, mi2, sum_log_j,
                        batch_time=batch_time))
            
            (re_loss, kl_divergence, flow_kld, mi1, mi2, mmd_loss, nll_ppl, nll, iw_nll, sum_log_j, _, _) = run(args, valid_iter, model, pad_id,optimizer, epoch,
                                                                    train=False)
            
            print(len(meta)*' ' + "| valid loss {:5.2f} ({:5.2f}) ({:5.2f}) | valid ppl {:5.2f} ({:5.2f} {:5.2f})"
                    "| mmd {:5.2f} | mi E {:5.2f} | mi R {:5.2f} | log J {:5.2f} \t".format(re_loss, kl_divergence, flow_kld,
                        nll_ppl, nll, iw_nll, mmd_loss, mi1, mi2, sum_log_j, flush=True))

            if dataset in ['yahoo'] and epoch in [15, 35] :
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
    
    except KeyboardInterrupt:
        print('-'*50)
        print('Quit training')

    (re_loss, kl_divergence, flow_kld, mi1, mi2, mmd_loss, nll_ppl, nll, iw_nll, sum_log_j, _, _) = run(args, test_iter, model, pad_id, optimizer, epoch,
                                                                      train=False)
    print('=' * 90)
    print("| Test results | test loss {:5.2f} ({:5.2f}) ({:5.2f}) | test ppl {:5.2f} ({:5.2f} {:5.2f}) | test mmd {:5.2f} | mi E {:5.2f} | mi R {:5.2f} | log J {:5.2f} ".format(
              re_loss, kl_divergence, flow_kld, nll_ppl, nll, iw_nll, mmd_loss, mi1, mi2, sum_log_j))
    print('=' * 90)
    
    with open(args.test_log_name, 'a') as fd:
        print('=' * 90, file=fd)
        print("{} | dist {} | ende {} | em {} | | kla {} | mmd {} | flow {} | center {} | n flow {} | ker {} | reg {} | band {} | t {} | mmd w {} | iw {} | gpu {} | log {} |".format(dataset, args.dist, args.de_type, args.embed_dim, args.kla, args.mmd, args.flow, args.center, args.n_flows, 
                                    args.kernel, args.reg, args.band, args.t, args.mmd_w, args.iw, args.device_id, args.test_log_name), file=fd)
        print('-'*90, file=fd)
        print("| Test results | test loss {:5.2f} ({:5.2f}) ({:5.2f}) | test ppl {:5.2f} ({:5.2f} {:5.2f}) | test mmd {:5.2f} | mi E {:5.2f} | mi R {:5.2f} | log J {:5.2f}".format(
              re_loss, kl_divergence, flow_kld, nll_ppl, nll, iw_nll, mmd_loss, mi1, mi2, sum_log_j), file=fd)
        print('=' * 90, file=fd)
    

    # generated = model.sample(10, args.sent_length, sos_id=sos_id)
    # sent_str = idx_to_word(generated, idx2word=corpus.idx2word, pad_idx=pad_id)
    # print(*sent_str, sep='\n')

if __name__ == "__main__":
    main(args)
    



