import argparse
import time
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model import LstmVAE, MultiNormalVAE
from data import Corpus, get_iterator, PAD_TOKEN
from data_yahoo import CorpusYahoo, get_iterator_Yahoo
from loss import seq_recon_loss, bow_recon_loss
from loss import total_kld
from loss import kld_decomp
from loss import compute_mmd


parser = argparse.ArgumentParser(description='Text VAE')
parser.add_argument('--data', type=str, default='data/ptb',
                    help="location of the data folder")
parser.add_argument('--max_vocab', type=int, default=20000,
                    help="maximum vocabulary size for the input")
parser.add_argument('--max_length', type=int, default=200,
                    help="maximum sequence length for the input")
parser.add_argument('--embed_size', type=int, default=200,
                    help="size of the word embedding")
parser.add_argument('--label_embed_size', type=int, default=8,
                    help="size of the label embedding")
parser.add_argument('--hidden_size', type=int, default=200,
                    help="number of hidden units for RNN")
parser.add_argument('--code_size', type=int, default=32,
                    help="latent code dimension")
parser.add_argument('--epochs', type=int, default=48,
                    help="maximum training epochs")
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help="batch size")
parser.add_argument('--dropout', type=float, default=0.2,
                    help="dropout applied to layers (0 = no dropout)")
parser.add_argument('--decomp', action='store_true',
                    help="use tc decomposition kl")
parser.add_argument('--alpha', type=float, default=1.0,
                    help="weight of the mutual information term")
parser.add_argument('--beta', type=float, default=1.0,
                    help="weight of the total correlation term")
parser.add_argument('--gamma', type=float, default=1.0,
                    help="weight of the dimension-wise kl term")
parser.add_argument('--c', type=float, default=0,
                    help="weight of the bow loss. default 0 (no bow loss)")
parser.add_argument('--lr', type=float, default=1e-3,
                    help="learning rate")
parser.add_argument('--wd', type=float, default=0,
                    help="weight decay used for regularization")
parser.add_argument('--epoch_size', type=int, default=2000,
                    help="number of training steps in an epoch")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed")
parser.add_argument('--kla', action='store_true',
                    help="use kl annealing")
parser.add_argument('--copa', action='store_true',
                    help="use copula annealing")  
parser.add_argument('--recopa', action='store_true', help="rverse copula annealing")
parser.add_argument('--copat', type=float, default=0.5, help="threshold for copula annealing")                  
parser.add_argument('--nocuda', action='store_true',
                    help="do not use CUDA")
parser.add_argument('--save_name', type=str, default="trained_model.pt")
parser.add_argument('--loss_name', type=str, default="copula_loss.pt")
parser.add_argument('--load', action='store_true', 
                    help="load pre-trained model")
parser.add_argument('--copula', action='store_true', help="use gaussian copula")
parser.add_argument('--factor', type=str, default='ldl')
parser.add_argument('--mmd', action='store_true', help="use mmd distance")
parser.add_argument('--method', type=str, default='ldl')
parser.add_argument('--cw', type=float, default=0.5, help="weight of copula density + marginal density")
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--multi', action='store_true', help="use multi GPU")
parser.add_argument('--nsamples', type=int, default=20)
parser.add_argument('--diag', action='store_true', help='use diagonal posterior')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)


def evaluate(data_iter, model, pad_id):
    model.eval()
    data_iter.init_epoch()
    size = len(data_iter.data())
    seq_loss = 0.0
    bow_loss = 0.0
    kld = 0.0
    mi = 0.0
    tc = 0.0
    dwkl = 0.0
    seq_words = 0
    bow_words = 0
    log_c = 0
    mmd_loss = 0
    if args.multi:
        model = model.module
    for batch in data_iter:
        texts, lengths = batch.text
        batch_size = texts.size(0)
        inputs = texts[:, :-1].clone()
        targets = texts[:, 1:].clone()
        (posterior, prior, z, seq_outputs,
         bow_outputs, bow_targets, log_copula, log_marginals) = model(inputs, lengths-1, pad_id)
        batch_seq = seq_recon_loss(
            seq_outputs, targets, pad_id
        )
        batch_bow = bow_recon_loss(bow_outputs, bow_targets)
        # kld terms are averaged across the mini-batch
        batch_kld = total_kld(posterior, prior) / batch_size
        batch_mi, batch_tc, batch_dwkl = kld_decomp(
             posterior, prior, z
        )

        prior_samples = torch.randn(args.batch_size, z.size(-1)).to(z.device)
        mmd = compute_mmd(prior_samples, z)

        
        iw_nll = model.iw_nll(posterior, prior, inputs, targets, lengths-1, pad_id, args.nsamples)
        
        seq_loss += batch_seq.item() / size
        # bow_loss += batch_bow.item() / size  
        bow_loss += iw_nll.item() * batch_size / size
        kld += batch_kld.item() * batch_size / size
        mi += batch_mi.item() * batch_size / size
        tc += batch_tc.item() * batch_size / size
        dwkl += batch_dwkl.item() * batch_size / size
        seq_words += torch.sum(lengths-1).item()
        bow_words += torch.sum(bow_targets)
        log_c += torch.sum(log_copula).item() / size
        mmd_loss += mmd.item() / size
    # seq_ppl = math.exp(seq_loss * size / seq_words)
    seq_ppl = math.exp((seq_loss + kld - log_c) * size / seq_words)
    bow_ppl = math.exp(bow_loss * size / bow_words)
    return (seq_loss, bow_loss, kld,
            mi, tc, dwkl,
            seq_ppl, bow_ppl, log_c, mmd_loss)


def train(data_iter, model, pad_id, optimizer, epoch):
    model.train()
    data_iter.init_epoch()
    batch_time = AverageMeter()
    size = min(len(data_iter.data()), args.epoch_size * args.batch_size)
    seq_loss = 0.0
    bow_loss = 0.0
    kld = 0.0
    mi = 0.0
    tc = 0.0
    dwkl = 0.0
    seq_words = 0
    bow_words = 0
    log_c = 0
    mmd_loss = 0
    end = time.time()
    # if args.multi:
    #     model = model.module
    for i, batch in enumerate(tqdm(data_iter)):
        if i == args.epoch_size:
            break
        texts, lengths = batch.text
        batch_size = texts.size(0)
        inputs = texts[:, :-1].clone()
        targets = texts[:, 1:].clone()
        posterior, prior, z, seq_outputs, bow_outputs, bow_targets, log_copula, log_marginals = model(inputs, lengths-1, pad_id)
        # print('debug')
        # posterior, prior = model.get_dist()
        batch_seq = seq_recon_loss(
            seq_outputs, targets, pad_id
        )
        batch_bow = bow_recon_loss(bow_outputs, bow_targets)
        # kld terms are averaged across the mini-batch
        batch_kld = total_kld(posterior, prior) / batch_size
        batch_mi, batch_tc, batch_dwkl = kld_decomp(
             posterior, prior, z
        )

        prior_samples = torch.randn(args.batch_size, z.size(-1)).to(z.device)
        mmd = compute_mmd(prior_samples, z)

        kld_weight = weight_schedule(args.epoch_size * (epoch - 1) + i) if args.kla else 1.
        # kld_weight = min(1./10 * epoch, 1) if args.kla else 1
        if args.copa is True:
            # copula_weight = weight_schedule(args.epoch_size * (epoch- 1 )+ i)
            copula_weight = min(1./args.epochs * epoch, 1)
            if args.copat is not None:
                copula_weight = copula_weight if copula_weight < args.copat else args.copat
        elif args.recopa is True:
            copula_weight = weight_schedule(args.epoch_size * (epoch- 1 )+ i)
            if args.copat is not None:
                copula_weight = 1 - copula_weight if copula_weight > args.copat else args.copat
            else:
                copula_weight = 1 - copula_weight
        else:
            copula_weight = args.cw
        optimizer.zero_grad()
        if args.decomp:
            kld_term = args.alpha * batch_mi + args.beta * batch_tc +\
                       args.gamma * batch_dwkl
        else:
            kld_term = batch_kld

        if args.copula is True:
            loss = (batch_seq + args.c * batch_bow) / batch_size + kld_weight * kld_term - copula_weight * (log_copula.sum() + 1* log_marginals.sum()) / batch_size
        else:
            loss = (batch_seq + args.c * batch_bow) / batch_size + kld_weight * kld_term
        if args.mmd is True:
            loss += (2.5 - kld_weight) * mmd
        
        loss.backward()
        optimizer.step()

        seq_loss += batch_seq.item() / size
        bow_loss += batch_bow.item() / size  
        kld += batch_kld.item() * batch_size / size
        mi += batch_mi.item() * batch_size / size
        tc += batch_tc.item() * batch_size / size
        dwkl += batch_dwkl.item() * batch_size / size
        seq_words += torch.sum(lengths-1).item()
        bow_words += torch.sum(bow_targets)
        log_c += torch.sum(log_copula).item() / size
        mmd_loss += mmd.item() / size

        batch_time.update(time.time() - end)
    
    # seq_ppl = math.exp(seq_loss * size / seq_words)
    seq_ppl = math.exp((seq_loss + kld - log_c) * size / seq_words)
    # bow_ppl = math.exp(bow_loss * size / bow_words)
    bow_ppl = math.exp(bow_loss * size / seq_words)        
    return (seq_loss, bow_loss, kld,
            mi, tc, dwkl,
            seq_ppl, bow_ppl, log_c, mmd_loss, batch_time)


def interpolate(i, start, duration):
    return max(min((i - start) / duration, 1), 0)


def weight_schedule(t):
    """Scheduling of the KLD annealing weight. """
    return interpolate(t, 6000, 40000)


def get_savepath(args):
    dataset = args.data.rstrip('/').split('/')[-1]
    path = './saves/emb{0:d}.hid{1:d}.z{2:d}{3}{4}.{5}.pt'.format(
        args.embed_size, args.hidden_size, args.code_size,
        '.wd{:.0e}'.format(args.wd) if args.wd > 0 else '',
        '.kla' if args.kla else '', dataset)
    return path

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

def main(args):
    print("Loading data")
    dataset = args.data.rstrip('/').split('/')[-1]
    if dataset in ['yahoo']:
        with_label = True
    else:
        with_label = False
    if dataset in ['yahoo']:
        corpus = CorpusYahoo(
        args.data, max_vocab_size=args.max_vocab,
        max_length=args.max_length, with_label=with_label
        )
        pad_id = corpus.word2idx['_PAD']
    else:
        corpus = Corpus(
            args.data, max_vocab_size=args.max_vocab,
            max_length=args.max_length, with_label=with_label
        )
        pad_id = corpus.word2idx[PAD_TOKEN]
    vocab_size = len(corpus.word2idx)
    print("\ttraining data size: ", len(corpus.train))
    print("\tvocabulary size: ", vocab_size)
    print("Constructing model")
    print(args)
    device = torch.device('cpu' if args.nocuda else 'cuda')
    torch.cuda.set_device(args.cuda)
    if args.diag:
        model = MultiNormalVAE(
            vocab_size, args.embed_size, args.hidden_size,
            args.code_size, args.dropout, batch_size=args.batch_size,
            decomp=args.method, copula=args.copula
        )
    else:
        model = LstmVAE(
            vocab_size, args.embed_size, args.hidden_size,
            args.code_size, args.dropout, batch_size=args.batch_size, 
            decomp=args.method, copula=args.copula)
    if args.multi:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_loss = None

    train_iter = get_iterator(corpus.train, args.batch_size, True,  device)
    valid_iter = get_iterator(corpus.valid, args.batch_size, False, device)
    test_iter  = get_iterator(corpus.test,  args.batch_size, False, device)

    start_epoch = 0
    if args.load is True:
        start_epoch, model, optimizer, losslogger = load_checkpoint(model, optimizer, device, args.save_name)

    tr_loggers = []
    va_loggers = []
    print("\nStart training")
    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_start_time = time.time()
            (tr_seq_loss, tr_bow_loss, tr_kld,
             tr_mi, tr_tc, tr_dwkl,
             tr_seq_ppl, tr_bow_ppl, tr_log_copula, tr_mmd, batch_time) = train(
                 train_iter, model, pad_id, optimizer, epoch
             )
            (va_seq_loss, va_bow_loss, va_kld,
             va_mi, va_tc, va_dwkl,
             va_seq_ppl, va_bow_ppl, va_log_copula, va_mmd) = evaluate(
                 valid_iter, model, pad_id
             )
            
            tr_losslogger = {"epoch": epoch,
                          "seq_loss": tr_seq_loss,
                          "bow_loss": tr_bow_loss,
                          "kld": tr_kld,
                          "mutual info": tr_mi,
                          "tc": tr_tc,
                          "dwkl": tr_dwkl,
                          "seq_ppl": tr_seq_ppl,
                          "bow_ppl": tr_bow_ppl,
                          "log_copula": tr_log_copula,
                          "mmd": tr_mmd,
                          "time": batch_time}
            tr_loggers.append(tr_losslogger)

            losslogger = {"epoch": epoch,
                          "seq_loss": va_seq_loss,
                          "bow_loss": va_bow_loss,
                          "kld": va_kld,
                          "mutual info": va_mi,
                          "tc": va_tc,
                          "dwkl": va_dwkl,
                          "seq_ppl": va_seq_ppl,
                          "bow_ppl": va_bow_ppl,
                          "log_copula": va_log_copula,
                          "mmd": va_mmd,
                          "time": batch_time}
            va_loggers.append(losslogger)

            save_checkpoint(model, optimizer, losslogger, args.save_name)
            print('-' * 90)
            meta = "| epoch {:2d} | time {:5.2f}s ".format(epoch, time.time()-epoch_start_time)
            print(meta + "| train loss {:5.2f} {:5.2f} ({:5.2f}) "
                  "| {:5.2f} {:5.2f} {:5.2f} "
                  "| train ppl {:5.2f} {:5.2f} | log copula {:5.2f} | mmd {:5.2f}"
                  "| Time {batch_time.val:5.2f} ({batch_time.avg:5.2f})\t".format(
                      tr_seq_loss, tr_bow_loss, tr_kld,
                      tr_mi, tr_tc, tr_dwkl,
                      tr_seq_ppl, tr_bow_ppl, tr_log_copula, tr_mmd, batch_time=batch_time))
            print(len(meta)*' ' + "| valid loss {:5.2f} {:5.2f} ({:5.2f}) "
                  "| {:5.2f} {:5.2f} {:5.2f} "
                  "| valid ppl {:5.2f} {:5.2f} | valid log copula {:5.2f} | valid mmd {:5.2f}"
                  "| joint NLL {:5.2f}".format(
                      va_seq_loss, va_bow_loss, va_kld,
                      va_mi, va_tc, va_dwkl,
                      va_seq_ppl, va_bow_ppl, va_log_copula, va_mmd, va_seq_loss+va_kld-va_log_copula), flush=True)
            epoch_loss = va_seq_loss + va_kld
            if best_loss is None or epoch_loss < best_loss:
                best_loss = epoch_loss
                # with open(get_savepath(args), 'wb') as f:
                #     torch.save(model, f)
                
    except KeyboardInterrupt:
        print('-' * 90)
        print('Exiting from training early')

    save_logger(tr_loggers, va_loggers, args.loss_name)

    # with open(get_savepath(args), 'rb') as f:
    #     model = torch.load(f)
    (te_seq_loss, te_bow_loss, te_kld,
     te_mi, te_tc, te_dwkl,
     te_seq_ppl, te_bow_ppl, te_log_copula, te_mmd) = evaluate(test_iter, model, pad_id)
    print('=' * 90)
    print("| End of training | test loss {:5.2f} {:5.2f} ({:5.2f}) "
          "| {:5.2f} {:5.2f} {:5.2f} "
          "| test ppl {:5.2f} {:5.2f}"
          "| test log copula {:5.2f}"
          "| test mmd {:5.2f}"
          "| test nll {:5.2f}".format(
              te_seq_loss, te_bow_loss, te_kld,
              te_mi, te_tc, te_dwkl,
              te_seq_ppl, te_bow_ppl,
               te_log_copula, te_mmd, te_seq_loss+te_kld-te_log_copula))
    print('=' * 90)

    te_losslogger = {"seq_loss": te_seq_loss,
                    "bow_loss": te_bow_loss,
                    "kld": te_kld,
                    "seq_ppl": te_seq_ppl,
                    "bow_ppl": te_bow_ppl,
                    "log_copula": te_log_copula,
                    "mmd": te_mmd,
                    }

def save_checkpoint(model, optimizer, losslogger, filename):
    state = {"model_state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "losslogger": losslogger}
    
    torch.save(state, filename)

def save_logger(tr_lossloggers, va_lossloggers, filename):
    state = {"tr_losslogger": tr_lossloggers,
             "va_losslogger": va_lossloggers}
    torch.save(state, filename)

def load_checkpoint(model, optimizer, device, filename):
    checkpoint = torch.load(filename)
    losslogger = checkpoint["losslogger"]
    start_epoch = losslogger["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    return start_epoch, model, optimizer, losslogger

def load_logger(filename):
    checkpoint = torch.load(filename)
    losslogger = checkpoint["losslogger"]
    return losslogger

if __name__ == "__main__":
    main(args)
