import torch
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator
import torchtext.datasets as datasets
import torchtext.data as data
import spacy

PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'


class Corpus(object):
    def __init__(self, datadir, min_freq=2, max_vocab_size=None, max_length=None, with_label=False):
        # tokenize = data.get_tokenizer('spacy')
        tokenize = lambda x: x.split()
        if max_length is None:
            preprocessing = None
        else:
            preprocessing = lambda x: x[:max_length]
        TEXT = Field(
            sequential=True, tokenize=tokenize,
            init_token=SOS_TOKEN, eos_token=EOS_TOKEN,
            pad_token=PAD_TOKEN, unk_token=UNK_TOKEN,
            preprocessing=preprocessing, lower=True,
            include_lengths=True, batch_first=True
        )
        LABEL = Field(sequential=False, use_vocab=False)
        if with_label:
            datafields = [('label', LABEL), ('text', TEXT)]
        else:
            datafields = [('text', TEXT)]
        self.train, self.valid, self.test = TabularDataset.splits(
            path=datadir, train='train.txt', validation='valid.txt',
            test='test.txt', format='tsv', fields=datafields
        )
        TEXT.build_vocab(
            self.train, self.valid, max_size=max_vocab_size,
            min_freq=min_freq
        )
        self.word2idx = TEXT.vocab.stoi
        self.idx2word = TEXT.vocab.itos
        self.with_label = with_label
    
        
def get_iterator(dataset, batch_size, train, device):
    sort_key = lambda x: len(x.text)
    dataset_iter = BucketIterator(
        dataset, batch_size=batch_size, device=device,
        train=train, shuffle=train, repeat=False,
        sort_key=sort_key, sort_within_batch=True
    )
    return dataset_iter