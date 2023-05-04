import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
import numpy as np
import spacy 
import random
from torch.utils.tensorboard import SummaryWriter

spacy_ger = spacy.load('de_core_news_sm')
spacy_eng = spacy.load('en_core_web_sm')

def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

text = 'hello how are you iam fine here'
tokenizer_ger(text)
