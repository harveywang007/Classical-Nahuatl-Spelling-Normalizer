# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:02:55 2024

@author: Harvey
"""

import os

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer


DATA_DIR = "./data/"
VOCAB_PATH = DATA_DIR + "to_tokenize/"
TOKENIZER_PATH = "./tokenizers/"

SPECIAL = ["<START>", "<PAD>", "<END>", "<MASK>"]
UNK_TOKEN = "<UNK>"


def create_vocab_path():
    if not os.path.isdir(VOCAB_PATH):
        os.makedirs(VOCAB_PATH)
        
        
def create_tokenizer_path():
    if not os.path.isdir(TOKENIZER_PATH):
        os.makedirs(TOKENIZER_PATH)
    

def prepare_vocab(corpus, output_name):   
    corpus.to_csv(VOCAB_PATH + output_name + ".txt",
                  header=False,
                  sep="\t",
                  index=False)
    

def train_tokenizer(file_name):
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    
    tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    
    trainer = BpeTrainer(min_frequency=10000,
                         initial_alphabet=ByteLevel.alphabet(),
                         special_tokens=SPECIAL)
    
    tokenizer.train([VOCAB_PATH + file_name + ".txt"], trainer)
    
    vocab_size = tokenizer.get_vocab_size()
    
    return tokenizer, vocab_size


def add_tokens(tokenizer):    
    tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    
    tok.bos_token = SPECIAL[0]
    tok.bos_token_id = tokenizer.token_to_id(SPECIAL[0])
    tok.eos_token = SPECIAL[2]
    tok.eos_token_id = tokenizer.token_to_id(SPECIAL[2])
    
    tok.pad_token = SPECIAL[1]
    tok.pad_token_id = tokenizer.token_to_id(SPECIAL[1])
    
    tok.mask_token = SPECIAL[3]
    tok.mask_token_id = tokenizer.token_to_id(SPECIAL[3])
    
    tok.unk_token = UNK_TOKEN
    tok.unk_token_id = tokenizer.token_to_id(UNK_TOKEN)
    
    return tok
    
    
def get_tokenizer(file_name, dir_name):
    tokenizer, vocab_size = train_tokenizer(file_name)
    
    tok = add_tokens(tokenizer)
    
    tok.save_pretrained(TOKENIZER_PATH + dir_name)
    
    return vocab_size


def load_tokenizer(dir_name):
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH + dir_name)