# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:02:55 2024

@author: Harvey Wang
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
    """Creates a path to store tokenizer vocabulary files."""
    if not os.path.isdir(VOCAB_PATH):
        os.makedirs(VOCAB_PATH)
        
        
def create_tokenizer_path():
    """Creates a path to store the tokenizer."""
    if not os.path.isdir(TOKENIZER_PATH):
        os.makedirs(TOKENIZER_PATH)
    

def prepare_vocab(corpus, output_name):
    """From the corpus, creates a text file containing the examples to train the tokenizer."""
    corpus.to_csv(VOCAB_PATH + output_name + ".txt",
                  header=False,
                  sep="\t",
                  index=False)
    

def train_tokenizer(file_name):
    """
    Trains a BPE tokenizer for a language.
    
    Args:
        file_name (str): The name of the file containing the vocabulary.
    Returns:
        tokenizer (Tokenizer): The trained tokenizer for the language.
        vocab_size (int): The number of tokens.
    """
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    
    tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    
    # The minimum frequency is given in lieu of a vocabulary size
    trainer = BpeTrainer(min_frequency=10000,
                         initial_alphabet=ByteLevel.alphabet(),
                         special_tokens=SPECIAL)
    
    tokenizer.train([VOCAB_PATH + file_name + ".txt"], trainer)
    
    # The vocab size is taken from the trained tokenizer
    vocab_size = tokenizer.get_vocab_size()
    
    return tokenizer, vocab_size


def add_tokens(tokenizer):
    """
    Wraps the tokenizer into a more useable class.
    Adds the tokens from the trained tokenizer into the wrapped tokenizer.
    
    Args:
        tokenizer (Tokenizer): The trained tokenizer for the language.
    Returns:
        tok (PreTrainedTokenizerFast): The wrapped tokenizer for the language.
    """
    # The tokenizer is wrapped in the PreTrainedTokenizerFast class
    # This class allows encoding and decoding to be implemented, among other things.
    tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    
    # The <START> token is added to the wrapped tokenizer
    tok.bos_token = SPECIAL[0]
    tok.bos_token_id = tokenizer.token_to_id(SPECIAL[0])
    # The <END> token is added into the wrapped tokenizer
    tok.eos_token = SPECIAL[2]
    tok.eos_token_id = tokenizer.token_to_id(SPECIAL[2])
    
    # The <PAD> token is added into the wrapped tokenizer
    tok.pad_token = SPECIAL[1]
    tok.pad_token_id = tokenizer.token_to_id(SPECIAL[1])
    
    # The <MASK> token is added into the wrapped tokenizer
    tok.mask_token = SPECIAL[3]
    tok.mask_token_id = tokenizer.token_to_id(SPECIAL[3])
    
    # The <UNK> (unknown) token is added into the wrapped tokenizer
    tok.unk_token = UNK_TOKEN
    tok.unk_token_id = tokenizer.token_to_id(UNK_TOKEN)
    
    return tok
    
    
def get_tokenizer(file_name, dir_name):
    """
    Creates and saves the tokenizer for a language.
    
    Args:
        file_name (str): The name of the file containing the vocabulary.
        dir_name (str): The folder to save the tokenizer.
    Returns:
        vocab_size (int): The size of the tokenizer's vocabulary.
    """
    # The BPE tokenizer is trained
    tokenizer, vocab_size = train_tokenizer(file_name)
    
    # The tokenizer is wrapped
    # The tokens are added to the wrapped tokenizer
    tok = add_tokens(tokenizer)
    
    # The wrapped tokenizer is saved
    tok.save_pretrained(TOKENIZER_PATH + dir_name)
    
    return vocab_size


def load_tokenizer(dir_name):
    """Loads the wrapped and saved tokenizer for use."""
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH + dir_name)
