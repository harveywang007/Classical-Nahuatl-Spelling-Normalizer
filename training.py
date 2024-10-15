# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:13:24 2024

@author: Harvey
"""

import os

import torch
import torch.nn as nn
from torch import Tensor
from torchdata.datapipes.iter import IterableWrapper, FileOpener
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from translation.transformer import Seq2SeqTransformer

from timeit import default_timer as timer


DATA_DIR = "./data/"
RAW_DATA_PATH = DATA_DIR + "raw_data/"
CSV_PATH = DATA_DIR + "CSVs/"

SAVED_MODEL_PATH = "./trained_transformers/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPECIAL = ["<START>", "<PAD>", "<END>", "<MASK>"]

NEG_INFTY = -1e9

EMB_SIZE = 256
MAX_LEN = 4096
NUM_HEADS = 4
FFN_HID_DIM = 1024
BATCH_SIZE = 20
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DROPOUT = 0.3


def create_tokens(sentence, tokenizer):
    sentence = tokenizer(sentence)["input_ids"]

    def add_tokens(sentence, tokenizer):
        sentence = torch.cat((torch.tensor(tokenizer(SPECIAL[0])["input_ids"]),
                              torch.tensor(sentence),
                              torch.tensor(tokenizer(SPECIAL[2])["input_ids"])))
        return sentence
    return add_tokens(sentence, tokenizer)


def collate(batch, source_tokenizer, target_tokenizer):    
    sources, inputs, labels = [], [], []

    for source_sentence, target_sentence in batch:
        sources.append(create_tokens(source_sentence, source_tokenizer))
        target = create_tokens(target_sentence, target_tokenizer)
        inputs.append(target[:-1])
        labels.append(target[1:])

    sources = pad_sequence(sources, padding_value=source_tokenizer(SPECIAL[1])["input_ids"][0])
    inputs = pad_sequence(inputs, padding_value=target_tokenizer(SPECIAL[1])["input_ids"][0])
    labels = pad_sequence(labels, padding_value=target_tokenizer(SPECIAL[1])["input_ids"][0])
    return sources, inputs, labels


def square_masks(size):
    mask = torch.ones((size, size), device=DEVICE)
    mask = torch.triu(mask).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(NEG_INFTY)).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(source, target, source_tokenizer, target_tokenizer):  
    target_mask = square_masks(target.shape[0])
    source_mask = torch.full((source.shape[0], source.shape[0]), False, device=DEVICE)

    source_padding_mask = (source == source_tokenizer(SPECIAL[1])["input_ids"][0]).transpose(0, 1)
    target_padding_mask = (target == target_tokenizer(SPECIAL[1])["input_ids"][0]).transpose(0, 1)
    return source_mask, target_mask, source_padding_mask, target_padding_mask


def train_transformer(source_vocab_size, target_vocab_size,
                      training_path, validation_path,
                      source_tokenizer, target_tokenizer,                      
                      num_epochs, file_name):
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, MAX_LEN,
                                     NUM_HEADS, source_vocab_size, target_vocab_size, FFN_HID_DIM,
                                     DROPOUT)
    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    transformer = transformer.to(DEVICE)
    
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=source_tokenizer(SPECIAL[1])["input_ids"][0])
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001)
    
    train_model(transformer, training_path, validation_path,
                source_tokenizer, target_tokenizer,
                optimizer, loss_func, num_epochs)
    
    if not os.path.isdir(SAVED_MODEL_PATH):
        os.makedirs(SAVED_MODEL_PATH)        
    
    torch.save(transformer.state_dict(), SAVED_MODEL_PATH + file_name + ".pt")
    

def train_model(model, training_path, validation_path,
                source_tokenizer, target_tokenizer,
                optimizer, loss_func, num_epochs):
    for epoch in range(1, num_epochs+1):
        start_time = timer()
        train_loss = train_epoch(model, training_path,
                                 source_tokenizer, target_tokenizer,
                                 optimizer, loss_func)
        end_time = timer()
    
        val_loss = evaluate(model, validation_path,
                            source_tokenizer, target_tokenizer,
                            loss_func)

        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))


def train_epoch(model, training_path,
                source_tokenizer, target_tokenizer,
                optimizer, loss_func):
    model.train()
    losses = 0

    train_iter = IterableWrapper([RAW_DATA_PATH + training_path + ".csv"])
    train_iter = FileOpener(train_iter, mode="b")
    train_iter = train_iter.parse_csv(skip_lines=1, as_tuple=True)

    train_dataloader = DataLoader(train_iter,
                                  batch_size=BATCH_SIZE,
                                  collate_fn=lambda x: collate(x, source_tokenizer, target_tokenizer))

    for source, input, label in train_dataloader:
        source = source.to(DEVICE)
        input = input.to(DEVICE)
        label = label.to(DEVICE)

        source_mask, input_mask, source_padding_mask, input_padding_mask = create_mask(source, input,
                                                                                       source_tokenizer, target_tokenizer)

        logits = model(source, input,
                       source_mask, input_mask,
                       source_padding_mask, input_padding_mask,
                       source_padding_mask)

        optimizer.zero_grad()

        loss = loss_func(logits.reshape(-1, logits.shape[-1]), label.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))
    

def evaluate(model, validation_path,
             source_tokenizer, target_tokenizer,
             loss_func):
    model.eval()
    losses = 0

    val_iter = IterableWrapper([RAW_DATA_PATH + validation_path + ".csv"])
    val_iter = FileOpener(val_iter, mode="b")
    val_iter = val_iter.parse_csv(skip_lines=1, as_tuple=True)

    val_dataloader = DataLoader(val_iter,
                                batch_size=BATCH_SIZE,
                                collate_fn=lambda x: collate(x, source_tokenizer, target_tokenizer))

    for source, input, label in val_dataloader:
        source = source.to(DEVICE)
        input = input.to(DEVICE)
        label = label.to(DEVICE)

        source_mask, input_mask, source_padding_mask, input_padding_mask = create_mask(source, input,
                                                                                       source_tokenizer, target_tokenizer)

        logits = model(source, input,
                       source_mask, input_mask,
                       source_padding_mask, input_padding_mask, source_padding_mask)

        loss = loss_func(logits.reshape(-1, logits.shape[-1]), label.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))

        
def main():
    """
    create_vocab_path()
    
    prepare_vocab(corpus, "unregularized_to_tokenize")
    prepare_vocab(corpus, "regularized_to_tokenize")
    
    create_tokenizer_path()
    
    unreg_vocav_size = get_tokenizer(file_name, dir_name)
    reg_vocav_size = get_tokenizer(file_name, dir_name)
    
    unreg_tokenizer = load_tokenizer(dir_name)
    reg_tokenizer = load_tokenizer(dir_name)
    """
    pass


if __name__ == "__main__":
    main()