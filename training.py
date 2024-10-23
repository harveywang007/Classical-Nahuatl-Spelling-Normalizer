# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 14:13:24 2024

@author: Harvey Wang
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
    """Tokenizes a sentence"""
    sentence = tokenizer(sentence)["input_ids"]

    def add_tokens(sentence, tokenizer):
        """Adds the <START> and <END> tokens to the tokenized sentence"""
        sentence = torch.cat((torch.tensor(tokenizer(SPECIAL[0])["input_ids"]), # The <START> token
                              torch.tensor(sentence), # The tokenized sentence
                              torch.tensor(tokenizer(SPECIAL[2])["input_ids"]))) # The <END> token
        return sentence
    return add_tokens(sentence, tokenizer)


def collate(batch, source_tokenizer, target_tokenizer):
    """
    Tokenizes and pads batches of tokenized source and target sentences.
    
    Args:
        batch (DataFrame): The batch of source and target sentence pairs.
        source_tokenizer (Tokenizer): The tokenizer for the source language.
        target_tokenizer (Tokenizer): The tokenizer for the target language.
    
    Returns:
        sources (torch.Tensor): The padded tensors of source sentences.
        inputs (torch.Tensor): The padded tensors of target sentences to learn inputs.
        labels (torch.Tensor): The padded tensors of target sentences to learn the next tokens.
    """
    sources, inputs, labels = [], [], []

    for source_sentence, target_sentence in batch:
        # The source sentences are prepared
        sources.append(create_tokens(source_sentence, source_tokenizer))
        # The target sentences are tokenized
        target = create_tokens(target_sentence, target_tokenizer)
        inputs.append(target[:-1]) # The sentences for learning inputs
        labels.append(target[1:]) # The sentences for learning to predict the next tokens

    # The sentences are padded
    sources = pad_sequence(sources, padding_value=source_tokenizer(SPECIAL[1])["input_ids"][0])
    inputs = pad_sequence(inputs, padding_value=target_tokenizer(SPECIAL[1])["input_ids"][0])
    labels = pad_sequence(labels, padding_value=target_tokenizer(SPECIAL[1])["input_ids"][0])
    return sources, inputs, labels


def square_masks(size):
    """Creates an attention mask for the sentence."""
    mask = torch.ones((size, size), device=DEVICE) # This creates a square matrix
    mask = torch.triu(mask).transpose(0, 1) # This leaves the lower half and diagonal of the matrix with 1s
    mask = mask.float().masked_fill(mask == 0, float(NEG_INFTY)).masked_fill(mask == 1, float(0.0)) # The matrix is fitted with negative infinities and 0s
    return mask


def create_mask(source, target, source_tokenizer, target_tokenizer):
    """
    Creates attention masks and padding masks for both source and target sentences.

    Args:
        source (torch.Tensor): The tensor of the source sentence.
        target (torch.Tensor): The tensor of the target sentence.
        source_tokenizer (Tokenizer): The tokenizer for the source language.
        target_tokenizer (Tokenizer): The tokenizer for the target language.

    Returns:
        source_mask (torch.Tensor): A boolean mask to indicate what to mask.
        target_mask (torch.Tensor): The attention mask for the target sentence.
        source_padding_mask (torch.Tensor): A boolean mask identifying the padding tokens in the sentence.
        target_padding_mask (torch.Tensor): A boolean mask identifying the padding tokens in the sentence.
        tuple: Source and target attention masks, and source and target padding masks.
    """
    target_mask = square_masks(target.shape[0])
    # The False-values indicate that the value should not be masked
    source_mask = torch.full((source.shape[0], source.shape[0]), False, device=DEVICE)

    source_padding_mask = (source == source_tokenizer(SPECIAL[1])["input_ids"][0]).transpose(0, 1)
    target_padding_mask = (target == target_tokenizer(SPECIAL[1])["input_ids"][0]).transpose(0, 1)
    return source_mask, target_mask, source_padding_mask, target_padding_mask


def train_transformer(source_vocab_size, target_vocab_size,
                      training_path, validation_path,
                      source_tokenizer, target_tokenizer,                      
                      num_epochs, file_name):
    """
    Initializes, trains, and saves the Seq2SeqTransformer model.
    
    Args:
        source_vocab_size (int): The size of the source language's vocabulary.
        target_vocab_size (int): The size of the target language's vocabulary.
        training_path (str): The path to the training dataset.
        validation_path (str): The path to the validation dataset.
        source_tokenizer (Tokenizer): The tokenizer for the source language.
        target_tokenizer (Tokenizer): The tokenizer for the target language.
        num_epochs (int): The number of epochs for training.
        file_name (str): The name the trained model is saved as.
    """
    # The transformer model is initialized
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, MAX_LEN,
                                     NUM_HEADS, source_vocab_size, target_vocab_size, FFN_HID_DIM,
                                     DROPOUT)
    
    # The weights are initialized with Xavier initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    transformer = transformer.to(DEVICE)
    
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=source_tokenizer(SPECIAL[1])["input_ids"][0]) # The loss function
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001) # The optimizer
    
    # Now, the model is trained
    train_model(transformer, training_path, validation_path,
                source_tokenizer, target_tokenizer,
                optimizer, loss_func, num_epochs)
    
    # This creates the folder where the model is saved
    if not os.path.isdir(SAVED_MODEL_PATH):
        os.makedirs(SAVED_MODEL_PATH)        
    
    # The model is saved
    torch.save(transformer.state_dict(), SAVED_MODEL_PATH + file_name + ".pt")
    

def train_model(model, training_path, validation_path,
                source_tokenizer, target_tokenizer,
                optimizer, loss_func, num_epochs):
    """
    Trains and validates the transformer model.

    Args:
        model (Seq2SeqTransformer): The transformer model.
        training_path (str): The path to the training dataset.
        validation_path (str): The path to the validation dataset.
        source_tokenizer (Tokenizer): The tokenizer for the source language.
        target_tokenizer (Tokenizer): The tokenizer for the target language.
        optimizer (optim.Adam): The optimizer for training.
        loss_func (nn.CrossEntropyLoss): The loss function.
        num_epochs (int): The number of epochs for training.
    """
    for epoch in range(1, num_epochs+1):
        start_time = timer() # This times how long each training epoch takes
        train_loss = train_epoch(model, training_path,
                                 source_tokenizer, target_tokenizer,
                                 optimizer, loss_func)
        end_time = timer() # This times how long each training epoch takes
    
        val_loss = evaluate(model, validation_path,
                            source_tokenizer, target_tokenizer,
                            loss_func)

        # Some information is printed out to show the progress of training
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))


def train_epoch(model, training_path,
                source_tokenizer, target_tokenizer,
                optimizer, loss_func):
    """
    Trains the model for a single epoch.

    Args:
        model (Seq2SeqTransformer): The transformer model.
        training_path (str): The path to the training dataset.
        source_tokenizer (Tokenizer): The tokenizer for the source language.
        target_tokenizer (Tokenizer): The tokenizer for the target language.
        optimizer (optim.Adam): The optimizer for training.
        loss_func (nn.CrossEntropyLoss): The loss function.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train() # This sets the model in training mode
    losses = 0 # The losses are initialized

    # This prepares and parses the CSV file with the dataset
    train_iter = IterableWrapper([RAW_DATA_PATH + training_path + ".csv"])
    train_iter = FileOpener(train_iter, mode="b")
    train_iter = train_iter.parse_csv(skip_lines=1, as_tuple=True)

    # This loads the dataset
    train_dataloader = DataLoader(train_iter,
                                  batch_size=BATCH_SIZE,
                                  collate_fn=lambda x: collate(x, source_tokenizer, target_tokenizer))

    # The batch of training data is iterated through
    for source, input, label in train_dataloader:
        source = source.to(DEVICE)
        input = input.to(DEVICE)
        label = label.to(DEVICE)

        # The masks are created
        source_mask, input_mask, source_padding_mask, input_padding_mask = create_mask(source, input,
                                                                                       source_tokenizer, target_tokenizer)

        # A forward pass through the model is made
        logits = model(source, input,
                       source_mask, input_mask,
                       source_padding_mask, input_padding_mask,
                       source_padding_mask)

        optimizer.zero_grad() # The gradients are reset for the next epoch

        loss = loss_func(logits.reshape(-1, logits.shape[-1]), label.reshape(-1)) # The loss is calculated
        loss.backward() # Backpropagation

        optimizer.step() # The parameters are updated
        losses += loss.item() # The loss of this epoch is added to the total loss value

    return losses / len(list(train_dataloader)) # The average loss for the epoch
    

def evaluate(model, validation_path,
             source_tokenizer, target_tokenizer,
             loss_func):
    """
    Validates the model for a single epoch.

    Args:
        model (Seq2SeqTransformer): The transformer model.
        validation_path (str): The path to the validation dataset.
        source_tokenizer (Tokenizer): The tokenizer for the source language.
        target_tokenizer (Tokenizer): The tokenizer for the target language.
        loss_func (nn.CrossEntropyLoss): The loss function.

    Returns:
        float: The average validation loss for the epoch.
    """
    model.eval() # This sets the model in validation mode
    losses = 0 # The losses are initialized

    # This prepares and parses the CSV file with the dataset
    val_iter = IterableWrapper([RAW_DATA_PATH + validation_path + ".csv"])
    val_iter = FileOpener(val_iter, mode="b")
    val_iter = val_iter.parse_csv(skip_lines=1, as_tuple=True)

    # This loads the dataset
    val_dataloader = DataLoader(val_iter,
                                batch_size=BATCH_SIZE,
                                collate_fn=lambda x: collate(x, source_tokenizer, target_tokenizer))

    # The batch of validation data is iterated through
    for source, input, label in val_dataloader:
        source = source.to(DEVICE)
        input = input.to(DEVICE)
        label = label.to(DEVICE)

        # The masks are created
        source_mask, input_mask, source_padding_mask, input_padding_mask = create_mask(source, input,
                                                                                       source_tokenizer, target_tokenizer)

        # A forward pass through the model is made
        logits = model(source, input,
                       source_mask, input_mask,
                       source_padding_mask, input_padding_mask, source_padding_mask)

        loss = loss_func(logits.reshape(-1, logits.shape[-1]), label.reshape(-1)) # The loss is calculated
        losses += loss.item() # The loss of this epoch is added to the total loss value

    return losses / len(list(val_dataloader)) # The average loss for the epoch
