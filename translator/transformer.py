# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 08:01:12 2024

@author: Harvey Wang
"""

import math

import torch
import torch.nn as nn
from torch.nn import Transformer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    """
    A class to implement the positional encoding of a transformer.
    
    Attributes:
        emb_size (int): The size of the embedding dimension.
        embeddings (nn.Embedding): The embedding layer.
        positions (torch.Tensor): The positional encoding.
    """
    def __init__(self, vocab_size, emb_size, max_len):
        """
        Initializes the PositionalEncoding object.
        
        Args:
            vocab_size (int): The size of the vocabulary (the number of unique tokens).
            emb_size (int): The size of the embedding dimension.
            max_len (int): The maximum length of input sequences.
        """
        super(PositionalEncoding, self).__init__()

        self.emb_size = emb_size # The embedding dimension size
        self.embeddings = nn.Embedding(vocab_size, emb_size) # The embedding layer
        self.positions = torch.zeros((max_len, emb_size)) # The positional encoding tensor
        
        # This calculation is to be passed into the sinosoidal function
        denom = torch.exp(-1 * torch.arange(0, emb_size, 2) * math.log(10000) / emb_size) # This calculates the denominator
        pos = torch.arange(0, max_len).reshape(max_len, 1) # This calculates the numerator
        
        # This applies the respective sinosoidal functions based on even and odd positions
        self.positions[:, 0::2] = torch.sin(pos * denom) # For even positions
        self.positions[:, 1::2] = torch.cos(pos * denom) # For odd positions
        # The reshaping allows the tensor to align with the input dimensions of the transformer
        self.positions = self.positions.unsqueeze(-2) # A dimension is added in the penultimate position
        

    def forward(self, x):
        """Implements the forward pass to add positional encoding to the embeddings."""
        outputs = self.embeddings(x.long()) * math.sqrt(self.emb_size) # The embeddings are scaled
        return outputs + self.positions[:outputs.size(0), :].to(DEVICE) # The positional encoding is added
    
    
class Seq2SeqTransformer(nn.Module):
    """
    A class to implement a transformer with positional encoding.
    
    Attributes:
        transformer (Transformer): The transformer.
        linear (nn.Linear): The linear layer.
        source_pos_enc (PositionalEncoding): The positional encoding for the source language sentences.
        target_pos_enc (PositionalEncoding): The positional encoding for the target language sentences.
    """
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, max_len,
                 num_heads, source_vocab_size, target_vocab_size, dim_feed_forward, dropout):
        """
        Initializes the Seq2SeqTransformer object.
        
        Args:
            num_encoder_layers (int): The number of layers in the encoder.
            num_decoder_layers (int): The number of layers in the decoder.
            emb_size (int): The embedding dimension size.
            max_len (int): The maximum length of input sequences.
            num_heads (int): The number of attention heads.
            source_vocab_size (int): The size of the source langauge's vocabulary (the number of unique tokens).
            target_vocab_size (int): The size of the target langauge's vocabulary (the number of unique tokens).
            dim_feed_forward (int): The dimension of the feed-forward layer.
            dropout (float): The dropout rate.
        """
        super(Seq2SeqTransformer, self).__init__()

        self.transformer = Transformer(d_model=emb_size, nhead=num_heads,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feed_forward,
                                       dropout=dropout)

        self.linear = nn.Linear(emb_size, target_vocab_size)
        self.source_pos_enc = PositionalEncoding(source_vocab_size, emb_size, max_len)
        self.target_pos_enc = PositionalEncoding(target_vocab_size, emb_size, max_len)


    def forward(self, source, target,
                source_mask, target_mask,
                source_padding_mask, target_padding_mask,
                memory_key_padding_mask):
        """Implements the forward pass through the transformer."""
        source_emb = self.source_pos_enc(source) # The source sentence is positionally encoded
        target_emb = self.target_pos_enc(target) # The target sentence is positionally encoded
        # The sentences are passed through the transformer
        outputs = self.transformer(source_emb, target_emb, source_mask, target_mask, None,
                                   source_padding_mask, target_padding_mask, memory_key_padding_mask)
        # The transformer outputs are passed through the linear layer
        return self.linear(outputs)


    def encode(self, source, source_mask):
        """Encodes the source sentence with the encoder."""
        return self.transformer.encoder(self.source_pos_enc(source), source_mask)


    def decode(self, target, memory, target_mask):
        """Decodes the target sentence with the decoder, using the encoded sentence as memory"""
        return self.transformer.decoder(self.target_pos_enc(target), memory, target_mask)
