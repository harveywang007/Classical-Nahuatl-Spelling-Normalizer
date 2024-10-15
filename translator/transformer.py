# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 08:01:12 2024

@author: Harvey
"""

import math

import torch
import torch.nn as nn
from torch.nn import Transformer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, vocab_size, emb_size, max_len):
        super(PositionalEncoding, self).__init__()

        self.emb_size = emb_size
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        self.positions = torch.zeros((max_len, emb_size))

        denom = torch.exp(-1 * torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        self.positions[:, 0::2] = torch.sin(pos * denom)
        self.positions[:, 1::2] = torch.cos(pos * denom)
        self.positions = self.positions.unsqueeze(-2)
        

    def forward(self, x):
        outputs = self.embeddings(x.long()) * math.sqrt(self.emb_size)
        return outputs + self.positions[:outputs.size(0), :].to(DEVICE)
    
    
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, emb_size, max_len,
                 num_heads, source_vocab_size, target_vocab_size, dim_feed_forward, dropout):
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
        source_emb = self.source_pos_enc(source)
        target_emb = self.target_pos_enc(target)
        outputs = self.transformer(source_emb, target_emb, source_mask, target_mask, None,
                                   source_padding_mask, target_padding_mask, memory_key_padding_mask)
        return self.linear(outputs)


    def encode(self, source, source_mask):
        return self.transformer.encoder(self.source_pos_enc(source), source_mask)


    def decode(self, target, memory, target_mask):
        return self.transformer.decoder(self.target_pos_enc(target), memory, target_mask)