# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:50:40 2024

@author: Harvey
"""

import os
import re

import pandas as pd


DATA_DIR = "./data/"
CSV_PATH = DATA_DIR + "CSVs/"
RAW_DATA_PATH = DATA_DIR + "raw_data/"


def prepare_data(file_name):
    corpus = pd.read_csv(CSV_PATH + file_name + ".csv", encoding="utf-8-sig")
    
    corpus = corpus.sample(frac=1).reset_index(drop=True)
    
    for column in corpus:
        corpus[column] = corpus[column].str.lower()
    
    return corpus


def create_raw_data_path():
    if not os.path.isdir(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)


def create_data_split(corpus, backtranslate, return_dfs=True):
    if backtranslate:
        corpus = corpus[["Regularized", "Unregularized"]]
        
        train = corpus[:int(corpus.shape[0]*0.75)]
        train.to_csv(RAW_DATA_PATH + "Backtranslation_Train.csv",
                     index=False,
                     encoding="utf-8-sig")
        
        val = corpus[int(corpus.shape[0]*0.75):]
        val.to_csv(RAW_DATA_PATH + "Backtranslation_Val.csv",
                   index=False,
                   encoding="utf-8-sig")
        
        if return_dfs:
            return train, val
        else:
            pass
    
    else:
        train = corpus[:int(corpus.shape[0]*0.7)]
        train.to_csv(RAW_DATA_PATH + "Train.csv",
                     index=False,
                     encoding="utf-8-sig")
        
        val = corpus[int(corpus.shape[0]*0.7):int(corpus.shape[0]*0.95)]
        val.to_csv(RAW_DATA_PATH + "Val.csv",
                   index=False,
                   encoding="utf-8-sig")
        
        test = corpus[int(corpus.shape[0]*0.95):]
        test.to_csv(RAW_DATA_PATH + "Test.csv",
                    index=False,
                    encoding="utf-8-sig")
        
        if return_dfs:
            return train, val, test
        else:
            pass
    
    
def load_data(pathway, file_name):
    return pd.read_csv(pathway + file_name + ".csv", encoding="utf-8-sig")


def split_sentence(sentence):
    return re.split(r"[\.\?,;:]+\s", sentence)


def pad_sentences(row):
    max_length = max(len(row["Unregularized"]), len(row["Regularized"]))
    row["Unregularized"] += [None] * (max_length - len(row["Unregularized"]))
    row["Regularized"] += [None] * (max_length - len(row["Regularized"]))
    
    return row


def expand_bilingual_corpus(corpus, output_file):
    corpus["Unregularized"] = corpus["Unregularized"].apply(split_sentence)
    corpus["Regularized"] = corpus["Regularized"].apply(split_sentence)
    
    corpus = corpus.apply(pad_sentences, axis=1)
    
    corpus = corpus.explode(["Unregularized", "Regularized"])
    
    corpus.to_excel(CSV_PATH + output_file + ".xlsx",
                    index=False,
                    header=["Unregularized", "Regularized"])


def expand_monolingual_corpus(corpus, output_file):
    corpus["To Backtranslate"] = corpus["To Backtranslate"].apply(split_sentence)
    
    corpus = corpus.explode("To Backtranslate")
    
    corpus.to_excel(CSV_PATH + output_file + ".xlsx",
                    index=False,
                    header=["To Backtranslate"])