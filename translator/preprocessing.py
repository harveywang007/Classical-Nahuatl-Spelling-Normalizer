# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:50:40 2024

@author: Harvey Wang
"""

import os
import re

import pandas as pd


DATA_DIR = "./data/"
CSV_PATH = DATA_DIR + "CSVs/"
RAW_DATA_PATH = DATA_DIR + "raw_data/"


def prepare_data(file_name):
    """
    Randomizes and sets all the rows to lower-case.
    
    Args:
        file_name (str): The name of the file.
    Returns:
        corpus (DataFrame): The file loaded as a DataFrame.
    """
    corpus = pd.read_csv(CSV_PATH + file_name + ".csv", encoding="utf-8-sig")
    
    # Randomizes the dataset
    corpus = corpus.sample(frac=1).reset_index(drop=True)
    
    # Converts the data into lower-case.
    for column in corpus:
        corpus[column] = corpus[column].str.lower()
    
    return corpus


def create_raw_data_path():
    """Creates a path to store text data files."""
    if not os.path.isdir(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)


def create_data_split(corpus, backtranslate, return_dfs=True):
    """
    Splits the corpus to backtranslate into 75% training and 25% validation.
    Splits the bilingual corpus into 70% training, 25% validation, and 5% testing.
    
    Args:
        corpus (DataFrame): The DataFrame containing the dataset.
        backtranslate (bool): The boolean indicating if the dataset is to train the backtranslator.
        return_dfs (bool): The boolean indicating if DataFrames should be returned.
    Returns:
        train (DataFrame): The DataFrame containing the training set.
        val (DataFrame): The DataFrame containing the validation set.
        test (DataFrame): The DataFrame containing the test set.
    """
    if backtranslate:
        # Switches the columns
        corpus = corpus[["Regularized", "Unregularized"]]
        
        # Splits the first 75% to be for training
        train = corpus[:int(corpus.shape[0]*0.75)]
        train.to_csv(RAW_DATA_PATH + "Backtranslation_Train.csv",
                     index=False,
                     encoding="utf-8-sig")
        
        # Splits the last 25% to be for validation
        val = corpus[int(corpus.shape[0]*0.75):]
        val.to_csv(RAW_DATA_PATH + "Backtranslation_Val.csv",
                   index=False,
                   encoding="utf-8-sig")
        
        if return_dfs:
            return train, val
        else:
            pass
    
    else:
        # Splits the first 70% to be for training
        train = corpus[:int(corpus.shape[0]*0.7)]
        train.to_csv(RAW_DATA_PATH + "Train.csv",
                     index=False,
                     encoding="utf-8-sig")
        
        # Splits the next 20% to be for validation
        val = corpus[int(corpus.shape[0]*0.7):int(corpus.shape[0]*0.95)]
        val.to_csv(RAW_DATA_PATH + "Val.csv",
                   index=False,
                   encoding="utf-8-sig")
        
        # Splits the last 5% to be for testing
        test = corpus[int(corpus.shape[0]*0.95):]
        test.to_csv(RAW_DATA_PATH + "Test.csv",
                    index=False,
                    encoding="utf-8-sig")
        
        if return_dfs:
            return train, val, test
        else:
            pass
    
    
def load_data(pathway, file_name):
    """Loads the specified data file in the specified pathway."""
    return pd.read_csv(pathway + file_name + ".csv", encoding="utf-8-sig")


def split_sentence(sentence):
    """Splits the sentence by punctuation."""
    return re.split(r"[\.\?,;:]+\s", sentence)


def pad_sentences(row):
    """Pads each list to the maximum list length in that row."""
    max_length = max(len(row["Unregularized"]), len(row["Regularized"]))
    row["Unregularized"] += [None] * (max_length - len(row["Unregularized"]))
    row["Regularized"] += [None] * (max_length - len(row["Regularized"]))
    
    return row


def expand_bilingual_corpus(corpus, output_file):
    """
    Increases the amount of training data for the bilingual corpus.
    
    Args:
        corpus (DataFrame): The bilingual corpus.
        output_file (string): The name of the Excel file to be saved.
    """
    # Splits each column's sentences by punctuation
    corpus["Unregularized"] = corpus["Unregularized"].apply(split_sentence)
    corpus["Regularized"] = corpus["Regularized"].apply(split_sentence)
    
    # Pads the lsit with split sentences
    corpus = corpus.apply(pad_sentences, axis=1)
    
    # Converts each element in the list to its own row
    corpus = corpus.explode(["Unregularized", "Regularized"])
    
    # Saves the DataFrame in an Excel file
    corpus.to_excel(CSV_PATH + output_file + ".xlsx",
                    index=False,
                    header=["Unregularized", "Regularized"])


def expand_monolingual_corpus(corpus, output_file):
    """
    Increases the amount of training data for the monolingual corpus.
    
    Args:
        corpus (DataFrame): The monolingual corpus.
        output_file (string): The name of the Excel file to be saved.
    """
    # Splits each column's sentences by punctuation
    corpus["To Backtranslate"] = corpus["To Backtranslate"].apply(split_sentence)
    
    # Converts each element in the list to its own row
    corpus = corpus.explode("To Backtranslate")
    
    # Saves the DataFrame in an Excel file
    corpus.to_excel(CSV_PATH + output_file + ".xlsx",
                    index=False,
                    header=["To Backtranslate"])
