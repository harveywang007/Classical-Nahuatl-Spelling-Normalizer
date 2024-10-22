# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 15:23:46 2024

@author: Harvey Wang
"""


import pandas as pd

import torch
from torch import Tensor
from torchdata.datapipes.iter import IterableWrapper, FileOpener
from torch.utils.data import DataLoader
from torchmetrics.text import CharErrorRate

from translation.transformer import Seq2SeqTransformer
from training import create_tokens, square_masks


DATA_DIR = "./data/"
RAW_DATA_PATH = DATA_DIR + "raw_data/"
CSV_PATH = DATA_DIR + "CSVs/"

SAVED_MODEL_PATH = "./trained_transformers/"

UNNORMALIZED_PATH = DATA_DIR + "Unnormalized/"
NORMALIZED_PATH = DATA_DIR + "Normalized/"

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


def load_trained_transformer(file_name, source_vocab_size=271, target_vocab_size=271):
    """
    Loads the trained transformer model.
    
    Args:
        file_name (str): The name of the file.
        source_vocab_size (int): The size of the source language's tokenizer's vocabulary.
        target_vocab_size (int): The size of the target language's tokenizer's vocabulary.
    Returns:
        transformer (Seq2SeqTransformer): The trained transformer.
    """
    # A transformer is initialized to load the trained one
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, MAX_LEN,
                                     NUM_HEADS, source_vocab_size, target_vocab_size, FFN_HID_DIM,
                                     DROPOUT)
    
    # The trained transformer is loaded
    transformer.load_state_dict(torch.load(SAVED_MODEL_PATH + file_name + ".pt",
                                           map_location=torch.device(DEVICE)))
    
    transformer.to(DEVICE)
    
    return transformer


def greedy_decode(model, source, source_mask,
                  target_tokenizer, max_len):
    """
    Generates a "translation" from a source sentence using the transformer.
    Finds the token with the highest probability at each step.
    
    Args:
        model (Seq2SeqTransformer): The trained transformer.
        source (torch.Tensor): The source sentence.
        source_mask (torch.Tensor): The attention mask for the source sentence.
        target_tokenizer (Tokenizer): The tokenizer for the target language.
        max_len (int): The maximum length of the generated target sentence.
    Returns:
        normalization (torch.Tensor): The generated target sentence as tokens.
    """
    source = source.to(DEVICE)
    source_mask = source_mask.to(DEVICE)

    # The source sentence is encoded
    memory = model.encode(source, source_mask)

    # The target sentence is initialized
    normalization = torch.ones(1, 1)
    normalization = normalization.fill_(torch.tensor(target_tokenizer(SPECIAL[0])["input_ids"][0])).type(torch.long).to(DEVICE) # The <START> token is added
    
    # The sentence is generated iteratively
    for _ in range(max_len-1):
        memory = memory.to(DEVICE)

        # This creates a mask for the output
        output_mask = (square_masks(normalization.size(0)).type(torch.bool)).to(DEVICE)

        # The next token is decoded
        output = model.decode(normalization, memory, output_mask)
        output = output.transpose(0, 1) # The sequence is transposed to make it easier to read

        # The next token is selected
        prob = model.linear(output[:, -1]) # The probabilities are taken
        _, next_word = torch.max(prob, dim=1) # The token with the highest probability
        next_word = next_word.item() # The next token

        # This appends the next token to the target sentence
        normalization = torch.cat([normalization, torch.ones(1, 1).type_as(source.data).fill_(next_word)], dim=0)

        # The loop stops once the <END> token is generated
        if next_word == target_tokenizer(SPECIAL[2])["input_ids"][0]:
            break

    return normalization


def translate_sentence(model, source_sentence,
                       source_tokenizer, target_tokenizer):
    """
    Translates a single sentence using the trained model.
    
    Args:
        model (Seq2SeqTransformer): The trained transformer model.
        source_sentence (str): The source sentence to be translated.
        source_tokenizer (Tokenizer): The tokenizer for the source language.
        target_tokenizer (Tokenizer): The tokenizer for the target language.
    
    Returns:
        str: The translated sentence.
    """
    model.eval() # The model is selected as evaluation mode

    # The source sentence is tokenized
    source = create_tokens(source_sentence, source_tokenizer)
    source = source.view(-1, 1) # The sentence is reshaped for the model

    # The attention mask is created
    num_tokens = source.shape[0]
    source_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

    # The "translation" is generated
    output_tokens = greedy_decode(model, source, source_mask,
                                  target_tokenizer, max_len=num_tokens + 5).flatten()
    output_tokens.tolist() # The tensor is converted to a list

    # The detokenized sentence is returned as a string
    return "".join(target_tokenizer.decode(output_tokens) # The decoding
                   .replace(SPECIAL[0], "") # Removing the <START> token
                   .replace(SPECIAL[2], "") # Removing the <END> token
                   .strip()) # Removing whitespaces


def translate_test(model, test_path,
                   source_tokenizer, target_tokenizer,
                   give_avg_cer=True):
    """
    Tests the trained model on a test dataset and calculates the Character Error Rate (CER).
    
    Args:
        model (Seq2SeqTransformer): The trained transformer model.
        test_path (str): The path to the test data file.
        source_tokenizer (Tokenizer): The tokenizer for the source language.
        target_tokenizer (Tokenizer): The tokenizer for the target language.
        give_avg_cer (bool): Whether to calculate and print the average CER.
    """
    cer = CharErrorRate() # This initializes the CER metric
    cers = [] # This creates a list to store the CERs of the sentences
    
    # This prepares and parses the CSV file with the dataset
    test_iter = IterableWrapper([RAW_DATA_PATH + test_path + ".csv"])
    test_iter = FileOpener(test_iter, mode="b")
    test_iter = test_iter.parse_csv(skip_lines=1, as_tuple=True)
    
    # This loads the dataset
    test_dataloader = DataLoader(test_iter, batch_size=BATCH_SIZE)
    
    # This iterates through the test data
    for input, ground_truth in test_dataloader:
        # The sentence is "translated
        translation = translate_sentence(model, input[0],
                                         source_tokenizer, target_tokenizer)
        char_error_rate = cer([translation], [ground_truth[0]]).item() # The CER is calculated
        cers.append(char_error_rate) # The CER is appended to the list
        
        # The translations are printed, along with other comparative information
        print("Input: " + input[0])
        print("Predicted: " + translation)
        print("Expected: " + ground_truth[0])
        print("Character Error Rate: " + str(char_error_rate))
        print("\n")
    
    # This calculates and prints the average CER
    if give_avg_cer:
        avg_cer = sum(cers) / len(cers)
        print("The average character error rate is " + str(avg_cer))
        
        
def normalize(model, file,
              source_tokenizer, target_tokenizer):
    """
    Normalizes ("translates") the sentences from an input file and saves the results to a new Excel file.
    
    Args:
        model (Seq2SeqTransformer): The trained transformer model.
        file (str): The input file containing unnormalized sentences.
        source_tokenizer (Tokenizer): The tokenizer for the source language.
        target_tokenizer (Tokenizer): The tokenizer for the target language.
    """
    normalizations = [] # This creates a list to store the normalized sentences
    
    # This prepares and parses the CSV file with the dataset
    source_iter = IterableWrapper([UNNORMALIZED_PATH + file + " - Unnormalized.csv"])
    source_iter = FileOpener(source_iter, mode="b")
    source_iter = source_iter.parse_csv(skip_lines=1, as_tuple=True)
    
    # This loads the dataset
    source_dataloader = DataLoader(source_iter)
    
    # This iterates through the data
    for input in source_dataloader:
        # The sentences are normalized and appended to the list
        normalizations.append(translate_sentence(model, input[0][0],
                                                 source_tokenizer, target_tokenizer))
    
    # The normalized sentences are converted to a DataFrame
    translation_df = pd.DataFrame({"Nahuatl": normalizations})
    
    # The DataFrame is converted to an Excel file
    translation_df.to_excel(NORMALIZED_PATH + file + " - Normalized.xlsx", index=False)


def back_translate(model, bt_path,
                   source_tokenizer, target_tokenizer,
                   return_dfs=True):
    """
    Performs backtranslation on a dataset and combines the backtranslated sentences with the original training set.

    Args:
        model (Seq2SeqTransformer): The trained transformer model.
        bt_path (str): The path to the backtranslation data file.
        source_tokenizer (Tokenizer): The tokenizer for the source language.
        target_tokenizer (Tokenizer): The tokenizer for the target language.
        return_dfs (bool): Whether to return the DataFrames with the back-translated sentences.

    Returns:
        back_tanslations (DataFrame): The DataFrame containing the backtranslations.
        full_set (DataFrame): The DataFrame containing the backtranslations combined with the training set.
    """
    inputs = [] # This creates a list to store the normalized sentences
    translations = [] # This creates a list to store the unnormalized (backtranslated) sentences
    
    # This prepares and parses the CSV file with the dataset 
    bt_iter = IterableWrapper([CSV_PATH + bt_path + ".csv"])
    bt_iter = FileOpener(bt_iter, mode="b")
    bt_iter = bt_iter.parse_csv(skip_lines=1, as_tuple=True)
    
    # This loads the dataset
    bt_dataloader = DataLoader(bt_iter)
    
    # This iterates through the data 
    for input in bt_dataloader:
        inputs.append(input[0][0]) # The normalized sentences are appended
        # The backtranslations/unnormalized sentences are appended
        translations.append(translate_sentence(model, input[0][0],
                                               source_tokenizer, target_tokenizer))
    
    # A dictionary is created from the two lists
    # The unnormalized and normalized sentences were originally called "unregularized" and "regularized", respectively
    back_tanslations = {"Unregularized": translations,
                        "Regularized": inputs}
    
    # The dictionary is converted into a DataFrame
    back_tanslations = pd.DataFrame(back_tanslations)
    
    # The training dataset is loaded
    training = pd.read_csv(RAW_DATA_PATH + "Train.csv")
    
    # The backtranslations are concatenated to the training data
    full_set = pd.concat([training, back_tanslations])
    
    # The DataFrame with the combined backtranslations and the training data is converted to a CSV file
    full_set.to_csv(RAW_DATA_PATH + "Combined Training.csv",
                    index=False, encoding="utf-8-sig")
    
    if return_dfs:
        return back_tanslations, full_set
    else:
        pass
