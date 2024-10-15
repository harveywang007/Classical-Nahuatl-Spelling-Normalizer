# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 15:23:46 2024

@author: Harvey
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
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, MAX_LEN,
                                     NUM_HEADS, source_vocab_size, target_vocab_size, FFN_HID_DIM,
                                     DROPOUT)

    transformer.load_state_dict(torch.load(SAVED_MODEL_PATH + file_name + ".pt",
                                           map_location=torch.device(DEVICE)))
    
    transformer.to(DEVICE)
    
    return transformer


def greedy_decode(model, source, source_mask,
                  target_tokenizer, max_len):
    source = source.to(DEVICE)
    source_mask = source_mask.to(DEVICE)

    memory = model.encode(source, source_mask)

    normalization = torch.ones(1, 1)
    normalization = normalization.fill_(torch.tensor(target_tokenizer(SPECIAL[0])["input_ids"][0])).type(torch.long).to(DEVICE)

    for _ in range(max_len-1):
        memory = memory.to(DEVICE)

        output_mask = (square_masks(normalization.size(0)).type(torch.bool)).to(DEVICE)

        output = model.decode(normalization, memory, output_mask)
        output = output.transpose(0, 1)

        prob = model.linear(output[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        normalization = torch.cat([normalization, torch.ones(1, 1).type_as(source.data).fill_(next_word)], dim=0)

        if next_word == target_tokenizer(SPECIAL[2])["input_ids"][0]:
            break

    return normalization


def translate_sentence(model, source_sentence,
                       source_tokenizer, target_tokenizer):
    model.eval()

    source = create_tokens(source_sentence, source_tokenizer)
    source = source.view(-1, 1)

    num_tokens = source.shape[0]
    source_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

    output_tokens = greedy_decode(model, source, source_mask,
                                  target_tokenizer, max_len=num_tokens + 5).flatten()
    output_tokens.tolist()

    return "".join(target_tokenizer.decode(output_tokens)
                   .replace(SPECIAL[0], "")
                   .replace(SPECIAL[2], "")
                   .strip())


def translate_test(model, test_path,
                   source_tokenizer, target_tokenizer,
                   give_avg_cer=True):    
    cer = CharErrorRate()
    cers = []
    
    test_iter = IterableWrapper([RAW_DATA_PATH + test_path + ".csv"])
    test_iter = FileOpener(test_iter, mode="b")
    test_iter = test_iter.parse_csv(skip_lines=1, as_tuple=True)
    
    test_dataloader = DataLoader(test_iter, batch_size=BATCH_SIZE)
    
    for input, ground_truth in test_dataloader:
        translation = translate_sentence(model, input[0],
                                         source_tokenizer, target_tokenizer)
        char_error_rate = cer([translation], [ground_truth[0]]).item()
        cers.append(char_error_rate)
    
        print("Input: " + input[0])
        print("Predicted: " + translation)
        print("Expected: " + ground_truth[0])
        print("Character Error Rate: " + str(char_error_rate))
        print("\n")
        
    if give_avg_cer:
        avg_cer = sum(cers) / len(cers)
        print("The average character error rate is " + str(avg_cer))
        
        
def normalize(model, file,
              source_tokenizer, target_tokenizer):
    normalizations = []
    
    source_iter = IterableWrapper([UNNORMALIZED_PATH + file + " - Unnormalized.csv"])
    source_iter = FileOpener(source_iter, mode="b")
    source_iter = source_iter.parse_csv(skip_lines=1, as_tuple=True)
    
    source_dataloader = DataLoader(source_iter)
    
    for input in source_dataloader:
        normalizations.append(translate_sentence(model, input[0][0],
                                                 source_tokenizer, target_tokenizer))
        
    translation_df = pd.DataFrame({"Nahuatl": normalizations})
        
    translation_df.to_excel(NORMALIZED_PATH + file + " - Normalized.xlsx", index=False)


def back_translate(model, bt_path,
                   source_tokenizer, target_tokenizer,
                   return_dfs=True):
    inputs = []
    translations = []
    
    bt_iter = IterableWrapper([CSV_PATH + bt_path + ".csv"])
    bt_iter = FileOpener(bt_iter, mode="b")
    bt_iter = bt_iter.parse_csv(skip_lines=1, as_tuple=True)
    
    bt_dataloader = DataLoader(bt_iter)
    
    for input in bt_dataloader:
        inputs.append(input[0][0])
        translations.append(translate_sentence(model, input[0][0],
                                               source_tokenizer, target_tokenizer))
        
    back_tanslations = {"Unregularized": translations,
                        "Regularized": inputs}
    
    back_tanslations = pd.DataFrame(back_tanslations)
    
    training = pd.read_csv(RAW_DATA_PATH + "Train.csv")
    
    full_set = pd.concat([training, back_tanslations])
    
    full_set.to_csv(RAW_DATA_PATH + "Combined Training.csv",
                    index=False, encoding="utf-8-sig")
    
    if return_dfs:
        return back_tanslations, full_set
    else:
        pass
    

def main():
    pass

if __name__ == "__main__":
    main()