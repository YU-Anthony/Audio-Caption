import torch
import torch.nn as nn
import time

from data_handling import get_clotho_loader, get_test_data_loader
from model import TransformerModel  # , RNNModel, RNNModelSmall
import itertools
import numpy as np
import os
import sys
import logging
import csv

from util import get_file_list, get_padding, print_hparams, greedy_decode, \
    calculate_bleu, calculate_spider, LabelSmoothingLoss, beam_search, align_word_embedding, gen_str
from hparams import hparams

import argparse

hp = hparams()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# device = torch.device(hp.device)
np.random.seed(hp.seed)
torch.manual_seed(hp.seed)


def test_with_beam(model,data, max_len=30, eos_ind=9, beam_size=3):
    model.eval()
    test_data = np.array([data])
    src = torch.from_numpy(test_data)
    src = src.to(device)
    
    output = beam_search(model, src, max_len, start_symbol_ind=0, beam_size=beam_size) 
    output_sentence_ind_batch = []
    for single_sample in output:
        output_sentence_ind = []
        for sym in single_sample:
            if sym == eos_ind: break
            output_sentence_ind.append(sym.item())
        output_sentence_ind_batch.append(output_sentence_ind)
    out_str = gen_str(output_sentence_ind_batch, hp.word_dict_pickle_path)
    
    return out_str



def run_test(audio):
    parser = argparse.ArgumentParser(description='hparams for model')
    parser.add_argument('--load_pretrain_cnn', action='store_true')
    parser.add_argument('--freeze_cnn', action='store_true')
    parser.add_argument('--load_pretrain_emb', action='store_true')
    parser.add_argument('--load_pretrain_model', action='store_true')
    parser.add_argument('--pretrain_emb_path', type=str, default=hp.pretrain_emb_path)
    parser.add_argument('--pretrain_cnn_path', type=str, default=hp.pretrain_cnn_path)
    args = parser.parse_args(args=[])
    for k, v in vars(args).items():
        setattr(hp, k, v)
    args = parser.parse_args(args=[])

    pretrain_emb = align_word_embedding(hp.word_dict_pickle_path, hp.pretrain_emb_path, hp.ntoken,
                                        hp.nhid) if hp.load_pretrain_emb else None
    pretrain_cnn = torch.load(hp.pretrain_cnn_path) if hp.load_pretrain_cnn else None


    model = TransformerModel(hp.ntoken, hp.ninp, hp.nhead, hp.nhid, hp.nlayers, hp.batch_size, dropout=0.2,
                             pretrain_cnn=pretrain_cnn, pretrain_emb=pretrain_emb,freeze_cnn=hp.freeze_cnn).to(device)


    test_data = audio
    
    # Generate caption(in test_out.csv)
    model.load_state_dict(torch.load("best.pt",map_location=device))
    text = test_with_beam(model,test_data, beam_size=3)
    return text
