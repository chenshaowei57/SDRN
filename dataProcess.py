from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import json
import pickle
import argparse
import collections
sys.path.append("../")
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import bert.tokenization as tokenization

try:
    import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy
    from xml.sax.saxutils import escape
except:
    sys.exit('Some package is missing... Perhaps <re>?')

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 tokens,
                 token_ids,
                 token_mask,
                 segmentId,
                 labels,
                 label_ids,
                 relations,
                 gold_relations,
                 token_to_orig_map):
        self.tokens = tokens #
        self.token_ids = token_ids
        self.token_mask = token_mask
        self.segmentId = segmentId
        self.labels = labels #
        self.label_ids = label_ids
        self.relations = relations
        self.gold_relations = gold_relations #
        self.token_to_orig_map = token_to_orig_map

def readDataFromFile(path):
    f = open(path, "r", encoding="utf-8")
    lines = f.readlines()
    f.close()
    seq = 0
    datasets = []
    words = []
    labels = []
    relations = []
    relations_gold = []
    for l in lines:
        if l.strip() == "#Relations":
            continue
        elif l.strip() == "" and len(words)>0:
            # if "B-T" in labels or "B-P" in labels:
            datasets.append({"words": words, "labels": labels, "relations": relations, "relations_gold": relations_gold})
            if len(words)>seq:
                seq = len(words)
            words = []
            labels = []
            relations = []
            relations_gold = []
        elif len(l.strip().split("\t")) == 2:
            tempLine = l.strip().split("\t")
            # WORD
            words.append(tempLine[0].lower())
            # LABEL
            labels.append(tempLine[1])
        elif len(l.strip().split("\t")) == 4:
            rel = list(map(int, l.strip().split("\t")))
            relations_gold.append([rel[2],rel[3],rel[0],rel[1]])
            if -1 not in rel:
                relations.append(rel)
    print("max_seq_length"+str(seq))
    return datasets

def convert_examples_to_features(examples, tokenizer, max_seq_length=100):
    seq = 0
    features = []
    labelDic = {"O":1, "B-T":2, "I-T":3,"B-P":4,"I-P":5}
    num = 0
    relationlen = 0
    aspectlen = 0
    opinionlen = 0
    sentlen = 0
    for (example_index, example) in enumerate(examples):
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        labels = []
        gold_relations = []
        #### split words and labels ####
        for (i, token) in enumerate(example["words"]):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            label = example["labels"][i]
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
            if label == "B-T" or label == "B-P":
                for i in range(len(sub_tokens)):
                    if i == 0:
                        labels.append(label)
                    else:
                        labels.append("I-"+label.split("-")[-1])
            else:
                for i in range(len(sub_tokens)):
                    labels.append(label)
        #### update relations ####
        for r in example["relations"]:
            temp = []
            for rr in r:
                if rr < len(example["words"]):
                    temp.append(orig_to_tok_index[rr])
                else:
                    temp.append(len(all_doc_tokens))
            gold_relations.append(temp)


        # Account for [CLS] and [SEP] with "- 2"
        if len(all_doc_tokens) > max_seq_length - 2:
            all_doc_tokens = all_doc_tokens[0:(max_seq_length - 2)]

        #### add start and end to token, make segment_ids ####
        tokens = []
        token_to_orig_map = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for index, token in enumerate(all_doc_tokens):
            token_to_orig_map[len(tokens)] = tok_to_orig_index[index]
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        if len(tokens)>seq:
            seq = len(tokens)
        if len(tokens)>=120:
            num+=1
            continue
        #### make token_ids ####
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        #### make label_id and relations ####
        label_ids = [0] * len(input_ids)
        label_ids[0] = 1
        relations = []
        for i in range(len(input_ids)):
            relations.append([0]*len(input_ids))
        for idx in range(len(labels)):
            if idx+1>len(labels):
                print(1)
            label_ids[idx+1] = labelDic[labels[idx]]
        label_ids[len(labels)+1] = 1
        for gr in gold_relations:
            for idx in range(gr[0]+1,gr[1]+1):
                for idy in range(gr[2]+1,gr[3]+1):
                    relations[idx][idy] = 1
            for idx in range(gr[2]+1,gr[3]+1):
                for idy in range(gr[0]+1,gr[1]+1):
                    relations[idx][idy] = 1
        #
        #
        sentlen+=1
        relationlen+=len(gold_relations)
        aspectlen+=labels.count("B-T")
        opinionlen+=labels.count("B-P")
        features.append(
            InputFeatures(
                tokens,
                input_ids,
                input_mask,
                segment_ids,
                labels,
                label_ids,
                relations,
                gold_relations,
                token_to_orig_map))
    print(sentlen)
    print(aspectlen)
    print(opinionlen)
    print(relationlen)
    print("\n")
    return features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--vocab_file", type=str, required=True)
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Whether to lower case the input text. Should be True for uncased "
                             "models and False for cased models.")
    parser.add_argument("--max_seq_length", default=150, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    args = parser.parse_args()
    train_set = readDataFromFile(args.train_file)
    dev_set = readDataFromFile(args.dev_file)
    test_set = readDataFromFile(args.test_file)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    # A = tokenizer.tokenize("disappointed")
    # print(A)
    train_features = convert_examples_to_features(train_set, tokenizer, max_seq_length=120)
    dev_features = convert_examples_to_features(dev_set, tokenizer, max_seq_length=120)
    test_features = convert_examples_to_features(test_set, tokenizer, max_seq_length=120)
    # torch.save({"train":train_features, "test":test_features, "dev":dev_features}, args.output_file)
    # print(1)