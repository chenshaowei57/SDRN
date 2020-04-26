# -*- coding: utf-8 -*-
# @Author: Shaowei Chen,     Contact: chenshaowei0507@163.com
# @Date:   2020-4-26 16:47:32

import time
import gc
import torch
from alphabet import Alphabet
from opinionMining import opinionMining
import sys
import numpy as np
import random
import os
import argparse
from bert.modeling import BertConfig
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from bert.optimization import BERTAdam

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed_num = 57
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


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
        self.tokens = tokens  #
        self.token_ids = token_ids
        self.token_mask = token_mask
        self.segmentId = segmentId
        self.labels = labels  #
        self.label_ids = label_ids
        self.relations = relations
        self.gold_relations = gold_relations  #
        self.token_to_orig_map = token_to_orig_map


#### target token level precision ####
def targetPredictCheck(targetPredict, batch_target_label, mask):
    pred = targetPredict.cpu().data.numpy()
    gold = batch_target_label.cpu().data.numpy()
    mask = mask.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    return right_token, total_token


#### relation token level precision ####
def relationPredictCheck(relationPredict, batch_relation):
    relationCheck = torch.ones(relationPredict.size(0), relationPredict.size(1), relationPredict.size(2)) * 0.1
    pred = relationPredict.cpu()
    pred = torch.gt(pred, relationCheck).data.numpy()
    gold = batch_relation.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * gold)
    total_token = gold.sum()
    return right_token, total_token


def recover_label(targetPredict, all_labels, all_input_mask):
    pred_variable = targetPredict
    gold_variable = all_labels
    mask_variable = all_input_mask
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [pred_tag[idx][idy] - 1 for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [gold_tag[idx][idy] - 1 for idy in range(seq_len) if mask[idx][idy] != 0]
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def get_ner_fmeasure(gold_results, pred_results, tagScheme):
    target_gold, opinion_gold = splitTandO(gold_results)
    target_pred, opinion_pred = splitTandO(pred_results)

    assert (len(target_gold) == len(target_pred))
    assert (len(opinion_gold) == len(opinion_pred))

    if tagScheme == "BIO":
        TP, TR, TF = evalForBIO(target_gold, target_pred)
        OP, OR, OF = evalForBIO(opinion_gold, opinion_pred)
    else:
        print("erro tagScheme!")
        exit(0)

    return TP, TR, TF, OP, OR, OF


def evalForBIO(gold, pred):
    correct = 0
    predicted = 0
    relevant = 0
    # count correct
    for num in range(len(gold)):
        if gold[num] == '1':
            if num < len(gold) - 1:
                if gold[num + 1] != '2':
                    if pred[num] == '1' and pred[num + 1] != '2':
                        correct += 1
                else:
                    if pred[num] == '1':
                        for j in range(num + 1, len(gold)):
                            if gold[j] == '2':
                                if pred[j] == '2' and j < len(gold) - 1:
                                    continue
                                elif pred[j] == '2' and j == len(gold) - 1:
                                    correct += 1
                                    break
                                else:
                                    break

                            else:
                                if pred[j] != '2':
                                    correct += 1
                                break
            else:
                if pred[num] == '1':
                    correct += 1
    # count predict
    for num in range(len(pred)):
        if pred[num] == '1':
            predicted += 1
    # count gold
    for num in range(len(gold)):
        if gold[num] == '1':
            relevant += 1

    precision = float(correct) / (predicted + 1e-6)
    recall = float(correct) / (relevant + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1


def splitTandO(result):
    target = []
    opinion = []
    for idx in range(len(result)):
        for idy in range(len(result[idx])):
            if result[idx][idy] == 0 or result[idx][idy] == -1:
                target.append('0')
                opinion.append('0')
            elif result[idx][idy] == 1:
                target.append('1')
                opinion.append('0')
            elif result[idx][idy] == 2:
                target.append('2')
                opinion.append('0')
            elif result[idx][idy] == 3:
                target.append('0')
                opinion.append('1')
            elif result[idx][idy] == 4:
                target.append('0')
                opinion.append('2')
    return target, opinion


def fmeasure_strict(pred_relations, raw_Ids):
    goldTotal = 0
    correct = 0
    predictTotal = 0
    for idx in range(len(pred_relations)):
        standard = raw_Ids[idx].gold_relations
        pred = pred_relations[idx]
        goldTotal += len(standard)
        predictTotal += len(pred)
        for r in standard:
            if r in pred:
                correct += 1
    precision = float(correct) / (predictTotal + 1e-6)
    recall = float(correct) / (goldTotal + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1

def make_relation(R_tensor, instance_text, thred):
    total_result = []
    for idx in range(len(instance_text)):
        opinionList = []
        targetList = []
        relationResult = []
        for idy in range(len(instance_text[idx])):
            if instance_text[idx][idy] == 3:
                if idy == len(instance_text[idx]) - 1:
                    opinionList.append([idy, idy + 1])
                else:
                    for k in range(idy + 1, len(instance_text[idx])):
                        if instance_text[idx][k] != 4:
                            opinionList.append([idy, k])
                            break
                        elif instance_text[idx][k] == 4 and k == len(instance_text[idx]) - 1:
                            opinionList.append([idy, k + 1])
                            break
            elif instance_text[idx][idy] == 1:
                if idy == len(instance_text[idx]) - 1:
                    targetList.append([idy - 1, idy])
                else:
                    for k in range(idy + 1, len(instance_text[idx])):
                        if instance_text[idx][k] != 2:
                            targetList.append([idy, k])
                            break
                        elif instance_text[idx][k] == 2 and k == len(instance_text[idx]) - 1:
                            targetList.append([idy, k + 1])
                            break
        for o in opinionList:
            for t in targetList:
                score1 = np.sum(R_tensor[idx][o[0]:o[1], t[0]:t[1]]) / (o[1] - o[0])  # *(t[1]-t[0]))
                score2 = np.sum(R_tensor[idx][t[0]:t[1], o[0]:o[1]]) / (t[1] - t[0])  # *(t[1]-t[0]))
                if (score1 + score2) / 2 > thred:
                    if [o[0] - 1, o[1] - 1, t[0] - 1, t[1] - 1] not in relationResult:
                        relationResult.append([o[0] - 1, o[1] - 1, t[0] - 1, t[1] - 1])
        total_result.append(relationResult)
    return total_result


def evaluate(eval_dataloader, test_set, model, output_file_path, args):
    pred_results = []
    gold_results = []
    relation_result = []

    model.eval()

    for step, batch in enumerate(eval_dataloader):
        if args.ifgpu:
            batch = tuple(t.cuda() for t in batch)  # multi-gpu does scattering it-self
        all_input_ids, all_input_mask, all_segment_ids, all_relations, all_labels = batch
        max_seq_len = torch.max(torch.sum(all_input_mask, dim=1))
        all_input_ids = all_input_ids[:, :max_seq_len].contiguous()
        all_input_mask = all_input_mask[:, :max_seq_len].contiguous()
        all_segment_ids = all_segment_ids[:, :max_seq_len].contiguous()
        all_labels = all_labels[:, :max_seq_len].contiguous()
        targetPredict, relationPredict = model(all_input_ids, all_segment_ids, all_input_mask)

        # get real label
        pred_label, gold_label = recover_label(targetPredict, all_labels, all_input_mask)
        pred_results += pred_label
        gold_results += gold_label
        relation_result += list(relationPredict.cpu().data.numpy())

    # evaluate
    TP, TR, TF, OP, OR, OF = get_ner_fmeasure(gold_results, pred_results, args.tagScheme)
    pred_relations = make_relation(relation_result, pred_results, args.inference_threds)
    RP, RR, RF = fmeasure_strict(pred_relations, test_set)

    # write to file
    labelDic = ["O", "B-T", "I-T", "B-P", "I-P", "O"]
    output_file = open(output_file_path, "w", encoding="utf-8")
    for k in range(len(test_set)):
        words = test_set[k].tokens
        pred = pred_results[k]
        gold = test_set[k].labels
        relations = pred_relations[k]
        for j in range(len(gold)):
            output_file.write(words[j + 1] + "\t" + gold[j] + "\t" + labelDic[pred[j + 1]] + "\n")
        output_file.write("#Relations\n")
        for r in relations:
            output_file.write(str(r[0]) + "\t" + str(r[1]) + "\t" + str(r[2]) + "\t" + str(r[3]) + "\n")
        output_file.write("\n")
    output_file.close()


    return RP, RR, RF, TP, TR, TF, OP, OR, OF



def bert_load_state_dict(model, state_dict):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
    return model


def read_data(train_features, type, batchsize):
    assert type in ["train", "test"]
    all_input_ids = torch.tensor([f.token_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.token_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segmentId for f in train_features], dtype=torch.long)
    all_relations = torch.tensor([f.relations for f in train_features], dtype=torch.long)
    all_labels = torch.tensor([f.label_ids for f in train_features], dtype=torch.long)
    all_labels[:, :1] = torch.ones(all_labels.size(0), 1).long()
    if type == "train":
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_relations, all_labels)
        train_sampler = RandomSampler(train_data)
    else:
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_relations, all_labels)
        train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batchsize)
    return train_dataloader


def main(args):
    if not os.path.exists(args.test_eval_dir):
        os.makedirs(args.test_eval_dir)
    if not os.path.exists(args.eval_dir):
        os.makedirs(args.eval_dir)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    #### print config ####
    print(args)

    #### add label ####
    label_alphabet = Alphabet('label', True)
    label_alphabet.add("O")
    label_alphabet.add("B-T")
    label_alphabet.add("I-T")
    label_alphabet.add("B-P")
    label_alphabet.add("I-P")

    #### read data
    print("Loading data....")
    datasets = torch.load(args.data)
    train_set = datasets["train"]
    test_set = datasets["test"]
    train_dataloader = read_data(train_set, "train", args.batchSize)
    eval_dataloader = read_data(test_set, "test", args.batchSize)

    #### load BERT config ####
    print("Loading BERT config....")
    bert_config = BertConfig.from_json_file(args.bert_json_dir)

    #### defined model ####
    model = opinionMining(args, bert_config, label_alphabet)
    if args.mode == "test":
        assert args.test_model != ""
        model = torch.load(args.test_model)
        test_start = time.time()
        # evaluate
        RP, RR, RF, TP, TR, TF, OP, OR, OF = evaluate(
            eval_dataloader, test_set, model, args.test_eval_dir + "/test_output", args)
        test_finish = time.time()
        test_cost = test_finish - test_start
        print("test: time: %.2fs, speed: %.2fst/s" % (test_cost, 0))
        print("relation result: Precision: %.4f; Recall: %.4f; F1: %.4f" % (RP, RR, RF))
        print("target result: Precision: %.4f; Recall: %.4f; F1: %.4f" % (TP, TR, TF))
        print("opinion result: Precision: %.4f; Recall: %.4f; F1: %.4f" % (OP, OR, OF))
    else:
        print("Loading model from pretrained checkpoint: " + args.bert_checkpoint_dir)
        model = bert_load_state_dict(model, torch.load(args.bert_checkpoint_dir, map_location='cpu'))

        #### define optimizer ####
        num_train_steps = int(len(train_set) / args.batchSize * args.iteration)
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if "bert" in n], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if "bert" not in n], 'lr': args.lr_rate, 'weight_decay': 0.01}]
        optimizer_grouped_parameters_r = [
            {'params': [p for n, p in param_optimizer if "bert" in n], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if "relation" in n], 'lr': args.R_lr_rate, 'weight_decay': 0.01}]
        optimizer = BERTAdam(optimizer_grouped_parameters,
                             lr=2e-05,
                             warmup=0.1,
                             t_total=num_train_steps)
        optimizer_r = BERTAdam(optimizer_grouped_parameters_r,
                              lr=2e-05,
                              warmup=0.1,
                              t_total=num_train_steps)

        #### train ####
        print("start training......")
        best_Score = -10000
        lr = args.lr_rate
        for idx in range(args.iteration):
            epoch_start = time.time()
            temp_start = epoch_start
            print("Epoch: %s/%s" % (idx, args.iteration))

            if idx>10:
                lr = lr*args.lr_decay
                print(lr)
                optimizer.param_groups[1]["lr"] = lr
                optimizer_r.param_groups[1]["lr"] = lr

            sample_loss = 0
            total_loss = 0
            right_target_token = 0
            whole_target_token = 0
            right_relation_token = 0
            whole_relation_token = 0

            model.train()
            model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                if args.ifgpu:
                    batch = tuple(t.cuda() for t in batch)
                all_input_ids, all_input_mask, all_segment_ids, all_relations, all_labels = batch
                max_seq_len = torch.max(torch.sum(all_input_mask, dim=1))
                all_input_ids = all_input_ids[:, :max_seq_len].contiguous()
                all_input_mask = all_input_mask[:, :max_seq_len].contiguous()
                all_segment_ids = all_segment_ids[:, :max_seq_len].contiguous()
                all_relations = all_relations[:, :max_seq_len, :max_seq_len].contiguous()
                all_labels = all_labels[:, :max_seq_len].contiguous()
                tloss, rloss, targetPredict, relationPredict = model.neg_log_likelihood_loss(all_input_ids,
                                                                                             all_segment_ids,
                                                                                             all_labels,
                                                                                             all_relations,
                                                                                             all_input_mask)
                # check right number
                targetRight, targetWhole = targetPredictCheck(targetPredict, all_labels, all_input_mask)
                relationRight, relationWhole = relationPredictCheck(relationPredict, all_relations)

                # cal right and whole label number
                right_target_token += targetRight
                whole_target_token += targetWhole
                right_relation_token += relationRight
                whole_relation_token += relationWhole
                # cal loss
                sample_loss += rloss.data[0] + tloss.data[0]
                total_loss += rloss.data[0] + tloss.data[0]
                # print train info
                if step % 20 == 0:
                    temp_time = time.time()
                    temp_cost = temp_time - temp_start
                    temp_start = temp_time
                    print("     Instance: %s; Time: %.2fs; loss: %.4f; target_acc: %s/%s=%.4f; relation_acc: %s/%s=%.4f" % (
                        step * args.batchSize, temp_cost, sample_loss, right_target_token, whole_target_token,
                        (right_target_token + 0.) / whole_target_token, right_relation_token, whole_relation_token,
                        (right_relation_token + 0.) / whole_relation_token))
                    if sample_loss > 1e8 or str(sample_loss) == "nan":
                        print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                        exit(1)
                    sys.stdout.flush()
                    sample_loss = 0

                if step % 2 == 0:
                    loss = 9*rloss + tloss  #
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    rloss.backward()
                    optimizer_r.step()
                    optimizer_r.zero_grad()

            temp_time = time.time()
            temp_cost = temp_time - temp_start
            print("     Instance: %s; Time: %.2fs; loss: %.4f; target_acc: %s/%s=%.4f; relation_acc: %s/%s=%.4f" % (
                step * args.batchSize, temp_cost, sample_loss, right_target_token, whole_target_token,
                (right_target_token + 0.) / whole_target_token, right_relation_token, whole_relation_token,
                (right_relation_token + 0.) / whole_relation_token))

            epoch_finish = time.time()
            epoch_cost = epoch_finish - epoch_start
            print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
                idx, epoch_cost, len(train_set) / epoch_cost, total_loss))
            print("totalloss:", total_loss)
            if total_loss > 1e8 or str(total_loss) == "nan":
                print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                exit(1)

            # evaluate
            RP, RR, RF, TP, TR, TF, OP, OR, OF = evaluate(
                eval_dataloader, test_set, model, args.eval_dir + "/test_output_" + str(idx), args)
            test_finish = time.time()
            test_cost = test_finish - epoch_finish
            current_Score = RF

            print("test: time: %.2fs, speed: %.2fst/s" % (test_cost, 0))
            print("relation result: Precision: %.4f; Recall: %.4f; F1: %.4f" % (RP, RR, RF))
            print("target result: Precision: %.4f; Recall: %.4f; F1: %.4f" % (TP, TR, TF))
            print("opinion result: Precision: %.4f; Recall: %.4f; F1: %.4f" % (OP, OR, OF))

            if current_Score > best_Score:
                print("Exceed previous best f score with target f: %.4f and opinion f: %.4f and relation f: %.4f" % (
                    TF, OF, RF))
                model_name = args.model_dir + "/modelFinal.model"
                print("Save current best model in file:", model_name)
                torch.save(model, model_name)
                best_Score = current_Score

            gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="test", choices=["train", "test"])
    parser.add_argument('--data', type=str, default="./data/2014Lap.pt")

    ## if test
    parser.add_argument('--test_model', type=str, default="./model/2014Lap2/modelFinal.model")
	parser.add_argument('--test_eval_dir', type=str, default="./test_eval/2014Lap2")

    ## if train
    parser.add_argument('--model_dir', type=str, default="./model/2014Lap2")
    parser.add_argument('--eval_dir', type=str, default="./eval/2014Lap2")
    parser.add_argument('--bert_json_dir', type=str,
                        default="/home/ramon/chenshaowei_summer/IJCAI2020_Rebuttal/bert-base-uncased/bert_config.json")
    parser.add_argument('--bert_checkpoint_dir', type=str,
                        default="/home/ramon/chenshaowei_summer/IJCAI2020_Rebuttal/bert-base-uncased/pytorch_model.bin")

    parser.add_argument('--tagScheme', type=str, default="BIO")
    parser.add_argument('--ifgpu', type=bool, default=True)

    parser.add_argument('--target_hidden_dim', type=int, default=250)
    parser.add_argument('--relation_hidden_dim', type=int, default=250)
    parser.add_argument('--relation_attention_dim', type=int, default=250)
    parser.add_argument('--relation_threds', type=float, default=0.1)
    parser.add_argument('--inference_threds', type=float, default=0.5)
    parser.add_argument('--iteration', type=int, default=70)
    parser.add_argument('--batchSize', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr_rate', type=float, default=0.001)
    parser.add_argument('--R_lr_rate', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.98)
    parser.add_argument('--step', type=int, default=1)

    args = parser.parse_args()
    main(args)