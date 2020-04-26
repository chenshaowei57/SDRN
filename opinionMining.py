# -*- coding: utf-8 -*-
# @Author: Shaowei Chen,     Contact: chenshaowei0507@163.com
# @Date:   2020-4-26 16:47:32

import torch
import torch.nn as nn
from relationAttention import RelationAttention
from crf_new import CRF
from bert.modeling import BertModel, BERTLayerNorm
import threading
import torch.nn.functional as F
import torch.nn.init as init


class opinionMining(nn.Module):
    def __init__(self, args, config, label_alphabet):
        super(opinionMining, self).__init__()
        print("build network...")
        self.gpu = args.ifgpu
        self.label_size = label_alphabet.size()
        self.bert_encoder_dim = config.hidden_size
        self.target_hidden_dim = args.target_hidden_dim
        self.relation_hidden_dim = args.relation_hidden_dim
        self.relation_threds = args.relation_threds
        self.drop = args.dropout
        self.step = args.step

        # encoder
        self.bert = BertModel(config)

        # target syn
        self.targetSyn_r = nn.Parameter(torch.Tensor(self.target_hidden_dim, self.bert_encoder_dim))
        self.targetSyn_s = nn.Parameter(torch.Tensor(self.target_hidden_dim, self.bert_encoder_dim))
        # relation syn
        self.relationSyn_u = nn.Parameter(torch.Tensor(self.relation_hidden_dim, self.bert_encoder_dim))
        self.relationSyn_s = nn.Parameter(torch.Tensor(self.relation_hidden_dim, self.bert_encoder_dim))
        init.xavier_uniform(self.targetSyn_r)
        init.xavier_uniform(self.targetSyn_s)
        init.xavier_uniform(self.relationSyn_u)
        init.xavier_uniform(self.relationSyn_s)

        # crf
        self.targetHidden2Tag = nn.Parameter(torch.Tensor(self.label_size + 2, self.target_hidden_dim))
        self.targetHidden2Tag_b = nn.Parameter(torch.Tensor(1, self.label_size + 2))
        init.xavier_uniform(self.targetHidden2Tag)
        init.xavier_uniform(self.targetHidden2Tag_b)

        self.crf = CRF(self.label_size, self.gpu)

        # relation
        self.relationAttention = RelationAttention(args)

        # other
        self.dropout = nn.Dropout(self.drop)
        self.softmax = nn.Softmax(dim=2)

        if self.gpu:
            self.bert = self.bert.cuda()
            self.targetSyn_r.data = self.targetSyn_r.cuda()
            self.targetSyn_s.data = self.targetSyn_s.cuda()
            self.relationSyn_u.data = self.relationSyn_u.cuda()
            self.relationSyn_s.data = self.relationSyn_s.cuda()
            self.targetHidden2Tag.data = self.targetHidden2Tag.cuda()
            self.targetHidden2Tag_b.data = self.targetHidden2Tag_b.cuda()
            self.relationAttention = self.relationAttention.cuda()
            self.dropout = self.dropout.cuda()
            self.softmax = self.softmax.cuda()

        def init_weights(module):
           if isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)

        self.apply(init_weights)

    def neg_log_likelihood_loss(self, all_input_ids, all_segment_ids, all_labels, all_relations, all_input_mask):
        batch_size = all_input_ids.size(0)
        seq_len = all_input_ids.size(1)
        maskTemp1 = all_input_mask.view(batch_size, 1, seq_len).repeat(1, seq_len, 1)
        maskTemp2 = all_input_mask.view(batch_size, seq_len, 1).repeat(1, 1, seq_len)
        maskMatrix = maskTemp1 * maskTemp2

        targetPredictScore, r_tensor = self.mainStructure(maskMatrix, all_input_ids, all_segment_ids, self.step,
                                                          all_input_mask)
        # target Loss
        target_loss = self.crf.neg_log_likelihood_loss(targetPredictScore, all_input_mask.byte(), all_labels)
        scores, tag_seq = self.crf._viterbi_decode(targetPredictScore, all_input_mask.byte())
        target_loss = target_loss / batch_size

        # relation Loss
        weight = torch.FloatTensor([0.01, 1.0]).cuda()
        relation_loss_function = nn.CrossEntropyLoss(weight=weight)
        relationScoreLoss = r_tensor.view(-1, 1)
        relationlabelLoss = all_relations.view(batch_size * seq_len * seq_len)
        relationScoreLoss = torch.cat([1 - relationScoreLoss, relationScoreLoss], 1)
        relation_loss = relation_loss_function(relationScoreLoss, relationlabelLoss)

        return target_loss, relation_loss, tag_seq, r_tensor

    def forward(self, all_input_ids, all_segment_ids, all_input_mask):
        batch_size = all_input_ids.size(0)
        seq_len = all_input_ids.size(1)
        maskTemp1 = all_input_mask.view(batch_size, 1, seq_len).repeat(1, seq_len, 1)
        maskTemp2 = all_input_mask.view(batch_size, seq_len, 1).repeat(1, 1, seq_len)
        maskMatrix = maskTemp1 * maskTemp2

        targetPredictScore, r_tensor = self.mainStructure(maskMatrix, all_input_ids, all_segment_ids, self.step,
                                                          all_input_mask)
        scores, tag_seq = self.crf._viterbi_decode(targetPredictScore, all_input_mask.byte())

        return tag_seq, r_tensor

    def mainStructure(self, maskMatrix, all_input_ids, all_segment_ids, steps, all_input_mask):

        batch_size = all_input_ids.size(0)
        seq_len = all_input_ids.size(1)
        # bert
        all_encoder_layers, _ = self.bert(all_input_ids, all_segment_ids, all_input_mask)
        sequence_output = all_encoder_layers[-1]
        sequence_output = self.dropout(sequence_output)

        # T tensor and R tensor
        t_tensor = torch.zeros(batch_size, seq_len, seq_len)
        r_tensor = torch.zeros(batch_size, seq_len, seq_len)
        if self.gpu:
            t_tensor = t_tensor.cuda()
            r_tensor = r_tensor.cuda()

        for i in range(steps):
            # target syn
            r_temp = r_tensor.ge(self.relation_threds).float()
            r_tensor = r_tensor * r_temp # b x s x s
            target_weighted = torch.bmm(r_tensor, sequence_output)
            target_div = torch.sum(r_tensor, 2)
            targetIfZero = target_div.eq(0).float()
            target_div = target_div + targetIfZero
            target_div = target_div.unsqueeze(2).repeat(1, 1, self.bert_encoder_dim)
            target_r = torch.div(target_weighted, target_div)
            target_hidden = F.linear(sequence_output, self.targetSyn_s, None) + F.linear(target_r, self.targetSyn_r, None)
            target_hidden = F.tanh(target_hidden)

            # relation syn
            relation_weighted = torch.bmm(t_tensor, sequence_output)
            relation_div = torch.sum(t_tensor, 2)
            relationIfZero = relation_div.eq(0).float()
            relation_div = relation_div + relationIfZero
            relation_div = relation_div.unsqueeze(2).repeat(1, 1, self.bert_encoder_dim)
            relation_a = torch.div(relation_weighted, relation_div)
            relation_hidden = F.linear(sequence_output, self.relationSyn_s, None)+F.linear(relation_a, self.relationSyn_u, None)
            relation_hidden = F.tanh(relation_hidden)

            # crf
            targetPredictInput = F.linear(target_hidden, self.targetHidden2Tag, self.targetHidden2Tag_b)#self.targetHidden2Tag(target_hidden)

            # Relation Attention
            relationScore = self.relationAttention(relation_hidden)


            # update T_tensor
            tag_score, tag_seq = self.crf._viterbi_decode(targetPredictInput, all_input_mask.byte())
            threads = []
            temp_T_tensor = torch.zeros(batch_size, seq_len, seq_len)
            if self.gpu:
                temp_T_tensor = temp_T_tensor.cuda()
            for i in range(batch_size):
                t = threading.Thread(target=self.makeEntity, args=(i, tag_seq[i, :], temp_T_tensor, seq_len))
                threads.append(t)
            for i in range(batch_size):
                threads[i].start()
            for i in range(batch_size):
                threads[i].join()
            tag_score_final = tag_score.unsqueeze(2).repeat(1, 1, seq_len)+tag_score.unsqueeze(1).repeat(1, seq_len, 1)
            t_tensor = tag_score_final * temp_T_tensor

            # Update R_tensor
            r_tensor = relationScore * (maskMatrix.float())

        return targetPredictInput, r_tensor
    def makeEntity(self, idx, tag_seq, temp_T_tensor, seq_len):
        # don't consider the entity which starts with "I-X"
        tag_seq = tag_seq.cpu()
        Abegin = -1
        Aend = -1
        Obegin = -1
        Oend = -1
        for idy in range(seq_len):
            if tag_seq[idy] in [0, 1, 2, 4]:
                if Abegin != -1:
                    temp_T_tensor[idx, Abegin:Aend, Abegin:Aend] = torch.ones(Aend - Abegin, Aend - Abegin)
                    Abegin = -1
                    Aend = -1
                if Obegin != -1:
                    temp_T_tensor[idx, Obegin:Oend, Obegin:Oend] = torch.ones(Oend - Obegin, Oend - Obegin)
                    Obegin = -1
                    Oend = -1
            if tag_seq[idy] == 2:
                Abegin = idy
                Aend = idy + 1
            if tag_seq[idy] == 3 and Abegin != -1:
                Aend += 1
            if tag_seq[idy] == 4:
                Obegin = idy
                Oend = idy + 1
            if tag_seq[idy] == 5 and Obegin != -1:
                Oend += 1
        if Abegin != -1:
            temp_T_tensor[idx, Abegin:Aend, Abegin:Aend] = torch.ones(Aend - Abegin, Aend - Abegin)
        if Obegin != -1:
            temp_T_tensor[idx, Obegin:Oend, Obegin:Oend] = torch.ones(Oend - Obegin, Oend - Obegin)

        return temp_T_tensor