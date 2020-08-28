# Copyright (c) 2020-present, Royal Bank of Canada.
# Copyright (c) 2020-present, Juxian He
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#################################################################################################
# Code is based on the VAE lagging encoder (https://arxiv.org/abs/1901.05534) implementation
# from https://github.com/jxhe/vae-lagging-encoder by Junxian He
#################################################################################################


import random
import datetime as dt
import time
import numpy as np
from config.model_config import default_mconf
from .base_network import LSTMEncoder, LSTMDecoder, SemMLPEncoder, SemLSTMEncoder
from .utils import uniform_initializer, value_initializer, gumbel_softmax
import torch
from torch import optim, set_flush_denormal
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")


def log_sum_exp(value, dim=None, keepdim=False):
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


class uniform_initializer(object):
    def __init__(self, stdv):
        self.stdv = stdv

    def __call__(self, tensor):
        nn.init.uniform_(tensor, -self.stdv, self.stdv)


class value_initializer(object):
    def __init__(self, value):
        self.value = value

    def __call__(self, tensor):
        with torch.no_grad():
            tensor.fill_(0.)
            tensor += self.value


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape, requires_grad=True).cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard


class VAE(nn.Module):
    def __init__(self, ni, nz, enc_nh, dec_nh, dec_dropout_in, dec_dropout_out, vocab, device):
        super(VAE, self).__init__()
        model_init = uniform_initializer(0.01)
        enc_embed_init = uniform_initializer(0.1)
        dec_embed_init = uniform_initializer(0.1)
        self.encoder = LSTMEncoder(ni, enc_nh, nz, len(
            vocab), model_init, enc_embed_init)
        self.decoder = LSTMDecoder(
            ni, dec_nh, nz, dec_dropout_in, dec_dropout_out, vocab,
            model_init, dec_embed_init, device)

    def cuda(self):
        self.encoder.cuda()
        self.decoder.cuda()

    def encode(self, x, nsamples=1):
        return self.encoder.encode(x, nsamples)

    def decode(self, x, z):
        return self.decoder(x, z)

    def loss(self, x, nsamples=1):
        z, KL = self.encode(x, nsamples)
        outputs = self.decode(x[:-1], z)
        return outputs, KL

    def calc_mi_q(self, x):
        return self.encoder.calc_mi(x)


class DecomposedVAE(nn.Module):
    def __init__(self, lstm_ni, lstm_nh, lstm_nz, mlp_ni, mlp_nz,
                 dec_ni, dec_nh, dec_dropout_in, dec_dropout_out,
                 vocab, n_vars, device, text_only):
        super(DecomposedVAE, self).__init__()
        model_init = uniform_initializer(0.01)
        enc_embed_init = uniform_initializer(0.1)
        dec_embed_init = uniform_initializer(0.1)
        self.lstm_encoder = LSTMEncoder(
            lstm_ni, lstm_nh, lstm_nz, len(vocab), model_init, enc_embed_init)
        if text_only:
            self.mlp_encoder = SemLSTMEncoder(
                lstm_ni, lstm_nh, mlp_nz, len(vocab), n_vars, model_init, enc_embed_init, device)
        else:
            self.mlp_encoder = SemMLPEncoder(
                mlp_ni, mlp_nz, n_vars, model_init, device)
        self.decoder = LSTMDecoder(
            dec_ni, dec_nh, lstm_nz + mlp_nz, dec_dropout_in, dec_dropout_out, vocab,
            model_init, dec_embed_init, device)

    def encode_syntax(self, x, nsamples=1):
        return self.lstm_encoder.encode(x, nsamples)

    def encode_semantic(self, x, nsamples=1):
        return self.mlp_encoder.encode(x, nsamples)

    def decode(self, x, z):
        return self.decoder(x, z)

    def var_loss(self, pos, neg, neg_samples):
        r, _ = self.mlp_encoder(pos, True)
        pos = self.mlp_encoder.encode_var(r)
        pos_scores = (pos * r).sum(-1)
        pos_scores = pos_scores / torch.norm(r, 2, -1)
        pos_scores = pos_scores / torch.norm(pos, 2, -1)
        neg, _ = self.mlp_encoder(neg)
        neg_scores = (neg * r.repeat(neg_samples, 1)).sum(-1)
        neg_scores = neg_scores / torch.norm(r.repeat(neg_samples, 1), 2, -1)
        neg_scores = neg_scores / torch.norm(neg, 2, -1)
        neg_scores = neg_scores.view(neg_samples, -1)
        pos_scores = pos_scores.unsqueeze(0).repeat(neg_samples, 1)
        raw_loss = torch.clamp(1 - pos_scores + neg_scores, min=0.).mean(0)
        srec_loss = raw_loss.mean()
        reg_loss = self.mlp_encoder.orthogonal_regularizer()
        return srec_loss, reg_loss, raw_loss.sum()

    def get_var_prob(self, inputs):
        _, p = self.mlp_encoder.encode_var(inputs, True)
        return p

    def loss(self, x, feat, tau=1.0, nsamples=1, no_ic=True):
        z1, KL1 = self.encode_syntax(x, nsamples)
        z2, KL2 = self.encode_semantic(feat, nsamples)
        z = torch.cat([z1, z2], -1)
        outputs = self.decode(x[:-1], z)
        if no_ic:
            reg_ic = torch.zeros(10)
        else:
            soft_outputs = gumbel_softmax(outputs, tau)
            log_density = self.lstm_encoder.eval_inference_dist(
                soft_outputs, z1)
            logit = log_density.exp()
            reg_ic = -torch.log(torch.sigmoid(logit))
        return outputs, KL1, KL2, reg_ic

    def calc_mi_q(self, x, feat):
        mi1 = self.lstm_encoder.calc_mi(x)
        mi2 = self.mlp_encoder.calc_mi(feat)
        return mi1, mi2


class CP_VAE(nn.Module):
    def __init__(self, device, vocab, mconf=default_mconf):
        super().__init__()
        self.mconf = mconf
        self.device = device
        model_init = uniform_initializer(0.01)
        enc_embed_init = uniform_initializer(0.1)
        dec_embed_init = uniform_initializer(0.1)

        # args
        self.enc_lr = self.mconf.enc_lr
        self.dec_lr = self.mconf.dec_lr
        self.srec_weight = self.mconf.srec_weight
        self.reg_weight = self.mconf.reg_weight
        self.text_only = False
        self.kl_weight = self.mconf.kl_start
        self.ic_weight = self.mconf.ic_weight
        self.beta1 = self.mconf.beta1
        self.beta2 = self.mconf.beta2
        self.warm_up = self.mconf.warm_up
        self.anneal_rate = (1.0 - self.kl_weight) / \
            (self.warm_up * 544)

        # syntax encoder
        self.lstm_encoder = LSTMEncoder(self.mconf.lstm_ni, self.mconf.lstm_nh,
                                        self.mconf.lstm_nz, len(vocab), model_init, enc_embed_init)
        # semantic encoder
        self.mlp_encoder = SemMLPEncoder(
            self.mconf.mlp_ni, self.mconf.mlp_nz, self.mconf.n_vars, model_init, device)
        # decoder
        self.decoder = LSTMDecoder(self.mconf.dec_ni, self.mconf.dec_nh, self.mconf.lstm_nz+self.mconf.mlp_nz,
                                   self.mconf.dec_dropout_in, self.mconf.dec_dropout_out, vocab, model_init, dec_embed_init, device)
        self.enc_param = list(self.lstm_encoder.parameters()) + \
            list(self.mlp_encoder.parameters())
        # param list
        self.lstm_enc_param = list(self.lstm_encoder.parameters())
        self.mlp_param = [p for name, p in self.mlp_encoder.named_parameters() if name != "var_embedding"]
        self.var_embedding_param = [self.mlp_encoder.var_embedding]
        self.lstm_dec_param = list(self.decoder.parameters())
        self.nn_param = self.lstm_enc_param + self.mlp_param + self.lstm_dec_param
        self.to(device=self.device)

    def encode_syntax(self, x, nsamples=1):
        return self.lstm_encoder.encode(x, nsamples)

    def encode_semantic(self, x, nsamples=1):
        return self.mlp_encoder.encode(x, nsamples)

    def decode(self, x, z):
        return self.decoder(x, z)

    def var_loss(self, pos, neg, neg_samples):
        r, _ = self.mlp_encoder(pos, True)
        pos = self.mlp_encoder.encode_var(r)
        pos_scores = (pos * r).sum(-1)
        pos_scores = pos_scores / torch.norm(r, 2, -1)
        pos_scores = pos_scores / torch.norm(pos, 2, -1)
        neg, _ = self.mlp_encoder(neg)
        neg_scores = (neg * r.repeat(neg_samples, 1)).sum(-1)
        neg_scores = neg_scores / torch.norm(r.repeat(neg_samples, 1), 2, -1)
        neg_scores = neg_scores / torch.norm(neg, 2, -1)
        neg_scores = neg_scores.view(neg_samples, -1)
        pos_scores = pos_scores.unsqueeze(0).repeat(neg_samples, 1)
        raw_loss = torch.clamp(1 - pos_scores + neg_scores, min=0.).mean(0)
        srec_loss = raw_loss.mean()
        reg_loss = self.mlp_encoder.orthogonal_regularizer()
        return srec_loss, reg_loss, raw_loss.sum()

    def get_var_prob(self, inputs):
        _, p = self.mlp_encoder.encode_var(inputs, True)
        return p

    def loss(self, x, feat, tau=1.0, nsamples=1, no_ic=True):
        z1, KL1 = self.encode_syntax(x, nsamples)
        z2, KL2 = self.encode_semantic(feat, nsamples)
        z = torch.cat([z1, z2], -1)
        outputs = self.decode(x[:-1].clone(), z)
        if no_ic:
            reg_ic = torch.zeros(10)
        else:
            soft_outputs = gumbel_softmax(outputs, tau)
            log_density = self.lstm_encoder.eval_inference_dist(
                soft_outputs, z1)
            logit = log_density.exp()
            reg_ic = -torch.log(torch.sigmoid(logit))
        return outputs, KL1, KL2, reg_ic

    def calc_mi_q(self, x, feat):
        mi1 = self.lstm_encoder.calc_mi(x)
        mi2 = self.mlp_encoder.calc_mi(feat)
        return mi1, mi2

    def _feed_batch(self, feat, batch_data, batch_feat):
        sent_len, batch_size = batch_data.size()

        shift = np.random.randint(max(1, sent_len - 9))
        batch_data = batch_data[shift:min(sent_len, shift + 10), :]
        sent_len, batch_size = batch_data.size()

        target = batch_data[1:]
        num_words = (sent_len - 1) * batch_size
        num_sents = batch_size
        self.kl_weight = min(1.0, self.kl_weight + self.anneal_rate)
        beta1 = self.beta1 if self.beta1 else self.kl_weight
        beta2 = self.beta2 if self.beta2 else self.kl_weight

        loss = 0

        sub_iter = 1
        batch_data_enc = batch_data
        batch_feat_enc = batch_feat
        burn_num_words = 0
        burn_pre_loss = 1e4
        burn_cur_loss = 0

        vae_logits, vae_kl1_loss, vae_kl2_loss, reg_ic = self.loss(
            batch_data, batch_feat, no_ic=self.ic_weight == 0)
        vae_logits = vae_logits.view(-1, vae_logits.size(2))
        vae_rec_loss = F.cross_entropy(
            vae_logits, target.view(-1), reduction="none")
        vae_rec_loss = vae_rec_loss.view(-1, batch_size).sum(0)
        vae_loss = vae_rec_loss + beta1 * vae_kl1_loss + beta2 * vae_kl2_loss
        if self.ic_weight > 0:
            vae_loss += self.ic_weight * reg_ic
        vae_loss = vae_loss.mean()
        total_rec_loss = vae_rec_loss.sum().item()
        total_kl1_loss = vae_kl1_loss.sum().item()
        total_kl2_loss = vae_kl2_loss.sum().item()
        loss = loss + vae_loss

        if self.text_only:
            while True:
                idx = np.random.choice(self.nbatch)
                neg_feat = feat[idx]
                if neg_feat.size(1) >= batch_size:
                    break
            idx = np.random.choice(batch_size, batch_size, replace=False)
            neg_feat = neg_feat[:, idx]
            var_loss, reg_loss, var_raw_loss = self.var_loss(
                batch_feat, neg_feat, 1)
        else:
            idx = np.random.choice(feat.shape[1], batch_size * 10)
            neg_feat = torch.tensor(feat[idx], dtype=torch.float,
                                    requires_grad=False, device=self.device)
            srec_loss, reg_loss, srec_raw_loss = self.var_loss(
                batch_feat, neg_feat, 10)
        total_srec_loss = srec_raw_loss.item()
        loss = loss + self.srec_weight * srec_loss + self.reg_weight * reg_loss
        return loss, num_words, num_sents, total_rec_loss, total_kl1_loss, total_kl2_loss, total_srec_loss

    def _train(self, feat, batch_generator, epochs, init_epoch):
        self.train()
        self.batch_generator = batch_generator
        self.nbatch = len(batch_generator[0])
        self.anneal_rate = (1.0 - self.kl_weight) / \
            (self.warm_up * self.nbatch)
        enc_optimizer = optim.Adam(self.enc_param, lr=self.enc_lr)
        dec_optimizer = optim.SGD(self.decoder.parameters(), lr=self.dec_lr)
        for epoch in range(init_epoch, epochs):
            total_rec_loss = 0
            total_kl1_loss = 0
            total_kl2_loss = 0
            total_srec_loss = 0
            num_words = 0
            num_sents = 0
            start_time = time.time()
            # shuffle the batches after every epoch
            batch_ids = list(range(self.nbatch))
            random.shuffle(batch_ids)
            for b in range(self.nbatch):
                batch_data, batch_feat = self.batch_generator[0][batch_ids[b]
                                                                 ], self.batch_generator[1][batch_ids[b]]
                loss, b_num_words, b_num_sents, b_total_rec_loss, b_total_kl1_loss, b_total_kl2_loss, b_total_srec_loss = self._feed_batch(feat,
                                                                                                                                           batch_data, batch_feat)
                num_words += b_num_words
                num_sents += b_num_sents
                total_rec_loss += b_total_rec_loss
                total_kl1_loss += b_total_kl1_loss
                total_kl2_loss += b_total_kl2_loss
                total_srec_loss += b_total_srec_loss
                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                enc_optimizer.step()
                dec_optimizer.step()

                cur_rec_loss = total_rec_loss / num_sents
                cur_kl1_loss = total_kl1_loss / num_sents
                cur_kl2_loss = total_kl2_loss / num_sents
                cur_vae_loss = cur_rec_loss + cur_kl1_loss + cur_kl2_loss
                cur_srec_loss = total_srec_loss / num_sents
                elapsed = time.time() - start_time
                timestamp = dt.datetime.now().isoformat()

                msg = "[{}]: batch {}/{}, epoch {}/{}".format(
                    timestamp, b+1, self.nbatch, epoch+1, epochs)
                msg += "\n\t vae_loss: {:g}, rec_loss: {:g}, kl1_loss: {:g}, kl2_loss: {:g}, srec_loss: {:g}".format(
                    cur_vae_loss, cur_rec_loss, cur_kl1_loss, cur_kl2_loss, cur_srec_loss)
                print(msg)
                total_rec_loss = 0
                total_kl1_loss = 0
                total_kl2_loss = 0
                total_srec_loss = 0
                num_words = 0
                num_sents = 0
                start_time = time.time()

    def evaluate(self, eval_data, eval_feat):
        """
        eval_data: List[batch]
        eval_feat: List[batch]
        """
        self.eval()
        total_rec_loss = 0
        total_kl1_loss = 0
        total_kl2_loss = 0
        total_mi1 = 0
        total_mi2 = 0
        num_sents = 0
        num_words = 0

        with torch.no_grad():
            for batch_data, batch_feat in zip(eval_data, eval_feat):
                sent_len, batch_size = batch_data.size()
                shift = np.random.randint(max(1, sent_len - 9))
                batch_data = batch_data[shift:min(sent_len, shift + 10), :]
                sent_len, batch_size = batch_data.size()
                target = batch_data[1:]

                num_sents += batch_size
                num_words += (sent_len - 1) * batch_size

                vae_logits, vae_kl1_loss, vae_kl2_loss, _ = self.loss(
                    batch_data, batch_feat)
                vae_logits = vae_logits.view(-1, vae_logits.size(2))
                vae_rec_loss = F.cross_entropy(
                    vae_logits, target.view(-1), reduction="none")
                total_rec_loss += vae_rec_loss.sum().item()
                total_kl1_loss += vae_kl1_loss.sum().item()
                total_kl2_loss += vae_kl2_loss.sum().item()

                mi1, mi2 = self.calc_mi_q(batch_data, batch_feat)
                total_mi1 += mi1 * batch_size
                total_mi2 += mi2 * batch_size

        cur_rec_loss = total_rec_loss / num_sents
        cur_kl1_loss = total_kl1_loss / num_sents
        cur_kl2_loss = total_kl2_loss / num_sents
        cur_vae_loss = cur_rec_loss + cur_kl1_loss + cur_kl2_loss
        cur_mi1 = total_mi1 / num_sents
        cur_mi2 = total_mi2 / num_sents
        return cur_vae_loss, cur_rec_loss, cur_kl1_loss, cur_kl2_loss, cur_mi1, cur_mi2

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print("saved model to {}".format(path))

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        print("loaded model from {}".format(path))

    def infer(self):
        raise NotImplementedError
