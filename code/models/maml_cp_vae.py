import math
import utils.data_processor
import utils.nn
import models.vae
from torch.cuda import init
from torch import optim
import numpy as np
import datetime as dt
import copy
import torch
import os
from utils.text_utils import MonoTextData
from config.model_config import default_maml_mconf
import sys
sys.path.append("..")


# ----------------

def get_coordinates(a, b, p):
    pa = p - a
    ba = b - a
    t = torch.sum(pa * ba) / torch.sum(ba * ba)
    d = torch.norm(pa - t * ba, 2)
    return t, d


def cal_log_density(mu, logvar, z):
    nz = z.shape[1]
    dev = z.expand_as(mu) - mu
    var = logvar.exp()
    log_density = -0.5 * ((dev ** 2) / var).sum(-1) - \
        0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))
    return log_density.mean().item()


class MAMLCP_VAE(object):

    def __init__(self, device, vocab, num_tasks=7, mconf=default_maml_mconf):
        self.num_tasks = num_tasks
        self.device = device
        self.m = models.vae.CP_VAE(device, vocab, mconf)
        self.mconf = mconf

    def train_maml(self, support_batch_generator, support_feat, query_batch_generator, query_feat, epochs, init_epoch=0):
        num_batches = len(support_batch_generator[0])

        # optimizers
        meta_enc_optimizer = optim.Adam(
            self.m.enc_param, lr=self.mconf.meta_enc_lr)
        meta_dec_optimizer = optim.SGD(
            self.m.decoder.parameters(), lr=self.mconf.meta_dec_lr)

        sub_enc_optimizer = optim.Adam(
            self.m.enc_param, lr=self.mconf.sub_enc_lr)
        sub_dec_optimizer = optim.SGD(
            self.m.decoder.parameters(), lr=self.mconf.sub_dec_lr)

        # meta-train from init_epoch to epochs
        for epoch in range(init_epoch, epochs):
            total_epoch_loss = 0.0
            init_state = copy.deepcopy(self.m.state_dict())
            # for each batch
            for b in range(num_batches):
                support_batch = support_batch_generator[0][b], support_batch_generator[1][b]
                query_batch = query_batch_generator[0][b], query_batch_generator[1][b]
                support_loss = []
                query_loss = []
                # for each sub-task
                for t in range(self.mconf.num_tasks):
                    batch_task = [support_batch[i][t]
                                  for i in range(len(support_batch))]
                    feat_task = support_feat[t]
                    query_feat_task = query_feat[t]
                    self.m.load_state_dict(init_state)
                    sub_dec_optimizer.zero_grad()
                    sub_enc_optimizer.zero_grad()

                    init_state = copy.deepcopy(self.m.state_dict())

                    # inner-loop updates
                    for step in range(self.mconf.num_updates):
                        loss, * \
                            _ = self.m._feed_batch(
                                feat_task, batch_task[0], batch_task[1])
                        sub_dec_optimizer.zero_grad()
                        sub_enc_optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(
                            self.m.parameters(), 5.0)
                        sub_dec_optimizer.step()
                        sub_enc_optimizer.step()

                    # compute task-specific query loss
                    batch_task = [query_batch[i][t]
                                  for i in range(len(query_batch))]
                    loss, *_ = self.m._feed_batch(
                        query_feat_task, batch_task[0], batch_task[1])
                    query_loss.append(loss)

                # restore the initial parameters
                self.m.load_state_dict(init_state)

                # average query loss for each sub-task
                avg_query_loss = torch.stack(
                    query_loss, dim=0).sum() / self.num_tasks
                total_epoch_loss += avg_query_loss.item()
                meta_dec_optimizer.zero_grad()
                meta_enc_optimizer.zero_grad()
                avg_query_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.m.parameters(), 5.0)
                meta_enc_optimizer.step()
                meta_dec_optimizer.step()

                # set the new initial parameters
                init_state = copy.deepcopy(self.m.state_dict())

                # logging
                timestamp = dt.datetime.now().isoformat()
                msg = "[{}]: batch {}/{}, epoch {}/{}, meta_total_loss {:g}".format(
                    timestamp, b+1, num_batches, epoch, epochs, avg_query_loss.detach().item())
                print(msg)
                if b == 2:
                    print(f'finish {b+1} batch')
                    exit(0)
            print("--------")
            print("epoch {}/{}: acc_total_loss {:g}".format(epoch +
                                                            1, epochs, total_epoch_loss))

    def fine_tune(self, feat, batch_generator, epochs, init_epoch=0):
        self.m._train(feat, batch_generator, epochs, init_epoch)

    def evaluate(self, val_batch_generator):
        return self.m.evaluate(*val_batch_generator)

    def infer(self, task_id, train_data_pth, train_feat_pth, dev_data_pth, dev_feat_pth, test_data_pth, test_feat_pth, device):
        # with style label
        train_data = MonoTextData(train_data_pth, True)
        train_feat = np.load(train_feat_pth)
        vocab = train_data.vocab
        dev_data = MonoTextData(dev_data_pth, True, vocab=vocab)
        dev_feat = np.load(dev_feat_pth)
        test_data = MonoTextData(test_data_pth, True, vocab=vocab)
        test_feat = np.load(test_feat_pth)

        self.m.eval()
        train_data, train_feat = train_data.create_data_batch_feats(
            32, train_feat, device)
        print("Collecting training distributions...")
        mus, logvars = [], []
        step = 0
        for batch_data, batch_feat in zip(train_data, train_feat):
            mu1, logvar1 = self.m.lstm_encoder(batch_data)
            mu2, logvar2 = self.m.mlp_encoder(batch_feat)
            r, _ = self.m.mlp_encoder(batch_feat, True)
            p = self.m.get_var_prob(r)
            mu = torch.cat([mu1, mu2], -1)
            logvar = torch.cat([logvar1, logvar2], -1)
            mus.append(mu.detach().cpu())
            logvars.append(logvar.detach().cpu())
            step += 1
            if step % 100 == 0:
                torch.cuda.empty_cache()
        mus = torch.cat(mus, 0)
        logvars = torch.cat(logvars, 0)

        neg_sample = dev_feat[:10]
        neg_inputs = torch.tensor(
            neg_sample, dtype=torch.float, requires_grad=False, device=device)
        r, _ = self.m.mlp_encoder(neg_inputs, True)
        p = self.m.get_var_prob(r).mean(0, keepdim=True)
        neg_idx = torch.max(p, 1)[1].item()

        pos_sample = dev_feat[-10:]
        pos_inputs = torch.tensor(
            pos_sample, dtype=torch.float, requires_grad=False, device=device)
        r, _ = self.m.mlp_encoder(pos_inputs, True)
        p = self.m.get_var_prob(r).mean(0, keepdim=True)
        top2 = torch.topk(p, 2, 1)[1].squeeze()
        if top2[0].item() == neg_idx:
            print("Collision!!! Use second most as postive.")
            pos_idx = top2[1].item()
        else:
            pos_idx = top2[0].item()
        other_idx = -1
        for i in range(3):
            if i not in [pos_idx, neg_idx]:
                other_idx = i
                break

        print("Negative: %d" % neg_idx)
        print("Positive: %d" % pos_idx)

        sep_id = -1
        for idx, x in enumerate(test_data.labels):
            if x == 1:
                sep_id = idx
                break

        bsz = 64
        ori_logps = []
        tra_logps = []
        pos_z2 = self.m.mlp_encoder.var_embedding[pos_idx:pos_idx + 1]
        neg_z2 = self.m.mlp_encoder.var_embedding[neg_idx:neg_idx + 1]
        other_z2 = self.m.mlp_encoder.var_embedding[other_idx:other_idx + 1]
        _, d0 = get_coordinates(pos_z2[0], neg_z2[0], other_z2[0])
        ori_obs = []
        tra_obs = []
        o_pth = f"../output/{self.mconf.corpus}/t{task_id}"
        if not os.path.exists(os.path.dirname(o_pth)):
            os.mkdir(os.path.dirname(o_pth))
        with open(os.path.join(o_pth, 'generated_results.txt'), "w") as f:
            idx = 0
            step = 0
            n_samples = len(test_data.labels)
            while idx < n_samples:
                label = test_data.labels[idx]
                _idx = idx + bsz if label else min(idx + bsz, sep_id)
                _idx = min(_idx, n_samples)
                var_id = neg_idx if label else pos_idx
                text, _ = test_data._to_tensor(
                    test_data.data[idx:_idx], batch_first=False, device=device)
                feat = torch.tensor(
                    test_feat[idx:_idx], dtype=torch.float, requires_grad=False, device=device)
                z1, _ = self.m.lstm_encoder(text[:min(text.shape[0], 10)])
                ori_z2, _ = self.m.mlp_encoder(feat)
                tra_z2 = self.m.mlp_encoder.var_embedding[var_id:var_id + 1, :].expand(
                    _idx - idx, -1)
                texts = self.m.decoder.beam_search_decode(z1, tra_z2)
                for text in texts:
                    f.write("%d\t%s\n" % (1 - label, " ".join(text[1:-1])))

                ori_z = torch.cat([z1, ori_z2], -1)
                tra_z = torch.cat([z1, tra_z2], -1)
                for i in range(_idx - idx):
                    ori_logps.append(cal_log_density(
                        mus, logvars, ori_z[i:i + 1].cpu()))
                    tra_logps.append(cal_log_density(
                        mus, logvars, tra_z[i:i + 1].cpu()))

                idx = _idx
                step += 1
                if step % 100 == 0:
                    print(step, idx)
        print("inference for corpus {}, task {} finished".format(
            self.mconf.corpus, task_id))

    def save_model(self, path):
        self.m.save_model(path)

    def load_model(self, path):
        self.m.load_model(path)
