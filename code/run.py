from utils.text_utils import MonoTextData
import utils.data_processor as data_processor
import utils.data_loader as data_loader
import models.maml_cp_vae as maml_cp_vae
import models.vae as vae
from config import model_config
import os
import sys
import json
import pprint
import pickle
import torch
import numpy as np
import datetime as dt
import logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s", datefmt="%a, %d %b %Y %H:%M:%S")
logger = logging.getLogger(__name__)

# ----------------


def _train_maml(net, mconf, support_batch_generators, support_feats, query_batch_generators, query_feats, vocab, device=torch.device('cpu'), maml_batch_size=16, total_epochs=10, epochs_per_val=2, support_batch_size=32, query_batch_size=8, dump_embeddings=False):
    logger.info("maml learning start......")
    turns = total_epochs // epochs_per_val
    if total_epochs % epochs_per_val:
        turns += 1
    logger.info("preparing val batch generator for each sub-task")
    val_batch_generators = []
    for t in range(1, mconf.num_tasks+1):
        data_pth = f"../data/{mconf.corpus}/val/t{t}.all"
        feat_pth = f"../data/{mconf.corpus}/val/t{t}_glove.npy"
        data = MonoTextData(data_pth, False, vocab=vocab)
        feat = np.load(feat_pth)
        _batch_generator = data.create_data_batch_feats(
            maml_batch_size, feat, device)
        val_batch_generators.append(_batch_generator)
    logger.info(f"{len(val_batch_generators)} val btach_generator obtained")

    best_val_loss = 1e9
    for turn in range(turns):
        init_epoch = turn * epochs_per_val
        end_epoch = min(total_epochs, (turn + 1) * epochs_per_val)
        net.train_maml(
            support_batch_generators,
            support_feats,
            query_batch_generators,
            query_feats,
            epochs=end_epoch,
            init_epoch=init_epoch
        )
        model_file = "epoch-{}.maml".format(end_epoch)
        model_path = mconf.model_save_dir_prefix + model_file
        net.save_model(model_path)
        logger.info(
            "maml-training checkpoint file {} saved at {} epoch".format(model_path, end_epoch))
        mconf.last_maml_ckpt = model_file
        logger.info("maml training epoch {} done".format(end_epoch))
        logger.info("evaluation\n--------")
        val_losses = .0
        for t in range(mconf.num_tasks):
            logger.info("inferring task {} ...".format(t+1))
            losses = net.evaluate(val_batch_generators[t])
            vae_loss, rec_loss, kl1_loss, kl2_loss, *_ = losses
            logger.info("vae loss {}, rec loss {}, kl1 loss {}, kl2 loss {}".format(
                vae_loss, rec_loss, kl1_loss, kl2_loss))
            val_losses += vae_loss
        if best_val_loss > val_losses / mconf.num_tasks:
            best_val_loss = val_losses / mconf.num_tasks
        logger.info("avg val loss for epoch [{}-{}] is {}".format(
            init_epoch+1, end_epoch, val_losses / mconf.num_tasks))


def _fine_tune(net, mconf, feat, batch_generator, vocab, device=torch.device('cpu'), total_epochs=6, epochs_per_val=2, batch_size=64, task_id=1, dump_embeddings=False):

    logger.info("Adapting model to task {} start".format(task_id))
    turns = total_epochs // epochs_per_val
    if total_epochs % epochs_per_val:
        turns += 1
    data_pth = f"../data/{mconf.corpus}/val/t{task_id}.all"
    feat_pth = f"../data/{mconf.corpus}/val/t{task_id}_glove.npy"
    data = MonoTextData(data_pth, False, vocab=vocab)
    val_feat = np.load(feat_pth)
    val_batch_generator = data.create_data_batch_feats(
        batch_size, val_feat, device)
    best_loss = 1e9
    not_improved = 0
    patience = 3
    early_stop_cnt = 0
    for turn in range(turns):
        init_epoch = turn * epochs_per_val
        end_epoch = min(total_epochs, (turn + 1) * epochs_per_val)
        net.fine_tune(
            feat=feat,
            batch_generator=batch_generator,
            epochs=end_epoch,
            init_epoch=init_epoch
        )
        model_file = "epoch-{}.t{}".format(end_epoch, task_id)
        model_path = mconf.model_save_dir_prefix + model_file
        net.save_model(model_path)
        logger.info("maml-fine-tuning checkpoint file {} for task {} saved at epoch {}".format(
            model_path, task_id, end_epoch))
        mconf.last_tsf_ckpts["t{}".format(task_id)] = model_file
        logger.info("evaluatoin\n------")
        vae_loss, rec_loss, kl1_loss, kl2_loss, * \
            _ = net.evaluate(val_batch_generator)
        logger.info("vae loss {}, rec loss {}, kl1 loss {}, kl2 loss {}".format(
            vae_loss, rec_loss, kl1_loss, kl2_loss))
        if vae_loss > best_loss:
            not_improved += 1
            if not_improved >= patience:
                early_stop_cnt += 1
                not_improved = 0
                net.m.dec_lr *= 0.5
        if early_stop_cnt >= 2:
            break
        if vae_loss < best_loss:
            best_loss = vae_loss
            not_improved = 0
            net.save_model(mconf.model_save_dir_prefix + "best.t{}".format(task_id))
            logger.info("best model for task {} saved at epoch {}".format(task_id, end_epoch))


def run_maml(mconf, device, load_data=False, load_model=False, maml_epochs=10, transfer_epochs=6, epochs_per_val=2, infer_task='', maml_batch_size=8, sub_batch_size=32, train_batch_size=64, dump_embeddings=False):
    torch.random.manual_seed(42)
    corpus = mconf.corpus
    data_path = "../data/{}".format(corpus)
    corpus_data_pth = os.path.join(data_path, "text.pretrain")
    corpus_data = MonoTextData(corpus_data_pth, False)
    vocab = corpus_data.vocab
    logger.info("size of whole vocab: {}".format(len(vocab)))
    if maml_epochs > 0 or transfer_epochs > 0:
        logger.info("loading data from maml_cp_vae")
        support_batch_generators = [[], []]
        query_batch_generators = [[], []]
        support_feats = []
        query_feats = []
        for label in ['train', 'val']:
            for t in range(1, mconf.num_tasks+1):
                data_pth = os.path.join(data_path+f"/{label}", f"t{t}.all")
                feat_pth = os.path.join(
                    data_path+f"/{label}", f"t{t}_glove.npy")
                data = MonoTextData(data_pth, False, vocab=vocab)
                feat = np.load(feat_pth)
                if label == 'train':
                    batch_generator = data.create_data_batch_feats(
                        sub_batch_size, feat, device)
                    support_batch_generators[0].append(batch_generator[0])
                    support_batch_generators[1].append(batch_generator[1])
                    support_feats.append(feat)
                else:
                    batch_generator = data.create_data_batch_feats(
                        maml_batch_size, feat, device)
                    query_batch_generators[0].append(batch_generator[0])
                    query_batch_generators[1].append(batch_generator[1])
                    query_feats.append(feat)
        assert len(support_batch_generators[0]) == len(
            query_batch_generators[0]) == mconf.num_tasks == len(support_feats) == len(query_feats)
        support_batch_generators[0] = list(zip(*support_batch_generators[0]))
        support_batch_generators[1] = list(zip(*support_batch_generators[1]))
        query_batch_generators[0] = list(zip(*query_batch_generators[0]))
        query_batch_generators[1] = list(zip(*query_batch_generators[1]))
        logger.info("preparing data for maml_cp_vae done")
        logger.info("Num of support batches: {}".format(
            len(support_batch_generators[0])))
        logger.info("Num of query batches: {}".format(
            len(query_batch_generators[0])))
        logger.info("Num of tasks:{}".format(
            len(support_batch_generators[0][0])))
    elif infer_task != '':
        print("inference mode")
        # with open(mconf.processed_data_save_dir_prefix + "{}t/vocab".format(mconf.num_tasks), "rb") as f:
        #     vocab = pickle.load(f)
        mconf.vocab_size = len(vocab)

    else:
        print("no operation, exiting ...")
        exit(0)

    printer = pprint.PrettyPrinter(indent=4)
    print(">>>>>>> Model Config <<<<<<<")
    printer.pprint(vars(mconf))

    if mconf.wordvec_path is None or (maml_epochs <= 0 or transfer_epochs <= 0) or load_model:
        # will be loading model parameters
        init_embedding = None
    else:
        print("loading initial embedding from {} ...".format(mconf.wordvec_path))
        init_embedding = data_loader.load_embedding_from_wdv(
            vocab, mconf.wordvec_path)
        mconf.embedding_size = init_embedding.shape[1]

    net = maml_cp_vae.MAMLCP_VAE(
        device=device, num_tasks=mconf.num_tasks,
        vocab=vocab, mconf=mconf
    )
    if load_model:
        net.load_model(mconf.model_save_dir_prefix + mconf.last_ckpt)

    logger.info('maml_cp_vae created')

    # meta training
    if maml_epochs > 0:
        # use all tasks for maml learning, and specified tasks for fine-tuning
        _train_maml(net, mconf, support_batch_generators, support_feats, query_batch_generators, query_feats, device=device, total_epochs=maml_epochs, vocab=vocab,
                    epochs_per_val=epochs_per_val, support_batch_size=sub_batch_size, query_batch_size=maml_batch_size, maml_batch_size=maml_batch_size)
        model_file = "epoch-{}.maml".format(maml_epochs)
        model_path = mconf.model_save_dir_prefix + model_file
        net.save_model(model_path)
        logger.info(
            "maml-training checkpoint file {} saved at epoch {}".format(model_path, maml_epochs))
        mconf.last_ckpt = model_file
        mconf.last_maml_ckpt = model_file

    # adapt to each sub-task using task-specific training data
    if transfer_epochs > 0:
        for t in mconf.tsf_tasks:
            net.load_model(mconf.model_save_dir_prefix + mconf.last_maml_ckpt)
            # net.load_model(mconf.model_save_dir_prefix + "epoch-15.maml")
            # batch generator
            train_data_pth = "../data/{}/train/t{}.all".format(mconf.corpus, t)
            train_feat_pth = "../data/{}/train/t{}_glove.npy".format(
                mconf.corpus, t)
            train_data = MonoTextData(train_data_pth, False)
            train_feat = np.load(train_feat_pth)
            t_batch_generator = train_data.create_data_batch_feats(
                train_batch_size, train_feat, device)
            _fine_tune(
                net, mconf, train_feat, t_batch_generator, device=device, vocab=vocab,
                total_epochs=transfer_epochs, epochs_per_val=epochs_per_val,
                batch_size=train_batch_size, task_id=t, dump_embeddings=dump_embeddings
            )
            model_file = "epoch-{}.t{}".format(transfer_epochs, t)
            model_path = mconf.model_save_dir_prefix + model_file
            net.save_model(model_path)
            logger.info("maml-fine-tuning checkpoint file {} for task {} saved at epoch {}".format(
                model_path, t, transfer_epochs))
            mconf.last_ckpt = model_file
            mconf.last_tsf_ckpts["t{}".format(t)] = model_file

    # perform inference(style transfer) for a specific sub-task(specified as infer_task)
    if infer_task != '':
        infer_task = int(infer_task)
        # net.load_model(mconf.model_save_dir_prefix +
        #                mconf.last_tsf_ckpts["t{}".format(infer_task)])
        net.load_model(mconf.model_save_dir_prefix +
                       "best.t{}".format(infer_task))
        logger.info("model loaded from {} for task {}".format(mconf.model_save_dir_prefix+"best.t{}".format(infer_task), infer_task))
        train_data_pth = "../data/{}/train/t{}.label.all".format(
            corpus, infer_task)
        train_feat_pth = "../data/{}/train/t{}_glove.npy".format(
            corpus, infer_task)
        dev_data_pth = "../data/{}/val/t{}.label.all".format(
            corpus, infer_task)
        dev_feat_pth = "../data/{}/val/t{}_glove.npy".format(
            corpus, infer_task)
        test_data_pth = "../data/{}/val/t{}.label.all".format(
            corpus, infer_task)
        test_feat_pth = "../data/{}/val/t{}_glove.npy".format(
            corpus, infer_task)
        net.infer(infer_task, train_data_pth, train_feat_pth, dev_data_pth,
                  dev_feat_pth, test_data_pth, test_feat_pth, vocab, device)
    return net


def run_online_inference(mconf, ckpt, tgt_file, device):

    with open(mconf.processed_data_save_dir_prefix + "{}t/vocab".format(mconf.num_tasks), "rb") as f:
        vocab = pickle.load(f)

    seqs, lengths, _, _ = data_processor.get_seq_data_from_file(
        filename=tgt_file, vocab=vocab, mconf=mconf
    )

    net = maml_cp_vae.MAMLAdvAutoencoder(
        device=device, num_tasks=mconf.num_tasks,
        mconf=mconf
    )
    model_path = mconf.model_save_dir_prefix + ckpt
    net.load_model(model_path)

    _, style_embeddings = net.get_batch_embeddings(
        input_sequences=seqs,
        lengths=lengths
    )
    style_conditioning_embedding = torch.mean(style_embeddings, axis=0)

    while True:
        sys.stdout.write("> ")
        sys.stdout.flush()

        cmd = sys.stdin.readline().rstrip()
        if cmd in ["quit", "exit"]:
            print("exiting ...")
            break
        seq = np.array(vocab.encode_sents(
            [cmd], length=mconf.max_seq_length, pad_token=False), dtype="int32")
        length = data_processor.get_sequence_lengths(
            seq, mconf.min_seq_length, mconf.max_seq_length)
        tsf, pred = net.infer(
            seq, length,
            style_conditioning_embedding=style_conditioning_embedding.cpu().detach().numpy()
        )
        tsf = vocab.decode_sents(tsf)
        print("[tsf]: {} (pred = {})".format(tsf[0], pred[0].item()))
