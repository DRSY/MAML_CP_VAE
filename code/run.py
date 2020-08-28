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
        mconf.last_maml_ckpt = model_file
        logger.info("evaluation\n--------")
        for t in range(mconf.num_tasks):
            logger.info("inferring task {} ...".format(t+1))
            losses = net.evaluate(val_batch_generators[t])
            vae_loss, rec_loss, kl1_loss, kl2_loss, *_ = losses
            logger.info("vae loss {}, rec loss {}, kl1 loss {}, kl2 loss {}".format(
                vae_loss, rec_loss, kl1_loss, kl2_loss))


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
        mconf.train_batch_size, val_feat, device)
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
        mconf.last_tsf_ckpts["t{}".format(task_id)] = model_file
        logger.info("evaluatoin\n------")
        vae_loss, rec_loss, kl1_loss, kl2_loss, * \
            _ = net.evaluate(val_batch_generator)
        logger.info("vae loss {}, rec loss {}, kl1 loss {}, kl2 loss {}".format(
            vae_loss, rec_loss, kl1_loss, kl2_loss))


def run_maml(mconf, device, load_data=False, load_model=False, maml_epochs=10, transfer_epochs=6, epochs_per_val=2, infer_task='', maml_batch_size=8, sub_batch_size=32, train_batch_size=64, dump_embeddings=False):

    if maml_epochs > 0 or transfer_epochs > 0:
        logger.info("loading data from maml_cp_vae")
        corpus = mconf.corpus
        support_batch_generators = [[], []]
        query_batch_generators = [[], []]
        support_feats = []
        query_feats = []
        data_path = "../data/{}".format(corpus)
        corpus_data_pth = os.path.join(data_path, "text.pretrain")
        corpus_data = MonoTextData(corpus_data_pth, False)
        vocab = corpus_data.vocab
        logger.info("size of whole vocab: {}".format(len(vocab)))
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
        with open(mconf.processed_data_save_dir_prefix + "{}t/vocab".format(mconf.num_tasks), "rb") as f:
            vocab = pickle.load(f)

        mconf.vocab_size = vocab._size
        mconf.bow_size = vocab._bows

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
        mconf.last_ckpt = model_file
        mconf.last_maml_ckpt = model_file

    # adapt to each sub-task using task-specific training data
    if transfer_epochs > 0:
        for t in mconf.tsf_tasks:
            net.load_model(mconf.model_save_dir_prefix + mconf.last_maml_ckpt)
            # batch generator
            train_data_pth = "../data/{}/train/t{}.all"
            train_feat_pth = "../data/{}/train/t{}_glove.npy"
            train_data = MonoTextData(train_data_pth, False)
            train_feat = np.load(train_feat_pth)
            t_batch_generator = train_data.create_data_batch_feats(
                mconf.train_batch_size, train_feat, device)
            _fine_tune(
                net, mconf, train_feat, t_batch_generator, device=device, vocab=vocab,
                total_epochs=transfer_epochs, epochs_per_val=epochs_per_val,
                batch_size=train_batch_size, task_id=t, dump_embeddings=dump_embeddings
            )
            model_file = "epoch-{}.t{}".format(transfer_epochs, t)
            model_path = mconf.model_save_dir_prefix + model_file
            net.save_model(model_path)
            mconf.last_ckpt = model_file
            mconf.last_tsf_ckpts["t{}".format(t)] = model_file

    if infer_task != '':
        infer_task = int(infer_task)
        net.load_model(mconf.model_save_dir_prefix +
                       mconf.last_tsf_ckpts["t{}".format(infer_task)])
        s0, s1, l0, l1, lb0, lb1, bow0, bow1 = data_processor.load_task_data(
            infer_task, mconf.data_dir_prefix, vocab,
            label="infer", mconf=mconf
        )
        infer_seqs = [s0, s1]
        infer_lengths = [l0, l1]
        infer_labels = [lb0, lb1]
        infer_bows = [bow0, bow1]

        content_embeddings, style_embeddings = net.get_batch_embeddings(
            input_sequences=np.concatenate(infer_seqs, axis=0),
            lengths=np.concatenate(infer_lengths, axis=0)
        )
        style_conditioning_embeddings = [
            torch.mean(style_embeddings[:infer_lengths[0].shape[0]], axis=0),
            torch.mean(style_embeddings[infer_lengths[0].shape[0]:], axis=0)
        ]

        if dump_embeddings:

            style_embeddings = style_embeddings.cpu().detach().numpy()
            content_embeddings = content_embeddings.cpu().detach().numpy()

            style_embeddings = [
                style_embeddings[:infer_lengths[0].shape[0]],
                style_embeddings[infer_lengths[0].shape[0]:]
            ]
            content_embeddings = [
                content_embeddings[:infer_lengths[0].shape[0]],
                content_embeddings[infer_lengths[0].shape[0]:]
            ]
            with open(mconf.emb_save_dir_prefix + "t{}/infer.emb".format(infer_task), "wb") as f:
                embeddings = {
                    "style": style_embeddings,
                    "content": content_embeddings
                }
                pickle.dump(embeddings, f)
                print("dumped embeddings to {}".format(
                    mconf.emb_save_dir_prefix + "t{}/infer.emb".format(infer_task)))
        for s in [0, 1]:
            inferred_seqs, style_preds = net.infer(
                infer_seqs[s], infer_lengths[s],
                style_conditioning_embedding=style_conditioning_embeddings[1-s].cpu(
                ).detach().numpy()
            )
            sents = vocab.decode_sents(inferred_seqs)
            with open(mconf.output_dir_prefix + "infer_t{}_{}-{}".format(infer_task, s, 1-s), 'w', encoding="utf-8") as f:
                for sent, pred in zip(sents, style_preds):
                    f.write(sent + '\t' + str(pred.item()) + '\n')

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


def extract_embeddings(mconf, ckpt, task_id, device, pretrain=False, sample_size=1000):

    net = maml_cp_vae.MAMLAdvAutoencoder(
        device=device, num_tasks=mconf.num_tasks,
        mconf=mconf
    )

    model_path = mconf.model_save_dir_prefix + ckpt
    net.load_model(model_path)

    if pretrain:
        dir_prefix = mconf.processed_data_save_dir_prefix + "pretrain/"
    else:
        dir_prefix = mconf.processed_data_save_dir_prefix + \
            "{}t/".format(mconf.num_tasks)

    with open(dir_prefix + "vocab", "rb") as f:
        vocab = pickle.load(f)

    with open(dir_prefix + "t{}.val".format(task_id), "rb") as f:
        data = pickle.load(f)
        s0, s1 = data["s0"], data["s1"]
        l0, l1 = data["l0"], data["l1"]
        lb0, lb1 = data["lb0"], data["lb1"]
        bow0, bow1 = data["bow0"], data["bow1"]

    inds0 = np.random.choice(list(range(l0.shape[0])), sample_size)
    inds1 = np.random.choice(list(range(l1.shape[0])), sample_size)

    content_embeddings, style_embeddings = net.get_batch_embeddings(
        input_sequences=np.concatenate([s0[inds0], s1[inds1]], axis=0),
        lengths=np.concatenate([l0[inds0], l1[inds1]], axis=0)
    )

    style_embeddings = style_embeddings.cpu().detach().numpy()
    content_embeddings = content_embeddings.cpu().detach().numpy()

    style_embeddings = [
        style_embeddings[:l0.shape[0]],
        style_embeddings[l0.shape[0]:]
    ]
    content_embeddings = [
        content_embeddings[:l0.shape[0]],
        content_embeddings[l0.shape[0]:]
    ]

    with open(mconf.emb_save_dir_prefix + "t{}/extract.emb".format(task_id), "wb") as f:
        embeddings = {
            "style": style_embeddings,
            "content": content_embeddings
        }
        pickle.dump(embeddings, f)
        print("dumped embeddings to {}t{}/extract.emb".format(mconf.emb_save_dir_prefix, task_id))
