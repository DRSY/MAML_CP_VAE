import pickle
import os
import numpy as np

# ----------------
import utils.data_processor
import utils.vocab

def print_data_info(vocab, seqs, lengths, labels, bow_representations, num_tasks):

	print("vocab_size = {}, bow_size = {}".format(vocab._size, vocab._bows))
	for key in ["train", "val"]:
		print("{}\n-------".format(key))
		for t in range(num_tasks):
			print("task {} data:".format(t+1))
			for s in [0, 1]:
				print(
					"\t {}:".format(s), seqs[key][t][s].shape, lengths[key][t][s].shape, 
					labels[key][t][s].shape, bow_representations[key][t][s].shape
				)


def load_data(mconf, load_data=False, save=False):
	
	if load_data:
		with open(mconf.processed_data_save_dir_prefix + "{}t/vocab".format(mconf.num_tasks), "rb") as f:
			vocab = pickle.load(f)
		seqs = {"train": [], "val": []}
		lengths = {"train": [], "val": []}
		labels = {"train": [], "val": []}
		bow_representations = {"train": [], "val": []}
		for t in range(mconf.num_tasks):
			for label in ["train", "val"]:
				with open(mconf.processed_data_save_dir_prefix + "{}t/t{}.{}".format(mconf.num_tasks, t+1, label), "rb") as f:
					data = pickle.load(f)
					s0, s1 = data["s0"], data["s1"]
					l0, l1 = data["l0"], data["l1"]
					lb0, lb1 = data["lb0"], data["lb1"]
					bow0, bow1 = data["bow0"], data["bow1"]
					seqs[label].append([s0, s1])
					lengths[label].append([l0, l1])
					labels[label].append([lb0, lb1])
					bow_representations[label].append([bow0, bow1])
	else:
		vocab, seqs, lengths, labels, bow_representations = _load_data(mconf, save=save)

	mconf.vocab_size = vocab._size
	mconf.bow_size = vocab._bows

	return vocab, seqs, lengths, labels, bow_representations


def _load_data(mconf, save=False):

	vocab = utils.vocab.Vocabulary(mconf=mconf)
	if os.path.exists(mconf.data_dir_prefix + "text.pretrain"):
		# directly update via predefined text
		print("updating vocab from {} ...".format(mconf.data_dir_prefix + "text.pretrain"))
		vocab.update_vocab(mconf.data_dir_prefix + "text.pretrain")
	else:
		for t in range(mconf.num_tasks):
			print("updating vocab from task {} ...".format(t+1))
			for s in [0, 1]:
				vocab.update_vocab(mconf.data_dir_prefix + "train/t{}.{}".format(t+1, s))
				vocab.update_vocab(mconf.data_dir_prefix + "val/t{}.{}".format(t+1, s))

	seqs, lengths, labels, bow_representations = utils.data_processor.load_all_tasks_data(mconf, vocab, save)

	return vocab, seqs, lengths, labels, bow_representations


def load_embedding_from_wdv(vocab, path):

	emb_size = None
	c = 0
	with open(path, 'r', encoding="utf-8") as f:
		for line in f:
			line = line.split()
			try:
				assert len(line) > 2
			except:
				continue
			word = line[0]
			emb = np.array(line[1:], dtype="float32")
			if emb_size is None:
				emb_size = emb.shape[0]
				embedding = np.asarray(np.random.uniform(size=(vocab._size, emb_size), low=-0.05, high=0.05), dtype="float32")
			if word in vocab._word2id:
				embedding[vocab._word2id[word]] = emb
				c += 1
	print("updates {} words from pretrained embeddings".format(c))

	return embedding
