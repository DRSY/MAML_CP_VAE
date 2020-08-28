import numpy as np
import torch
import pickle

# ----------------
import utils.vocab


def makeup_seqs(X, lengths, labels, bow_representations, n):

	ratio = n // X.shape[0]

	if n % X.shape[0]:
		ratio += 1

	X = np.concatenate([X] * ratio, axis=0)
	lengths = np.concatenate([lengths] * ratio, axis=0)
	labels = np.concatenate([labels] * ratio, axis=0)
	bow_representations = np.concatenate([bow_representations] * ratio, axis=0)

	inds = list(range(X.shape[0]))
	np.random.shuffle(inds)

	sample = np.random.choice(inds, n)

	return X[sample], lengths[sample], labels[sample], bow_representations[sample]


def get_pretrain_batch_generator(seqs, lengths, batch_size, device, shuffle=True):

	n = lengths.shape[0]

	num_batches = n // batch_size
	if n % batch_size:
		n += 1

	batch_generator = _pretrain_batch_generator(
		seqs, lengths, batch_size, n, 
		num_batches, device, shuffle=shuffle
	)

	return batch_generator, num_batches

def _pretrain_batch_generator(seqs, lengths, batch_size, data_size, num_batches, device, shuffle=True):

	b = 0
	inds = list(range(data_size))
	if shuffle:
		np.random.shuffle(inds)
	while True:
		start = b * batch_size
		end = min(data_size, (b + 1) * batch_size)
		seqs_batch = []
		lengths_batch = []

		seqs_batch = torch.tensor(seqs[start:end], dtype=torch.int32, device=device)
		lengths_batch = torch.tensor(lengths[start:end], dtype=torch.int32, device=device)

		yield seqs_batch, lengths_batch # , (end - start) # yield also batch_size ?

		if b == num_batches - 1:
			if shuffle:
				print("shuffling ...")
				np.random.shuffle(inds)
			b = 0
		else:
			b += 1


def get_batch_generator(seqs, lengths, labels, bow_representations, batch_size, device, shuffle=True):

	sizes = [seq.shape[0] for seq in seqs]
	n = max(sizes)

	for i in range(len(seqs)):
		if seqs[i].shape[0] < n:
			seqs[i], lengths[i], labels[i], bow_representations[i] = makeup_seqs(
				seqs[i], lengths[i], labels[i], 
				bow_representations[i], n
			)

	num_batches = n // batch_size
	if n % batch_size:
		num_batches += 1

	batch_generator = _batch_generator(
		seqs, lengths, labels, bow_representations,
		batch_size, n, num_batches, device, shuffle=shuffle
	)

	return batch_generator, num_batches


def _batch_generator(seqs, lengths, labels, bow_representations, batch_size, data_size, num_batches, device, shuffle=True):

	b = 0
	inds = list(range(data_size))
	if shuffle:
		np.random.shuffle(inds)
	while True:
		start = b * batch_size
		end = min(data_size, (b + 1) * batch_size)
		seqs_batch = []
		lengths_batch = []
		labels_batch = []
		bow_representations_batch = []
		for seq, length, label, bow_representation in zip(seqs, lengths, labels, bow_representations):
			seqs_batch.append(seq[inds][start:end])
			lengths_batch.append(length[inds][start:end])
			labels_batch.append(label[inds][start:end])
			bow_representations_batch.append(bow_representation[inds][start:end])


		seqs_batch = np.concatenate(seqs_batch, axis=0)
		lengths_batch = np.concatenate(lengths_batch, axis=0)
		labels_batch = np.concatenate(labels_batch, axis=0)
		bow_representations_batch = np.concatenate(bow_representations_batch, axis=0)

		if shuffle:
			actual_batch_size = labels_batch.shape[0]
			batch_inds = list(range(actual_batch_size))
			np.random.shuffle(batch_inds)
			
			seqs_batch = seqs_batch[batch_inds]
			lengths_batch = lengths_batch[batch_inds]
			labels_batch = labels_batch[batch_inds]
			bow_representations_batch = bow_representations_batch[batch_inds]

		seqs_batch = torch.tensor(seqs_batch, dtype=torch.int32, device=device)
		lengths_batch = torch.tensor(lengths_batch, dtype=torch.int32, device=device)
		labels_batch = torch.tensor(labels_batch, dtype=torch.int32, device=device)
		bow_representations_batch = torch.tensor(bow_representations_batch, dtype=torch.float32, device=device)

		yield seqs_batch, lengths_batch, labels_batch, bow_representations_batch

		if b == num_batches - 1:
			if shuffle:
				print("shuffling ...")
				np.random.shuffle(inds)
			b = 0
		else:
			b += 1


def get_maml_batch_generator(seqs, lengths, labels, bow_representations, batch_size, num_tasks, device, shuffle=True):

	sizes = []
	for seqs_task in seqs:
		sizes += [seq.shape[0] for seq in seqs_task]
	# sizes = [seq.shape[0] for seq in seqs_task for seqs_task in seqs]
	n = max(sizes)

	for t in range(num_tasks):
		seqs_task, lengths_task, labels_task, bow_representations_task = seqs[t], lengths[t], labels[t], bow_representations[t]
		for i in range(len(seqs_task)):
			if seqs_task[i].shape[0] < n:
				seqs_task[i], lengths_task[i], labels_task[i], bow_representations_task[i] = makeup_seqs(
					seqs_task[i], lengths_task[i], labels_task[i],
					bow_representations_task[i], n
				)
		seqs[t], lengths[t], labels[t], bow_representations[t] = seqs_task, lengths_task, labels_task, bow_representations_task

	num_batches = n // batch_size
	if n % batch_size:
		num_batches += 1

	batch_generator = _maml_batch_generator(
		seqs, lengths, labels, bow_representations,
		batch_size, n, num_batches, num_tasks, device=device, shuffle=shuffle
	)

	return batch_generator, num_batches


def _maml_batch_generator(seqs, lengths, labels, bow_representations, batch_size, data_size, num_batches, num_tasks, device, shuffle=True):

	b = 0
	inds = list(range(data_size))
	if shuffle:
		np.random.shuffle(inds)
	while True:
		seqs_batch_all, lengths_batch_all, labels_batch_all, bow_representations_batch_all = [], [], [], []
		start = b * batch_size
		end = min(data_size, (b + 1) * batch_size)
		for t in range(num_tasks):
			seqs_batch = []
			lengths_batch = []
			labels_batch = []
			bow_representations_batch = []
			for seq, length, label, bow_representation  in zip(seqs[t], lengths[t], labels[t], bow_representations[t]):
				seqs_batch.append(seq[inds][start:end])
				lengths_batch.append(length[inds][start:end])
				labels_batch.append(label[inds][start:end])
				bow_representations_batch.append(bow_representation[inds][start:end])
			
			seqs_batch = np.concatenate(seqs_batch, axis=0)
			lengths_batch = np.concatenate(lengths_batch, axis=0)
			labels_batch = np.concatenate(labels_batch, axis=0)
			bow_representations_batch = np.concatenate(bow_representations_batch, axis=0)

			if shuffle:
				actual_batch_size = labels_batch.shape[0]
				batch_inds = list(range(actual_batch_size))
				np.random.shuffle(batch_inds)

				seqs_batch = seqs_batch[batch_inds]
				lengths_batch = lengths_batch[batch_inds]
				labels_batch = labels_batch[batch_inds]
				bow_representations_batch = bow_representations_batch[batch_inds]

			seqs_batch_all.append(torch.tensor(seqs_batch, dtype=torch.int32, device=device))
			lengths_batch_all.append(torch.tensor(lengths_batch, dtype=torch.int32, device=device))
			labels_batch_all.append(torch.tensor(labels_batch, dtype=torch.int32, device=device))
			bow_representations_batch_all.append(torch.tensor(bow_representations_batch, dtype=torch.float32, device=device))

		yield seqs_batch_all, lengths_batch_all, labels_batch_all, bow_representations_batch_all

		if b == num_batches - 1:
			if shuffle:
				print("shuffling ...")
				np.random.shuffle(inds)
			b = 0
		else:
			b += 1


def get_sequence_lengths(seqs, min_length, max_length):

	tmp = np.concatenate([seqs, np.ones((seqs.shape[0], 1), dtype="int32")], axis=1)
	l = np.clip(np.argmin(tmp, axis=1), min_length, max_length)

	return np.asarray(l, dtype="int32")


def get_sequence_bow_representations(bow_seqs, bow_size):

	bow_representations = np.asarray(
		np.zeros(shape=(len(bow_seqs), bow_size)),
		dtype="float32"
	)
	inds = []
	skipped = 0
	for i in range(len(bow_seqs)):
		seq = bow_seqs[i]
		for j in range(len(seq)):
			bow_representations[i][seq[j]] += 1.
		if len(seq) == 0:
			skipped += 1
			# print("empty:", i)
			continue
		inds.append(i)
		bow_representations[i] /= max(sum(bow_representations[i]), 1)

	print("[WARNING] skipped {} sentences".format(skipped))

	return bow_representations, inds


# ====================================================================================================
# get data from text files

def get_seq_data_from_file(filename, vocab, mconf, label=0, pretrain=False):

	with open(filename, 'r', encoding="utf-8") as f:
		lines = f.readlines()

	seqs = np.array(vocab.encode_sents(lines, length=mconf.max_seq_length, pad_token=False), dtype="int32")
	lengths = get_sequence_lengths(seqs, mconf.min_seq_length, mconf.max_seq_length)

	if pretrain:
		bow_representations = []
		labels = []
	else:
		bow_seqs = vocab.get_bow_seqs(lines, maxlen=mconf.max_seq_length) # no padding
		# bow_seqs = vocab.get_bow_seqs(lines)
		bow_representations, inds = get_sequence_bow_representations(bow_seqs, vocab._bows)
		# bow_representations = get_sequence_bow_representations(bow_seqs, vocab._bows)
		labels = np.full(lengths.shape[0], label, dtype="int32")

	return seqs[inds], lengths[inds], labels[inds], bow_representations
	# return seqs, lengths, labels, bow_representations


def load_task_data(task_id, data_dir, vocab, label, mconf):

	s0, l0, lb0, bow0 = get_seq_data_from_file("{}/{}/t{}.0".format(data_dir, label, task_id), vocab, mconf, label=0)
	s1, l1, lb1, bow1 = get_seq_data_from_file("{}/{}/t{}.1".format(data_dir, label, task_id), vocab, mconf, label=1)

	return s0, s1, l0, l1, lb0, lb1, bow0, bow1


def load_all_tasks_data(mconf, vocab, save=False):

	seqs = {"train": [], "val": []}
	lengths = {"train": [], "val": []}
	labels = {"train": [], "val": []}
	bow_representations = {"train": [], "val": []}
	for t in range(1, mconf.num_tasks + 1):
		for label in ["train", "val"]:
			print("loading {} data for task {} ...".format(label, t))
			s0, s1, l0, l1, lb0, lb1, bow0, bow1 = load_task_data(
				t, mconf.data_dir_prefix, vocab, 
				label=label, mconf=mconf
			)
			seqs[label].append([s0, s1])
			lengths[label].append([l0, l1])
			labels[label].append([lb0, lb1])
			bow_representations[label].append([bow0, bow1])

	if save:
		with open(mconf.processed_data_save_dir_prefix + "{}t/vocab".format(mconf.num_tasks), "wb") as f:
			print("saved vocab to {}t/vocab".format(mconf.num_tasks))
			pickle.dump(vocab, f)
		for t in range(mconf.num_tasks):
			data_train, data_val = dict(), dict()
			for s in [0, 1]:
				data_train["s{}".format(s)] = seqs["train"][t][s]
				data_train["l{}".format(s)] = lengths["train"][t][s]
				data_train["lb{}".format(s)] = labels["train"][t][s]
				data_train["bow{}".format(s)] = bow_representations["train"][t][s]
				
				data_val["s{}".format(s)] = seqs["val"][t][s]
				data_val["l{}".format(s)] = lengths["val"][t][s]
				data_val["lb{}".format(s)] = labels["val"][t][s]
				data_val["bow{}".format(s)] = bow_representations["val"][t][s]

			with open(mconf.processed_data_save_dir_prefix + "{}t/t{}.train".format(mconf.num_tasks, t+1), "wb") as f:
				pickle.dump(data_train, f)
			with open(mconf.processed_data_save_dir_prefix + "{}t/t{}.val".format(mconf.num_tasks, t+1), "wb") as f:
				pickle.dump(data_val, f)

			print("saved data for task {}".format(t+1))

	return seqs, lengths, labels, bow_representations
