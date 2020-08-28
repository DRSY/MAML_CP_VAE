from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords

import string
import pickle

from config.model_config import default_mconf

class Vocabulary:

	def __init__(self, mconf=default_mconf):

		self._word2id = {'<pad>': 0, '<eos>': 1, '<sos>': 2, '<unk>': 3}
		self._id2word = ['<pad>', '<eos>', '<sos>', '<unk>']
		self._bow2id = {}
		self._id2bow = []

		self._tokenize = word_tokenize
		self._size = 4
		self.filter_list = list(string.punctuation)

		self.bow_filter_list = list(set(stopwords.words("english"))) + self.filter_list
		# self.bow_filter_list = self.filter_list
		# self.bow_tokenizer = RegexpTokenizer(r"\w+")
		self._bows = 0

		self.vocab_cutoff = mconf.vocab_cutoff
		self.bow_cutoff = mconf.bow_cutoff

	def init_from_saved_vocab(self, path):

		f = open(path, 'rb')
		v = pickle.load(f)

		self._word2id = v._word2id
		self._id2word = v._id2word
		self._bow2id = v._bow2id
		self._id2bow = v._id2bow

		self._tokenize = v._tokenize
		self._size = v._size
		self.filter_list = v.filter_list

		self.bow_filter_list = v.bow_filter_list
		# self.bow_tokenizer = v.bow_tokenizer
		self._bows = v._bows

		self.vocab_cutoff = v.vocab_cutoff
		self.bow_cutoff = v.bow_cutoff

		f.close()
		print("initialized vocabulary from {}".format(path))

	def update_vocab(self, path):

		f = open(path, 'r', encoding='utf-8')
		lines = f.readlines()
		print('lines: ' + str(len(lines)))

		added_words, added_bows = 0, 0
		sentences = 0
		words = {}
		bows = {}

		for line in lines:
			sentences += 1
			line = line.lower()
			try:
				# in `label + \t + sent` format
				line = line.split('\t')[1]
			except:
				pass
			tokenized = self._tokenize(line)
			if sentences % 5000 == 0:
				print("words: {}, bows: {}, lines: {}".format(added_words, added_bows, sentences))
			# update vocab
			for word in tokenized:
				if word in self._word2id or word in self.filter_list:
					continue
				if word not in words:
					words[word] = 1
					added_words += 1
				else:
					words[word] += 1

				if word in self._bow2id or word in self.bow_filter_list:
					continue
				if word not in bows:
					bows[word] = 1
					added_bows += 1
				else:
					bows[word] += 1

		added_words, added_bows = 0, 0
		for word in words:
			if words[word] >= self.vocab_cutoff and word not in self._word2id:
				self._word2id[word] = self._size + added_words
				self._id2word.append(word)
				added_words += 1
		for word in bows:
			if bows[word] >= self.bow_cutoff and word not in self._bow2id:
				self._bow2id[word] = self._bows + added_bows
				self._id2bow.append(word)
				added_bows += 1

		# debug
		if added_bows > added_words:
			print("[DEBUG]: ", [w for w in bows if w not in self._word2id])

		self._size += added_words
		self._bows += added_bows

		print("updated: words {}, bows {}".format(added_words, added_bows))
		print("current: words {}, bows {}".format(self._size, self._bows))


	def word2id(self, word):

		word = word.lower()

		if word in self._id2word:
			return self._word2id[word]
		else:
			return self._word2id['<unk>']


	def id2word(self, ind):

		if ind >= self._size or ind < 0:
			return '<unk>'
		else:
			return self._id2word[ind]


	def save_vocab(self, path):

		f = open(path, 'wb')
		pickle.dump(self, f)
		f.close()

		print("saved vocabulary to " + path)


	def encode_sents(self, sents, length=None, pad_token=False):

		seqs = []

		for sent in sents:
			sent = self._tokenize(sent.lower())
			if self.filter_list != []:
				sent = [w for w in sent if w not in self.filter_list]

			if pad_token:
				encoded = [0] * (len(sent) + 2)
				encoded[0] = self._word2id['<sos>']
				encoded[-1] = self._word2id['<eos>']
				pos = 1
			else:
				encoded = [0] * len(sent)
				pos = 0

			for word in sent:
				encoded[pos] = self.word2id(word)
				pos += 1

			if length is not None:
				if length <= len(encoded):
					encoded = encoded[:length]
					# encoded[-1] = 0
				else:
					appended = [0] * (length - len(encoded))
					encoded += appended

			seqs.append(encoded)

		return seqs


	def decode_sents(self, seqs):

		sents = []

		stop = False

		for seq in seqs:
			sent = ''
			for ind in seq[:-1]:
				word = self.id2word(ind)
				
				if word == '<eos>':
					stop = True
					sent += word
					break
				elif word == '<unk>' or word == '<pad>':
					pass
				else:
					sent += word + ' '

			if not stop:
				word = self.id2word(seq[-1])
				if word != '<unk>' and word != '<pad>':
					sent += word
				else:
					sent = sent[:-1]

			sents.append(sent)

		return sents


	def get_bow_seqs(self, sents, maxlen=None):

		bow_seqs = []

		for sent in sents:
			sent = self._tokenize(sent.lower())
			sent = [w for w in sent if w not in self.bow_filter_list]
			bow = []
			for k in range(len(sent)):
				if sent[k] in self._bow2id:
					bow.append(self._bow2id[sent[k]])

			if maxlen is not None:
				if maxlen <= len(bow):
					bow = bow[:maxlen]

			bow_seqs.append(bow)

		return bow_seqs
