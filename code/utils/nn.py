import numpy as np
import torch

# ==================================================================
'''
def gumbel_softmax(logits, gamma, eps=1e-20):

	U = torch.randn(logits.shape)
	G = -torch.log(-torch.log(U + eps) + eps)

	return torch.nn.functional.softmax((logits + G) / gamma)
'''

def softsample_word(dropout, proj, embedding, gamma):

	def loop_func(output):

		output = torch.nn.functional.dropout(output, dropout)
		logits = proj(output)
		prob = torch.nn.functional.gumbel_softmax(logits, gamma)

		x = torch.mm(prob, embedding)

		return x, logits

	return loop_func


def softmax_word(dropout, proj, embedding, gamma):

	def loop_func(output):

		output = torch.nn.functional.dropout(output, dropout)
		logits = proj(output)
		prob = torch.nn.functional.softmax(logits/gamma, dim=1)
		x = torch.mm(prob, embedding)

		return x, logits

	return loop_func


def argmax_word(dropout, proj, embedding):

	def loop_func(output):

		output = torch.nn.functional.dropout(output, dropout)
		logits = proj(output)
		word = torch.argmax(logits, axis=1)
		x = embedding(word)

		return x, logits

	return loop_func

'''
def rnn_decode(h, x, length, cell, loop_func):

	h_seq, logits_seq = [], []

	for t in range(length):
		h_seq.append(h.view(-1, 1, cell.hidden_size))
		output, h = cell(x.unsqueeze(1), h)
		x, logits = loop_func(output.view(-1, cell.hidden_size))
		logits_seq.append(logits.unsqueeze(1))

	return torch.cat(h_seq, axis=1), torch.cat(logits_seq, axis=1)
'''

# ==================================================================
# vae utils

def calc_kl_loss(mu, log_sigma):

	return torch.mean(-0.5 * torch.sum(1 + log_sigma - mu**2 - torch.exp(log_sigma), axis=1))


def compute_batch_entropy(x, eps=1e-20):

	return torch.mean(torch.sum(-x * torch.log(x + eps), axis=1))


def rnn_decode_with_latent_vec(h, x, length, cell, loop_func, latent_vec):

	h_seq, logits_seq = [], []
	latent_vec = latent_vec.unsqueeze(1)

	for t in range(length):
		
		# h_seq.append(h.view(-1, 1, cell.hidden_size))
		
		output, h = cell(
			torch.cat(
				(latent_vec, x.unsqueeze(1)), axis=-1
			), h
		)
		x, logits = loop_func(output.view(-1, cell.hidden_size))
		logits_seq.append(logits.unsqueeze(1))

		h_seq.append(h.view(-1, 1, cell.hidden_size))

	return torch.cat(h_seq, axis=1), torch.cat(logits_seq, axis=1)



def reduced_cross_entropy_loss(onehot, pred, eps=1e-8):

	return -torch.mean(torch.sum(onehot * torch.log(pred + eps), axis=1))


