# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>
import os
import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import string

all_chars = string.ascii_letters+string.punctuation+string.digits+' '

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")


class WordCharCNNEmbedding(nn.Module):
	def __init__(self, vocab_size, d_emb, char_size, c_emb, conv_l, char_padding_idx=1, padding_size=2, kernel_size=3):
		super(WordCharCNNEmbedding, self).__init__()
		self.char_embedding = nn.Embedding(char_size, c_emb).to(device)
		self._init_char_embedding(char_padding_idx)
		self.conv_embedding = nn.Sequential(nn.Conv1d(in_channels=c_emb,
													  out_channels=conv_l,kernel_size=kernel_size, padding=padding_size).to(device),nn.ReLU()).to(device)
		self.word_embedding = nn.Embedding(vocab_size, d_emb).to(device)

	def _init_char_embedding(self, padding_idx):
		nn.init.xavier_normal_(self.char_embedding.weight).to(device)
		self.char_embedding.weight.data[padding_idx].uniform_(0, 0)

	def forward(self, X, X_word):
		char_embeddings = []
		for x in X:
			char_embedding = self.char_embedding(x)
			char_embedding = char_embedding.transpose(1, 0).unsqueeze(0)
			char_embedding = self.conv_embedding(char_embedding)
			char_embedding = nn.MaxPool1d(char_embedding.size()[2])(char_embedding).to(device)
			char_embeddings.append(char_embedding[:, :, 0])
		final_char_embedding = torch.cat(char_embeddings, dim=0).to(device)
		word_embedding = self.word_embedding(X_word)
		result = torch.cat([final_char_embedding, word_embedding], 1).to(device)
		return result


class POSTagger(nn.Module):
	def __init__(self, embedding, n_emb, hidden_dim, ntags, word2idx, num_layers):
		super(POSTagger, self).__init__()
		self.embedding = embedding
		# self.tagger_rnn = nn.LSTM(input_size=n_emb, hidden_size=hidden_dim, num_layers=2, dropout=0.2,bidirectional=True).to(device)
		self.tagger_rnn = nn.LSTM(input_size=n_emb, hidden_size=hidden_dim, bidirectional=True).to(device)
		self._init_rnn_weights()

		self.hidden2tag = nn.Sequential(nn.Linear(in_features=hidden_dim * 2, out_features=ntags)).to(device)
		self._init_linear_weights_and_bias()
		self.word2idx = word2idx

	def _init_rnn_weights(self):
		for idx in range(len(self.tagger_rnn.all_weights[0])):
			dim = self.tagger_rnn.all_weights[0][idx].size()
			if len(dim) < 2:
				nn.init.constant_(self.tagger_rnn.all_weights[0][idx], 1).to(device)
			elif len(dim) == 2:
				nn.init.xavier_uniform_(self.tagger_rnn.all_weights[0][idx]).to(device)

	def _init_linear_weights_and_bias(self):
		nn.init.xavier_uniform_(self.hidden2tag[0].weight).to(device)
		nn.init.constant_(self.hidden2tag[0].bias, 1).to(device)

	def forward(self, sent):
		x_word = sent2tensor(sent, self.word2idx)
		x = []
		for word in sent:
			x.append(word2tensor(word))
		embeds = self.embedding(x, x_word)
		out, _ = self.tagger_rnn(embeds.view(len(sent), 1, -1))
		tag_outputs = self.hidden2tag(out.view(len(sent), -1))
		tag_scores = F.log_softmax(tag_outputs, dim=1)
		return tag_scores


def sent2tensor(sent, to_idx):
	idxs = []
	for w in sent:
		w = w.lower()
		if w not in to_idx:
			w = 'UNK'
		idxs.append(to_idx[w])
	return torch.tensor(idxs, dtype=torch.long).to(device)


def tags2tensor(tags, to_idx):
	idxs = []
	for tag in tags:
		idxs.append(to_idx[tag])
	return torch.tensor(idxs, dtype=torch.long).to(device)


def word2tensor(word):
	tensor = torch.zeros(len(word)).to(device)
	for i, char in enumerate(word):
		tensor[i] = all_chars.find(char)
	return tensor.long()

def tag_sentence(test_file, model_file, out_file):
	d_emb, c_emb, hidden_dim, conv_l, word2idx, tag2idx, idx2tag, num_layers, embedding_state_dict, tagger_state_dict = torch.load(model_file)
	embedding = WordCharCNNEmbedding(len(word2idx), d_emb, len(all_chars), c_emb, conv_l).to(device)
	tagger = POSTagger(embedding, d_emb+conv_l, hidden_dim, len(tag2idx), word2idx, num_layers).to(device)
	embedding.load_state_dict(embedding_state_dict)
	tagger.load_state_dict(tagger_state_dict)
	test_file = open(test_file, 'r')
	out_file = open(out_file, 'w+')
	for line in test_file.readlines():
		line = line.split()
		temp = ''
		tag_scores = tagger(line)
		_, predicted_tags = torch.max(tag_scores,1)
		for i in range(len(line)):
			tag = idx2tag[predicted_tags[i].item()]
			temp += '{}/{} '.format(line[i], tag)
		temp += '\n'
		out_file.write(temp)

	print('Finished...')

if __name__ == "__main__":
	# make no changes here
	test_file = sys.argv[1]
	model_file = sys.argv[2]
	out_file = sys.argv[3]
	tag_sentence(test_file, model_file, out_file)
