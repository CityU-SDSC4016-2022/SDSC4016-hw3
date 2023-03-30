import numpy as np
import torch
from gensim.models import word2vec


class Preprocess():
    def __init__(self, sentences: list[list[str]] | list[str], sen_len: int, model: word2vec.Word2Vec):
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.model = model
        self.model_dim = 0
        self.model_matrix = []

    def get_w2v_model(self):
        # load word to vector model
        self.model = self.model
        self.model_dim = self.model.vector_size

    def add_embedding(self, word: str):
        # add word into embedding
        vector = torch.empty(1, self.model_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.model_matrix = torch.cat([self.model_matrix, vector], 0)

    def make_embedding(self) -> torch.Tensor:
        print("Get embedding ...")
        self.get_w2v_model()

        for _, word in enumerate(self.model.wv.key_to_index):
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.model_matrix.append(self.model.wv[word])
        self.model_matrix = np.array(self.model_matrix)
        self.model_matrix = torch.tensor(self.model_matrix)
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print(f"Total words: {len(self.model_matrix)}")
        return self.model_matrix

    def pad_sequence(self, sentence: list) -> list:
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence

    def sentence_word2idx(self) -> torch.Tensor:
        sentence_list = []
        for _, sen in enumerate(self.sentences):
            sentence_idx = []
            for word in sen:
                if word in self.word2idx.keys():
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        print(f"Sentence count #{len(sentence_list)}")
        return torch.LongTensor(sentence_list)

    def labels_to_tensor(self, y_var: list[str]) -> torch.Tensor:
        # turn labels into tensors
        y_var = [int(label) for label in y_var]
        return torch.LongTensor(y_var)
