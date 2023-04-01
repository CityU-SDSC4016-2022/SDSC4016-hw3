import os
from test import testing
from sklearn.model_selection import train_test_split
from torch.utils import data
from gensim.models import word2vec
import torch
import pandas as pd
from train import training
from data import TwitterDataset
from model import LSTMNet
from preprocess import Preprocess
from utils import load_data


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data from txt
    # train_x_n = load_data("data/Train_nolabel.txt", True, False)
    train_x, train_y = load_data("data/Train_label.txt", True, True)
    test_x = load_data("data/Test.txt", False)

    # Model configuration
    sen_len = 32
    split_size = 0.2 # split size of the Twitter testing dataset
    batch_size = 256 # training batch size
    vec_size = 256 # dimension of the word vectors
    w2v_win = 8 # window size: Maximum distance between the current and predicted word within a sentence
    w2v_mc = 8 # Min count: Ignores all words with total frequency lower than this
    w2v_epoch = 20 # Number of epochs for the word2vec model
    train_epoch = 8 # Number of epochs for the LSTM model
    train_lr = 1e-3 # learning rate for the LSTM model
    lstm_hidden_dim = 256 # hidden dimentions of the LSTM model
    lstm_num_layers = 3 # number of layers of the LSTM model
    lstm_dropout = 0.5 # Add dropout to the LSTM model
    fix_embedding = True # fix embedding during training

    # Word2Vec model
    w2v_model = word2vec.Word2Vec(train_x + test_x, vector_size=vec_size, window=w2v_win, min_count=w2v_mc, workers=16, epochs=w2v_epoch)

    # Preprocessing
    preprocess = Preprocess(train_x, sen_len, w2v_model)
    embedding = preprocess.make_embedding()
    train_x = preprocess.sentence_word2idx()
    train_y = preprocess.labels_to_tensor(train_y)

    # Twitter dataset
    x_train, x_val, y_train, y_val = train_test_split(train_x, train_y, test_size=split_size)
    train_dataset = TwitterDataset(x_train, y_train)
    val_dataset = TwitterDataset(x_val, y_val)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # LSTM model
    lstm_model = LSTMNet(embedding, w2v_model.vector_size, lstm_hidden_dim, lstm_num_layers, lstm_dropout, fix_embedding)
    lstm_model = lstm_model.to(device)

    # Training
    best_acc, lstm_model = training(batch_size, train_epoch, train_lr, train_loader, val_loader, lstm_model, device)

    # Testing
    print("\nPredicting")
    preprocess = Preprocess(test_x, sen_len, w2v_model)
    embedding = preprocess.make_embedding()
    test_x = preprocess.sentence_word2idx()
    test_dataset = TwitterDataset(test_x, None)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    outputs = testing(test_loader, lstm_model, device)

    # save as csv
    tmp = pd.DataFrame({"id": [str(i) for i in range(len(test_x))], "labels": outputs})
    tmp.to_csv(os.path.join(f"predict_{best_acc * 100}.csv"), index=False)


if __name__ == "__main__":
    main()
