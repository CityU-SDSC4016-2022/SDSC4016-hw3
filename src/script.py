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
    split_size = 0.2
    batch_size = 256
    vec_size = 256
    w2v_win = 8
    w2v_mc = 8
    w2v_epoch = 20
    train_epoch = 8
    train_lr = 1e-3
    lstm_hidden_dim = 256
    lstm_num_layers = 3
    lstm_dropout = 0.5
    fix_embedding = True

    w2v_model = word2vec.Word2Vec(train_x + test_x, vector_size=vec_size, window=w2v_win, min_count=w2v_mc, workers=16, epochs=w2v_epoch)

    # Preprocessing
    preprocess = Preprocess(train_x, sen_len, w2v_model)
    embedding = preprocess.make_embedding()
    train_x = preprocess.sentence_word2idx()
    train_y = preprocess.labels_to_tensor(train_y)

    x_train, x_val, y_train, y_val = train_test_split(train_x, train_y, test_size=split_size)
    train_dataset = TwitterDataset(x_train, y_train)
    val_dataset = TwitterDataset(x_val, y_val)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

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
