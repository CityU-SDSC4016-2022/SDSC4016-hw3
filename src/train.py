import copy
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from model import LSTMNet
from utils import evaluation


def training(batch_size: int, n_epoch: int, lr: float, train: DataLoader, valid: DataLoader, model: LSTMNet, device: torch.device) -> tuple[float, LSTMNet]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameter:{total}, trainable:{trainable}", end="\n\n")

    model.train()  # set training mode
    criterion = nn.BCELoss()  # Define loss function
    t_batch = len(train)
    v_batch = len(valid)
    optimizer = optim.AdamW(model.parameters(), lr=lr)  # set optimizer as AdamW
    total_loss, total_acc, best_acc = 0, 0, 0
    best_model = None

    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0

        # For training
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct = evaluation(outputs, labels)  # calculate accuracy
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            print(f"[ Epoch{epoch + 1}: {i + 1}/{t_batch} ]", end="\r")
        print(f"\nTrain | Loss:{total_loss / t_batch:.5f} Acc: {total_acc / t_batch:.5f}")

        # For validation
        model.eval()  # set validation mode
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.float)
                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                correct = evaluation(outputs, labels)
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            print(f"Valid | Loss:{total_loss / v_batch:.5f} Acc: {total_acc / v_batch:.5f}")
            if total_acc > best_acc:
                # if the result of validation is better than previous model, save the new model
                best_acc = total_acc
                best_model = copy.deepcopy(model)
                print(f"saving model with acc {total_acc / v_batch:.5f}")
        model.train()
    return best_acc / v_batch, best_model
