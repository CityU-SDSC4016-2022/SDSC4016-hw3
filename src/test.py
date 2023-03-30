import torch
from torch.utils import data
from model import LSTMNet


def testing(test_loader: data.DataLoader, model: LSTMNet, device: torch.device) -> list:
    model.eval()
    ret_output = []
    with torch.no_grad():
        for _, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0
            ret_output += outputs.int().tolist()
    return ret_output
