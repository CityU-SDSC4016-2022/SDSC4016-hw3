import torch


def evaluation(outputs, labels):
    # outputs => probability (float)
    # labels => labels
    outputs[outputs >= 0.5] = 1  # Negtive Sentiment
    outputs[outputs < 0.5] = 0  # Positive Sentiment
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct


def load_data(path: str, train: bool, labeled: bool = None):
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        if train:
            lines = [line.strip("\n").split(" ") for line in lines]
            if labeled:
                x_var = [line[2:] for line in lines]
                y_var = [line[0] for line in lines]
                return x_var, y_var
            else:
                return lines
        else:
            lines = ["".join(line.strip("\n").split(",")[1:]).strip() for line in lines[1:]]
            x_var = [sen.split(" ") for sen in lines]
            return x_var
