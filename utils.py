import os
import pickle
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np
import torch


def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print('No checkpoint!')
        return
    checkpoint = torch.load(checkpoint_path)
    checkpoint_new = model.state_dict()
    for param in checkpoint_new:
        checkpoint_new[param] = checkpoint[param]
    model.load_state_dict(checkpoint_new)


def save_vocab(vocab, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(vocab, f)


def load_vocab(file_path):
    with open(file_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def report_validation_scores(valid_dataloader, model):
    labels = []
    preds = []
    for data in tqdm(valid_dataloader):
        s1 = data['s1']
        s2 = data['s2']
        label = data['label']
        predication = model(s1, s2)
        labels.append(label.detach().numpy())
        predication = (predication > 0.5).type(torch.float)
        preds.append(predication.detach().numpy())
    print(classification_report(np.concatenate(labels, 0), np.concatenate(preds, 0),
                                target_names=["no duplicate", "duplicate"]))
