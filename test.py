import torch
from torch.utils.data import DataLoader
from dataloader.test_dataset import QADataset_Test
from networks import Base_Net
from options import get_opts
import numpy as np
from sklearn.metrics import classification_report

from utils import load_vocab, load_checkpoint

if __name__ == '__main__':
    opt = get_opts()
    vocab = load_vocab('checkpoints/vocab.pickle')
    dataset = QADataset_Test(opt, 'dataset/dataset.csv', vocab)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    model = Base_Net(opt, dataset.get_vocab_len())
    load_checkpoint(model, 'checkpoints/final_model.pth')
    labels = []
    preds = []
    for step, data in enumerate(dataloader, 0):
        # get data
        s1 = data['s1']
        s2 = data['s2']
        label = data['label']
        # input
        predication = model(s1, s2)
        # loss backward
        labels.append(label.detach().numpy())
        predication = (predication > 0.5).type(torch.float)
        preds.append(predication.detach().numpy())

    print(classification_report(np.concatenate(labels, 0), np.concatenate(preds, 0),
                                target_names=["no duplicate", "duplicate"]))
