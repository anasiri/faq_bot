import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import QADataset
from loss import ContrastiveLoss
from networks import SiameseNet
from options import get_opts
import numpy as np
from sklearn.metrics import classification_report

if __name__ == '__main__':
    opt = get_opts()
    dataset = QADataset(opt, 'dataset/data.pickle')
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    model = SiameseNet(opt, dataset.get_vocab_len())

    learning_rate = opt.lr_rate
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=learning_rate)

    criterion = torch.nn.BCELoss()  # ContrastiveLoss()

    loss_weights = Variable(torch.FloatTensor([1, 3]))
    if torch.cuda.is_available():
        loss_weights = loss_weights.cuda()
    # criterion = torch.nn.CrossEntropyLoss(loss_weights)
    for epoch in range(opt.epochs):
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
            optimizer.zero_grad()
            loss = criterion(predication, label)
            loss.backward()
            optimizer.step()
            labels.append(label.detach().numpy())
            predication = (predication > 0.5).type(torch.float)
            acc = (predication.reshape(-1).detach().numpy().round() == label.detach().numpy()).mean()
            preds.append(predication.detach().numpy())
            if step % 10 == 0:
                print(f"Step {step}/{opt.training_size//opt.batch_size}  \t loss:{loss} accuracy: {acc} ")
        print(classification_report(np.concatenate(labels, 0), np.concatenate(preds, 0), target_names=["no duplicate", "duplicate"]))
        print(f"Finished epoch {epoch}/{opt.epochs}")
