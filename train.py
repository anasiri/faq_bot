import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloader.train_dataset import  QADataset_Train
from networks import SiameseNet
from options import get_opts
import numpy as np
from sklearn.metrics import classification_report,accuracy_score,f1_score,recall_score,precision_score

from utils import save_checkpoint, save_vocab

if __name__ == '__main__':
    opt = get_opts()
    dataset = QADataset_Train(opt, 'dataset/dataset.csv')
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
            preds.append(predication.detach().numpy())

            label = label.detach().numpy()
            predication = predication.reshape(-1).detach().numpy()

            accuracy = accuracy_score(label,predication)
            recall = recall_score(label,predication)
            precision = precision_score(label,predication)
            f1 = f1_score(label,predication)

            if step % 10 == 0:
                print(f"Step {step}/{opt.training_size//opt.batch_size} accuracy: {accuracy:.3f} precision: {precision:.3f}"
                      f" recall: {recall:.3f} f1-score: {f1:.3f} loss:{loss:.3f}")
        print(classification_report(np.concatenate(labels, 0), np.concatenate(preds, 0), target_names=["no duplicate", "duplicate"]))
        print(f"Finished epoch {epoch}/{opt.epochs}")
    save_checkpoint(model, 'checkpoints/final_model.pth')
    save_vocab(dataset.word_to_ix, 'checkpoints/vocab.pickle')
