import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataloader.train_dataset import QADataset_Train
from networks import SiameseNet
from options import get_opts
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from utils import save_checkpoint, save_vocab, report_validation_scores

if __name__ == '__main__':
    opt = get_opts()
    dataset = QADataset_Train(opt, 'dataset/dataset.csv')
    train_percent = int(opt.training_size * 0.8)
    val_percent = opt.training_size -train_percent
    train_set, val_set = torch.utils.data.random_split(dataset, [train_percent,val_percent])
    train_dataloader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    model = SiameseNet(opt, dataset.get_vocab_len())

    learning_rate = opt.lr_rate
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=learning_rate,weight_decay=0.01)

    criterion = torch.nn.BCELoss()  # ContrastiveLoss()

    loss_weights = Variable(torch.FloatTensor([1, 3]))
    if torch.cuda.is_available():
        loss_weights = loss_weights.cuda()
    # criterion = torch.nn.CrossEntropyLoss(loss_weights)
    for epoch in range(1, opt.epochs + 1):
        labels = []
        preds = []
        for step, data in enumerate(train_dataloader, 0):
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

            accuracy = accuracy_score(label, predication)
            recall = recall_score(label, predication)
            precision = precision_score(label, predication)
            f1 = f1_score(label, predication)

            if step % 10 == 0:
                print(
                    f"Step {step}/{opt.training_size // opt.batch_size} accuracy: {accuracy:.3f} precision: {precision:.3f}"
                    f" recall: {recall:.3f} f1-score: {f1:.3f} loss:{loss:.3f}")
        print(f"Finished epoch {epoch}/{opt.epochs}")
        print(f"Testing on validation set...")
        report_validation_scores(valid_dataloader,model.eval())
    save_checkpoint(model, 'checkpoints/final_model.pth')
    save_vocab(dataset.word_to_ix, 'checkpoints/vocab.pickle')
