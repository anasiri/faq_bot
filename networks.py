from torch import nn
import torch


class SiameseNet(nn.Module):

    def __init__(self,opt, vocab_size, target_size=1):
        super(SiameseNet, self).__init__()
        self.hidden_dim = opt.hidden_dim
        self.max_no_tokens = opt.max_no_tokens
        self.word_embeddings = nn.Embedding(vocab_size, opt.embedding_dim)
        self.lstm = nn.LSTM(opt.embedding_dim, opt.hidden_dim)

        self.classifier = nn.Sequential(nn.Linear(self.hidden_dim * 5 * self.max_no_tokens, self.hidden_dim * 3),
                                        nn.Dropout(0.8),
                                        nn.BatchNorm1d( self.hidden_dim * 3),
                                        nn.Linear(self.hidden_dim * 3, self.hidden_dim),
                                        nn.Dropout(0.8),
                                        nn.BatchNorm1d( self.hidden_dim),
                                        nn.Linear(self.hidden_dim, target_size),
                                        nn.Sigmoid())

    def forward(self, s1, s2):
        emb1 = self.word_embeddings(s1)
        v1, _ = self.lstm(emb1)

        emb2 = self.word_embeddings(s2)
        v2, _ = self.lstm(emb2)

        features = torch.cat((v1, torch.abs(v1 - v2), v2, v1 * v2, (v1 + v2) / 2), 1)
        # output = self.classifier(torch.cat([v1,v2],1))
        features = features.reshape(features.shape[0],-1)
        output = self.classifier(features)
        return output
