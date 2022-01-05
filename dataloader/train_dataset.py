import math
import torch
import pandas as pd
from hazm import word_tokenize, Normalizer
from dataloader.base_dataset import QADataset


class QADataset_Train(QADataset):
    def __init__(self, opt, dataset_path, START_TAG="<START>", STOP_TAG="<STOP>", PAD_TAG="<PAD>"):
        super().__init__(opt.max_no_tokens, START_TAG, STOP_TAG, PAD_TAG)

        df = pd.read_csv(dataset_path)
        positive_data = df[df['is_duplicate'] == 1].iloc[:math.floor(opt.training_size / 2)]
        negative_data = df[df['is_duplicate'] == 0].iloc[:math.ceil(opt.training_size / 2)]
        balanced_dataset = pd.concat([positive_data, negative_data])
        self.y = balanced_dataset["is_duplicate"]
        sentence_pairs = balanced_dataset[["question1_p", "question2_p"]]

        self.X = []
        for i in range(len(sentence_pairs)):
            sent1 = sentence_pairs.iloc[i]["question1_p"]
            sent1 = self.preprocess(sent1)
            sent1 = self.pad_sentence(sent1)

            sent2 = sentence_pairs.iloc[i]["question2_p"]
            sent2 = self.preprocess(sent2)
            sent2 = self.pad_sentence(sent2)

            self.X.append((sent1, sent2))
        self.word_to_ix = {PAD_TAG: 0, START_TAG: 1, STOP_TAG: 2}
        # For each words-list (sentence) and tags-list in each tuple of training_data
        for sentences in self.X:
            for sent in sentences:
                for word in sent:
                    if word not in self.word_to_ix:  # word has not been assigned an index yet
                        self.word_to_ix[word] = len(self.word_to_ix)
    def __getitem__(self, index):
        s1 = self.to_tensor(self.X[index][0], self.word_to_ix)
        s2 = self.to_tensor(self.X[index][1], self.word_to_ix)
        label = torch.tensor(self.y.iloc[index], dtype=torch.float).unsqueeze(0)
        return {"s1": s1, "s2": s2, "label": label}
