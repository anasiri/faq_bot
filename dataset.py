import math
import pickle
import torch
import pandas as pd
from torch.utils.data import Dataset


class QADataset(Dataset):
    def __init__(self, opt, dataset_path, START_TAG="<START>", STOP_TAG="<STOP>", PAD_TAG="<PAD>"):
        self.START_TAG = START_TAG
        self.STOP_TAG = STOP_TAG
        self.PAD_TAG = PAD_TAG
        self.max_no_tokens = opt.max_no_tokens
        with open(dataset_path, 'rb') as f:
            cleaned_df = pickle.load(f)
        positive_data = cleaned_df[cleaned_df['is_duplicate'] == 1].iloc[:math.floor(opt.training_size / 2)]
        negative_data = cleaned_df[cleaned_df['is_duplicate'] == 0].iloc[:math.ceil(opt.training_size / 2)]
        balanced_dataset = pd.concat([positive_data, negative_data])
        self.y = balanced_dataset["is_duplicate"]
        sentence_pairs = balanced_dataset.drop("is_duplicate", 1)

        self.X = []
        for i in range(len(sentence_pairs)):
            sent1 = self.process_sentence(sentence_pairs.iloc[i]["question1_p"])
            sent2 = self.process_sentence(sentence_pairs.iloc[i]["question2_p"])
            self.X.append((sent1, sent2))
        self.word_to_ix = {PAD_TAG: 0, START_TAG: 1, STOP_TAG: 2}
        # For each words-list (sentence) and tags-list in each tuple of training_data
        for sentences in self.X:
            for sent in sentences:
                for word in sent:
                    if word not in self.word_to_ix:  # word has not been assigned an index yet
                        self.word_to_ix[word] = len(self.word_to_ix)

    def process_sentence(self, sentence):
        new_sentence = [self.START_TAG]
        new_sentence = new_sentence + sentence[:min(self.max_no_tokens - 2, len(sentence))]
        new_sentence.append(self.STOP_TAG)
        new_sentence = new_sentence + [self.PAD_TAG] * max(self.max_no_tokens - len(new_sentence), 0)
        return new_sentence

    def get_vocab_len(self):
        return len(self.word_to_ix)

    def to_tensor(self, seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        s1 = self.to_tensor(self.X[index][0], self.word_to_ix)
        s2 = self.to_tensor(self.X[index][1], self.word_to_ix)
        label = torch.tensor(self.y.iloc[index], dtype=torch.float).unsqueeze(0)
        return {"s1": s1, "s2": s2, "label": label}
