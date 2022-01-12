import random

import pandas as pd
from collections import defaultdict
from dataloader.base_dataset import QADataset


class QADataset_Train(QADataset):
    def __init__(self, opt, dataset_path, START_TAG="<START>", STOP_TAG="<STOP>", PAD_TAG="<PAD>", OOV_TAG="<OOV>"):
        super().__init__(opt.max_no_tokens, START_TAG, STOP_TAG, PAD_TAG, OOV_TAG)

        df = pd.read_csv(dataset_path)
        self.triplets = []
        for i in range(df.shape[0]):
            current_triplet = []
            for j in range(3):
                sent = df.iloc[i][j]
                sent = self.preprocess(sent)
                sent = self.pad_sentence(sent)
                current_triplet.append(sent)
            self.triplets.append(current_triplet)
        # For each words-list (sentence) and tags-list in each tuple of training_data
        word_count = defaultdict(int)
        for sentences in self.triplets:
            for sent in sentences:
                for word in sent:
                    word_count[word] += 1
        for word, count in word_count.items():
            if word not in self.word_to_ix:  # word has not been assigned an index yet
                self.word_to_ix[word] = len(self.word_to_ix)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        anchor = self.to_tensor(self.triplets[index][0])
        positive = self.to_tensor(self.triplets[index][1])
        negative = self.to_tensor(self.triplets[index][2])
        return {"anchor": anchor, "positive": positive, "negative": negative}
