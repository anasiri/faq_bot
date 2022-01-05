import math
import torch
import pandas as pd
from dataloader.base_dataset import QADataset


class QADataset_Test(QADataset):
    def __init__(self, opt, dataset_path, word_to_ix, START_TAG="<START>", STOP_TAG="<STOP>", PAD_TAG="<PAD>",OOV_TAG="<OOV>"):
        super().__init__(opt.max_no_tokens, START_TAG, STOP_TAG, PAD_TAG,OOV_TAG)

        df = pd.read_csv(dataset_path)
        self.y = df["is_duplicate"]
        sentence_pairs = df[["question1_p", "question2_p"]]
        self.word_to_ix = word_to_ix
        self.X = []
        for i in range(len(sentence_pairs)):
            sent1 = sentence_pairs.iloc[i]["question1_p"]
            sent1 = self.preprocess(sent1)
            sent1 = self.pad_sentence(sent1)

            sent2 = sentence_pairs.iloc[i]["question2_p"]
            sent2 = self.preprocess(sent2)
            sent2 = self.pad_sentence(sent2)

            self.X.append((sent1, sent2))

    def __getitem__(self, index):
        s1 = self.to_tensor(self.X[index][0])
        s2 = self.to_tensor(self.X[index][1])
        label = torch.tensor(self.y.iloc[index], dtype=torch.float).unsqueeze(0)
        return {"s1": s1, "s2": s2, "label": label}
