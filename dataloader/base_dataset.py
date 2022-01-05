import torch
from torch.utils.data import Dataset
from hazm import word_tokenize, Normalizer


class QADataset(Dataset):
    def __init__(self, max_no_tokens, START_TAG="<START>", STOP_TAG="<STOP>", PAD_TAG="<PAD>"):
        self.START_TAG = START_TAG
        self.STOP_TAG = STOP_TAG
        self.PAD_TAG = PAD_TAG
        self.max_no_tokens = max_no_tokens
        self.normalizer = Normalizer()
        self.X = []
        self.y = []
        self.word_to_ix = {PAD_TAG: 0, START_TAG: 1, STOP_TAG: 2}

    def preprocess(self, x):
        out = self.normalizer.normalize(x)
        out = word_tokenize(out)
        out = [i for i in out if i not in ["؟", "!", ".", "،", ",", "?", ":", "<", ">", "(", ")", "{", "}"]]
        return out

    def pad_sentence(self, sentence):
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
        pass
