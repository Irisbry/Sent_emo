import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter


def simple_tokenizer(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


class Vocab:
    def __init__(self, min_freq=20, max_size=15000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = ["<pad>", "<unk>"]

    def build(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(simple_tokenizer(text))

        most_common = counter.most_common(self.max_size)
        for word, freq in most_common:
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)

    def encode(self, tokens):
        return [
            self.word2idx.get(token, self.word2idx["<unk>"])
            for token in tokens
        ]

    def __len__(self):
        return len(self.idx2word)


class IMDBDataset(Dataset):
    def __init__(self, root_dir, split, vocab=None, max_len=500):
        self.samples = []
        self.max_len = max_len
        self.vocab = vocab

        for label, folder in [(0, "pos"), (1, "neg")]:
            dir_path = os.path.join(root_dir, split, folder)
            for fname in os.listdir(dir_path):
                with open(os.path.join(dir_path, fname), encoding="utf-8") as f:
                    self.samples.append((label, f.read()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, text = self.samples[idx]
        tokens = simple_tokenizer(text)[:self.max_len]

        if self.vocab:
            tokens = self.vocab.encode(tokens)

        return label, tokens


def collate_fn(batch):
    labels, sequences = zip(*batch)
    max_len = max(len(seq) for seq in sequences)

    padded = [
        seq + [0] * (max_len - len(seq))
        for seq in sequences
    ]

    return (
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(padded, dtype=torch.long),
    )


def get_dataloader(
    imdb_root,
    split,
    batch_size,
    vocab=None,
    shuffle=True,
):
    dataset = IMDBDataset(imdb_root, split, vocab)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    return loader
