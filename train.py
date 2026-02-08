import pickle
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


class Vocabulary:
    PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"

    def __init__(self, min_freq=2):
        self.word2idx = {self.PAD: 0, self.START: 1, self.END: 2, self.UNK: 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.min_freq = min_freq

    def _tokenize(self, text):
        return text.lower().strip().split()

    def build(self, captions):
        count = Counter()
        for cap in captions:
            count.update(self._tokenize(cap))
        for word, freq in count.items():
            if freq >= self.min_freq and word not in self.word2idx:
                i = len(self.word2idx)
                self.word2idx[word] = i
                self.idx2word[i] = word

    def encode(self, caption, add_special=True):
        out = [self.word2idx[self.START]] if add_special else []
        for w in self._tokenize(caption):
            out.append(self.word2idx.get(w, self.word2idx[self.UNK]))
        if add_special:
            out.append(self.word2idx[self.END])
        return out

    def decode(self, ids):
        return " ".join(self.idx2word.get(i, self.UNK) for i in ids if i not in {0, 1, 2})


def load_captions(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[1 if "image" in lines[0].lower() else 0:]:
        if "," in line:
            img, cap = line.strip().split(",", 1)
            cap = cap.strip('"').strip()
            if cap:
                pairs.append((img.strip(), cap))
    return pairs


class Encoder(nn.Module):
    def __init__(self, input_dim=2048, hidden=512, dropout=0.5):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.dropout(x)
        return x


class ImageCaptioner(nn.Module):
    def __init__(self, vocab_size, hidden=1024, embed=512, dropout=0.5):
        super().__init__()
        self.encoder = Encoder(2048, hidden)
        self.embed = nn.Embedding(vocab_size, embed, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed, hidden, batch_first=True, dropout=0)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, feats, caps):
        h = self.encoder(feats).unsqueeze(0)
        c = h.new_zeros(h.size())
        emb = self.embed_dropout(self.embed(caps[:, :-1]))
        out, _ = self.lstm(emb, (h, c))
        out = self.fc_dropout(out)
        return self.fc(out)


def greedy_search(model, feat, vocab, max_len=50, device="cpu"):
    model.eval()
    feat = feat.to(device).unsqueeze(0)
    h = model.encoder(feat).unsqueeze(0)
    c = h.new_zeros(h.size())
    tokens = [1]
    with torch.no_grad():
        for _ in range(max_len - 1):
            x = torch.tensor([[tokens[-1]]], dtype=torch.long, device=device)
            emb = model.embed(x)
            out, (h, c) = model.lstm(emb, (h, c))
            logits = model.fc(out.squeeze(1))
            next_id = logits.argmax(dim=-1).item()
            tokens.append(next_id)
            if next_id == 2:
                break
    return vocab.decode(tokens)


def compute_bleu4(refs, hyps):
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    except ImportError:
        return 0.0
    refs = [[r.split()] for r in refs]
    hyps = [h.split() for h in hyps]
    return corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25),
                      smoothing_function=SmoothingFunction().method1)


class CaptionDataset(Dataset):
    def __init__(self, feat_path, pairs, vocab, max_len=50):
        with open(feat_path, "rb") as f:
            self.feats = pickle.load(f)
        self.pairs = [(img, cap) for img, cap in pairs if img in self.feats]
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img, cap = self.pairs[i]
        ids = self.vocab.encode(cap)
        if len(ids) > self.max_len:
            ids = ids[: self.max_len - 1] + [2]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(self.feats[img], dtype=torch.float32), torch.tensor(ids, dtype=torch.long)


def collate_batch(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = load_captions("data/captions.txt")
    vocab = Vocabulary(min_freq=2)
    vocab.build([cap for _, cap in pairs])

    ds = CaptionDataset("flickr30k_features.pkl", pairs, vocab)
    train_ds, val_ds = random_split(ds, [int(0.9 * len(ds)), len(ds) - int(0.9 * len(ds))])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=64, collate_fn=collate_batch)

    model = ImageCaptioner(len(vocab.word2idx)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = {"train_loss": [], "val_loss": []}
    for epoch in range(10):
        model.train()
        total = 0.0
        for feats, caps in tqdm(train_loader, leave=False):
            feats, caps = feats.to(device), caps.to(device)
            optimizer.zero_grad()
            logits = model(feats, caps)
            loss = criterion(logits.reshape(-1, logits.size(-1)), caps[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        train_loss = total / len(train_loader)
        history["train_loss"].append(train_loss)

        model.eval()
        total = 0.0
        with torch.no_grad():
            for feats, caps in val_loader:
                feats, caps = feats.to(device), caps.to(device)
                logits = model(feats, caps)
                loss = criterion(logits.reshape(-1, logits.size(-1)), caps[:, 1:].reshape(-1))
                total += loss.item()
        val_loss = total / len(val_loader)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch + 1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    torch.save({"model_state": model.state_dict(), "vocab": vocab, "history": history}, "caption_model.pt")
    print("Saved to caption_model.pt")

    model.eval()
    refs, hyps = [], []
    for idx in tqdm(val_ds.indices[: min(500, len(val_ds.indices))], desc="BLEU"):
        img, gt = val_ds.dataset.pairs[idx]
        refs.append(gt)
        hyps.append(greedy_search(model, torch.tensor(val_ds.dataset.feats[img]), vocab, device=device))
    print(f"BLEU-4: {compute_bleu4(refs, hyps):.4f}")
