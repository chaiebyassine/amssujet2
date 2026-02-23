import math, random, argparse, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

def load_lines(path):
    return [l.strip() for l in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]

def build_vocab(lines):
    chars = set()
    for s in lines: chars.update(s)
    vocab = ["<PAD>","<SOS>","<EOS>"] + sorted(chars)
    stoi = {c:i for i,c in enumerate(vocab)}
    itos = {i:c for c,i in stoi.items()}
    return vocab, stoi, itos

def encode(s, stoi):
    return [stoi["<SOS>"]] + [stoi[c] for c in s if c in stoi] + [stoi["<EOS>"]]

def make_dataset(lines, stoi, max_len, limit):
    data=[]
    for s in lines[:limit]:
        ids = encode(s, stoi)
        if len(ids) >= 3 and len(ids) <= max_len:
            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)
            data.append((x,y))
    return data

def batchify(dataset, bs, pad_id, device):
    random.shuffle(dataset)
    for i in range(0, len(dataset), bs):
        batch = dataset[i:i+bs]
        xs=[x for x,_ in batch]
        ys=[y for _,y in batch]
        xpad = pad_sequence(xs, batch_first=True, padding_value=pad_id).to(device)
        ypad = pad_sequence(ys, batch_first=True, padding_value=pad_id).to(device)
        yield xpad, ypad

class Chomp1d(nn.Module):
    def __init__(self, chomp):
        super().__init__()
        self.chomp = chomp
    def forward(self, x):
        return x[:, :, :-self.chomp] if self.chomp > 0 else x

class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k, dilation, dropout):
        super().__init__()
        pad = (k-1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x):
        out = self.net(x)
        res = x if self.down is None else self.down(x)
        return out + res

class TCNLM(nn.Module):
    def __init__(self, vocab_size, emb=64, channels=128, layers=4, k=3, dropout=0.2, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=pad_id)
        blocks=[]
        c_in = emb
        for i in range(layers):
            d = 2**i
            blocks.append(TCNBlock(c_in, channels, k=k, dilation=d, dropout=dropout))
            c_in = channels
        self.tcn = nn.Sequential(*blocks)
        self.fc = nn.Linear(channels, vocab_size)

    def forward(self, x):
        # x: (B,T)
        e = self.emb(x)            # (B,T,E)
        h = e.transpose(1,2)       # (B,E,T)
        h = self.tcn(h)            # (B,C,T)
        h = h.transpose(1,2)       # (B,T,C)
        return self.fc(h)          # (B,T,V)

@torch.no_grad()
def evaluate(model, dataset, bs, pad_id, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    total_loss, total_tok = 0.0, 0
    for x,y in batchify(dataset, bs, pad_id, device):
        logits = model(x)
        B,T,V = logits.shape
        loss = loss_fn(logits.reshape(B*T,V), y.reshape(B*T))
        total_loss += loss.item()
        total_tok += (y != pad_id).sum().item()
    nll = total_loss / max(1, total_tok)
    ppl = math.exp(min(50.0, nll))
    return nll, ppl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=20)
    ap.add_argument("--limit_train", type=int, default=999999999)
    ap.add_argument("--limit_eval", type=int, default=999999999)
    ap.add_argument("--dropout", type=float, default=0.3)
    args = ap.parse_args()

    device = torch.device("cpu")
    train_lines = load_lines("train.txt")
    eval_lines  = load_lines("eval.txt")

    vocab, stoi, itos = build_vocab(train_lines)
    pad_id = stoi["<PAD>"]

    train_ds = make_dataset(train_lines, stoi, args.max_len, args.limit_train)
    eval_ds  = make_dataset(eval_lines,  stoi, args.max_len, args.limit_eval)

    model = TCNLM(len(vocab), emb=64, channels=128, layers=4, k=3, dropout=args.dropout, pad_id=pad_id).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    best = float("inf")
    for ep in range(1, args.epochs+1):
        model.train()
        total_loss, total_tok = 0.0, 0
        start = time.time()

        for x,y in batchify(train_ds, args.batch, pad_id, device):
            opt.zero_grad()
            logits = model(x)
            B,T,V = logits.shape
            loss = loss_fn(logits.reshape(B*T,V), y.reshape(B*T))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tok = (y != pad_id).sum().item()
            total_loss += loss.item() * tok
            total_tok += tok

        train_nll = total_loss / max(1, total_tok)
        train_ppl = math.exp(min(50.0, train_nll))
        eval_nll, eval_ppl = evaluate(model, eval_ds, args.batch, pad_id, device)
        print(f"Epoch {ep}/{args.epochs} | train_ppl={train_ppl:.2f} | eval_ppl={eval_ppl:.2f} | {int(time.time()-start)}s")

        if eval_nll < best:
            best = eval_nll
            torch.save(model.state_dict(), "model_tcn.pt")

if __name__ == "__main__":
    main()
