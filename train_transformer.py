import math, random, argparse, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

def load_lines(path):
    return [l.strip() for l in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]

def build_vocab(lines):
    chars = set()
    for s in lines:
        chars.update(list(s))
    vocab = ["<PAD>", "<SOS>", "<EOS>"] + sorted(chars)
    stoi = {ch:i for i,ch in enumerate(vocab)}
    itos = {i:ch for ch,i in stoi.items()}
    return vocab, stoi, itos

def encode(line, stoi):
    return [stoi["<SOS>"]] + [stoi[c] for c in line if c in stoi] + [stoi["<EOS>"]]

def make_dataset(lines, stoi, max_len, limit):
    data = []
    for s in lines[:limit]:
        ids = encode(s, stoi)
        if len(ids) < 3:
            continue
        if max_len and len(ids) > max_len:
            ids = ids[:max_len-1] + [stoi["<EOS>"]]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        data.append((x, y))
    return data

def batchify(dataset, batch_size, pad_id, device):
    random.shuffle(dataset)
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        xs = [x for x,_ in batch]
        ys = [y for _,y in batch]
        xpad = pad_sequence(xs, batch_first=True, padding_value=pad_id).to(device)
        ypad = pad_sequence(ys, batch_first=True, padding_value=pad_id).to(device)
        yield xpad, ypad

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CausalTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=2, layers=1, ff=128, dropout=0.3, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=dropout,
            activation="gelu"
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (B,T)
        B, T = x.shape
        h = self.emb(x) * math.sqrt(self.emb.embedding_dim)
        h = self.pos(h)
        h = self.drop(h)
        h = h.transpose(0, 1)  # (T,B,E)

        # causal mask: bloque le futur
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        # padding mask: ignore PAD
        key_padding = (x == self.pad_id)  # (B,T)

        out = self.tr(h, mask=causal, src_key_padding_mask=key_padding)  # (T,B,E)
        out = out.transpose(0, 1)  # (B,T,E)
        out = self.drop(out)
        return self.fc(out)  # (B,T,V)

@torch.no_grad()
def evaluate(model, dataset, batch_size, pad_id, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    total_loss, total_tok = 0.0, 0
    for x, y in batchify(dataset, batch_size, pad_id, device):
        logits = model(x)
        B,T,V = logits.shape
        loss = loss_fn(logits.reshape(B*T, V), y.reshape(B*T))
        total_loss += loss.item()
        total_tok += (y != pad_id).sum().item()
    nll = total_loss / max(1, total_tok)
    ppl = math.exp(min(50.0, nll))
    return nll, ppl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.txt")
    ap.add_argument("--eval", default="eval.txt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=20)
    ap.add_argument("--limit_train", type=int, default=999999999)
    ap.add_argument("--limit_eval", type=int, default=999999999)

    ap.add_argument("--d_model", type=int, default=32)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--ff", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    train_lines = load_lines(args.train)
    eval_lines  = load_lines(args.eval)

    vocab, stoi, itos = build_vocab(train_lines)
    pad_id = stoi["<PAD>"]

    train_ds = make_dataset(train_lines, stoi, args.max_len, args.limit_train)
    eval_ds  = make_dataset(eval_lines,  stoi, args.max_len, args.limit_eval)

    model = CausalTransformerLM(
        vocab_size=len(vocab),
        d_model=args.d_model,
        nhead=args.heads,
        layers=args.layers,
        ff=args.ff,
        dropout=args.dropout,
        pad_id=pad_id
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=args.label_smoothing)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=1, verbose=True)

    print(f"Device: {device}")
    print(f"Transformer | Vocab: {len(vocab)} | Train seq: {len(train_ds)} | Eval seq: {len(eval_ds)}")
    print(f"d_model={args.d_model} heads={args.heads} layers={args.layers} ff={args.ff} dropout={args.dropout}")
    print(f"lr={args.lr} weight_decay={args.weight_decay} label_smoothing={args.label_smoothing}")

    best_eval = float("inf")
    bad = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss, total_tok = 0.0, 0
        start = time.time()

        for x, y in batchify(train_ds, args.batch, pad_id, device):
            opt.zero_grad()
            logits = model(x)
            B,T,V = logits.shape
            loss = loss_fn(logits.reshape(B*T, V), y.reshape(B*T))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tok = (y != pad_id).sum().item()
            total_loss += loss.item() * tok
            total_tok += tok

        train_nll = total_loss / max(1, total_tok)
        train_ppl = math.exp(min(50.0, train_nll))
        eval_nll, eval_ppl = evaluate(model, eval_ds, args.batch, pad_id, device)
        scheduler.step(eval_nll)

        print(f"Epoch {epoch}/{args.epochs} | train_ppl={train_ppl:.2f} | eval_ppl={eval_ppl:.2f} | {int(time.time()-start)}s")

        if eval_nll < best_eval:
            best_eval = eval_nll
            bad = 0
            torch.save({"state": model.state_dict(), "args": vars(args)}, "best_transformer.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping (patience={args.patience}). Best eval_nll={best_eval:.4f}")
                break

if __name__ == "__main__":
    main()
