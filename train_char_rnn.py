import math, random, argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

def load_lines(path: str):
    return Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()

def build_vocab(lines):
    chars = set()
    for s in lines:
        chars.update(list(s))
    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    vocab = [PAD, SOS, EOS] + sorted(chars)
    stoi = {ch:i for i,ch in enumerate(vocab)}
    itos = {i:ch for ch,i in stoi.items()}
    return vocab, stoi, itos

def encode(line, stoi):
    ids = [stoi["<SOS>"]] + [stoi[c] for c in line if c in stoi] + [stoi["<EOS>"]]
    return torch.tensor(ids, dtype=torch.long)

def make_dataset(lines, stoi, max_len):
    data = []
    for s in lines:
        if not s:
            continue
        t = encode(s, stoi)
        if len(t) < 3:
            continue
        if max_len and len(t) > max_len:
            t = t[:max_len-1]
            t = torch.cat([t, torch.tensor([stoi["<EOS>"]], dtype=torch.long)], dim=0)
        data.append((t[:-1], t[1:]))
    return data

def batchify(dataset, batch_size, pad_id, device):
    random.shuffle(dataset)
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        xs = [x for x,_ in batch]
        ys = [y for _,y in batch]
        lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
        xs_pad = pad_sequence(xs, batch_first=True, padding_value=pad_id).to(device)
        ys_pad = pad_sequence(ys, batch_first=True, padding_value=pad_id).to(device)
        lengths = lengths.to(device)
        yield xs_pad, ys_pad, lengths

class CharLM(nn.Module):
    def __init__(self, vocab_size, emb=64, hidden=256, layers=1, dropout=0.1, cell="lstm"):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        cell = cell.lower()
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(
            input_size=emb,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x, lengths):
        e = self.emb(x)
        packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        logits = self.fc(out)
        return logits

@torch.no_grad()
def evaluate(model, dataset, batch_size, pad_id, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    for xs, ys, lengths in batchify(dataset, batch_size, pad_id, device):
        logits = model(xs, lengths)
        B,T,V = logits.shape
        loss = loss_fn(logits.reshape(B*T, V), ys.reshape(B*T))
        total_loss += loss.item()
        total_tokens += (ys != pad_id).sum().item()
    if total_tokens == 0:
        return float("inf"), float("inf")
    avg_nll = total_loss / total_tokens
    ppl = math.exp(min(50.0, avg_nll))
    return avg_nll, ppl

@torch.no_grad()
def generate(model, stoi, itos, device, max_len=16, temperature=1.0):
    model.eval()
    pad_id = stoi["<PAD>"]
    sos = stoi["<SOS>"]
    eos = stoi["<EOS>"]

    x = torch.tensor([[sos]], dtype=torch.long, device=device)
    lengths = torch.tensor([1], dtype=torch.long, device=device)
    out = []
    for _ in range(max_len):
        logits = model(x, lengths)
        last = logits[0, -1] / max(1e-6, temperature)
        probs = torch.softmax(last, dim=0)
        probs[pad_id] = 0
        probs = probs / probs.sum()
        idx = torch.multinomial(probs, 1).item()
        if idx == eos:
            break
        out.append(itos[idx])
        x = torch.cat([x, torch.tensor([[idx]], device=device)], dim=1)
        lengths = torch.tensor([x.shape[1]], dtype=torch.long, device=device)
    return "".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.txt")
    ap.add_argument("--eval", default="eval.txt")
    ap.add_argument("--cell", choices=["rnn","lstm","gru"], default="lstm")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--emb", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gen_n", type=int, default=5)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔥 تحميل البيانات
    train_lines = load_lines(args.train)
    eval_lines  = load_lines(args.eval)

    # 🔥 دمج + خلط
    all_lines = train_lines + eval_lines
    random.shuffle(all_lines)

    # 🔥 تقسيم 80/20 جديد
    split = int(0.8 * len(all_lines))
    train_lines = all_lines[:split]
    eval_lines  = all_lines[split:]

    # 🔥 بناء vocab من الكل
    vocab, stoi, itos = build_vocab(all_lines)
    pad_id = stoi["<PAD>"]

    train_ds = make_dataset(train_lines, stoi, args.max_len)
    eval_ds  = make_dataset(eval_lines,  stoi, args.max_len)

    model = CharLM(
        vocab_size=len(vocab),
        emb=args.emb,
        hidden=args.hidden,
        layers=args.layers,
        dropout=0.2,
        cell=args.cell
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    print(f"Device: {device}")
    print(f"Cell: {args.cell} | Vocab: {len(vocab)}")
    print(f"Train seq: {len(train_ds)} | Eval seq: {len(eval_ds)}")

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for xs, ys, lengths in batchify(train_ds, args.batch, pad_id, device):
            opt.zero_grad()
            logits = model(xs, lengths)
            B,T,V = logits.shape
            loss = loss_fn(logits.reshape(B*T, V), ys.reshape(B*T))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tokens = (ys != pad_id).sum().item()
            total_loss += loss.item() * tokens
            total_tokens += tokens

        train_nll = total_loss / max(1, total_tokens)
        train_ppl = math.exp(min(50.0, train_nll))

        eval_nll, eval_ppl = evaluate(model, eval_ds, args.batch, pad_id, device)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_ppl={train_ppl:.2f} | "
              f"eval_ppl={eval_ppl:.2f}")

    print("\nGenerated examples:")
    for _ in range(args.gen_n):
        print(generate(model, stoi, itos, device, max_len=16, temperature=0.8))
if __name__ == "__main__":
    main()
