import math, random, argparse, time
from pathlib import Path

import torch
import torch.nn as nn

# -------------------------
# Utils
# -------------------------
def load_lines(path):
    return [l.strip("\n") for l in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]

def build_vocab(lines):
    chars = set()
    for s in lines:
        chars.update(list(s))
    PAD, SOS, EOS = "<PAD>", "<SOS>", "<EOS>"
    vocab = [PAD, SOS, EOS] + sorted(chars)
    stoi = {ch:i for i,ch in enumerate(vocab)}
    itos = {i:ch for ch,i in stoi.items()}
    return vocab, stoi, itos

def encode(s, stoi):
    return [stoi["<SOS>"]] + [stoi[c] for c in s if c in stoi] + [stoi["<EOS>"]]

def batchify(seqs, batch_size, pad_id, device):
    random.shuffle(seqs)
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i+batch_size]
        lens = [len(x) for x in batch]
        T = max(lens)
        x = torch.full((len(batch), T-1), pad_id, dtype=torch.long)
        y = torch.full((len(batch), T-1), pad_id, dtype=torch.long)
        for b, ids in enumerate(batch):
            ids = torch.tensor(ids, dtype=torch.long)
            x[b, :len(ids)-1] = ids[:-1]
            y[b, :len(ids)-1] = ids[1:]
        yield x.to(device), y.to(device)

# -------------------------
# Model
# -------------------------
class CharRNNLM(nn.Module):
    def __init__(self, vocab_size, emb=64, hidden=128, layers=1, cell="lstm", dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=pad_id)
        cell = cell.lower()
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(
            input_size=emb,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        e = self.emb(x)
        out, _ = self.rnn(e)
        return self.fc(out)

@torch.no_grad()
def evaluate(model, eval_seqs, batch_size, pad_id, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    total_loss, total_tok = 0.0, 0
    for x, y in batchify(eval_seqs, batch_size, pad_id, device):
        logits = model(x)
        B,T,V = logits.shape
        loss = loss_fn(logits.reshape(B*T, V), y.reshape(B*T))
        total_loss += loss.item()
        total_tok += (y != pad_id).sum().item()
    nll = total_loss / max(1, total_tok)
    ppl = math.exp(min(50.0, nll))
    return nll, ppl

@torch.no_grad()
def generate(model, stoi, itos, device, max_len=12, temperature=1.0):
    model.eval()
    pad_id = stoi["<PAD>"]
    sos = stoi["<SOS>"]
    eos = stoi["<EOS>"]

    x = torch.tensor([[sos]], dtype=torch.long, device=device)
    out = []
    for _ in range(max_len):
        logits = model(x)  # (1,T,V)
        last = logits[0, -1] / max(1e-6, temperature)
        probs = torch.softmax(last, dim=0)
        probs[pad_id] = 0
        probs = probs / probs.sum()
        idx = torch.multinomial(probs, 1).item()
        if idx == eos:
            break
        out.append(itos[idx])
        x = torch.cat([x, torch.tensor([[idx]], device=device)], dim=1)
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
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_len", type=int, default=20)
    ap.add_argument("--limit_train", type=int, default=20000)
    ap.add_argument("--limit_eval", type=int, default=2000)
    ap.add_argument("--mode", choices=["train","gen"], default="train")
    ap.add_argument("--out", default="10k.txt")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_lines = load_lines(args.train)
    eval_lines  = load_lines(args.eval)

    vocab, stoi, itos = build_vocab(train_lines)
    pad_id = stoi["<PAD>"]

    train_seqs = []
    for s in train_lines[:args.limit_train]:
        ids = encode(s, stoi)
        if len(ids) >= 3 and len(ids) <= args.max_len:
            train_seqs.append(ids)

    eval_seqs = []
    for s in eval_lines[:args.limit_eval]:
        ids = encode(s, stoi)
        if len(ids) >= 3 and len(ids) <= args.max_len:
            eval_seqs.append(ids)

    model = CharRNNLM(
        vocab_size=len(vocab),
        emb=args.emb,
        hidden=args.hidden,
        layers=args.layers,
        cell=args.cell,
        dropout=args.dropout,
        pad_id=pad_id
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)

    ckpt = f"model_{args.cell}.pt"

    if args.mode == "train":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
        best_eval = float("inf")

        print(f"Train seq: {len(train_seqs)} | Eval seq: {len(eval_seqs)} | Vocab: {len(vocab)} | cell={args.cell}")

        for ep in range(1, args.epochs+1):
            model.train()
            total_loss, total_tok = 0.0, 0
            start = time.time()

            for x, y in batchify(train_seqs, args.batch, pad_id, device):
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
            eval_nll, eval_ppl = evaluate(model, eval_seqs, args.batch, pad_id, device)

            print(f"Epoch {ep}/{args.epochs} | train_ppl={train_ppl:.2f} | eval_ppl={eval_ppl:.2f} | {int(time.time()-start)}s")

            if eval_nll < best_eval:
                best_eval = eval_nll
                torch.save({"state": model.state_dict(), "stoi": stoi, "itos": itos, "args": vars(args)}, ckpt)

        print("Saved best model to", ckpt)

    else:  # gen
        if not Path(ckpt).exists():
            print("ERROR: model file not found:", ckpt)
            return

        saved = torch.load(ckpt, map_location=device)
        model.load_state_dict(saved["state"])
        stoi = saved["stoi"]
        itos = saved["itos"]
        model.eval()

        print(f"Generating {args.n} passwords with temperature={args.temperature}")

        with open(args.out, "w", encoding="utf-8") as f:
            for _ in range(args.n):
                pw = generate(model, stoi, itos, device, max_len=args.max_len, temperature=args.temperature)
                f.write(pw + "\n")

        print("Saved generated passwords to:", args.out)

if __name__ == "__main__":
    main()
