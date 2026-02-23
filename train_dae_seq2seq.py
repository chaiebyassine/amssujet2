import math, random, argparse, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# -----------------------------
# Data / Vocab
# -----------------------------
def load_lines(path):
    return [l.strip() for l in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]

def build_vocab(lines):
    chars = set()
    for s in lines:
        chars.update(list(s))
    vocab = ["<PAD>", "<SOS>", "<EOS>"] + sorted(chars)
    stoi = {c:i for i,c in enumerate(vocab)}
    itos = {i:c for c,i in stoi.items()}
    return vocab, stoi, itos

def encode(line, stoi):
    return [stoi["<SOS>"]] + [stoi[c] for c in line if c in stoi] + [stoi["<EOS>"]]

def maybe_truncate(ids, max_len, eos_id):
    if max_len and len(ids) > max_len:
        return ids[:max_len-1] + [eos_id]
    return ids

# -----------------------------
# Denoising noise
# -----------------------------
def denoise(ids, pad_id, sos_id, eos_id, drop_p=0.1, rep_p=0.1, vocab_min=3, vocab_size=100):
    """
    ids = [SOS, ..., EOS]
    On ne modifie pas SOS/EOS.
    - drop_p : supprime des tokens
    - rep_p : remplace par un token aléatoire (hors PAD/SOS/EOS)
    """
    if len(ids) <= 3:
        return ids[:]  # trop court

    core = ids[1:-1]
    out = []
    for t in core:
        if random.random() < drop_p:
            continue
        if random.random() < rep_p:
            t = random.randint(vocab_min, vocab_size-1)
        out.append(t)

    if len(out) == 0:
        out = [random.randint(vocab_min, vocab_size-1)]

    return [sos_id] + out + [eos_id]

# -----------------------------
# Batching
# -----------------------------
def make_dataset(lines, stoi, max_len, limit):
    eos = stoi["<EOS>"]
    seqs = []
    for s in lines[:limit]:
        ids = encode(s, stoi)
        ids = maybe_truncate(ids, max_len, eos)
        if len(ids) >= 3:
            seqs.append(ids)
    return seqs

def batchify(seqs, batch_size, pad_id, device):
    random.shuffle(seqs)
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i+batch_size]
        lens = [len(x) for x in batch]
        x = [torch.tensor(b, dtype=torch.long) for b in batch]
        xpad = pad_sequence(x, batch_first=True, padding_value=pad_id)
        yield xpad.to(device), torch.tensor(lens, dtype=torch.long).to(device)

# -----------------------------
# Model
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb=64, hidden=128, layers=1, dropout=0.1, pad_id=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=pad_id)
        self.rnn = nn.GRU(
            input_size=emb,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=False
        )

    def forward(self, x, lengths):
        e = self.emb(x)
        packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)  # h: (layers, B, hidden)
        return h

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb=64, hidden=128, layers=1, dropout=0.1, pad_id=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=pad_id)
        self.rnn = nn.GRU(
            input_size=emb,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, y_inp, h0):
        # y_inp: (B,T) => logits (B,T,V)
        e = self.emb(y_inp)
        out, _ = self.rnn(e, h0)
        return self.fc(out)

class Seq2SeqDAE(nn.Module):
    def __init__(self, vocab_size, emb=64, hidden=128, layers=1, dropout=0.1, pad_id=0):
        super().__init__()
        self.enc = Encoder(vocab_size, emb, hidden, layers, dropout, pad_id)
        self.dec = Decoder(vocab_size, emb, hidden, layers, dropout, pad_id)

# -----------------------------
# Train / Eval
# -----------------------------
def make_noised_batch(xpad, lengths, stoi, drop_p, rep_p, vocab_size, device):
    pad = stoi["<PAD>"]; sos = stoi["<SOS>"]; eos = stoi["<EOS>"]
    noised = []
    new_lens = []
    for b in range(xpad.size(0)):
        ids = xpad[b, :lengths[b]].tolist()
        ids2 = denoise(ids, pad, sos, eos, drop_p=drop_p, rep_p=rep_p, vocab_min=3, vocab_size=vocab_size)
        noised.append(torch.tensor(ids2, dtype=torch.long))
        new_lens.append(len(ids2))
    x_noised = pad_sequence(noised, batch_first=True, padding_value=pad)
    return x_noised.to(device), torch.tensor(new_lens, dtype=torch.long).to(device)

def teacher_forcing_inputs_targets(x_clean, lengths, pad_id):
    # x_clean contient [SOS ... EOS]
    # decoder input = tout sauf dernier token
    # target = tout sauf premier token
    ys_inp = []
    ys_tgt = []
    for b in range(x_clean.size(0)):
        ids = x_clean[b, :lengths[b]]
        ys_inp.append(ids[:-1])
        ys_tgt.append(ids[1:])
    y_inp = pad_sequence(ys_inp, batch_first=True, padding_value=pad_id)
    y_tgt = pad_sequence(ys_tgt, batch_first=True, padding_value=pad_id)
    return y_inp, y_tgt

@torch.no_grad()
def evaluate(model, eval_seqs, batch_size, pad_id, stoi, drop_p, rep_p, vocab_size, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    total_loss, total_tok = 0.0, 0

    for x_clean, lengths in batchify(eval_seqs, batch_size, pad_id, device):
        x_noised, nlen = make_noised_batch(x_clean, lengths, stoi, drop_p, rep_p, vocab_size, device)
        h = model.enc(x_noised, nlen)
        y_inp, y_tgt = teacher_forcing_inputs_targets(x_clean, lengths, pad_id)
        logits = model.dec(y_inp, h)
        B,T,V = logits.shape
        loss = loss_fn(logits.reshape(B*T, V), y_tgt.reshape(B*T))
        total_loss += loss.item()
        total_tok += (y_tgt != pad_id).sum().item()

    nll = total_loss / max(1, total_tok)
    ppl = math.exp(min(50.0, nll))
    return nll, ppl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.txt")
    ap.add_argument("--eval", default="eval.txt")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=20)
    ap.add_argument("--limit_train", type=int, default=999999999)
    ap.add_argument("--limit_eval", type=int, default=999999999)

    ap.add_argument("--emb", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--lr", type=float, default=1e-3)

    # bruit
    ap.add_argument("--drop_p", type=float, default=0.10, help="probabilité suppression token")
    ap.add_argument("--rep_p", type=float, default=0.10, help="probabilité remplacement token")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    print("device:", device)

    train_lines = load_lines(args.train)
    eval_lines  = load_lines(args.eval)

    vocab, stoi, itos = build_vocab(train_lines)
    pad_id = stoi["<PAD>"]

    train_seqs = make_dataset(train_lines, stoi, args.max_len, args.limit_train)
    eval_seqs  = make_dataset(eval_lines,  stoi, args.max_len, args.limit_eval)

    print(f"Seq2Seq DAE | Vocab: {len(vocab)} | Train seq: {len(train_seqs)} | Eval seq: {len(eval_seqs)}")
    print(f"emb={args.emb} hidden={args.hidden} layers={args.layers} dropout={args.dropout} drop_p={args.drop_p} rep_p={args.rep_p}")

    model = Seq2SeqDAE(len(vocab), emb=args.emb, hidden=args.hidden, layers=args.layers, dropout=args.dropout, pad_id=pad_id).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")

    best_eval = float("inf")

    for ep in range(1, args.epochs+1):
        model.train()
        start = time.time()
        total_loss, total_tok = 0.0, 0

        for x_clean, lengths in batchify(train_seqs, args.batch, pad_id, device):
            x_noised, nlen = make_noised_batch(x_clean, lengths, stoi, args.drop_p, args.rep_p, len(vocab), device)
            h = model.enc(x_noised, nlen)
            y_inp, y_tgt = teacher_forcing_inputs_targets(x_clean, lengths, pad_id)

            opt.zero_grad()
            logits = model.dec(y_inp, h)
            B,T,V = logits.shape
            loss = loss_fn(logits.reshape(B*T, V), y_tgt.reshape(B*T))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()
            total_tok += (y_tgt != pad_id).sum().item()

        train_nll = total_loss / max(1, total_tok)
        train_ppl = math.exp(min(50.0, train_nll))
        eval_nll, eval_ppl = evaluate(model, eval_seqs, args.batch, pad_id, stoi, args.drop_p, args.rep_p, len(vocab), device)

        print(f"Epoch {ep}/{args.epochs} | train_ppl={train_ppl:.2f} | eval_ppl={eval_ppl:.2f} | {int(time.time()-start)}s")

        if eval_nll < best_eval:
            best_eval = eval_nll
            torch.save({"state": model.state_dict(), "args": vars(args)}, "best_dae_seq2seq.pt")

    print("Saved best model to best_dae_seq2seq.pt")

if __name__ == "__main__":
    main()
