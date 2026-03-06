import torch
import torch.nn as nn
import random
from pathlib import Path

from train_dae_seq2seq import Seq2SeqDAE, build_vocab, load_lines


def generate(model, stoi, itos, device, max_len=12):
    sos = stoi["<SOS>"]
    eos = stoi["<EOS>"]

    x = torch.tensor([[sos]], dtype=torch.long).to(device)

    out = []

    for _ in range(max_len):
        h = model.enc(x, torch.tensor([x.size(1)]).to(device))
        logits = model.dec(x, h)

        probs = torch.softmax(logits[0, -1], dim=0)
        idx = torch.multinomial(probs, 1).item()

        if idx == eos:
            break

        out.append(itos[idx])

        x = torch.cat([x, torch.tensor([[idx]]).to(device)], dim=1)

    return "".join(out)


def main():

    device = torch.device("cpu")

    train_lines = load_lines("train.txt")

    vocab, stoi, itos = build_vocab(train_lines)

    saved = torch.load("best_dae_seq2seq.pt", map_location=device)

    args = saved["args"]

    model = Seq2SeqDAE(
        len(vocab),
        emb=args["emb"],
        hidden=args["hidden"],
        layers=args["layers"],
        dropout=args["dropout"]
    ).to(device)

    model.load_state_dict(saved["state"])

    model.eval()

    Path("results").mkdir(exist_ok=True)

    with open("results/generated_dae.txt", "w") as f:
        for _ in range(20000):
            pw = generate(model, stoi, itos, device)
            f.write(pw + "\n")

    print("Generated 20000 passwords -> results/generated_dae.txt")


if __name__ == "__main__":
    main()
