import torch
from pathlib import Path
from train_char_rnn import CharLM, generate
import argparse

def load_rnn_model(path, device):
    saved = torch.load(path, map_location=device)

    args = saved["args"]
    stoi = saved["stoi"]
    itos = saved["itos"]

    model = CharLM(
        vocab_size=len(stoi),
        emb=args["emb"],
        hidden=args["hidden"],
        layers=args["layers"],
        dropout=0.1,
        cell=args["cell"]
    ).to(device)

    model.load_state_dict(saved["state"])
    model.eval()

    return model, stoi, itos


def generate_and_save(model_path, out_path, n, device):
    model, stoi, itos = load_rnn_model(model_path, device)

    print(f"Generating from {model_path}")

    with open(out_path, "w", encoding="utf-8") as f:
        for _ in range(n):
            pw = generate(model, stoi, itos, device)
            f.write(pw + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    args = parser.parse_args()

    device = torch.device("cpu")

    results = Path("results")
    results.mkdir(exist_ok=True)

    generate_and_save("model_rnn.pt",  results/"generated_rnn.txt",  args.n, device)
    generate_and_save("model_lstm.pt", results/"generated_lstm.txt", args.n, device)
    generate_and_save("model_gru.pt",  results/"generated_gru.txt",  args.n, device)

    print("Done generating passwords!")


if __name__ == "__main__":
    main()
