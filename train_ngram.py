import argparse, math, random
from collections import Counter, defaultdict
from pathlib import Path

def load_lines(p):
    return [l.strip() for l in Path(p).read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]

def add_tokens(s, n):
    # on met (n-1) symboles de début + symbole fin
    return ("^" * (n - 1)) + s + "$"

def train_ngram(lines, n, max_len=None, limit=None):
    counts = defaultdict(Counter)
    total_ctx = Counter()

    if limit is None:
        limit = len(lines)

    used = 0
    for s in lines[:limit]:
        if max_len is not None and len(s) > max_len:
            continue
        s = add_tokens(s, n)
        for i in range(len(s) - n + 1):
            ctx = s[i:i + n - 1]
            nxt = s[i + n - 1]
            counts[ctx][nxt] += 1
            total_ctx[ctx] += 1
        used += 1

    return counts, total_ctx, used

def eval_nll_ppl(lines, counts, total_ctx, n, alpha=0.1, max_len=None, limit=None):
    # vocab = tous les caractères vus + '$' (fin)
    vocab = set()
    for ctx in counts:
        vocab.update(counts[ctx].keys())
    vocab.add("$")
    V = len(vocab)

    if limit is None:
        limit = len(lines)

    total_nll = 0.0
    total_tok = 0

    for s in lines[:limit]:
        if max_len is not None and len(s) > max_len:
            continue
        s = add_tokens(s, n)
        for i in range(len(s) - n + 1):
            ctx = s[i:i + n - 1]
            nxt = s[i + n - 1]
            c = counts.get(ctx, Counter())
            num = c.get(nxt, 0) + alpha
            den = total_ctx.get(ctx, 0) + alpha * V
            p = num / den
            total_nll += -math.log(p)
            total_tok += 1

    nll = total_nll / max(1, total_tok)
    ppl = math.exp(min(50.0, nll))
    return nll, ppl

def sample_next(counts, total_ctx, ctx, alpha, V, vocab_list):
    # distribution lissée
    c = counts.get(ctx, Counter())
    den = total_ctx.get(ctx, 0) + alpha * V

    # sampling simple : on fait une roulette sur les caractères connus + lissage
    # pour rester simple et rapide, on approxime :
    # - on échantillonne d'abord parmi les caractères vus
    # - si on tombe dans la masse de lissage, on choisit un caractère au hasard
    seen_mass = (sum(c.values()) + alpha * len(c)) / den
    if random.random() > seen_mass:
        return random.choice(vocab_list)

    # roulette parmi vus
    r = random.random()
    acc = 0.0
    for ch, cnt in c.items():
        acc += (cnt + alpha) / den
        if acc >= r:
            return ch
    # fallback
    return random.choice(vocab_list)

def generate_one(counts, total_ctx, n, max_len=20, alpha=0.1):
    vocab = set()
    for ctx in counts:
        vocab.update(counts[ctx].keys())
    vocab.add("$")
    vocab_list = list(vocab)
    V = len(vocab_list)

    ctx = "^" * (n - 1)
    out = []
    for _ in range(max_len):
        nxt = sample_next(counts, total_ctx, ctx, alpha, V, vocab_list)
        if nxt == "$":
            break
        out.append(nxt)
        ctx = (ctx + nxt)[- (n - 1):]
    return "".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.txt")
    ap.add_argument("--eval", default="eval.txt")
    ap.add_argument("--n", type=int, default=4, help="ordre du n-gram (ex: 3,4,5)")
    ap.add_argument("--alpha", type=float, default=0.1, help="lissage Laplace (0.1 ou 1.0)")
    ap.add_argument("--max_len", type=int, default=20, help="ignore/tronque au-delà (comme les NN)")
    ap.add_argument("--limit_train", type=int, default=999999999)
    ap.add_argument("--limit_eval", type=int, default=999999999)

    ap.add_argument("--mode", choices=["eval","gen"], default="eval")
    ap.add_argument("--out", default="10k.txt")
    ap.add_argument("--num", type=int, default=10000)
    args = ap.parse_args()

    train_lines = load_lines(args.train)
    eval_lines = load_lines(args.eval)

    counts, total_ctx, used_train = train_ngram(
        train_lines, args.n, max_len=args.max_len, limit=args.limit_train
    )
    nll, ppl = eval_nll_ppl(
        eval_lines, counts, total_ctx, args.n, alpha=args.alpha,
        max_len=args.max_len, limit=args.limit_eval
    )

    print(f"N-gram n={args.n} | train_used={used_train} | eval_ppl={ppl:.2f}")

    if args.mode == "gen":
        with open(args.out, "w", encoding="utf-8") as f:
            for _ in range(args.num):
                f.write(generate_one(counts, total_ctx, args.n, max_len=args.max_len, alpha=args.alpha) + "\n")
        print(f"Generated {args.num} passwords -> {args.out}")

if __name__ == "__main__":
    main()
