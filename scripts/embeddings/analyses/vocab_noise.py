#!/usr/bin/env python3
"""
Why does one run's cosine drop to 0.1 for samples the other calls identical?

The widest disagreements on the scatter's top arm pair a clean new string
('agricultural soil') against Dany strings like 'Control_T0_a' or
'no treatment in 2000'. If Dany's vocabulary carries raw experiment labels,
a low Dany cosine is not preserved signal - it is noise, and the collapse is
a cleanup rather than a loss.

This measures how much of each run's sub-biome vocabulary is non-descriptive,
over the whole overlap, and shows what the biggest new strings absorb.

    MAP_ROOT=... python3 scripts/embeddings/analyses/vocab_noise.py
"""
import os, re, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os as _os, sys as _sys  # noqa: E401
_EMBEDDINGS_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # scripts/embeddings
_sys.path.insert(0, _EMBEDDINGS_DIR)
from embed_subbiomes_keywords import clean_text

ROOT = os.path.expanduser(os.environ.get("MAP_ROOT", "~/MicrobeAtlasProject"))
OLD, NEW = f"{ROOT}/sidequest", f"{ROOT}/sidequest/latest"

# a sub-biome string that is really an experiment / specimen code, not a habitat
NOISE = [("has a digit", re.compile(r"\d")),
         ("has _ or /", re.compile(r"[_/]")),
         ("all caps token", re.compile(r"\b[A-Z]{3,}\b")),
         ("single token, not a word", re.compile(r"^\S+$"))]


def stream(path):
    for line in open(path, encoding="utf-8", errors="replace"):
        sid, _, raw = line.rstrip("\n").partition("\t")
        if sid and raw.strip():
            yield sid, clean_text(raw, False)


def report(name, texts):
    vocab = Counter(texts.values())
    n_types, n_tokens = len(vocab), sum(vocab.values())
    print(f"\n  {name}: {n_tokens:,} samples, {n_types:,} distinct strings "
          f"({n_tokens / n_types:.0f} samples per string)")
    for label, rx in NOISE:
        types = [t for t in vocab if rx.search(t)]
        tokens = sum(vocab[t] for t in types)
        print(f"    {label:<24} {len(types):>7,} strings ({len(types)/n_types:5.1%})"
              f"   {tokens:>9,} samples ({tokens/n_tokens:5.1%})")
    any_rx = [t for t in vocab if any(rx.search(t) for _, rx in NOISE)]
    print(f"    {'any of the above':<24} {len(any_rx):>7,} strings ({len(any_rx)/n_types:5.1%})"
          f"   {sum(vocab[t] for t in any_rx):>9,} samples "
          f"({sum(vocab[t] for t in any_rx)/n_tokens:5.1%})")
    return vocab, set(any_rx)


def main():
    old = dict(stream(f"{OLD}/GPT_sub_biomes.txt"))
    new = dict(stream(f"{NEW}/GPT_sub_biomes.txt"))
    both = sorted(set(old) & set(new))
    print(f"overlap: {len(both):,} samples")
    old = {s: old[s] for s in both}
    new = {s: new[s] for s in both}

    print("\n" + "=" * 74 + "\nvocabulary noise, same samples in both runs\n" + "=" * 74)
    _, old_noise = report("Dany (GPT-3.5)", old)
    _, new_noise = report("new (GPT-5)", new)

    # what do the biggest new strings absorb, and how much of it is noise?
    absorbed = defaultdict(Counter)
    for s in both:
        absorbed[new[s]][old[s]] += 1
    print("\n" + "=" * 74 + "\nwhat the largest new strings absorb\n" + "=" * 74)
    print(f"  {'new string':<26} {'samples':>9} {'distinct old':>13} {'old noisy':>10}  examples of absorbed")
    for text, c in sorted(absorbed.items(), key=lambda kv: -sum(kv[1].values()))[:10]:
        noisy = sum(v for t, v in c.items() if t in old_noise)
        tot = sum(c.values())
        ex = ", ".join(repr(t[:24]) for t, _ in c.most_common(20)[3:6])
        print(f"  {text[:25]:<26} {tot:>9,} {len(c):>13,} {noisy/tot:>9.1%}  {ex}")

    moved = [s for s in both if old[s] in old_noise and new[s] not in new_noise]
    back = [s for s in both if new[s] in new_noise and old[s] not in old_noise]
    print(f"\n  noisy in Dany -> clean in new: {len(moved):>9,} samples ({len(moved)/len(both):.1%})")
    print(f"  clean in Dany -> noisy in new: {len(back):>9,} samples ({len(back)/len(both):.1%})")


if __name__ == "__main__":
    main()
