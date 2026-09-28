#!/usr/bin/env python3
"""
Embed every (non-obsolete) ENVO / Uberon term with the same code, model and
dimension as the GPT keyword / sub-biome embeddings, so terms and samples live
in one vector space (needed for zero-shot retrieval on those embeddings).

Term text = map_samples_to_ontology.term_text(): 'label; synonym; synonym'.
Output has the same layout as the GPT_*_unique_embeddings files (texts, embeddings)
and is resumable: rerunning only embeds texts that are not in the file yet.

python scripts/embed_ontology_terms.py --dry_run      # count tokens / cost, no API call
python scripts/embed_ontology_terms.py                # ~19k texts, text-embedding-3-large @ 1024
"""

import argparse
import os
import sys

import pandas as pd
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embed_subbiomes_keywords import MAX_BATCH, PRICE_PER_1M_TOKENS, embed_unique, estimate_tokens
from map_samples_to_ontology import term_text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ontology_terms", default="~/MicrobeAtlasProject/ontology_terms.tsv.gz")
    parser.add_argument("--model", default="text-embedding-3-large", choices=sorted(PRICE_PER_1M_TOKENS))
    parser.add_argument("--embedding_dim", type=int, default=1024)
    parser.add_argument("--api_key_path", default="~/MicrobeAtlasProject/my_api_key_embeddings")
    parser.add_argument("--output_dir", default="~/MicrobeAtlasProject/ontology_mapping")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    terms = pd.read_csv(os.path.expanduser(args.ontology_terms), sep="\t", keep_default_na=False)
    terms = terms[terms["obsolete"].astype(str) != "True"]
    texts = list(dict.fromkeys(term_text(terms)))  # distinct texts, stable order
    n_tokens, method = estimate_tokens(texts, args.model)
    print(f"{len(terms)} terms ({terms['ontology'].value_counts().to_dict()}), {len(texts)} distinct texts, "
          f"{n_tokens:,} tokens ({method}) = ${n_tokens / 1e6 * PRICE_PER_1M_TOKENS[args.model]:.3f}")
    if args.dry_run:
        return

    out_dir = os.path.expanduser(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ontology_terms_unique_embeddings__{args.model}__dim{args.embedding_dim}.h5")
    client = OpenAI(api_key=open(os.path.expanduser(args.api_key_path)).read().strip(), max_retries=8)
    embed_unique("ontology_terms", texts, out_path, client, args.model, args.embedding_dim, MAX_BATCH)


if __name__ == "__main__":
    main()
