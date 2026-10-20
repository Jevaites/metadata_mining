"""Toy data, file writers and a fake OpenAI Batch API for the tests of 7_rerank_atlas.py
(imported by test_7_rerank_atlas_unit.py and test_7_rerank_atlas_integration.py).

Run (from the repository root or from scripts/ontology_mapping):
    python -m pytest scripts/ontology_mapping/tests/test_7_rerank_atlas_unit.py \
                     scripts/ontology_mapping/tests/test_7_rerank_atlas_integration.py -v

Nothing here calls the network: the OpenAI client is replaced by FakeOpenAI, and the "LLM" is
keyword_llm(), a rule that reads the prompt and picks an option by its label.

------------------------------------------------------------------------------- the toy ontology
Plain names as term ids, so a candidate list reads "marine:0.5;freshwater:0.25;aquatic:0.25".

  biome slot                        feature slot      material slot
  aquatic        aquatic biome      sea               sea_water
  ├── marine     marine biome       pond              fresh_water
  │   └── ocean  ocean biome        river
  └── freshwater freshwater biome
      └── lake   lake biome

ANC below is the ancestor map common.term_ancestors() computes from that table, written out by hand.
"""

import gzip
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent  # scripts/ontology_mapping
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))  # 7_rerank_atlas.py imports common, hierarchy, rerank_prompt

import rerank_prompt  # noqa: E402  (needs the sys.path line above)


def load_rerank_atlas():
    """Import 7_rerank_atlas.py as the module `rerank_atlas` (a module name cannot start with a digit)."""
    if "rerank_atlas" not in sys.modules:
        spec = importlib.util.spec_from_file_location("rerank_atlas", SCRIPTS_DIR / "7_rerank_atlas.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["rerank_atlas"] = module
        spec.loader.exec_module(module)
    return sys.modules["rerank_atlas"]


# ============================================================================= toy ontology
TERMS = [
    # term_id      label               parent
    ("aquatic",    "aquatic biome",    ""),
    ("marine",     "marine biome",     "aquatic"),
    ("ocean",      "ocean biome",      "marine"),
    ("freshwater", "freshwater biome", "aquatic"),
    ("lake",       "lake biome",       "freshwater"),
    ("sea",        "sea",              ""),
    ("pond",       "pond",             ""),
    ("river",      "river",            ""),
    ("sea_water",  "sea water",        ""),
    ("fresh_water", "fresh water",     ""),
]
LABEL = {t: label for t, label, _ in TERMS}
ANC = {"aquatic": set(), "marine": {"aquatic"}, "ocean": {"marine", "aquatic"},
       "freshwater": {"aquatic"}, "lake": {"freshwater", "aquatic"},
       "sea": set(), "pond": set(), "river": set(), "sea_water": set(), "fresh_water": set()}


def write_terms(p):
    """The toy ontology as 1_build_term_index.py writes it (ontology_terms.tsv)."""
    pd.DataFrame([{"ontology": "TOY", "term_id": t, "label": label, "synonyms": "", "definition": "",
                   "parents": parent, "obsolete": "False"} for t, label, parent in TERMS]
                 ).to_csv(p, sep="\t", index=False)
    return str(p)


# ============================================================================= file writers
def write_jsonl(p, records):
    with open(p, "w") as handle:
        for r in records:
            handle.write(json.dumps(r) + "\n")
    return str(p)


def read_jsonl(p):
    with open(p) as handle:
        return [json.loads(line) for line in handle]


def write_tsv(p, rows):
    """rows: list of dicts -> TSV ('.gz' names are gzipped by pandas)."""
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return str(p)


def read_tsv(p):
    """As common.read_tsv: every cell a string, empty cells "" (never NaN)."""
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)


def write_vectors_npz(p, vectors):
    """{sample_id: vector} -> the .npz of 3_extract_sample_embeddings.py (common.load_npz reads it)."""
    ids = list(vectors)
    np.savez(p, sample_ids=np.array(ids), vectors=np.array([vectors[s] for s in ids], dtype=float),
             index=np.arange(len(ids)))
    return str(p)


def write_h5(p, rows):
    """Rows of the atlas embedding file (dataset 'embeddings'; row numbers are index.npz's kw_rows / sb_rows)."""
    with h5py.File(p, "w") as handle:
        handle.create_dataset("embeddings", data=np.array(rows, dtype=np.float32))
    return str(p)


def write_sample_info(p, records):
    """{sample_id: ['key=value', ...]} -> MicrobeAtlas sample.info.gz, laid out as the real file: an empty first
    line, then per record a '>ID' header, its key=value lines and an empty line."""
    with gzip.open(p, "wt") as handle:
        handle.write("\n")
        for sample_id, lines in records.items():
            handle.write(f">{sample_id}\n" + "".join(line + "\n" for line in lines) + "\n")
    return str(p)


# ============================================================================= a fake LLM
def option_letters(messages):
    """{option label: letter} of a reranker prompt (rerank_prompt.prompt), 'none of these' left out.
    Example: the lines 'A. freshwater biome', 'B. marine biome', 'C. none of these terms fits'
    -> {'freshwater biome': 'A', 'marine biome': 'B'}."""
    user = messages[-1]["content"]
    return {label: letter for letter, label in re.findall(r"^([A-Z])\. (.+)$", user, re.M)
            if label != "none of these terms fits"}


def keyword_llm(rules, p):
    """A fake LLM. rules: [(keyword, label)]. It reads the sample metadata in the prompt; the first rule whose
    keyword appears there decides the label it answers, with probability p; 1 - p is spread evenly over the other
    options ('none of these' gets 0). No rule matches -> None (the request fails).
    -> llm(messages) = {letter: probability} or None."""
    def llm(messages):
        user = messages[-1]["content"]
        metadata = user.split("Sample metadata:\n", 1)[1].split("\n", 1)[0]
        label = next((label for keyword, label in rules if keyword in metadata), None)
        letters = option_letters(messages)
        if label is None or label not in letters:
            return None
        others = [letter for lab, letter in letters.items() if lab != label]
        return {letters[label]: p, **{letter: (1 - p) / len(others) for letter in others}}
    return llm


def pilot_response(o, llm, model="gpt-4.1-mini"):
    """What experiments/rerank_pilot.py run writes for one pilot sample: the answer letter and the
    log-probabilities of the top letters (the input of 7_rerank_atlas.py fit --responses)."""
    probs = llm(rerank_prompt.prompt(o, "logprobs", "v2"))
    if probs is None:
        return {"slot": o["slot"], "sample_id": o["sample_id"], "model": model, "content": "", "top_logprobs_raw": None}
    return {"slot": o["slot"], "sample_id": o["sample_id"], "model": model, "content": max(probs, key=probs.get),
            "top_logprobs_raw": [[letter, math.log(q)] for letter, q in probs.items()]}


# ============================================================================= a fake OpenAI client
class FakeOpenAI:
    """A stand-in for openai.OpenAI() with only the Batch API calls that submit() and collect() make:

        client.files.create(file=..., purpose="batch")   -> .id          submit: upload the request lines
        client.batches.create(input_file_id=..., ...)     -> .id          submit: start a batch
        client.batches.retrieve(batch_id)                 -> .status, .request_counts, .errors,
                                                             .output_file_id, .error_file_id     collect
        client.files.content(file_id).text                -> JSONL of results                     collect

    A batch finishes as soon as it is retrieved, unless its id is in `still_running` (status
    "in_progress"). Each request is answered by `llm(body["messages"])`; None = the request fails with
    HTTP 500. A custom_id in `expired` was never run (the 24 h window ran out): its error-file line has
    "response": null. Result lines have the real Batch API layout (id, custom_id, response {status_code,
    request_id, body}, error), with only the body fields collect() reads plus usage.
    `sent` lists, per batch, the request lines that were uploaded (custom_id, body)."""

    def __init__(self, llm):
        self.llm = llm
        self.sent = []
        self.still_running = set()
        self.expired = set()
        self._files, self._batches = {}, {}
        self.files = SimpleNamespace(create=self._upload, content=self._content)
        self.batches = SimpleNamespace(create=self._create_batch, retrieve=self._retrieve)

    def _upload(self, file, purpose):
        file_id = f"file-{len(self._files)}"
        self._files[file_id] = file.read().decode()
        return SimpleNamespace(id=file_id)

    def _content(self, file_id):
        return SimpleNamespace(text=self._files[file_id])

    def _create_batch(self, input_file_id, endpoint, completion_window, metadata):
        batch_id = f"batch-{len(self._batches)}"
        lines = [json.loads(line) for line in self._files[input_file_id].splitlines()]
        self.sent.append(lines)
        self._batches[batch_id] = lines
        return SimpleNamespace(id=batch_id)

    def _retrieve(self, batch_id):
        lines = self._batches[batch_id]
        if batch_id in self.still_running:
            return SimpleNamespace(id=batch_id, status="in_progress", errors=None, output_file_id=None, error_file_id=None,
                                   request_counts=SimpleNamespace(completed=0, failed=0, total=len(lines)))
        ok, failed = [], []
        for n, line in enumerate(lines):
            body, head = line["body"], {"id": f"batch_req_{batch_id}_{n}", "custom_id": line["custom_id"]}
            if line["custom_id"] in self.expired:
                failed.append({**head, "response": None,
                               "error": {"code": "batch_expired", "message": "This request could not be executed before the completion window expired."}})
                continue
            probs = self.llm(body["messages"])
            if probs is None:
                failed.append({**head, "response": {"status_code": 500, "request_id": f"req_{n}", "body": {
                    "error": {"message": "The server had an error while processing your request.", "type": "server_error"}}},
                    "error": None})
                continue
            best = max(probs, key=probs.get)
            top = [{"token": letter, "logprob": math.log(q)} for letter, q in sorted(probs.items(), key=lambda x: -x[1])]
            ok.append({**head, "response": {"status_code": 200, "request_id": f"req_{n}", "body": {
                "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": best},
                             "logprobs": {"content": [{"token": best, "logprob": math.log(probs[best]), "top_logprobs": top}]},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 1, "total_tokens": 1001}}}, "error": None})
        out_id = self._store(ok)
        err_id = self._store(failed)
        return SimpleNamespace(id=batch_id, status="completed", errors=None, output_file_id=out_id, error_file_id=err_id,
                               request_counts=SimpleNamespace(completed=len(ok), failed=len(failed), total=len(lines)))

    def _store(self, results):
        if not results:
            return None
        file_id = f"file-{len(self._files)}"
        self._files[file_id] = "".join(json.dumps(r) + "\n" for r in results)
        return file_id


def output_file_client(lines):
    """A Batch API client with one finished batch whose output file holds `lines` (dicts written as given),
    for collect() on hand-written result lines. -> (client, batches.json content naming that batch)."""
    text = "".join(json.dumps(line) + "\n" for line in lines)
    batch = SimpleNamespace(id="batch-real", status="completed", errors=None, output_file_id="file-out", error_file_id=None,
                            request_counts=SimpleNamespace(completed=len(lines), failed=0, total=len(lines)))
    client = SimpleNamespace(batches=SimpleNamespace(retrieve=lambda batch_id: batch),
                             files=SimpleNamespace(content=lambda file_id: SimpleNamespace(text=text)))
    state = {"batches": [{"batch_id": "batch-real", "model": "gpt-4.1-mini",
                          "requests": [line["custom_id"] for line in lines], "collected": False}]}
    return client, state


# ============================================================================= mutants (test_7_rerank_atlas_mutations.py)
def load_mutant(old, new, name="mutant"):
    """7_rerank_atlas.py with one deliberate bug: the source with `old` replaced by `new`, loaded as a separate
    module (the real rerank_atlas module is untouched). `old` must occur exactly once in today's source, so a
    mutant whose line has since been rewritten fails loudly instead of silently testing the unchanged code."""
    import types
    source = (SCRIPTS_DIR / "7_rerank_atlas.py").read_text()
    assert source.count(old) == 1, f"mutant {name!r}: {old!r} occurs {source.count(old)} times in 7_rerank_atlas.py"
    module = types.ModuleType(f"rerank_atlas_{name}")
    module.__file__ = str(SCRIPTS_DIR / "7_rerank_atlas.py")
    exec(compile(source.replace(old, new), f"7_rerank_atlas.py ({name})", "exec"), module.__dict__)
    return module
