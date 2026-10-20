"""
Fakes and file helpers for the tests of 4_embed_terms.py (imported by its integration and mutation tests:
`from step4_helpers import FakeEmbeddingClient, step4, ...`; importing `step4` makes the fixture available).

Kept out of conftest.py on purpose: `from conftest import ...` is ambiguous when this folder and
scripts/embeddings/tests are collected in the same pytest run (both have a conftest.py).

Nothing here calls the OpenAI API: `FakeEmbeddingClient` replaces `openai.OpenAI()` and returns
hand-made, human-readable vectors (see `concept_vector`).
"""


import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

STEP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STEP_DIR))
sys.path.insert(0, str(STEP_DIR.parent / "embeddings"))


def import_script(filename, module_name):
    """Import a pipeline script whose file name is not a valid module name (e.g. '4_embed_terms.py')."""
    spec = importlib.util.spec_from_file_location(module_name, STEP_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def step4():
    """The 4_embed_terms.py module (its main() and the names it imports)."""
    return import_script("4_embed_terms.py", "step4_embed_terms")


CONCEPTS = {
    "water": {"water", "sea", "marine", "lake", "river"},
    "soil": {"soil", "earth", "dirt"},
    "gut": {"gut", "intestine", "feces", "fecal", "stool"},
}


def concept_vector(text):
    """Fake embedding: [#water words, #soil words, #gut words] in `text`."""
    words = re.findall(r"[a-z]+", text.lower())
    return [float(sum(w in vocabulary for w in words)) for vocabulary in CONCEPTS.values()]


class FakeEmbeddingClient:
    """Stands in for openai.OpenAI(): supports `client.embeddings.create(input=[...], model=..., dimensions=...)`.

    embed:        text -> list of floats (default: concept_vector, 3 dimensions)
    fail_on_call: raise RuntimeError on this call number (1 = first call), to simulate an API outage
    calls:        one dict per API call, e.g. {"input": ["soil; "], "model": "...", "dimensions": 3}
    """

    def __init__(self, embed=concept_vector, fail_on_call=None):
        self.embed = embed
        self.fail_on_call = fail_on_call
        self.calls = []
        self.constructed_with = None
        self.embeddings = self

    def create(self, input, model, **kwargs):
        self.calls.append({"input": list(input), "model": model, **kwargs})
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("simulated API outage")
        return SimpleNamespace(data=[SimpleNamespace(embedding=self.embed(t)) for t in input])

    @property
    def embedded_texts(self):
        """Every text sent to the API, in order, over all calls."""
        return [t for call in self.calls for t in call["input"]]


def install_fake_openai(monkeypatch, module, client):
    """Make `OpenAI(api_key=..., max_retries=...)` inside `module` return `client`."""
    def fake_openai(**kwargs):
        client.constructed_with = kwargs
        return client
    monkeypatch.setattr(module, "OpenAI", fake_openai)


TERM_COLUMNS = ["ontology", "term_id", "label", "synonyms", "definition", "parents", "obsolete"]


def write_term_table(path, rows):
    """Write a term table like 1_build_term_index.py does (gzipped if `path` ends with .gz).

    rows: (ontology, term_id, label, synonyms, obsolete) tuples; synonyms '||'-joined, obsolete "True"/"False".
    `definition` and `parents` are left empty: step 4 does not use them."""
    table = pd.DataFrame([(o, t, l, s, "", "", obs) for o, t, l, s, obs in rows], columns=TERM_COLUMNS)
    table.to_csv(path, sep="\t", index=False)
    return path


def read_h5(path):
    """(texts, embeddings) of an embeddings .h5 file: the rows exactly as stored, no normalisation."""
    with h5py.File(path, "r") as f:
        texts = [t.decode() if isinstance(t, bytes) else t for t in f["texts"][:]]
        return texts, np.asarray(f["embeddings"][:])
