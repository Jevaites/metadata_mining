"""pytest setup shared by every test in this folder.

Run from anywhere:   python -m pytest scripts/embeddings/tests -v

* scripts/embeddings goes on sys.path, so tests can `import embed_subbiomes_keywords as esk`
  (helpers.py does the same, so it can also be imported on its own).
* The tests never talk to OpenAI: they hand the code a FakeOpenAI (helpers.py). If the `openai`
  package is not installed, a stub module is registered so that the script's top-level
  `from openai import OpenAI` still works.
* tiktoken is blocked in every test (see `no_tiktoken`): with it, estimate_tokens() would download
  a tokenizer file on first use. Blocked, the token estimate is len(text) // 4, which a test can
  compute by hand.
"""

import sys
import types
from pathlib import Path

import pytest

EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent
if str(EMBEDDINGS_DIR) not in sys.path:
    sys.path.insert(0, str(EMBEDDINGS_DIR))

try:
    import openai  # noqa: F401
except ImportError:  # only the name is needed; every test replaces it with a fake
    sys.modules["openai"] = types.SimpleNamespace(OpenAI=None)


@pytest.fixture(autouse=True)
def no_tiktoken(monkeypatch):
    """`import tiktoken` raises ImportError inside a test, so estimate_tokens() uses ~4 chars/token."""
    monkeypatch.setitem(sys.modules, "tiktoken", None)
