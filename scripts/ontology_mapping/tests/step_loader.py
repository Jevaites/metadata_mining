"""Helpers for testing the numbered step scripts (1_build_term_index.py, ...).

1. Importing a step. A module name cannot start with a digit, so `import 1_build_term_index` is a
   syntax error. load_step() imports the file under an importable alias instead:
       bti = load_step("1_build_term_index.py", "build_term_index")
2. No network. The `fake_web` fixture replaces urllib.request.urlopen: "downloading" a URL returns the
   bytes the test put in fake_web.pages[url], and fake_web.requested lists the URLs that were asked for.
   Use it in a test module with `from step_loader import fake_web`.
"""

import importlib.util
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))  # steps import `common`, as when they are run from that folder


def load_step(filename, alias):
    """Import scripts/ontology_mapping/<filename> as the module <alias> (once; later calls reuse it)."""
    if alias not in sys.modules:
        spec = importlib.util.spec_from_file_location(alias, SCRIPTS_DIR / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
    return sys.modules[alias]


class FakeResponse:
    """What urlopen() returns: used as `with urlopen(url) as response: data = response.read()`."""

    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


@pytest.fixture
def fake_web(monkeypatch):
    web = SimpleNamespace(pages={}, requested=[])

    def urlopen(url):
        web.requested.append(url)
        return FakeResponse(web.pages[url])  # KeyError here = the code asked for a URL the test did not set up

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return web
