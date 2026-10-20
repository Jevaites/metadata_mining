"""
pytest setup shared by every test in this folder (pytest loads this file automatically).

Run from the repository root:
    pip install pytest h5py          # the rest (pandas, numpy, scikit-learn, openai) is in metadmin_env.yml
    pytest scripts/ontology_mapping/tests -v

Only sys.path is set here (the step scripts import `common`, `methods`, ... from their own folder, and
4_embed_terms.py imports embed_subbiomes_keywords from scripts/embeddings). Shared helpers live in
importable modules (step_loader.py, step2_helpers.py, step4_helpers.py, ...), never here, so that
`from conftest import ...` is never needed.
"""

import sys
from pathlib import Path

STEP_DIR = Path(__file__).resolve().parent.parent
for path in (STEP_DIR, STEP_DIR.parent / "embeddings"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
