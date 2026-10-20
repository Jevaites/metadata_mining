"""
Fixtures and input writers for the tests of 2_build_training_set.py (imported by its two test modules:
`from step2_helpers import step2, run_step2`, which makes the fixtures available there).

Run from the repository root (or from scripts/ontology_mapping):

    python -m pytest scripts/ontology_mapping/tests/test_2_build_training_set_unit.py \
                     scripts/ontology_mapping/tests/test_2_build_training_set_integration.py -v

The pipeline scripts start with a digit (2_build_training_set.py), so they cannot be imported with a
plain `import`; the `step2` fixture loads the file as a module instead. They also import `common`
from their own folder, which is put on sys.path here.

The file-writing helpers below produce the three inputs of step 2 in their real on-disk formats,
from small Python literals that the tests spell out:

    Metalog long tables  <domain>_all_long_<date>.tsv.gz   (sample_alias, metadata_item, curation_tier, value)
    MicrobeAtlas         sample.info(.gz)                   '>SAMPLE_ID' header, then 'key=value' lines
    term table           ontology_terms.tsv(.gz)            output of 1_build_term_index.py
"""

import gzip
import importlib.util
import os
import sys

import pandas as pd
import pytest

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ontology_mapping
sys.path.insert(0, SCRIPTS_DIR)


def load_script(filename):
    """Import a pipeline script whose file name is not a valid module name (e.g. '2_build_training_set.py')."""
    spec = importlib.util.spec_from_file_location("step_" + filename[:-3], os.path.join(SCRIPTS_DIR, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def step2():
    """The module 2_build_training_set.py (its functions, constants and main())."""
    return load_script("2_build_training_set.py")


# ----------------------------------------------------------------------------- input writers
def write_metalog_long(directory, filename, rows):
    """Write one Metalog long table. `rows` = [(sample_alias, metadata_item, value), ...].
    Columns as in the real files; curation_tier is 'core' (the tier of every environment_* row; step 2 ignores it)."""
    df = pd.DataFrame(rows, columns=["sample_alias", "metadata_item", "value"])
    df.insert(2, "curation_tier", "core")
    df.to_csv(os.path.join(directory, filename), sep="\t", index=False)  # .gz suffix -> gzip-compressed


def write_sample_info(path, records):
    """Write a MicrobeAtlas sample.info. `records` = {sample_id: ['key=value', ...]} (in file order), or the
    file's text as a str (written as is). A path ending in .gz is gzip-compressed, as the real file is."""
    text = records if isinstance(records, str) else "".join(f">{sample_id}\n" + "".join(line + "\n" for line in lines) for sample_id, lines in records.items())
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt") as handle:
        handle.write(text)


def write_terms(path, terms):
    """Write a term table as 1_build_term_index.py does. `terms` = [(term_id, label, obsolete), ...].
    Step 2 only reads term_id, label and obsolete; the other columns are there to match the real format."""
    df = pd.DataFrame(
        [{"ontology": term_id.split("_")[0], "term_id": term_id, "label": label, "synonyms": "",
          "definition": "", "parents": "", "obsolete": str(obsolete)} for term_id, label, obsolete in terms])
    df.to_csv(path, sep="\t", index=False)


# ----------------------------------------------------------------------------- run main() end to end
@pytest.fixture
def run_step2(step2, tmp_path, monkeypatch, capsys):
    """Write the inputs to a temporary folder, run step 2 exactly as the command line would, and return
    (output table, printed log).

        out, log = run_step2(metalog={"human_all_long_2026-09-20.tsv.gz": [(alias, item, value), ...]},
                             atlas={"SRS1": ["sample_env_material=soil"]},  # or sample.info text
                             terms=[("ENVO_00001998", "soil", False)],
                             max_chars=2000,                       # optional, like --max_chars
                             date="2026-09-20")                    # optional, like --date

    The output is read back with every cell as a string and empty cells as "" (as the next steps read it)."""
    def run(metalog, atlas, terms, max_chars=None, date=None, sample_info_name="sample.info"):
        metalog_dir = tmp_path / "metalog"
        metalog_dir.mkdir(exist_ok=True)
        for filename, rows in metalog.items():
            write_metalog_long(metalog_dir, filename, rows)
        write_sample_info(tmp_path / sample_info_name, atlas)
        write_terms(tmp_path / "ontology_terms.tsv", terms)
        output = tmp_path / "metalog_training_set.tsv.gz"

        argv = ["2_build_training_set.py",
                "--metalog_dir", str(metalog_dir),
                "--sample_info", str(tmp_path / sample_info_name),
                "--ontology_terms", str(tmp_path / "ontology_terms.tsv"),
                "--output", str(output)]
        if max_chars is not None:
            argv += ["--max_chars", str(max_chars)]
        if date is not None:
            argv += ["--date", date]
        monkeypatch.setattr(sys, "argv", argv)
        step2.main()

        out = pd.read_csv(output, sep="\t", dtype=str, keep_default_na=False)
        return out, capsys.readouterr().out
    return run
