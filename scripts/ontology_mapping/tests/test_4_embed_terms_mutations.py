"""
Mutation tests for 4_embed_terms.py: deliberate one-line bugs, and proof that each one is caught.

Why: a test suite can pass because the code is right, or because no test looks at the line that matters.
Each test below takes one line of step 4 (or of the embed_unique code step 4 relies on), breaks it in a plausible way,
runs the same small scenario with the real code and with the broken copy (the "mutant"), and asserts:
    the real code gives the right answer, and the mutant gives a specific, documented wrong one.
So each test also documents why the line is written the way it is, and what happens without it.
The docstring of each test names the regular tests that fail on that mutant: integration tests of this
folder, and (prefixed embeddings/) tests of scripts/embeddings/tests/test_embed_subbiomes_keywords_*.py.
Mutants of common.py (obsolete terms kept, term vectors not normalised) are in test_common_mutations.py.

A mutant is the source file with one snippet replaced, loaded as a separate module by load_mutant();
the real modules are never touched. Where the mutated line lives decides how the mutant is plugged in:
    4_embed_terms.py              run the mutant's main()
    embed_subbiomes_keywords.py   step 4 calls the mutant's embed_unique   (monkeypatch step4.embed_unique)

MUTANTS is the catalogue: name -> (file, original snippet, broken replacement).
test_every_mutated_line_still_exists fails when one of those lines is rewritten: update the snippet,
then check that its test still makes sense.

Scenarios use the term table of test_4_embed_terms_integration.py: lake / soil / feces plus an obsolete
'old pond', with fake vectors = [water, soil, gut] word counts (see its docstring).
"""

import types
from pathlib import Path

import h5py
import pytest

import common
from step4_helpers import FakeEmbeddingClient, concept_vector, read_h5, write_term_table, step4  # noqa: F401 (step4 is a fixture)
from test_4_embed_terms_integration import FECES, LAKE, OUTPUT_NAME, SOIL, TERMS, run_step4

SCRIPTS = Path(__file__).resolve().parents[2]  # scripts/
STEP4 = "ontology_mapping/4_embed_terms.py"
ESK = "embeddings/embed_subbiomes_keywords.py"

MUTANTS = {
    # 4_embed_terms.py
    "dry_run_does_not_stop":       (STEP4, "    if args.dry_run:\n        return\n", ""),
    "texts_not_deduplicated":      (STEP4, 'list(dict.fromkeys(terms["text"]))', 'list(terms["text"])'),
    "api_key_not_stripped":        (STEP4, ".read().strip()", ".read()"),
    "empty_table_accepted":        (STEP4, "    if not texts:", "    if False:"),          # the 2026-10-19 fix
    # embed_subbiomes_keywords.py: embed_unique and the .h5 helpers
    "resume_forgotten":            (ESK, "todo = [t for t in texts if t not in done]", "todo = list(texts)"),
    "no_repair_before_resume":     (ESK, '    h5_align(out_path, ("texts", "embeddings"))\n', ""),
    "missing_column_ignored":      (ESK, "if name in f else 0 for name in sorted(set(columns) | set(f))}", "for name in f}"),
    "dimension_never_sent":        (ESK, 'kwargs["dimensions"] = dim', "pass"),
}


def load_mutant(name):
    """The file of MUTANTS[name] with its snippet replaced, loaded as a new module (the real one is untouched).
    The snippet must occur exactly once, so a line that has since been rewritten fails loudly instead of
    silently testing unchanged code."""
    filename, old, new = MUTANTS[name]
    source = (SCRIPTS / filename).read_text()
    assert source.count(old) == 1, f"mutant {name!r}: the snippet occurs {source.count(old)} times in {filename}"
    module = types.ModuleType(f"mutant_{name}")
    module.__file__ = str(SCRIPTS / filename)  # 4_embed_terms.py uses it to find scripts/embeddings
    exec(compile(source.replace(old, new), f"{filename} (mutant {name})", "exec"), module.__dict__)
    return module


def make_workdir(path, terms=TERMS):
    """Same layout as the `workdir` fixture of the integration tests; one per run, so real and mutant
    runs never share files."""
    path.mkdir()
    write_term_table(path / "ontology_terms.tsv.gz", terms)
    (path / "api_key").write_text("  sk-test-key\n")
    return path


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_every_mutated_line_still_exists(name):
    load_mutant(name)


# ============================================================================= 4_embed_terms.py
def test_dry_run_must_stop_before_any_api_call(step4, monkeypatch, tmp_path):
    """Caught by: test_dry_run_reports_the_cost_and_touches_nothing."""
    real, broken = FakeEmbeddingClient(), FakeEmbeddingClient()

    run_step4(step4, monkeypatch, make_workdir(tmp_path / "real"), real, "--dry_run")
    run_step4(load_mutant("dry_run_does_not_stop"), monkeypatch, make_workdir(tmp_path / "mutant"), broken, "--dry_run")

    assert real.calls == []
    assert broken.embedded_texts == [LAKE, SOIL, FECES]           # a "dry" run that pays for every embedding
    assert (tmp_path / "mutant" / "out" / OUTPUT_NAME).exists()


def test_identical_texts_must_be_sent_once(step4, monkeypatch, tmp_path):
    """Caught by: test_identical_texts_in_two_ontologies_are_embedded_once."""
    terms = TERMS + [("ENVO", "ENVO_00002003", "feces", "stool||fecal matter", "False")]  # same text as UBERON feces
    real, broken = FakeEmbeddingClient(), FakeEmbeddingClient()

    run_step4(step4, monkeypatch, make_workdir(tmp_path / "real", terms), real)
    out = run_step4(load_mutant("texts_not_deduplicated"), monkeypatch, make_workdir(tmp_path / "mutant", terms), broken)

    assert real.embedded_texts == [LAKE, SOIL, FECES]
    assert broken.embedded_texts == [LAKE, SOIL, FECES, FECES]    # paid twice...
    assert read_h5(out)[0] == [LAKE, SOIL, FECES, FECES]          # ...and stored twice


def test_api_key_must_be_stripped(step4, monkeypatch, tmp_path):
    """Caught by: test_full_run_embeds_each_live_term_once. The key file holds '  sk-test-key\\n'."""
    real, broken = FakeEmbeddingClient(), FakeEmbeddingClient()

    run_step4(step4, monkeypatch, make_workdir(tmp_path / "real"), real)
    run_step4(load_mutant("api_key_not_stripped"), monkeypatch, make_workdir(tmp_path / "mutant"), broken)

    assert real.constructed_with["api_key"] == "sk-test-key"
    assert broken.constructed_with["api_key"] == "  sk-test-key\n"   # the real API answers 401 Unauthorized


def test_an_empty_term_table_must_be_refused(step4, monkeypatch, tmp_path):
    """Regression check for the 2026-10-19 fix.
    Caught by: test_a_table_with_only_obsolete_terms_is_refused_with_a_clear_message."""
    only_obsolete = [("ENVO", "ENVO_99999999", "old pond", "", "True")]

    with pytest.raises(SystemExit, match="has no non-obsolete terms"):
        run_step4(step4, monkeypatch, make_workdir(tmp_path / "real", only_obsolete), FakeEmbeddingClient())

    # mutant: the run "succeeds", writes nothing, and the next step fails on a missing file, with no hint why
    out = run_step4(load_mutant("empty_table_accepted"), monkeypatch, make_workdir(tmp_path / "mutant", only_obsolete),
                    FakeEmbeddingClient())
    assert not out.exists()
    with pytest.raises(FileNotFoundError):
        common.load_term_vectors(out, [])


# ============================================================================= embed_subbiomes_keywords.py
def test_resume_must_skip_texts_already_in_the_file(step4, monkeypatch, tmp_path):
    """Caught by: test_rerunning_a_finished_run_is_free, test_ontology_update_embeds_only_new_or_changed_texts,
    test_api_failure_mid_run_keeps_finished_batches_and_the_rerun_completes,
    embeddings/TestEmbedUnique::test_resume_skips_texts_already_in_file."""
    real_dir, mutant_dir = make_workdir(tmp_path / "real"), make_workdir(tmp_path / "mutant")
    run_step4(step4, monkeypatch, real_dir, FakeEmbeddingClient())
    real_rerun = FakeEmbeddingClient()
    run_step4(step4, monkeypatch, real_dir, real_rerun)

    monkeypatch.setattr(step4, "embed_unique", load_mutant("resume_forgotten").embed_unique)
    run_step4(step4, monkeypatch, mutant_dir, FakeEmbeddingClient())
    broken_rerun = FakeEmbeddingClient()
    out = run_step4(step4, monkeypatch, mutant_dir, broken_rerun)

    assert real_rerun.calls == []
    assert broken_rerun.embedded_texts == [LAKE, SOIL, FECES]     # every rerun pays for everything again
    assert read_h5(out)[0] == [LAKE, SOIL, FECES] * 2             # and the file grows by a full copy


def kill_while_writing(out, text):
    """Leave `out` as a kill inside h5_append would: `text` appended to `texts`, its vector never written."""
    with h5py.File(out, "a") as f:
        f["texts"].resize(f["texts"].shape[0] + 1, axis=0)
        f["texts"][-1] = text


def test_a_ragged_file_must_be_repaired_before_resuming(step4, monkeypatch, tmp_path):
    """Caught by: test_a_run_killed_while_writing_is_repaired_on_the_next_run,
    test_a_run_killed_during_its_very_first_write_starts_over,
    embeddings/TestResume::test_crash_during_first_write_of_unique_file_is_repaired."""
    with_river = TERMS + [("ENVO", "ENVO_00000022", "river", "", "False")]
    new_texts = [LAKE, SOIL, FECES, "river; "]

    def killed_run_then_rerun(workdir, rerun_client):
        out = run_step4(step4, monkeypatch, workdir, FakeEmbeddingClient())   # lake, soil, feces
        kill_while_writing(out, "river; ")                                   # 4 texts, 3 vectors
        write_term_table(workdir / "ontology_terms.tsv.gz", with_river)
        run_step4(step4, monkeypatch, workdir, rerun_client)
        return out

    real = FakeEmbeddingClient()
    out = killed_run_then_rerun(make_workdir(tmp_path / "real"), real)
    assert real.embedded_texts == ["river; "]
    assert common.load_term_vectors(out, new_texts).shape == (4, 3)

    monkeypatch.setattr(step4, "embed_unique", load_mutant("no_repair_before_resume").embed_unique)
    broken = FakeEmbeddingClient()
    out = killed_run_then_rerun(make_workdir(tmp_path / "mutant"), broken)
    assert broken.calls == []                    # 'river; ' is in `texts`, so it looks done
    with pytest.raises(IndexError):              # ...but row 3 has no vector: the next step crashes
        common.load_term_vectors(out, new_texts)


def test_a_column_that_was_never_created_must_count_as_empty(step4, monkeypatch, tmp_path):
    """A kill (or full disk) during the very first write: `texts` exists, `embeddings` was never created.
    Caught by: test_a_run_killed_during_its_very_first_write_starts_over,
               embeddings/TestH5Align::test_column_missing_after_crash_on_first_write."""
    def first_write_killed_then_rerun(workdir, rerun_client):
        out = workdir / "out" / OUTPUT_NAME
        out.parent.mkdir()
        with h5py.File(out, "w") as f:
            f.create_dataset("texts", data=[LAKE, SOIL, FECES], dtype=h5py.string_dtype(), maxshape=(None,))
        run_step4(step4, monkeypatch, workdir, rerun_client)
        return out

    real = FakeEmbeddingClient()
    out = first_write_killed_then_rerun(make_workdir(tmp_path / "real"), real)
    assert real.embedded_texts == [LAKE, SOIL, FECES]
    assert read_h5(out)[1].tolist() == [[2, 0, 0], [0, 1, 0], [0, 0, 3]]

    monkeypatch.setattr(step4, "embed_unique", load_mutant("missing_column_ignored").embed_unique)
    broken = FakeEmbeddingClient()
    out = first_write_killed_then_rerun(make_workdir(tmp_path / "mutant"), broken)
    assert broken.calls == []                    # only `texts` is looked at: the file seems complete
    with pytest.raises(KeyError):                # no `embeddings` column: the next step crashes
        common.load_term_vectors(out, [LAKE])


class DimensionAwareClient(FakeEmbeddingClient):
    """Behaves like the real API: `dimensions=d` gives d numbers per text, no `dimensions` gives the model's
    native 3072 (concept_vector padded with zeros)."""

    def create(self, input, model, **kwargs):
        size = kwargs.get("dimensions", 3072)
        self.embed = lambda text: (concept_vector(text) + [0.0] * size)[:size]
        return super().create(input, model, **kwargs)


def test_a_reduced_dimension_must_be_requested(step4, monkeypatch, tmp_path):
    """Caught by: test_full_run_embeds_each_live_term_once,
    embeddings/TestEmbedUnique::test_dimensions_sent_when_smaller_than_native."""
    out = run_step4(step4, monkeypatch, make_workdir(tmp_path / "real"), DimensionAwareClient())
    assert read_h5(out)[1].shape == (3, 3)

    # mutant: the API returns full-size vectors; the shape check in embed_unique stops the run before writing
    monkeypatch.setattr(step4, "embed_unique", load_mutant("dimension_never_sent").embed_unique)
    with pytest.raises(AssertionError, match=r"got \(3, 3072\), expected dim 3"):
        run_step4(step4, monkeypatch, make_workdir(tmp_path / "mutant"), DimensionAwareClient())
