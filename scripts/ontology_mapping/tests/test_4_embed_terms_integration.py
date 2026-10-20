"""
Integration tests: run `python 4_embed_terms.py ...` in-process on a 4-row term table and check the
.h5 it writes, then read it back the way the later steps do (common.load_term_vectors).

Where step 4 sits in the pipeline:

    1_build_term_index.py  ->  ontology_terms.tsv.gz                     (one row per term)
    4_embed_terms.py       ->  ontology_terms_unique_embeddings__<model>__dim<dim>.h5
                                 texts       one per distinct term text ('label; synonym; ...')
                                 embeddings  one vector per text, same row order
    6_predict_atlas.py, experiments/*  ->  load_term_vectors(h5, terms["text"])  (lookup by text)

The OpenAI client is replaced by FakeEmbeddingClient, whose vectors are word counts on three axes
[water, soil, gut] (see step4_helpers.concept_vector), so every expected vector below can be checked by eye.

The term table used by most tests (TERMS):

    ontology  term_id         label     synonyms             obsolete  text embedded                 fake vector
    ENVO      ENVO_00000020   lake      freshwater lake      False     'lake; freshwater lake'       [2, 0, 0]
    ENVO      ENVO_00001998   soil      (none)               False     'soil; '                      [0, 1, 0]
    UBERON    UBERON_0001988  feces     stool||fecal matter  False     'feces; stool; fecal matter'  [0, 0, 3]
    ENVO      ENVO_99999999   old pond  (none)               True      -- obsolete, never embedded --
"""

import sys

import numpy as np
import pytest

from common import load_term_vectors, load_terms
from step4_helpers import FakeEmbeddingClient, concept_vector, install_fake_openai, read_h5, write_term_table, step4  # noqa: F401 (step4 is a fixture)

TERMS = [
    ("ENVO", "ENVO_00000020", "lake", "freshwater lake", "False"),
    ("ENVO", "ENVO_00001998", "soil", "", "False"),
    ("UBERON", "UBERON_0001988", "feces", "stool||fecal matter", "False"),
    ("ENVO", "ENVO_99999999", "old pond", "", "True"),
]
LAKE, SOIL, FECES = "lake; freshwater lake", "soil; ", "feces; stool; fecal matter"
OUTPUT_NAME = "ontology_terms_unique_embeddings__text-embedding-3-large__dim3.h5"


@pytest.fixture
def workdir(tmp_path):
    """A term table, an API key file and an (empty) output folder location, as on the real machine."""
    write_term_table(tmp_path / "ontology_terms.tsv.gz", TERMS)
    (tmp_path / "api_key").write_text("  sk-test-key\n")
    return tmp_path


def run_step4(step4, monkeypatch, workdir, client, *extra_args):
    """Run `python 4_embed_terms.py --ontology_terms ... --embedding_dim 3 <extra_args>`, with the
    OpenAI client replaced by `client`. Returns the path of the .h5 the run writes to (dim 3 unless
    extra_args override it)."""
    argv = ["4_embed_terms.py",
            "--ontology_terms", str(workdir / "ontology_terms.tsv.gz"),
            "--api_key_path", str(workdir / "api_key"),
            "--output_dir", str(workdir / "out"),
            "--embedding_dim", "3",
            *extra_args]                                   # later flags win in argparse
    monkeypatch.setattr(sys, "argv", argv)
    install_fake_openai(monkeypatch, step4, client)
    step4.main()
    return workdir / "out" / OUTPUT_NAME


# ============================================================================= the basic workflow
def test_dry_run_reports_the_cost_and_touches_nothing(step4, monkeypatch, workdir, capsys):
    monkeypatch.setitem(sys.modules, "tiktoken", None)    # deterministic estimate: ~4 characters per token
    client = FakeEmbeddingClient()

    run_step4(step4, monkeypatch, workdir, client, "--dry_run")

    # 3 live terms; characters: 21 ('lake; freshwater lake') + 6 ('soil; ') + 26 (feces...) = 53 -> 53 // 4 = 13
    assert capsys.readouterr().out.strip() == \
        "3 terms ({'ENVO': 2, 'UBERON': 1}), 3 distinct texts, 13 tokens (~4 chars/token) = $0.000"
    assert client.constructed_with is None                # no OpenAI client created
    assert not (workdir / "out").exists()                 # no output folder, no file


def test_full_run_embeds_each_live_term_once(step4, monkeypatch, workdir):
    client = FakeEmbeddingClient()

    out = run_step4(step4, monkeypatch, workdir, client)

    # one API call (3 texts < MAX_BATCH = 2048), with the key read from the file and stripped
    assert client.constructed_with == {"api_key": "sk-test-key", "max_retries": 8}
    assert client.calls == [{"input": [LAKE, SOIL, FECES], "model": "text-embedding-3-large", "dimensions": 3}]

    # the file: one row per text, in term-table order, raw (unnormalised) API vectors
    texts, vectors = read_h5(out)
    assert texts == [LAKE, SOIL, FECES]                   # 'old pond' (obsolete) is not there
    assert vectors.tolist() == [[2, 0, 0],
                                [0, 1, 0],
                                [0, 0, 3]]


def test_later_steps_read_the_vectors_back_by_text_and_match_samples_to_terms(step4, monkeypatch, workdir):
    out = run_step4(step4, monkeypatch, workdir, FakeEmbeddingClient())

    # what 6_predict_atlas.py does: one L2-normalised vector per term, looked up by its text
    terms = load_terms(workdir / "ontology_terms.tsv.gz")
    term_vectors = load_term_vectors(out, list(terms["text"]))
    assert term_vectors.tolist() == [[1, 0, 0],          # [2, 0, 0] / 2
                                     [0, 1, 0],
                                     [0, 0, 1]]          # [0, 0, 3] / 3

    # The point of step 4: sample texts embedded with the same model land next to the right term.
    for sample_text, expected_term in [("water from a lake", "ENVO_00000020"),
                                       ("garden soil", "ENVO_00001998"),
                                       ("human stool, gut", "UBERON_0001988")]:
        query = np.array(concept_vector(sample_text))
        nearest = terms["term_id"][int(np.argmax(term_vectors @ query))]
        assert nearest == expected_term, sample_text


def test_rerunning_a_finished_run_is_free(step4, monkeypatch, workdir):
    out = run_step4(step4, monkeypatch, workdir, FakeEmbeddingClient())
    before = read_h5(out)

    rerun = FakeEmbeddingClient()
    run_step4(step4, monkeypatch, workdir, rerun)

    assert rerun.calls == []                              # every text is already in the file
    after = read_h5(out)
    assert after[0] == before[0] and np.array_equal(after[1], before[1])


# ============================================================================= edge cases
def test_ontology_update_embeds_only_new_or_changed_texts(step4, monkeypatch, workdir):
    out = run_step4(step4, monkeypatch, workdir, FakeEmbeddingClient())

    # New ontology release: 'soil' gains the synonym 'earth' and a new term 'river' appears.
    write_term_table(workdir / "ontology_terms.tsv.gz", [
        ("ENVO", "ENVO_00000020", "lake", "freshwater lake", "False"),
        ("ENVO", "ENVO_00001998", "soil", "earth", "False"),          # text 'soil; ' -> 'soil; earth'
        ("UBERON", "UBERON_0001988", "feces", "stool||fecal matter", "False"),
        ("ENVO", "ENVO_00000022", "river", "", "False"),              # new
    ])
    new_texts = list(load_terms(workdir / "ontology_terms.tsv.gz")["text"])

    # Before step 4 is rerun, the later steps refuse the stale file and say what to do.
    with pytest.raises(SystemExit, match=r"2 term texts are not in .* re-run 4_embed_terms.py"):
        load_term_vectors(out, new_texts)

    rerun = FakeEmbeddingClient()
    run_step4(step4, monkeypatch, workdir, rerun)

    assert rerun.embedded_texts == ["soil; earth", "river; "]
    texts, _ = read_h5(out)
    assert texts == [LAKE, SOIL, FECES, "soil; earth", "river; "]   # the stale 'soil; ' row stays (harmless)
    assert load_term_vectors(out, new_texts).tolist() == [[1, 0, 0],   # lake
                                                          [0, 1, 0],   # 'soil; earth' = [0, 2, 0] / 2
                                                          [0, 0, 1],   # feces
                                                          [1, 0, 0]]   # river


def test_identical_texts_in_two_ontologies_are_embedded_once(step4, monkeypatch, workdir):
    write_term_table(workdir / "ontology_terms.tsv.gz", TERMS + [
        ("ENVO", "ENVO_00002003", "feces", "stool||fecal matter", "False"),   # same text as UBERON feces
    ])
    client = FakeEmbeddingClient()
    out = run_step4(step4, monkeypatch, workdir, client)

    assert client.embedded_texts == [LAKE, SOIL, FECES]   # 4 live terms, 3 distinct texts
    terms = load_terms(workdir / "ontology_terms.tsv.gz")
    vectors = load_term_vectors(out, list(terms["text"]))
    assert list(terms["term_id"]) == ["ENVO_00000020", "ENVO_00001998", "UBERON_0001988", "ENVO_00002003"]
    assert vectors[2].tolist() == vectors[3].tolist() == [0, 0, 1]     # both terms share the vector


def test_multi_line_synonyms_are_embedded_and_found_verbatim(step4, monkeypatch, workdir):
    # Real case (shortened): 2 FOODON terms have synonyms spanning lines, with quotes; the term table
    # stores them as quoted TSV fields. The .h5 key must be the exact text, '\n' and '"' included.
    chicory = ("FOODON", "FOODON_00001730", "chicory food product",
               'Blue daisy, coffeeweed.\n\nNote: "Cornflower" is Centaurea cyanus.', "False")
    write_term_table(workdir / "ontology_terms.tsv.gz", TERMS + [chicory])
    client = FakeEmbeddingClient()

    out = run_step4(step4, monkeypatch, workdir, client)

    text = 'chicory food product; Blue daisy, coffeeweed.\n\nNote: "Cornflower" is Centaurea cyanus.'
    assert client.embedded_texts == [LAKE, SOIL, FECES, text]
    assert read_h5(out)[0][3] == text
    terms = load_terms(workdir / "ontology_terms.tsv.gz")
    assert load_term_vectors(out, list(terms["text"])).shape == (4, 3)  # every live term found by its text


def test_api_failure_mid_run_keeps_finished_batches_and_the_rerun_completes(step4, monkeypatch, workdir):
    monkeypatch.setattr(step4, "MAX_BATCH", 1)            # one text per API call, to have several batches

    with pytest.raises(RuntimeError, match="simulated API outage"):
        run_step4(step4, monkeypatch, workdir, FakeEmbeddingClient(fail_on_call=2))
    out = workdir / "out" / OUTPUT_NAME
    assert read_h5(out)[0] == [LAKE]                      # batch 1 was written before batch 2 failed

    rerun = FakeEmbeddingClient()
    run_step4(step4, monkeypatch, workdir, rerun)

    assert rerun.embedded_texts == [SOIL, FECES]          # only what was missing
    texts, vectors = read_h5(out)
    assert texts == [LAKE, SOIL, FECES]                   # same file as an uninterrupted run
    assert vectors.tolist() == [[2, 0, 0], [0, 1, 0], [0, 0, 3]]


def test_a_run_killed_while_writing_is_repaired_on_the_next_run(step4, monkeypatch, workdir):
    out = run_step4(step4, monkeypatch, workdir, FakeEmbeddingClient())

    # Simulate a kill inside h5_append: 'river; ' got into `texts` but its vector never got into `embeddings`.
    import h5py
    with h5py.File(out, "a") as f:
        f["texts"].resize(4, axis=0)
        f["texts"][3] = "river; "
    write_term_table(workdir / "ontology_terms.tsv.gz", TERMS + [("ENVO", "ENVO_00000022", "river", "", "False")])

    rerun = FakeEmbeddingClient()
    run_step4(step4, monkeypatch, workdir, rerun)

    # Without the repair (h5_align) 'river; ' would count as done and have no vector.
    assert rerun.embedded_texts == ["river; "]
    texts, vectors = read_h5(out)
    assert texts == [LAKE, SOIL, FECES, "river; "]
    assert vectors.tolist() == [[2, 0, 0], [0, 1, 0], [0, 0, 3], [1, 0, 0]]


def test_a_run_killed_during_its_very_first_write_starts_over(step4, monkeypatch, workdir):
    # The first h5_append creates `texts`, then `embeddings`. A kill (or a full disk) in between leaves a
    # file with 3 texts and no `embeddings` column at all.
    import h5py
    out = workdir / "out" / OUTPUT_NAME
    out.parent.mkdir()
    with h5py.File(out, "w") as f:
        f.create_dataset("texts", data=[LAKE, SOIL, FECES], dtype=h5py.string_dtype(), maxshape=(None,))

    client = FakeEmbeddingClient()
    run_step4(step4, monkeypatch, workdir, client)

    # The missing column counts as 0 rows, so the 3 texts are not "done" and are embedded again.
    assert client.embedded_texts == [LAKE, SOIL, FECES]
    texts, vectors = read_h5(out)
    assert texts == [LAKE, SOIL, FECES]
    assert vectors.tolist() == [[2, 0, 0], [0, 1, 0], [0, 0, 3]]


def test_each_model_and_dimension_gets_its_own_file(step4, monkeypatch, workdir):
    run_step4(step4, monkeypatch, workdir, FakeEmbeddingClient())

    # Same terms at dimension 2 (the fake keeps the first two axes): a new file, everything re-embedded.
    two_dims = FakeEmbeddingClient(embed=lambda text: concept_vector(text)[:2])
    run_step4(step4, monkeypatch, workdir, two_dims, "--embedding_dim", "2")

    assert two_dims.embedded_texts == [LAKE, SOIL, FECES]
    assert sorted(p.name for p in (workdir / "out").iterdir()) == [
        "ontology_terms_unique_embeddings__text-embedding-3-large__dim2.h5",
        "ontology_terms_unique_embeddings__text-embedding-3-large__dim3.h5",
    ]


def test_api_returning_the_wrong_dimension_stops_the_run_before_writing(step4, monkeypatch, workdir):
    # --embedding_dim 2, but the "API" answers with 3 numbers per text (e.g. `dimensions` ignored)
    with pytest.raises(AssertionError, match="expected dim 2"):
        run_step4(step4, monkeypatch, workdir, FakeEmbeddingClient(), "--embedding_dim", "2")
    assert not (workdir / "out" / "ontology_terms_unique_embeddings__text-embedding-3-large__dim2.h5").exists()


def test_a_table_with_only_obsolete_terms_is_refused_with_a_clear_message(step4, monkeypatch, workdir):
    # Without this check the run "succeeds" with no API call and no .h5, and the later steps then fail
    # on a missing file without saying why (fixed 2026-10-19; see test_4_embed_terms_mutations.py).
    write_term_table(workdir / "ontology_terms.tsv.gz", [("ENVO", "ENVO_99999999", "old pond", "", "True")])
    client = FakeEmbeddingClient()

    with pytest.raises(SystemExit, match="has no non-obsolete terms: nothing to embed"):
        run_step4(step4, monkeypatch, workdir, client)

    assert client.calls == []


def test_unknown_model_is_rejected_by_the_command_line(step4, monkeypatch, workdir):
    with pytest.raises(SystemExit) as stop:               # argparse: invalid choice -> exit code 2
        run_step4(step4, monkeypatch, workdir, FakeEmbeddingClient(), "--model", "text-embedding-ada-002")
    assert stop.value.code == 2


def test_paths_starting_with_tilde_are_expanded(step4, monkeypatch, tmp_path):
    # The defaults are all '~/MicrobeAtlasProject/...': check that '~' means the home folder everywhere.
    monkeypatch.setenv("HOME", str(tmp_path))
    write_term_table(tmp_path / "terms.tsv.gz", TERMS)
    (tmp_path / "key").write_text("sk-test-key")
    monkeypatch.setattr(sys, "argv", ["4_embed_terms.py", "--ontology_terms", "~/terms.tsv.gz",
                                      "--api_key_path", "~/key", "--output_dir", "~/out", "--embedding_dim", "3"])
    install_fake_openai(monkeypatch, step4, FakeEmbeddingClient())

    step4.main()

    assert read_h5(tmp_path / "out" / OUTPUT_NAME)[0] == [LAKE, SOIL, FECES]
