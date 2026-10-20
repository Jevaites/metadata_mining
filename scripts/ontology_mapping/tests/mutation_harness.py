"""
Shared harness of the subprocess-style mutation tests (test_1_..., test_2_..., test_3_..., test_6_predict_atlas_...,
test_6c_..., test_common_..., test_hierarchy_..., test_methods_mutations.py).

A mutation is a small, plausible bug written into a copy of a script: `edits` is a list of (find, replace)
pairs (each `find` must occur exactly once in the script, so a stale snippet fails loudly), `why` says
what the bug would do to the pipeline, and `caught_by` names the tests that must FAIL on it: a list of
node ids, or a dict {node id: text (or tuple of texts) the AssertionError message must contain}.

Each *_mutations.py file builds its own temporary copy (which files, where) and calls:
    build_copy(tmp_path, {relative path: source text or Path to copy}, tests_dir)  -> folder of the copy's tests
    run_pytest(tmp_path, cwd, targets)            -> {node id: Outcome(outcome, exception, message)}
    check_unmutated(outcomes, MUTATIONS, n)       the copy passes, and every caught_by test exists and passes
    check_caught(mutation, outcomes)              every caught_by test failed (on an assertion, with the text)

Every run is a fresh Python process, so a mutant never leaks into the real modules of this session.
Outcomes are collected by a small pytest plugin (PLUGIN) that writes them to $OUTCOMES_JSON.
"""

import json
import os
import shutil
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

HERE = Path(__file__).resolve().parent

Mutation = namedtuple("Mutation", "name edits why caught_by")
Outcome = namedtuple("Outcome", "outcome exception message")  # outcome: passed / failed / skipped / not run

PLUGIN = '''
import json, os
import pytest
OUTCOMES = {}
def pytest_runtest_logreport(report):
    if report.when == "call" or report.outcome != "passed":       # a setup error counts as "failed"
        OUTCOMES[report.nodeid] = [report.outcome, getattr(report, "exception", ""), str(report.longrepr or "")]
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):                         # keeps the exception type
    report = (yield).get_result()
    report.exception = call.excinfo.typename if call.excinfo else ""
def pytest_sessionfinish(session):
    with open(os.environ["OUTCOMES_JSON"], "w") as handle:
        json.dump(OUTCOMES, handle)
'''


def mutate(script, edits):
    """The text of `script` (a Path) with the (find, replace) edits applied."""
    source = script.read_text()
    for find, replace in edits:
        assert source.count(find) == 1, f"{script.name} changed: update the mutation snippet {find!r}"
        source = source.replace(find, replace)
    return source


def build_copy(tmp_path, files, tests_dir="tests"):
    """Write `files` ({path relative to tmp_path: text, or Path of a file to copy}) and the real conftest.py
    into tmp_path/tests_dir (it puts the copy's script folder, tests_dir/.., first on sys.path).
    Returns tmp_path/tests_dir."""
    for relative, content in files.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, Path):
            shutil.copy(content, target)
        else:
            target.write_text(content)
    tests = tmp_path / tests_dir
    tests.mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / "conftest.py", tests / "conftest.py")
    return tests


def run_pytest(tmp_path, cwd, targets, env=None, args=(), verbose=False):
    """Run pytest on `targets` (files or node ids) from `cwd` in a fresh Python.
    Returns {node id: Outcome}; a node id that did not run is absent. verbose=True (-vv) keeps full
    assertion diffs in Outcome.message (slow when a mutant makes long lists differ)."""
    plugin_dir = tmp_path / "_mutation_plugin"
    plugin_dir.mkdir(exist_ok=True)
    (plugin_dir / "mutation_outcomes.py").write_text(PLUGIN)
    outcomes = plugin_dir / "outcomes.json"
    environment = dict(os.environ, **(env or {}), OUTCOMES_JSON=str(outcomes),
                       PYTHONPATH=os.pathsep.join([str(plugin_dir), os.environ.get("PYTHONPATH", "")]))
    subprocess.run([sys.executable, "-m", "pytest", "-vv" if verbose else "-q", "-p", "no:cacheprovider",
                    "-p", "mutation_outcomes", *args, *targets], cwd=cwd, env=environment, capture_output=True, timeout=600)
    return {node: Outcome(*values) for node, values in json.loads(outcomes.read_text()).items()}


def named_tests(mutations):
    return sorted({node for m in mutations for node in m.caught_by})


def check_unmutated(outcomes, mutations, min_tests):
    """On an unchanged copy: at least `min_tests` ran (they were really collected), none failed (a skip, e.g.
    an optional package missing, is allowed), and every test named in a caught_by ran and passed, so it
    fails only because of its mutant (a renamed test would otherwise be "not run" and look like a catch)."""
    assert len(outcomes) >= min_tests, f"only {len(outcomes)} tests ran"
    assert {node: o.outcome for node, o in outcomes.items() if o.outcome not in ("passed", "skipped")} == {}
    missing = {node: outcomes[node].outcome if node in outcomes else "not run" for node in named_tests(mutations)}
    assert {node: o for node, o in missing.items() if o != "passed"} == {}


def check_caught(mutation, outcomes):
    """Every caught_by test failed. With a dict caught_by, it failed on an AssertionError (not a crash) whose
    message contains the given text(s), i.e. it caught the bug for the intended reason."""
    not_failed = {node: outcomes[node].outcome if node in outcomes else "not run"
                  for node in mutation.caught_by if node not in outcomes or outcomes[node].outcome != "failed"}
    assert not_failed == {}, f"{mutation.name} ({mutation.why}) was not caught by: {not_failed}"
    if isinstance(mutation.caught_by, dict):
        for node, expected in mutation.caught_by.items():
            _, exception, message = outcomes[node]
            assert exception == "AssertionError", f"{node} crashed ({exception}) instead of failing an assertion: {message}"
            missing = [text for text in ((expected,) if isinstance(expected, str) else expected) if text not in message]
            assert missing == [], f"{node} failed, but not in the expected way (no {missing}): {message}"
