"""Toy ontology shared by test_hierarchy_unit.py and test_hierarchy_integration.py.

One slot ('biome'), nine Metalog labels and two extra ontology terms below the slot root (above the root:
simplified, see the last paragraph):

    BFO_0000040                 not ENVO: never added as an answer
    └── environmental system    ENVO_01000254, in hierarchy.NO_ANSWER
        └── biome               ENVO_00000428, the real slot root: in NO_ANSWER, so never an answer
            ├── aquatic                 label
            │   ├── marine              label
            │   │   └── ocean           label
            │   └── freshwater          label
            │       └── lake            NOT a label (in the ontology, never used by Metalog in this slot)
            └── terrestrial             NOT a label
                ├── soil                label
                ├── desert              label
                ├── forest              label
                ├── grassland           label
                └── tundra              label

Why five terrestrial labels: in the default 'labels' mode, a label that has half or more of the slot's
labels at or below it counts as too vague to be an answer (Closure.informative). 'aquatic' has 4 of the 9
labels at or below it (aquatic, marine, ocean, freshwater), so it stays an allowed answer.

Ids below the root are made up but readable ("ENVO_ocean", real ones are ENVO_ + 8 digits): an assertion
error then shows names. They keep the ENVO_ prefix because Closure(answers="ontology") only adds
ENVO_/UBERON_ ancestors (terrestrial). As in the real ontology_terms.tsv.gz, the root has ancestors of its
own (real: 5, among them environmental system and the non-ENVO BFO_0000040); these are true of every label,
so they must never become answers either.
"""

import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))  # so the tests can `import hierarchy`

BIOME = "ENVO_00000428"  # slot root
SYSTEM, BFO = "ENVO_01000254", "BFO_0000040"  # ancestors of the root (real ids)
UP = {BIOME, SYSTEM, BFO}  # the root and everything above it
AQUATIC, MARINE, OCEAN = "ENVO_aquatic", "ENVO_marine", "ENVO_ocean"
FRESHWATER, LAKE = "ENVO_freshwater", "ENVO_lake"
TERRESTRIAL = "ENVO_terrestrial"
SOIL, DESERT, FOREST, GRASSLAND, TUNDRA = "ENVO_soil", "ENVO_desert", "ENVO_forest", "ENVO_grassland", "ENVO_tundra"

# {term: set of ALL its ancestors} (the transitive closure, as common.term_ancestors returns it)
ANC = {
    BFO: set(),
    SYSTEM: {BFO},
    BIOME: {SYSTEM, BFO},
    AQUATIC: UP,
    MARINE: {AQUATIC} | UP,
    OCEAN: {MARINE, AQUATIC} | UP,
    FRESHWATER: {AQUATIC} | UP,
    LAKE: {FRESHWATER, AQUATIC} | UP,
    TERRESTRIAL: UP,
    SOIL: {TERRESTRIAL} | UP,
    DESERT: {TERRESTRIAL} | UP,
    FOREST: {TERRESTRIAL} | UP,
    GRASSLAND: {TERRESTRIAL} | UP,
    TUNDRA: {TERRESTRIAL} | UP,
}

# The slot's training labels = the model's classes = the columns of every score / probability matrix
VOCAB = [OCEAN, MARINE, FRESHWATER, AQUATIC, SOIL, DESERT, FOREST, GRASSLAND, TUNDRA]


def probs(*rows, vocab=VOCAB):
    """Probability matrix (one row per dict) over `vocab`; labels missing from a dict get 0.
    probs({OCEAN: .55, MARINE: .45}) -> [[.55, .45, 0, 0, 0, 0, 0, 0, 0]]."""
    return np.array([[row.get(label, 0.0) for label in vocab] for row in rows], dtype=float)


def log_scores(*rows, vocab=VOCAB):
    """Scores that ARE log-probabilities: softmax(log_scores(row), T=1) gives the row back.
    A perfectly calibrated model, so the temperature fitted on it is 1. Zeros become 1e-9."""
    return np.log(np.maximum(probs(*rows, vocab=vocab), 1e-9))
