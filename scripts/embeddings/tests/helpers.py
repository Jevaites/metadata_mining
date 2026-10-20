"""Test doubles and small readers shared by the unit and integration tests.

FakeOpenAI      replaces openai.OpenAI: returns the vector the test wrote for each text, records every call.
read_h5(path)   an output .h5 file as {column name: python list}.
rows(path)      an output .h5 file as a list of tuples, one per row - the easiest thing to compare by eye:
                    unique file      -> [(text, embedding), ...]
                    per-sample file  -> [(sample_id, text, embedding), ...]
write_lines(p)  a source text file (sample_id<TAB>text, no header) from a list of lines.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import h5py

EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent
if str(EMBEDDINGS_DIR) not in sys.path:
    sys.path.insert(0, str(EMBEDDINGS_DIR))


class FakeOpenAI:
    """Stands in for `OpenAI(...)`. Only `client.embeddings.create(input=[...], model=..., [dimensions=...])`
    exists, as in the script.

    vectors        {text: vector}. The fake returns exactly that vector (it does not truncate to
                   `dimensions`; a test picks vectors of the right size). A text missing from the table
                   raises KeyError: the code sent something the test did not expect.
    fail_on_call   1-based number of the call that raises ConnectionError (a dropped connection
                   mid-run), e.g. fail_on_call=2 lets the first batch through and fails the second.
    calls          one dict per create() call: {"input": [...], "model": ..., "dimensions": ... if sent}
    """

    def __init__(self, vectors, fail_on_call=None):
        self.vectors = vectors
        self.fail_on_call = fail_on_call
        self.calls = []
        self.embeddings = self  # so that client.embeddings.create(...) is self.create(...)

    def create(self, input, model, **kwargs):
        self.calls.append({"input": list(input), "model": model, **kwargs})
        if len(self.calls) == self.fail_on_call:
            raise ConnectionError("simulated API failure")
        return SimpleNamespace(data=[SimpleNamespace(embedding=list(self.vectors[t])) for t in input])

    @property
    def texts_sent(self):
        """Every text sent to the API, in order, over all calls."""
        return [t for call in self.calls for t in call["input"]]

    @property
    def batches(self):
        """The inputs of each call, e.g. [["a", "b"], ["c"]] for two calls."""
        return [call["input"] for call in self.calls]


def read_h5(path):
    with h5py.File(path, "r") as f:
        out = {}
        for name in f:
            values = f[name][:]
            if values.dtype.kind in "OS":  # string column
                out[name] = [v.decode("utf-8") if isinstance(v, bytes) else v for v in values]
            else:
                out[name] = values.tolist()
        return out


def rows(path):
    data = read_h5(path)
    columns = ["sample_ids", "texts", "embeddings"] if "sample_ids" in data else ["texts", "embeddings"]
    return list(zip(*(data[c] for c in columns)))


def write_lines(path, lines):
    Path(path).write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return str(path)
