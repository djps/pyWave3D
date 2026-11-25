"""The options rules: case-insensitive keys, dotted paths, defaults and type errors."""

import json

import pytest

from pywave3d.options import Options, load_options

DOC = {
    "Model": {
        "Geometry": {"lx": 0.01, "ORIGIN": [0, 0, 1]},
        "Flags": {"a": True, "b": "yes", "c": "F", "bad": "maybe"},
        "Name": "  water  ",
        "N": 4,
        "List": ["a", "b"],
    },
    "Output": {"xyz": [{"name": "p0.txt"}, {"name": "p1.txt"}], "one": {"name": "only"}},
}


@pytest.fixture
def opts() -> Options:
    return Options.from_dict(DOC)


def test_case_insensitive_dotted_paths(opts: Options) -> None:
    assert opts.get("model.geometry.LX", float) == 0.01
    assert opts.child("MODEL").child("geometry").get("lx", float) == 0.01
    assert opts.get("Model.Geometry.origin", list[float]) == [0.0, 0.0, 1.0]


def test_missing_keys_give_defaults_and_empty_nodes(opts: Options) -> None:
    assert opts.get("Model.Geometry.ly", float, 0.5) == 0.5
    assert opts.get("nothing.here", int) is None
    node = opts.child("Model.Missing")
    assert not node.exists()
    assert node.get("anything", float, 2.0) == 2.0
    assert opts.get_option("Model.Missing") is None


def test_values_by_type(opts: Options) -> None:
    assert opts.get("Model.Name", str) == "water"
    assert opts.get("Model.N", int) == 4
    assert opts.get("Model.N", float) == 4.0
    assert opts.get("Model.Flags.a", bool) is True
    assert opts.get("Model.Flags.b", bool) is True
    assert opts.get("Model.Flags.c", bool) is False
    assert opts.get("Model.List", list[str]) == ["a", "b"]
    assert opts.get("Model.N", list[float]) == [4.0]  # a single value is a list of one


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        ("Model.Name", float),
        ("Model.N", str),
        ("Model.Flags.bad", bool),
        ("Model.Geometry.lx", int),
        ("Model.Flags.a", int),
    ],
)
def test_wrong_type_names_the_key(opts: Options, path: str, kind: type) -> None:
    with pytest.raises(TypeError, match=path):
        opts.get(path, kind)


def test_children(opts: Options) -> None:
    names = [c.get("name", str) for c in opts.children("Output.xyz")]
    assert names == ["p0.txt", "p1.txt"]
    assert [c.get("name", str) for c in opts.children("Output.one")] == ["only"]
    assert opts.children("Output.none") == []


def test_load_and_resolve(tmp_path) -> None:
    (tmp_path / "data.txt").write_text("1 2 3 4\n")
    f = tmp_path / "o.json"
    f.write_text(json.dumps({"Initial": {"InputFile": "data.txt"}}))
    o = load_options(f)
    assert o.child("Initial").resolve(o.get("Initial.InputFile", str)) == tmp_path / "data.txt"
    with pytest.raises(FileNotFoundError):
        o.resolve("missing.txt")


def test_reference_options_file(repo) -> None:
    o = load_options(repo / "options" / "woptions.json")
    assert o.get("Model.PhysicalProperties.SpeedofSound", float) == 1487.0
    assert o.get("Solver.Options.NLN", bool) is False
    assert len(o.children("Solver.Output.xyz")) == 6
