"""Adapter classes are resolved from an agent config: generic, a file or a module."""
import pytest

from odyssey_runtime.adapters import load_adapter
from odyssey_bridge.planners.base import NavsimPlanner


def test_empty_spec_is_the_generic_adapter():
    assert load_adapter("") is NavsimPlanner
    assert load_adapter(None) is NavsimPlanner


def test_file_spec_imports_the_file_and_its_siblings(tmp_path):
    (tmp_path / "helper.py").write_text("TAG = 'from-sibling'\n")
    (tmp_path / "my_adapter.py").write_text(
        "from odyssey_bridge.planners.base import NavsimPlanner\n"
        "import helper\n"
        "class MyPlanner(NavsimPlanner):\n"
        "    TAG = helper.TAG\n"
        "class NotAPlanner:\n"
        "    pass\n")
    cls = load_adapter(f"{tmp_path / 'my_adapter.py'}:MyPlanner")
    assert issubclass(cls, NavsimPlanner) and cls.TAG == "from-sibling"
    assert load_adapter(f"{tmp_path / 'my_adapter.py'}:MyPlanner") is cls   # cached module
    with pytest.raises(TypeError, match="NotAPlanner"):
        load_adapter(f"{tmp_path / 'my_adapter.py'}:NotAPlanner")
    with pytest.raises(TypeError, match="Missing"):
        load_adapter(f"{tmp_path / 'my_adapter.py'}:Missing")


def test_module_spec_imports_dotted_path():
    cls = load_adapter("odyssey_bridge.planners.base:NavsimPlanner")
    assert cls is NavsimPlanner


@pytest.mark.parametrize("spec", ["my_adapter.py", ":MyPlanner", "module:", "no-colon"])
def test_malformed_specs_are_refused(spec):
    with pytest.raises(ValueError, match="adapter must be"):
        load_adapter(spec)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="adapter file not found"):
        load_adapter(f"{tmp_path / 'absent.py'}:X")


def test_broken_adapter_file_does_not_stay_imported(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("raise ImportError('no torch here')\n")
    with pytest.raises(ImportError, match="no torch here"):
        load_adapter(f"{bad}:X")
    bad.write_text("from odyssey_bridge.planners.base import NavsimPlanner\n"
                   "class X(NavsimPlanner):\n    pass\n")
    assert load_adapter(f"{bad}:X").__name__ == "X"
