"""python -m odyssey_runtime: the command word may stand anywhere on the command line."""
import sys

import pytest

from odyssey_runtime import __main__ as entry


@pytest.mark.parametrize("argv, index", [
    (["eval", "--agent", "a.yaml"], 0),
    (["--agent", "a.yaml", "eval", "--gpus", "0"], 2),
    (["--agent", "eval"], None),                      # a config named "eval" is not the command
    (["--agent", "eval", "run"], 2),
    ([], None),
])
def test_command_index(argv, index):
    assert entry.command_index(argv) == index


def test_eval_after_options_runs_the_batch_runner_with_every_other_argument(monkeypatch):
    from odyssey_runtime import eval as ev
    seen = []
    monkeypatch.setattr(ev, "main", lambda argv: seen.append(argv) or 0)
    monkeypatch.setattr(sys, "argv", ["odyssey_runtime", "--agent", "a.yaml", "eval", "--gpus", "0"])
    with pytest.raises(SystemExit) as stop:
        entry.main()
    assert stop.value.code == 0 and seen == [["--agent", "a.yaml", "--gpus", "0"]]
