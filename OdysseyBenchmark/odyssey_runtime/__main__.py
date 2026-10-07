"""Run, check or batch-evaluate a model on the installed scenes."""

import argparse
from pathlib import Path
import sys

COMMANDS = ("run", "validate-profile", "check", "eval")


def command_index(argv):
    """Index of the command word in argv (the first command name that is not --agent's value), or None."""
    for i, arg in enumerate(argv):
        if arg in COMMANDS and (i == 0 or argv[i - 1] != "--agent"):
            return i
    return None


def main():
    root = Path(__file__).resolve().parents[2]          # the repository root
    parser = argparse.ArgumentParser(description=__doc__, add_help=False,
                                     usage="python -m odyssey_runtime {run,check,eval,validate-profile} "
                                           "--agent CONFIG [options]; add --help after a command for its options")
    parser.add_argument("command", nargs="?", choices=list(COMMANDS))
    parser.add_argument("--agent", type=Path, default=root / "OdysseyBenchmark/agents/ltf_sdroute.yaml",
                        help="agent config (OdysseyBenchmark/agents/<model>.yaml or your own)")
    argv = sys.argv[1:]
    i = command_index(argv)
    if i is not None and argv[i] == "eval":   # eval owns its whole command line (incl. --help)
        from .eval import main as evaluate

        raise SystemExit(evaluate(argv[:i] + argv[i + 1:]))
    args, remaining = parser.parse_known_args()
    # run and check print their own options for --help; the bare command prints this one.
    if args.command is None or (args.command == "validate-profile" and {"-h", "--help"} & set(remaining)):
        parser.print_help()
        raise SystemExit(0 if {"-h", "--help"} & set(remaining) else 2)
    if args.command == "check":
        from .check import main as check

        raise SystemExit(check(args.agent, remaining))
    if args.command == "validate-profile":
        if remaining:
            parser.error("unexpected extra arguments")
        from .agent_config import load

        agent = load(args.agent)
        profile, model = agent.profile, agent.model
        print(f"{model.name}: python={model.python} repo={model.repo} agent_config={model.agent_config} "
              f"checkpoint={model.checkpoint} adapter={model.adapter or 'generic'}")
        print(f"cameras={list(profile.cameras)} history_capacity={profile.history_capacity} "
              f"navigation={profile.navigation}")
        return
    from .launch import main as run

    raise SystemExit(run(args.agent, remaining))


if __name__ == "__main__":
    main()
