"""Dispatcher for the ``biom3`` command line interface.

``biom3 <command> [<args>]`` looks the command up in :mod:`biom3.cli.registry`,
imports its module, and hands the remaining arguments to that module's
``parse_arguments`` and ``main``. Command modules are imported only when their
command runs, so ``biom3 --help`` does not load torch.
"""

import contextlib
import difflib
import importlib
import sys

import biom3
from biom3.cli.registry import COMMANDS, OFFICIAL, TIERS

PROG = "biom3"
DESCRIPTION = "BioM3: protein sequence generation guided by natural language prompts."


class UsageError(Exception):
    pass


def find_command(argv):
    """Split ``argv`` into the command it names and the arguments that follow.

    Returns ``(None, argv)`` when ``argv`` does not start with a command path.
    """
    matches = [c for c in COMMANDS if tuple(argv[:len(c.path)]) == c.path]
    if not matches:
        return None, list(argv)
    command = max(matches, key=lambda c: len(c.path))
    return command, list(argv[len(command.path):])


def build_argv(command, user_args):
    """Return the argument list for ``command``: its preset arguments, then the user's."""
    for flag in command.preset_args:
        if not flag.startswith("--"):
            continue
        if any(a == flag or a.startswith(flag + "=") for a in user_args):
            raise UsageError(
                f"{flag} is set by this command ({' '.join(command.preset_args)}) "
                "and cannot be passed explicitly"
            )
    return [*command.preset_args, *user_args]


@contextlib.contextmanager
def _program_name(name):
    """Label argparse usage and error messages with ``name``.

    argparse reads the program name from ``sys.argv[0]``. The original value is
    restored on exit because Lightning re-executes ``sys.argv`` to launch worker
    processes, so it must be intact by the time the command runs.
    """
    original = sys.argv[0]
    sys.argv[0] = name
    try:
        yield
    finally:
        sys.argv[0] = original


def run_command(command, user_args):
    argv = build_argv(command, user_args)
    module = importlib.import_module(command.module)
    with _program_name(f"{PROG} {command.name}"):
        args = module.parse_arguments(argv)
    result = module.main(args)
    return result if isinstance(result, int) else 0


def _in_group(prefix):
    prefix = tuple(prefix)
    return [c for c in COMMANDS if c.path[:len(prefix)] == prefix]


def format_help(group=(), show_all=False):
    """Help text listing the commands under ``group`` (all commands when empty)."""
    commands = _in_group(group)
    if not group and not show_all:
        commands = [c for c in commands if c.tier == OFFICIAL]
    names = {c: " ".join(c.path[len(group):]) for c in commands}
    width = max(len(name) for name in names.values())

    lines = [f"usage: {' '.join((PROG, *group))} <command> [<args>]", ""]
    if not group:
        lines += [DESCRIPTION, ""]
    for tier in TIERS:
        tier_commands = [c for c in commands if c.tier == tier]
        if not tier_commands:
            continue
        lines.append("commands:" if tier == OFFICIAL else f"{tier} commands:")
        lines += [f"  {names[c]:<{width}}  {c.summary}" for c in tier_commands]
        lines.append("")

    lines.append("options:")
    lines.append("  -h, --help  show this message and exit")
    if not group:
        lines.append("  --all       also list advanced and experimental commands")
        lines.append("  --version   show the biom3 version and exit")
    lines += ["", f"Run `{' '.join((PROG, *group))} <command> --help` for a command's arguments."]
    return "\n".join(lines)


def _no_command(argv):
    """Handle an ``argv`` that names no command: help, version, or a usage error."""
    n_words = next((i for i, a in enumerate(argv) if a.startswith("-")), len(argv))
    words, options = tuple(argv[:n_words]), argv[n_words:]

    depth = 0
    while depth < len(words) and _in_group(words[:depth + 1]):
        depth += 1
    group = words[:depth]
    prog = " ".join((PROG, *group))

    def usage_error(message):
        print(f"{prog}: error: {message}", file=sys.stderr)
        if group:
            print("\n" + format_help(group), file=sys.stderr)
        else:
            print(f"Run `{PROG} --help --all` to list the commands.", file=sys.stderr)
        return 2

    if depth < len(words):
        choices = sorted({c.path[depth] for c in _in_group(group)})
        close = difflib.get_close_matches(words[depth], choices, n=1)
        hint = f" (did you mean '{close[0]}'?)" if close else ""
        return usage_error(f"unknown command '{words[depth]}'{hint}")

    known = {"-h", "--help"} if group else {"-h", "--help", "--all", "--version"}
    unrecognized = [a for a in options if a not in known]
    if group and (unrecognized or not options):
        return usage_error("a command is required")
    if unrecognized:
        return usage_error(f"unrecognized arguments: {' '.join(unrecognized)}")

    if "--version" in options:
        print(f"{PROG} {biom3.__version__}")
    else:
        print(format_help(group, show_all="--all" in options))
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    command, user_args = find_command(argv)
    if command is None:
        return _no_command(argv)
    try:
        return run_command(command, user_args)
    except UsageError as exc:
        print(f"{PROG} {command.name}: error: {exc}", file=sys.stderr)
        return 2
