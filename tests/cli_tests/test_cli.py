"""Tests for the ``biom3`` command line dispatcher."""

import importlib

import pytest

from biom3.cli.dispatch import UsageError, build_argv, find_command
from biom3.cli.registry import COMMANDS


@pytest.mark.parametrize("command", COMMANDS, ids=lambda c: "-".join(c.path))
def test_command_resolves(command):
    found, user_args = find_command([*command.path, "--some_flag", "value"])
    assert found is command
    assert user_args == ["--some_flag", "value"]

    module = importlib.import_module(command.module)
    assert callable(module.parse_arguments)
    assert callable(module.main)


def test_finetune_presets_finetune_flag():
    command, user_args = find_command(["finetune", "--epochs", "3"])
    argv = build_argv(command, user_args)
    assert argv == ["--finetune", "True", "--epochs", "3"]

    args = importlib.import_module(command.module).parse_arguments(argv)
    assert args.finetune is True


@pytest.mark.parametrize("user_args", [["--finetune", "False"], ["--finetune=False"]])
def test_finetune_rejects_explicit_finetune_flag(user_args):
    command, _ = find_command(["finetune"])
    with pytest.raises(UsageError):
        build_argv(command, user_args)
