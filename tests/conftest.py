"""Pytest Configuration File

"""

import pytest
import os
import shutil

DATDIR = "tests/_data"  # data directory for all tests.
TMPDIR = "tests/_tmp"  # output directory for all tests.

def remove_dir(dir:str):
    """Helper function to remove a directory recursively."""
    if not dir.startswith(TMPDIR):
        msg = f"Can only use function `remove_dir` from tests.conftest to \
        remove directories in the directory {TMPDIR}. Got: {dir}"
        raise RuntimeError(msg)
    shutil.rmtree(dir)

def get_args(fpath):
    """Read command line arguments from a text file."""
    with open(fpath, 'r') as f:
        # Strip whitespace and ignore empty lines
        lines = [line.strip() for line in f if line.strip()]
    # Join into a single string and split into arguments
    argstring = " ".join(lines)
    arglist = argstring.split()
    return arglist

@pytest.fixture(autouse=True)
def _isolate_matmul_precision():
    """Prevent one test's global float32 matmul precision from leaking to the next.

    The training entrypoints (Stage{1,2,3}/run_PL_training.py::main) call
    torch.set_float32_matmul_precision('medium') for production throughput. In a
    full-suite run that setting persists process-wide and silently lowers matmul
    precision for later tests — breaking exact-arithmetic assertions such as the
    LoRA merge-equivalence checks (folded weight vs runtime path diverge under
    bf16 accumulation). Snapshot and restore around every test.
    """
    import torch
    prev = torch.get_float32_matmul_precision()
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)

@pytest.fixture
def no_process_group(monkeypatch):
    """Take the single-process path despite a default process group that an
    earlier in-process training test left initialized. On XPU, Lightning's
    DeepSpeed strategy leaves one that cannot be destroyed: DeepSpeed caches a
    clone of the world group (deepspeed.utils.groups._WORLD_GROUP), and later
    Stage 3 runs fail on the stale handle."""
    import torch.distributed as dist
    monkeypatch.setattr(dist, "is_initialized", lambda: False)


def check_downloads(paths_to_check):
    """Returns list of missing files and a warning message."""
    issues = []
    for fpath in paths_to_check:
        if not os.path.exists(fpath):
            msg = f"Weight files not found: {fpath}"
            issues.append(msg)
    msg = ""
    if issues:
        msg = "Entrypoint test relies on downloaded weights!"
        msg += "\nThis test will be skipped until the following issues are resolved:"
        for issue in issues:
            msg += f"\n  {issue}"
    return issues, msg

#####################
##  Configuration  ##
#####################

def pytest_addoption(parser):
    parser.addoption(
        "--benchmark", action="store_true", default=False,
        help="run benchmarking tests"
    )
    parser.addoption(
        "--include_requires_gpu", action="store_true", default=False,
        help="run GPU specific tests"
    )
    parser.addoption(
        "--database_files", action="store_true", default=False,
        help="run tests that require full database files"
    )
    parser.addoption(
        "--network", action="store_true", default=False,
        help="run tests that require network access"
    )
    parser.addoption(
        "--quick", action="store_true", default=False,
        help="skip tests marked slow (entrypoint, training, pipeline)"
    )
    parser.addoption(
        "--multinode", action="store", type=int, default=0,
        help="number of nodes for multinode tests (0 = skip multinode-marked tests)"
    )
    parser.addoption(
        "--multidevice", action="store", type=int, default=0,
        help="number of devices per node for multinode tests "
             "(0 = skip multinode-marked tests)"
    )
    parser.addoption(
        "--skip_devices", action="store", default="",
        help="comma-separated parametrize device values to skip (e.g. 'cpu' or "
             "'cpu,xpu'). Drops any test variant whose @parametrize includes "
             "the listed value. Useful on Spark to skip slow cpu redundancies."
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "benchmark: mark test as benchmarking")
    config.addinivalue_line("markers", "requires_gpu: mark test as GPU specific")
    config.addinivalue_line("markers", "database_files: mark test as requiring full database files")
    config.addinivalue_line("markers", "network: mark test as requiring network access")
    config.addinivalue_line("markers", "slow: mark test as slow (skipped under --quick)")
    config.addinivalue_line("markers",
        "multinode: mark test as requiring an MPI launcher with --multinode/--multidevice")

def pytest_collection_modifyitems(config, items):
    benchmark_flag_given = False
    include_requires_gpu_given = False
    if config.getoption("--benchmark"):
        # --benchmark given in cli: do not skip benchmarking tests
        benchmark_flag_given = True
    if config.getoption("--include_requires_gpu"):
        # --include_requires_gpu given in cli: do not skip GPU tests
        include_requires_gpu_given = True
    database_files_flag_given = False
    if config.getoption("--database_files"):
        database_files_flag_given = True
    network_flag_given = False
    if config.getoption("--network"):
        network_flag_given = True
    quick_flag_given = config.getoption("--quick")
    skip_devices = {
        d.strip().lower()
        for d in str(config.getoption("--skip_devices") or "").split(",")
        if d.strip()
    }
    multinode_n = int(config.getoption("--multinode"))
    multidevice_n = int(config.getoption("--multidevice"))
    expected_world = multinode_n * multidevice_n
    # Resolve actual world size from the launcher's env vars. PALS / PMI /
    # OpenMPI don't set WORLD_SIZE — fall back to the same scan used by
    # biom3.core.distributed.get_world_size so the gate works under mpiexec.
    try:
        from biom3.core.distributed import get_world_size as _get_world_size
        actual_world = _get_world_size()
    except Exception:
        actual_world = int(os.environ.get("WORLD_SIZE", 1))
    skip_benchmark = pytest.mark.skip(reason="need --benchmark option to run")
    skip_requires_gpu = pytest.mark.skip(reason="need --include_requires_gpu option to run")
    skip_database_files = pytest.mark.skip(reason="need --database_files option to run")
    skip_network = pytest.mark.skip(reason="need --network option to run")
    skip_slow = pytest.mark.skip(reason="skipped under --quick; remove flag to run slow tests")
    skip_multinode = pytest.mark.skip(
        reason=f"multinode test needs --multinode N --multidevice M with "
               f"WORLD_SIZE=N*M (got --multinode={multinode_n} "
               f"--multidevice={multidevice_n} WORLD_SIZE={actual_world})"
    )
    for item in items:
        if "benchmark" in item.keywords and not benchmark_flag_given:
            item.add_marker(skip_benchmark)
        if "requires_gpu" in item.keywords and not include_requires_gpu_given:
            item.add_marker(skip_requires_gpu)
        if "database_files" in item.keywords and not database_files_flag_given:
            item.add_marker(skip_database_files)
        if "network" in item.keywords and not network_flag_given:
            item.add_marker(skip_network)
        if "slow" in item.keywords and quick_flag_given:
            item.add_marker(skip_slow)
        if "multinode" in item.keywords:
            if expected_world == 0 or actual_world != expected_world:
                item.add_marker(skip_multinode)
        if skip_devices:
            # Drop any parametrize variant whose values include a string
            # listed in --skip_devices. Matches against the rendered test
            # id (e.g. "test_foo[cpu-arg2]") so we cover @parametrize
            # over "device" or any other arg containing the literal name.
            params = getattr(item, "callspec", None)
            param_values = list(params.params.values()) if params else []
            if any(
                isinstance(v, str) and v.lower() in skip_devices
                for v in param_values
            ):
                item.add_marker(pytest.mark.skip(
                    reason=f"--skip_devices={','.join(sorted(skip_devices))}"
                ))