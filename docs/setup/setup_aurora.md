# Installation and setup instructions for Aurora

> **Containerized alternative:** to run BioM3 from an Apptainer `.sif` instead of
> a bare-metal venv, see [setup_aurora_container.md](./setup_aurora_container.md).
> The instructions below are the bare-metal `module load frameworks` path.

The Aurora cluster provides access to Intel GPUs, so a different installation is required. Like Polaris, there is a prebuilt environment that we can extend using a virtual environment.

In order to run on Intel GPUs, we have modified a custom `lightning` package, which must be included in the installation. This [lightning source code](https://github.com/addison-nm/lightning) is available on GitHub, if one needs to edit this directly. The commands below install this version of lightning, originally forked from the ALCF lightning repo.

Create the environment:

```bash
env_name="biom3-env"
module load frameworks
cd /path/to/BioM3-dev
# Create environment, using packages from prebuilt one
python -m venv venvs/${env_name} --system-site-packages
source "venvs/${env_name}/bin/activate"
# Install custom lightning package
python -m pip install git+https://github.com/addison-nm/lightning.git --no-build-isolation
# Install BioM3
python -m pip install -e '.[app]'
# Install additional dependencies
python -m pip install -r requirements/aurora.txt
# h5py must be venv-local: the frameworks build links an OpenSSL that
# conflicts with the system libssl once hashlib is imported first.
python -m pip install --ignore-installed --no-cache-dir h5py
```

Note that presently an error message may be raised due to package conflicts, but the installation should still work.

Verify the h5py install before going further, each line in a fresh interpreter:

```bash
python -c "import h5py; print(h5py.__file__)"           # must be under .../venvs/biom3-env/
python -c "import hashlib, pandas; import h5py; print('ok')"
```

The second line is the real check. The frameworks h5py links an OpenSSL that conflicts with
the system `libssl` as soon as `hashlib` is loaded first, and pandas imports hashlib — so
`biom3_train_stage3` fails on `import h5py` while a plain `import h5py` succeeds. Running the
test suite does not check this: pytest imports every test module at collection, and
`tests/data_prep_tests/` sorts before `tests/dbio_tests/`, so h5py always binds before pandas
and every later import is a no-op.

## Usage

Load the Aurora frameworks module, activate the environment, and source `environment.sh` at the start of each session. The script auto-detects Aurora and sets additional variables required for Intel GPUs (`NUMEXPR_MAX_THREADS`, `ONEAPI_DEVICE_SELECTOR`).

```bash
cd /path/to/BioM3-dev
module load frameworks
source venvs/biom3-env/bin/activate
source environment.sh
```

### Running tests

```bash
python -m pytest tests/test_imports.py           # Quick import check
python -m pytest tests                           # Full suite (may take some time)
python -m pytest tests -rs                       # Full suite, report skipped tests
```

Some tests will be skipped if pretrained weights have not been synced. See [setup_shared_weights.md](./setup_shared_weights.md) for the list of required files.
