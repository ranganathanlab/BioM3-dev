# Session: Validating BioM3 on Aurora's oneAPI 2026.1 stack

**Date:** 2026-10-04 to 2026-10-06
**Branch:** `dev`

## Goal

Validate BioM3 on Aurora from two angles, a fresh source checkout and an Apptainer image,
with PBS scripts that can be queued as they are. Aurora had moved to `frameworks/2026.1.0`
on 2026-10-02, and the container path turned out not to work on the new stack, so most of
the session went into fixing that.

## Host stack

`frameworks/2026.1.0`: oneAPI 2026.1.0, torch 2.13.0a0 (source build), Aurora MPICH, Cray
libfabric 2.3.1 at `/opt/cray/libfabric/2.3.1/lib64`, Level-Zero GPU driver 25.18,
apptainer 1.5.1. `/opt/cray/libfabric/1.22.0`, which the docs named, no longer exists on
the compute nodes.

## Validation kit

The scripts live under `_misc/aurora_validation/`, which is gitignored and deliberately not
committed.

- `checkout/`: clone `dev`, build the venv per `docs/setup/setup_aurora.md`, fetch the
  published weights, run the workload. Scripts for the test suite, the generation pipeline,
  Stage 3 finetuning on 1 node and on 2 nodes × 12 ranks, Stage 1 pretraining on 1 and 4
  nodes, and a 16-node job that runs four 4-node finetunes at once by slicing
  `$PBS_NODEFILE`.
- `container/`: the same workloads through `scripts/aurora/apptainer_run.sh` and
  `apptainer_mpi_run.sh`. The image path is set in one place, `container/image.conf`.
- `data/`: the 13-family Stage 1 debug subset the pretraining scripts use.

## What was wrong, and the fixes

| Commit | What |
| ------ | ---- |
| `bac17c3` | `docs:` Apptainer image build steps in the dev guide |
| `fccf427` | `chore:` `xpu-oneapi` image to oneAPI 2026.1 (base `intel/oneapi:2026.1.0-devel-ubuntu24.04`, torch 2.14.1+xpu, dpctl 0.22.1) |
| `15feb07` | `fix:` keep image libraries ahead of the host's `/usr/lib64` under `mpiexec` |
| `abd9941` | `fix:` single-device strategy for one-rank `ddp` runs in Stage 3 |
| `cc09c24` | `test:` the Stage 1 inference checkpoint case loads a checkpoint the test bundle ships |
| `b1a8363` | `docs:` Aurora container docs brought up to oneAPI 2026.1 |

**The image has to follow the host.** The oneAPI 2025.3 image, with the host's libfabric
2.3.1 bound in, got wrong data from multi-node collectives over CXI: DDP's parameter check
failed on some ranks at 2 nodes (Stage 3) and at 4 nodes (Stage 1). Rebuilding on oneAPI
2026.1 fixed it. The torch `+xpu` wheel's runtime has to equal the base image's oneAPI.

**Zero XPU devices under `mpiexec`.** `apptainer_mpi_run.sh` appended the host's
`/usr/lib64` (bound at `/hostevent`) to `LD_LIBRARY_PATH` for PMIx, which made the host's
Level-Zero driver win over the image's. The 2026.1 image then paired its own loader with
the host's older driver and saw no devices. An early diagnosis in this session, that the
image's driver did not match the host's, was wrong: on a node the image's own driver sees
all 12 devices.

**One-rank `ddp` hang.** In the 2026.1 image, Stage 3 with `--distributed_strategy ddp` and
one rank hung at the first training step, the main thread spinning inside the Level-Zero
driver while Lightning read a logged metric, with or without `mpiexec` and on both oneCCL
transports. One rank now uses `SingleDeviceStrategy`, as Stage 1 already did. The cause of
the hang itself is not known.

## Results

Checkout: generation, both finetunes, Stage 1 pretraining on 1 and 4 nodes and the 16-node
job all passed. The test suite had two failures, both the Stage 1 inference checkpoint case
naming a checkpoint the test-weights bundle does not ship; `cc09c24` fixes that.

Container, `biom3_xpu-oneapi-abd9941.sif`: one-rank finetune, 2-node finetune over CXI,
Stage 1 pretraining at 12 and 48 ranks over CXI, and generation all passed. The test suite
gave 1520 passed, 2 failed, 95 skipped; the two failures are the same case, from the tests
baked into the image before `cc09c24`.

## Open items

1. The cause of the one-rank `ddp` hang. It is avoided, not explained.
2. The single-node `xpu` image has not been run on the new stack.
3. The User Guide still names the `cuda-e34af20` tags.

## Notes for next time

- `apptainer build` works on a login node; `apptainer exec` only on a compute node.
- Do not infer node paths from a login node. The libfabric path was confirmed from a
  compute node before the docs were changed.
- `debug` and `debug-scaling` each allow one running and one queued job per user.
