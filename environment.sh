# BioM3 environment variables
# Source this file before running tests or scripts: source environment.sh
#
# Common variables are set unconditionally. Machine detection runs once, up
# front, using filesystem fingerprints (/flare : Aurora, /grand : Polaris). 
# Spark identified based on hostname. (TODO: this is fragile.)
# The block below the detection sets per-machine variables based on the 
# resolved BIOM3_MACHINE.


# --- Common (all machines) ---
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1


# --- Machine detection ---
# An explicit BIOM3_MACHINE (e.g. exported by the BioM3 Docker image) wins;
# auto-detect only when it is unset. The HPC checks (/flare, /grand) come
# first so ALCF apptainer runs resolve to aurora/polaris before the generic
# /.dockerenv container fallback.
if [[ -z "${BIOM3_MACHINE:-}" ]]; then
    if [[ -d /flare ]]; then
        BIOM3_MACHINE=aurora
    elif [[ -d /grand ]]; then
        BIOM3_MACHINE=polaris
    elif [[ -f /.dockerenv ]]; then
        BIOM3_MACHINE=container
    elif [[ "$(hostname)" == spark* ]]; then
        BIOM3_MACHINE=spark
    else
        BIOM3_MACHINE=unknown
    fi
fi
export BIOM3_MACHINE
echo "[environment.sh] Detected machine: $BIOM3_MACHINE"


# --- Machine-specific settings ---
if [[ "$BIOM3_MACHINE" == polaris ]]; then
    # --- Polaris (ALCF) — NVIDIA GPUs
    : # No Polaris-specific exports currently.

elif [[ "$BIOM3_MACHINE" == aurora ]]; then
    # --- Aurora (ALCF) — Intel GPUs
    
    # --- Documented by ALCF. Refer to:
    # --- https://docs.alcf.anl.gov/aurora/data-science/frameworks/scikit-learn
    # --- Per ALCF: "This is to resolve an issue due to a package called "numexpr".
    # --- It sets the variable
    # --- 'numexpr.nthreads' to available number of threads by default, in this case
    # --- to 208. However, the 'NUMEXPR_MAX_THREADS' is also set to 64 as a package
    # --- default. The solution is to either set the 'NUMEXPR_NUM_THREADS' to less than
    # --- or equal to '64' or to increase the 'NUMEXPR_MAX_THREADS' to the available
    # --- number of threads. Both of these variables can be set manually."
    export NUMEXPR_MAX_THREADS=64
    
    # --- Override the frameworks/2025.3.1 default `opencl:gpu;level_zero:gpu`,
    # --- ALCF explicitly flags as potentially problematic.
    # --- We have not found this to be necessary.
    # export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"

    # --- ALCF-recommended oneCCL environment. These don't appear to be necessary.
    # export CCL_PROCESS_LAUNCHER=pmix
    # export CCL_ATL_TRANSPORT=mpi
    # export CCL_KVS_MODE=mpi
    # export FI_MR_CACHE_MONITOR=userfaultfd
    
    # --- This appears to be necessary to avoid hangs, although after some 
    # --- further investigation, those "hangs" may have been actually just been
    # --- the IDE failing to refresh the log file. May not be necessary.
    # --- Hang avoidance — pairs with CCL_OP_SYNC (already set by frameworks module).
    export CCL_ATL_SYNC_COLL=1

    # --- Pin CCL progress threads to the last core of each rank's --cpu-bind range
    # --- (see ALCF-canonical 8-cores-per-rank binding in the train scripts).
    # --- Each rank therefore gets 7 framework cores + 1 CCL worker core within its
    # --- pin domain — equivalent to CCL_WORKER_AFFINITY=auto but explicit so we
    # --- can verify with `taskset`. Suggested by ALCF staff.
    export CCL_WORKER_AFFINITY=8,16,24,32,40,48,60,68,76,84,92,100
    
    # --- Avoid `AF_UNIX path too long` on Lightning DataLoader workers
    export TMPDIR=/tmp
    
    # --- Raise oneCCL's Level-Zero IPC-handle cache cap (default 1000). 
    # --- ("Sender cache limit is reached" warning seen in log file)
    export CCL_ZE_CACHE_GET_IPC_HANDLES_THRESHOLD=10000

elif [[ "$BIOM3_MACHINE" == spark ]]; then
    # --- DGX Spark — single NVIDIA GPU
    : # No Spark-specific exports currently.

elif [[ "$BIOM3_MACHINE" == container ]]; then
    # --- Docker container — NVIDIA GPU(s). torchrun handles multi-GPU and
    # multi-node rendezvous and sets a safe OMP_NUM_THREADS itself (see
    # scripts/launchers/container_{singlenode,multinode}.sh).
    # No HPC/MPI/PBS/CPU-binding settings apply here.
    : # No container-specific exports currently needed.

else
    echo "[environment.sh] Unknown machine: $(hostname) (using common settings only)"
fi
