#!/usr/bin/env bash
#=============================================================================
#
# FILE: scripts/_container_mounts.sh  (sourced, not run)
#
# The host-to-container mounts shared by every BioM3 container wrapper
# (docker/run.sh and scripts/{aurora,polaris}/apptainer_*run.sh), so each
# runtime mounts the same things at the same paths from the same settings:
#
#   BIOM3_WEIGHTS_DIR  host weights dir  -> /app/weights    (ro; default ./weights, if present)
#   BIOM3_DATA_DIR     host data dir     -> /app/data       (ro; default ./data, if present)
#   BIOM3_OUTPUTS_DIR  host outputs dir  -> /app/outputs    (rw; default ./outputs, created)
#   BIOM3_TESTS_TMP    test scratch dir  -> /app/tests/_tmp (rw; default <outputs>/tests_tmp)
#   BIOM3_CONFIGS_DIR  host configs dir  -> /app/configs    (ro; optional, replaces baked-in)
#   BIOM3_BIND_EXTRA   comma-separated host paths, each mounted read-only at the
#                      same path; an entry containing ':' is used as-is
#                      (src:dst[:opts])
#
# biom3_container_mounts [--no-weights] fills the array MOUNTS with
# src:dst[:opts] specs, which `docker run -v` and `apptainer --bind` both
# accept. It returns non-zero if a BIOM3_BIND_EXTRA source does not exist:
# docker would otherwise create it as an empty root-owned directory.
#
#=============================================================================

biom3_container_mounts() {
    local weights="${BIOM3_WEIGHTS_DIR:-$PWD/weights}"
    local data="${BIOM3_DATA_DIR:-$PWD/data}"
    local outputs="${BIOM3_OUTPUTS_DIR:-$PWD/outputs}"
    local tests_tmp="${BIOM3_TESTS_TMP:-${outputs}/tests_tmp}"
    local spec src
    local -a extra=()

    mkdir -p "${outputs}" "${tests_tmp}"
    MOUNTS=()
    if [[ "${1:-}" != "--no-weights" && -d "${weights}" ]]; then
        MOUNTS+=("${weights}:/app/weights:ro")
    fi
    if [[ -d "${data}" ]]; then
        MOUNTS+=("${data}:/app/data:ro")
    fi
    MOUNTS+=("${outputs}:/app/outputs" "${tests_tmp}:/app/tests/_tmp")
    if [[ -n "${BIOM3_CONFIGS_DIR:-}" ]]; then
        MOUNTS+=("${BIOM3_CONFIGS_DIR}:/app/configs:ro")
    fi

    if [[ -n "${BIOM3_BIND_EXTRA:-}" ]]; then
        IFS=, read -ra extra <<< "${BIOM3_BIND_EXTRA}"
    fi
    for spec in ${extra[@]+"${extra[@]}"}; do
        [[ -z "${spec}" ]] && continue
        src="${spec%%:*}"
        if [[ ! -e "${src}" ]]; then
            echo "ERROR: BIOM3_BIND_EXTRA path '${src}' does not exist." >&2
            return 1
        fi
        if [[ "${spec}" == *:* ]]; then
            MOUNTS+=("${spec}")
        else
            MOUNTS+=("${spec}:${spec}:ro")
        fi
    done
    return 0
}
