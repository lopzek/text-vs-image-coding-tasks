#!/bin/bash
# setup_vllm_env.sh — build the serving environment (.venv-vllm) exactly from the lock.
#
# vLLM is the inference backend of every GPU run. It pins its own torch (cu129) and
# transformers, so it lives in a separate Python 3.12 environment; the code is used from
# this environment through PYTHONPATH (run_cell.py sets it), nothing is pip-installed from
# the package.
#
#   VLLM_COMMIT_FULL=<40-hex vLLM commit> bash setup_vllm_env.sh --from-lock [--venv NAME] [--wheelhouse DIR] [--dry-run]
#
# The lock pins a vLLM nightly build (vllm==<version>+g<short sha>.cu129), which is served
# only from the per-commit index wheels.vllm.ai/<40-hex commit>/cu129; the rolling nightly
# index keeps only the newest build. VLLM_COMMIT_FULL gives that commit (the README has it);
# without it the script asks api.github.com for the full id of the lock's short sha.
# A wheel that has disappeared from its index can be placed in the wheelhouse directory
# (default wheelhouse-vllm/), which is searched before the indexes.
# The script refuses to finish unless the installed set matches the lock line for line.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO}"

DRY_RUN=0; VENV=.venv-vllm; WHEELHOUSE=wheelhouse-vllm
while [ $# -gt 0 ]; do
    case "$1" in
        --from-lock) : ;;   # the only mode; accepted for the README's invocation
        --dry-run)   DRY_RUN=1 ;;
        --venv|--wheelhouse)
            [ $# -ge 2 ] || { echo "[setup-vllm] ERROR: $1 needs a value" >&2; exit 2; }
            if [ "$1" = "--venv" ]; then VENV="$2"; else WHEELHOUSE="$2"; fi
            shift ;;
        -h|--help) sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "[setup-vllm] ERROR: unknown argument: $1" >&2
           echo "[setup-vllm] usage: bash setup_vllm_env.sh --from-lock [--venv NAME] [--wheelhouse DIR] [--dry-run]" >&2
           exit 2 ;;
    esac
    shift
done

lock="requirements-vllm.lock.txt"
idx_args=()
[ -f "${lock}" ] || { echo "[setup-vllm] ERROR: ${lock} not found in ${REPO}" >&2; exit 2; }
vline="$(grep -E '^vllm==' "${lock}" | head -1)"
[ -n "${vline}" ] || { echo "[setup-vllm] ERROR: no 'vllm==' line in ${lock}" >&2; exit 2; }
tline="$(grep -E '^torch==' "${lock}" | head -1)"
cu="$(printf '%s' "${tline}" | grep -oE '\+cu[0-9]+' | tr -d '+')"

short="$(printf '%s' "${vline}" | grep -oE '\+g[0-9a-f]{7,40}' | sed 's/^+g//')"
if [ -n "${short}" ]; then
    vcu="$(printf '%s' "${vline}" | grep -oE '\.cu[0-9]+$' | tr -d '.')"
    full="${VLLM_COMMIT_FULL:-}"
    if [ -z "${full}" ]; then
        full="$(curl -s --max-time 30 "https://api.github.com/repos/vllm-project/vllm/commits/${short}" \
            | grep -m1 -oE '"sha": *"[0-9a-f]{40}"' | grep -oE '[0-9a-f]{40}')"
    fi
    if ! printf '%s' "${full}" | grep -qE '^[0-9a-f]{40}$'; then
        echo "[setup-vllm] ERROR: need the 40-hex vLLM commit for +g${short} (got '${full}')." \
             "Set VLLM_COMMIT_FULL=<40-hex sha>." >&2
        exit 2
    fi
    case "${full}" in
        "${short}"*) : ;;
        *) echo "[setup-vllm] ERROR: VLLM_COMMIT_FULL=${full} does not match the lock's +g${short}" >&2
           exit 2 ;;
    esac
    idx_args+=(--extra-index-url "https://wheels.vllm.ai/${full}/${vcu:-${cu}}")
fi
[ -n "${cu}" ] && idx_args+=(--extra-index-url "https://download.pytorch.org/whl/${cu}")
[ -d "${WHEELHOUSE}" ] && idx_args+=(--find-links "${WHEELHOUSE}")

venv_cmd=(uv venv "${VENV}" --python 3.12)
pip_cmd=(uv pip install -r "${lock}" --python "${VENV}/bin/python"
         "${idx_args[@]}" --index-strategy unsafe-best-match)
if [ "${DRY_RUN}" -eq 1 ]; then
    echo "[setup-vllm] DRY RUN (nothing installed):"
    echo "  ${venv_cmd[*]}"
    echo "  ${pip_cmd[*]}"
    exit 0
fi
# An environment that may have produced results is never rebuilt in place.
if [ -e "${VENV}" ]; then
    echo "[setup-vllm] ERROR: ${VENV} already exists — build under a new name (--venv)." >&2
    exit 2
fi
"${venv_cmd[@]}" || exit 1
"${pip_cmd[@]}" || { echo "[setup-vllm] ERROR: install from ${lock} failed" >&2; exit 1; }
"${VENV}/bin/python" -c "import torch,transformers,vllm,tokenizers; \
print('torch',torch.__version__,'| transformers',transformers.__version__, \
'| vllm',vllm.__version__,'| tokenizers',tokenizers.__version__)" || exit 1
PYTHONPATH="${REPO}" "${VENV}/bin/python" -c "
import vllm, codecompress.main
print('[setup-vllm] OK: codecompress imports and vLLM is importable')
" || exit 1
ndiff="$(diff <(sort "${lock}") <(uv pip freeze --python "${VENV}/bin/python" | sort) | grep -c '^[<>]')"
if [ "${ndiff}" -ne 0 ]; then
    echo "[setup-vllm] ERROR: ${VENV} differs from ${lock} in ${ndiff} line(s)" >&2
    exit 1
fi
echo "[setup-vllm] done: ${VENV} matches ${lock} exactly."
