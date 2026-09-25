#!/usr/bin/env bash
# Container entrypoint. MUSE_MODE=serve (default) stages the model and serves it.
# MUSE_MODE=setup starts sshd and waits, for seeding the volume or quantizing.
set -Eeuo pipefail
# shellcheck disable=SC1091
. /usr/local/lib/muse/muse-lib.sh

exec > >(tee -a /tmp/muse.log) 2>&1
log "muse-serve starting; mode=${MUSE_MODE:-serve}; gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
[ -d "$MUSE_VOLUME" ] || die "volume not mounted at $MUSE_VOLUME"

start_sshd

if [ "${MUSE_MODE:-serve}" = "setup" ]; then
  log "setup mode: no model server. SSH in, then run muse-init-volume / muse-seed <profile>."
  exec sleep infinity
fi

resolve_profile
resolve_key
stage_model

nginx -t
nginx
log "nginx listening on 0.0.0.0:8000 -> vLLM 127.0.0.1:8002"

build_vllm_args
log "vllm serve ${VLLM_ARGS[*]}"
log "env: DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-} MOE_DEEP_GEMM=${VLLM_MOE_USE_DEEP_GEMM:-} FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-} TRITON_FP8_GEMM=${VLLM_USE_TRITON_FP8_GEMM:-}"
# vLLM becomes PID 1. If it exits, the container exits and the Pod shows EXITED instead of a silent 502.
exec vllm serve "${VLLM_ARGS[@]}"
