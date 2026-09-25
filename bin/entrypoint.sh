#!/usr/bin/env bash
# Container entrypoint. INFER_MODE=serve (default) stages the model and serves it.
# INFER_MODE=setup starts sshd and waits, for seeding the volume or quantizing.
set -Eeuo pipefail
# shellcheck disable=SC1091
. /usr/local/lib/infer/infer-lib.sh

exec > >(tee -a /tmp/infer.log) 2>&1
log "infer-serve starting; mode=${INFER_MODE:-serve}; gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
[ -d "$INFER_VOLUME" ] || die "volume not mounted at $INFER_VOLUME"

start_sshd

if [ "${INFER_MODE:-serve}" = "setup" ]; then
  log "setup mode: no model server. SSH in, then run infer-init-volume / infer-seed <profile>."
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
