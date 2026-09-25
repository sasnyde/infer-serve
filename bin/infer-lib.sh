#!/usr/bin/env bash
# Shared functions for entrypoint.sh, infer-seed and infer-init-volume.
# Precedence: profile file < local.env < local.<profile>.env < INFER_OVERRIDE env. Template/deploy env
# supplies INFER_PROFILE, INFER_OVERRIDE, INFER_API_KEY, INFER_MODE and PUBLIC_KEY.

INFER_VOLUME="${INFER_VOLUME:-/workspace}"
INFER_DIR="$INFER_VOLUME/infer"
LOCAL_MODELS="${LOCAL_MODELS:-/tmp/models}"

log() { printf '[infer %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { log "ERROR: $*"; exit 1; }

# Resolve which profile to run: INFER_PROFILE env wins, else the active-profile file.
resolve_profile() {
  if [ -z "${INFER_PROFILE:-}" ] && [ -s "$INFER_DIR/active-profile" ]; then
    INFER_PROFILE="$(tr -d '[:space:]' < "$INFER_DIR/active-profile")"
  fi
  [ -n "${INFER_PROFILE:-}" ] || die "no profile: set INFER_PROFILE or write $INFER_DIR/active-profile"
  PROFILE_FILE="$INFER_DIR/profiles/$INFER_PROFILE.env"
  [ -s "$PROFILE_FILE" ] || die "profile file missing: $PROFILE_FILE (run infer-init-volume, or check the name)"
  set -a
  # shellcheck disable=SC1090
  . "$PROFILE_FILE"
  local ov
  for ov in "$INFER_DIR/local.env" "$INFER_DIR/local.$INFER_PROFILE.env"; do
    if [ -s "$ov" ]; then
      # shellcheck disable=SC1090
      . "$ov"
      log "applied $(basename "$ov")"
    fi
  done
  # Deploy-form overrides, applied last: INFER_OVERRIDE="MAX_SEQS=8 GPU_UTIL=0.85"
  if [ -n "${INFER_OVERRIDE:-}" ]; then
    local kv
    for kv in $INFER_OVERRIDE; do
      case "$kv" in *=*) export "$kv"; log "override $kv" ;; *) die "bad INFER_OVERRIDE entry: $kv" ;; esac
    done
  fi
  set +a
  [ -n "${MODEL_NAME:-}" ] || die "profile $INFER_PROFILE sets no MODEL_NAME"
  [ -n "${MODEL_REPO:-}" ] || die "profile $INFER_PROFILE sets no MODEL_REPO"
  log "profile=$INFER_PROFILE model=$MODEL_NAME repo=$MODEL_REPO"
}

# Resolve the serving key: Runpod Secret (INFER_API_KEY) wins; the legacy key file is the fallback.
resolve_key() {
  case "${INFER_API_KEY:-}" in
    ""|*RUNPOD_SECRET*) INFER_API_KEY="" ;;
  esac
  if [ -n "$INFER_API_KEY" ]; then
    KEY_SOURCE=secret
  elif [ -s "$INFER_VOLUME/muse-api-key" ]; then
    INFER_API_KEY="$(tr -d '\r\n' < "$INFER_VOLUME/muse-api-key")"
    KEY_SOURCE=file
  else
    die "no API key: map the muse_api_key Secret in the template, or keep $INFER_VOLUME/muse-api-key"
  fi
  [ -n "$INFER_API_KEY" ] || die "API key resolved to an empty string"
  export INFER_API_KEY VLLM_API_KEY="$INFER_API_KEY"
  log "api key source: $KEY_SOURCE (${#INFER_API_KEY} chars)"
}

start_sshd() {
  if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p /root/.ssh /run/sshd
    chmod 700 /root/.ssh
    printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    ssh-keygen -A >/dev/null 2>&1 || true
    /usr/sbin/sshd
    log "sshd started"
  else
    log "PUBLIC_KEY not set; sshd not started (enable SSH terminal access on the deploy form)"
  fi
}

# Download MODEL_REPO into $1 (a local directory) using the Hub snapshot API.
download_model() {
  local dst="$1"
  python3 - "$MODEL_REPO" "$dst" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dst = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo, local_dir=dst)
print("downloaded", repo, "->", dst)
PY
}

# Stage the model onto local disk. Order: volume mirror -> local; else Hub -> local -> volume mirror.
# The volume is a FUSE object store: vLLM must never open safetensors on it directly.
stage_model() {
  local src="$INFER_DIR/models/$MODEL_NAME" dst="$LOCAL_MODELS/$MODEL_NAME"
  mkdir -p "$LOCAL_MODELS"
  if [ -d "$dst" ] && [ -f "$dst/.complete" ]; then
    log "local copy already present: $dst"
  elif [ -f "$src/.complete" ]; then
    log "copying $MODEL_NAME from volume mirror (sequential read)"
    rm -rf "$dst"
    local t0; t0=$(date +%s)
    cp -r "$src" "$dst"
    log "copy took $(( $(date +%s) - t0 ))s; $(du -sh "$dst" | cut -f1)"
  else
    log "no complete mirror at $src; downloading $MODEL_REPO from the Hub"
    rm -rf "$dst"
    local t0; t0=$(date +%s)
    download_model "$dst"
    touch "$dst/.complete"
    log "download took $(( $(date +%s) - t0 ))s; $(du -sh "$dst" | cut -f1)"
    if [ "${INFER_MIRROR:-1}" = "1" ]; then
      log "mirroring to volume: $src"
      rm -rf "$src"
      mkdir -p "$INFER_DIR/models"
      cp -r "$dst" "$src"
      log "mirror complete"
    fi
  fi
  export MODEL_PATH="$dst"
  # Everything vLLM needs is local now; keep Hub calls out of the serving path.
  export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1
}

build_vllm_args() {
  VLLM_ARGS=(
    "$MODEL_PATH"
    --served-model-name "${SERVED_MODEL_NAME:-$MODEL_REPO}"
    --host 127.0.0.1 --port 8002
    --tensor-parallel-size "${TP:-1}"
    --gpu-memory-utilization "${GPU_UTIL:-0.63}"
    --max-model-len "${MAX_LEN:-32768}"
    --max-num-seqs "${MAX_SEQS:-10}"
    --generation-config auto
  )
  if [ "${KV_CACHE_DTYPE:-auto}" != "auto" ]; then VLLM_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE"); fi
  if [ "${ENFORCE_EAGER:-0}" = "1" ]; then VLLM_ARGS+=(--enforce-eager); fi
  if [ "${TRUST_REMOTE_CODE:-0}" = "1" ]; then VLLM_ARGS+=(--trust-remote-code); fi
  if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    local extra=( ${EXTRA_ARGS} )
    VLLM_ARGS+=("${extra[@]}")
  fi
}
