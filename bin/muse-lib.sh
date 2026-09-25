#!/usr/bin/env bash
# Shared functions for entrypoint.sh, muse-seed and muse-init-volume.
# Precedence: profile file < local.env < local.<profile>.env < MUSE_OVERRIDE env. Template/deploy env
# supplies MUSE_PROFILE, MUSE_OVERRIDE, MUSE_API_KEY, MUSE_MODE and PUBLIC_KEY.

MUSE_VOLUME="${MUSE_VOLUME:-/workspace}"
MUSE_DIR="$MUSE_VOLUME/muse"
LOCAL_MODELS="${LOCAL_MODELS:-/tmp/models}"

log() { printf '[muse %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { log "ERROR: $*"; exit 1; }

# Resolve which profile to run: MUSE_PROFILE env wins, else the active-profile file.
resolve_profile() {
  if [ -z "${MUSE_PROFILE:-}" ] && [ -s "$MUSE_DIR/active-profile" ]; then
    MUSE_PROFILE="$(tr -d '[:space:]' < "$MUSE_DIR/active-profile")"
  fi
  [ -n "${MUSE_PROFILE:-}" ] || die "no profile: set MUSE_PROFILE or write $MUSE_DIR/active-profile"
  PROFILE_FILE="$MUSE_DIR/profiles/$MUSE_PROFILE.env"
  [ -s "$PROFILE_FILE" ] || die "profile file missing: $PROFILE_FILE (run muse-init-volume, or check the name)"
  set -a
  # shellcheck disable=SC1090
  . "$PROFILE_FILE"
  local ov
  for ov in "$MUSE_DIR/local.env" "$MUSE_DIR/local.$MUSE_PROFILE.env"; do
    if [ -s "$ov" ]; then
      # shellcheck disable=SC1090
      . "$ov"
      log "applied $(basename "$ov")"
    fi
  done
  # Deploy-form overrides, applied last: MUSE_OVERRIDE="MAX_SEQS=8 GPU_UTIL=0.85"
  if [ -n "${MUSE_OVERRIDE:-}" ]; then
    local kv
    for kv in $MUSE_OVERRIDE; do
      case "$kv" in *=*) export "$kv"; log "override $kv" ;; *) die "bad MUSE_OVERRIDE entry: $kv" ;; esac
    done
  fi
  set +a
  [ -n "${MODEL_NAME:-}" ] || die "profile $MUSE_PROFILE sets no MODEL_NAME"
  [ -n "${MODEL_REPO:-}" ] || die "profile $MUSE_PROFILE sets no MODEL_REPO"
  log "profile=$MUSE_PROFILE model=$MODEL_NAME repo=$MODEL_REPO"
}

# Resolve the serving key: Runpod Secret (MUSE_API_KEY) wins; the legacy key file is the fallback.
resolve_key() {
  case "${MUSE_API_KEY:-}" in
    ""|*RUNPOD_SECRET*) MUSE_API_KEY="" ;;
  esac
  if [ -n "$MUSE_API_KEY" ]; then
    KEY_SOURCE=secret
  elif [ -s "$MUSE_VOLUME/muse-api-key" ]; then
    MUSE_API_KEY="$(tr -d '\r\n' < "$MUSE_VOLUME/muse-api-key")"
    KEY_SOURCE=file
  else
    die "no API key: map the muse_api_key Secret in the template, or keep $MUSE_VOLUME/muse-api-key"
  fi
  [ -n "$MUSE_API_KEY" ] || die "API key resolved to an empty string"
  export MUSE_API_KEY VLLM_API_KEY="$MUSE_API_KEY"
  log "api key source: $KEY_SOURCE (${#MUSE_API_KEY} chars)"
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
  local src="$MUSE_DIR/models/$MODEL_NAME" dst="$LOCAL_MODELS/$MODEL_NAME"
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
    if [ "${MUSE_MIRROR:-1}" = "1" ]; then
      log "mirroring to volume: $src"
      rm -rf "$src"
      mkdir -p "$MUSE_DIR/models"
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
