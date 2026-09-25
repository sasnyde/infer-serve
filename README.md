# infer-serve

vLLM + Nginx gateway image for Runpod Pods (formerly muse-serve) with a global volume at /workspace.
See the runbook HTML for the full process. Layout:

    Dockerfile                image: vllm/vllm-openai + nginx + sshd + scripts
    nginx.conf                two-route gateway on 8000 -> vLLM 127.0.0.1:8002
    bin/entrypoint.sh         serve (default) or INFER_MODE=setup
    bin/infer-lib.sh           profile/key/model-staging functions
    bin/infer-seed             download + mirror one profile's model to the volume
    bin/infer-init-volume      create /workspace/infer layout, install profiles
    bin/infer-check            local gateway checks (401/200/404/chat)
    profiles/*.env            model x GPU settings; copied to /workspace/infer/profiles once
    mac/infer.py               Runpod API helper: list/wait/stop/start/terminate
