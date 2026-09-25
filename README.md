# muse-serve

vLLM + Nginx gateway image for Runpod Pods with a global volume at /workspace.
See the runbook HTML for the full process. Layout:

    Dockerfile                image: vllm/vllm-openai + nginx + sshd + scripts
    nginx.conf                two-route gateway on 8000 -> vLLM 127.0.0.1:8002
    bin/entrypoint.sh         serve (default) or MUSE_MODE=setup
    bin/muse-lib.sh           profile/key/model-staging functions
    bin/muse-seed             download + mirror one profile's model to the volume
    bin/muse-init-volume      create /workspace/muse layout, install profiles
    bin/muse-check            local gateway checks (401/200/404/chat)
    profiles/*.env            model x GPU settings; copied to /workspace/muse/profiles once
    mac/muse.py               Runpod API helper: list/wait/stop/start/terminate
