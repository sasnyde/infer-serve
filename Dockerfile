# muse-serve — vLLM + Nginx gateway for Runpod Pods (RTX PRO 6000 Blackwell / RTX 5090, sm120)
# Build for linux/amd64. Pinned base: the vLLM version verified on the 6000 PRO deployment.
ARG VLLM_TAG=v0.30.0
FROM vllm/vllm-openai:${VLLM_TAG}

ARG TRANSFORMERS_PIN=5.17.0
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends nginx openssh-server curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && rm -f /etc/nginx/sites-enabled/default \
 && mkdir -p /run/sshd /tmp/models

# Match the verified Pod's transformers pin and add hf_transfer for faster cold downloads.
RUN uv pip install --system "transformers==${TRANSFORMERS_PIN}" hf_transfer

COPY nginx.conf /etc/nginx/nginx.conf
COPY bin/muse-lib.sh /usr/local/lib/muse/muse-lib.sh
COPY bin/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY bin/muse-seed /usr/local/bin/muse-seed
COPY bin/muse-check /usr/local/bin/muse-check
COPY bin/muse-init-volume /usr/local/bin/muse-init-volume
COPY profiles/ /opt/muse/profiles/
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/muse-seed /usr/local/bin/muse-check /usr/local/bin/muse-init-volume \
 && nginx -t

ENV MUSE_VOLUME=/workspace \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    VLLM_USE_DEEP_GEMM=0 \
    VLLM_MOE_USE_DEEP_GEMM=0 \
    VLLM_USE_FLASHINFER_SAMPLER=0

EXPOSE 8000 22
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
