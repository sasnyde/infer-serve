#!/usr/bin/env python3
"""muse.py — Runpod v2 API helper for muse-serve Pods. Standard library only.

  export RUNPOD_API_KEY=...    # account API key (Runpod console)
  export MUSE_API_KEY=...      # model-serving bearer key (same value as the muse_api_key Secret)

  muse.py list                         pods: id, name, status, gpu, $/h
  muse.py status <pod_id>              one pod, raw fields that matter
  muse.py url <pod_id>                 print the public base URL
  muse.py env <pod_id>                 print the lines for the web app's .env file
  muse.py wait <pod_id> [--chat]       block until RUNNING and /v1/models returns 200; --chat sends one request
  muse.py stop <pod_id>                release the GPU, keep the Pod ID (container disk is wiped)
  muse.py start <pod_id>               boot a stopped Pod (same ID and URL; GPU may be unavailable)
  muse.py terminate <pod_id>           delete the Pod (the global volume is untouched)
  muse.py template <name> <image>      create a Pod template via API (console works too; see runbook)
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://api.runpod.io/v2"


def key(name):
    v = os.environ.get(name, "")
    if not v:
        sys.exit(f"{name} is not set in this shell")
    return v


def call(method, path, body=None, timeout=60):
    req = urllib.request.Request(
        API + path,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={
            "Authorization": f"Bearer {key('RUNPOD_API_KEY')}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            detail = json.loads(raw)
        except ValueError:
            detail = raw
        return e.code, detail


def base_url(pod_id):
    return f"https://{pod_id}-8000.proxy.runpod.net/v1"


def http_code(url, bearer=None, body=None, timeout=30):
    headers = {"Content-Type": "application/json"}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers=headers, method="POST" if body else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except (urllib.error.URLError, TimeoutError) as e:
        return 0, str(e)


def print_env(pod_id, model=None):
    print("Add to the web app .env (the key is the muse_api_key Secret value; it does not change):")
    print(f"MUSE_BASE_URL={base_url(pod_id)}")
    print("MUSE_API_KEY=<value of the muse_api_key Runpod Secret>")
    if model:
        print(f"MUSE_MODEL={model}")


def cmd_env(pod_id):
    bearer = os.environ.get("MUSE_API_KEY")
    model = None
    if bearer:
        code, body = http_code(f"{base_url(pod_id)}/models", bearer)
        if code == 200:
            ids = [m.get("id") for m in json.loads(body).get("data", [])]
            model = ids[0] if ids else None
        else:
            print(f"note: the Pod is not answering yet (HTTP {code}); MUSE_MODEL omitted")
    print_env(pod_id, model)


def cmd_list():
    code, data = call("GET", "/pods")
    if code != 200:
        sys.exit(f"list failed: {code} {data}")
    pods = data.get("pods", [])
    if not pods:
        print("no pods")
        return
    for p in pods:
        gpu = (p.get("gpu") or {}).get("id", "-")
        print(f"{p['id']:14} {p.get('status', '?'):12} {p.get('cost', 0):>6.2f}/h  {gpu:40} {p.get('name', '')}")


def cmd_status(pod_id):
    code, p = call("GET", f"/pods/{pod_id}")
    if code != 200:
        sys.exit(f"status failed: {code} {p}")
    keep = ["id", "name", "status", "actions", "image", "dataCenterId", "cloud", "cost", "ports", "mounts", "startedAt"]
    print(json.dumps({k: p.get(k) for k in keep}, indent=2))
    print("gpu:", json.dumps(p.get("gpu")))
    print("base url:", base_url(pod_id))


def cmd_wait(pod_id, chat=False, timeout_s=2400):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        code, p = call("GET", f"/pods/{pod_id}")
        if code != 200:
            sys.exit(f"status failed: {code} {p}")
        st = p.get("status")
        if st != last:
            print(f"[{int(time.time() - t0):4d}s] pod status: {st}")
            last = st
        if st in ("EXITED", "ERROR", "TERMINATED"):
            sys.exit(f"pod is {st}; check its logs in the Runpod console")
        if st == "RUNNING":
            break
        time.sleep(10)
    else:
        sys.exit("timed out waiting for RUNNING")

    url = base_url(pod_id)
    bearer = key("MUSE_API_KEY")
    print(f"[{int(time.time() - t0):4d}s] container is up; waiting for the model (weights copy + vLLM init)")
    while time.time() - t0 < timeout_s:
        code, body = http_code(f"{url}/models", bearer)
        if code == 200:
            ids = [m.get("id") for m in json.loads(body).get("data", [])]
            print(f"[{int(time.time() - t0):4d}s] READY  models={ids}")
            break
        print(f"[{int(time.time() - t0):4d}s] not ready yet (HTTP {code})")
        time.sleep(15)
    else:
        sys.exit("timed out waiting for /v1/models")

    code, _ = http_code(f"{url}/models")
    print("no-key check:", "ok (401)" if code == 401 else f"FAIL ({code})")
    code, _ = http_code(url.replace("/v1", "/invocations"))
    print("blocked-route check:", "ok (404)" if code == 404 else f"FAIL ({code})")

    if chat:
        model = ids[0]
        code, body = http_code(f"{url}/chat/completions", bearer, timeout=180, body={
            "model": model,
            "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            "max_tokens": 1024,
        })
        try:
            content = json.loads(body)["choices"][0]["message"]["content"]
            print("chat:", "ok ->", content.strip()[:80])
        except (ValueError, KeyError, IndexError):
            print("chat: FAIL", code, body[:300])
    print()
    print_env(pod_id, ids[0] if ids else None)


def cmd_action(pod_id, action):
    if action == "terminate":
        code, data = call("DELETE", f"/pods/{pod_id}")
        print("terminated" if code in (200, 204) else f"terminate failed: {code} {data}")
        return
    code, data = call("POST", f"/pods/{pod_id}/action", {"action": action})
    if code == 200:
        print(f"{action}: status now {data.get('status')}")
    else:
        print(f"{action} failed: {code} {data}")


def cmd_template(name, image):
    body = {
        "name": name,
        "image": image,
        "disk": 100,
        "ports": ["8000/http", "22/tcp"],
        "env": {
            "MUSE_API_KEY": "{{ RUNPOD_SECRET_muse_api_key }}",
            "MUSE_MODE": "serve",
            "MUSE_PROFILE": "muse-fp8-6000",
        },
    }
    code, data = call("POST", "/templates", body)
    if code in (200, 201):
        print("template id:", data.get("id"))
    else:
        print(f"create failed: {code} {json.dumps(data, indent=2)}")
        print("If a field is rejected (422 lists it), create the template in the console with the same values.")


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, args = argv[1], argv[2:]
    if cmd == "list":
        cmd_list()
    elif cmd == "status" and args:
        cmd_status(args[0])
    elif cmd == "url" and args:
        print(base_url(args[0]))
    elif cmd == "env" and args:
        cmd_env(args[0])
    elif cmd == "wait" and args:
        cmd_wait(args[0], chat="--chat" in args)
    elif cmd in ("stop", "start", "terminate") and args:
        cmd_action(args[0], cmd)
    elif cmd == "template" and len(args) == 2:
        cmd_template(args[0], args[1])
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv)
