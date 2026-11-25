# FurnForge AI — Comprehensive Deployment & Operations Guide

This guide covers everything required to deploy, operate, and present **FurnForge AI** across local machines, Docker environments, and cloud GPU infrastructure.

---

## Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │          Browser Client UI             │
                      │  (Vercel SPA or Docker Nginx on :5173) │
                      └──────────────────┬─────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │  Dynamic Backend URL Resolution Layer        │
                 │  1. ?backend=https://... URL query parameter  │
                 │  2. localStorage ('furnforge_dynamic_backend')│
                 │  3. Fallback: VITE_BACKEND_URL / relative /api│
                 └───────────────────────┬───────────────────────┘
                                         │
             ┌───────────────────────────┼───────────────────────────┐
             │                           │                           │
             ▼                           ▼                           ▼
   ┌───────────────────┐       ┌───────────────────┐       ┌───────────────────┐
   │  Local Docker     │       │ Cloudflare Tunnel │       │ Cloud Serverless  │
   │  Compose Backend  │       │ (Live Demos)      │       │ (Modal / RunPod)  │
   │  http://localhost:│       │ https://*.try     │       │ Dedicated GPU     │
   │  8005             │       │ cloudflare.com    │       │ Instance          │
   └─────────┬─────────┘       └─────────┬─────────┘       └─────────┬─────────┘
             │                           │                           │
             └───────────────────────────┼───────────────────────────┘
                                         │
                                         ▼
                      ┌─────────────────────────────────────┐
                      │          FastAPI Backend            │
                      │   - OpenCV Bilateral / Canny Edge   │
                      │   - ResNet CNN + Grad-CAM Extractor │
                      │   - SD 1.5 + ControlNet Scribble    │
                      │   - TripoSR / Hunyuan3D-2 3D Engine │
                      └──────────────────┬──────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
      ┌─────────────────────┐                         ┌─────────────────────┐
      │ Persistent Volumes  │                         │ Cloudflare R2 / S3  │
      │ /uploads  /outputs  │                         │ Zero-Egress Storage │
      │ /.hf_cache (Weights)│                         │ Photoreal + 3D GLBs │
      └─────────────────────┘                         └─────────────────────┘
```

---

## 1. One-Command Local Docker Compose Deployment

The repository includes a ready-to-run multi-container setup orchestrating the Vite frontend (served via production Nginx) and the FastAPI backend.

### Quick Start (CPU Mode)

```bash
# Clone and enter directory
git clone https://github.com/whysooraj/FurnForgeAI.git
cd FurnForgeAI

# Launch containers in detached mode
docker compose up --build -d
```

Once running:
- **Web UI:** [http://localhost:5173](http://localhost:5173) (also available on [http://localhost:8080](http://localhost:8080))
- **FastAPI Backend & Swagger Docs:** [http://localhost:8005/docs](http://localhost:8005/docs)
- **Health Check:** `curl http://localhost:8005/api/health`

### GPU Accelerated Mode (NVIDIA CUDA)

To enable GPU acceleration for Stable Diffusion 1.5 and TripoSR:

1. Ensure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed on the host.
2. In `docker-compose.yml`, change `DEVICE=cpu` to `DEVICE=cuda`.
3. Uncomment the `deploy.resources.reservations.devices` block in `docker-compose.yml`:

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

4. Rebuild and launch:
```bash
docker compose up --build -d
```

### Volume Persistence & Hugging Face Cache

The `docker-compose.yml` mounts:
- `./backend/uploads`: User uploaded sketches.
- `./backend/outputs`: Generated Canny edge maps, photoreal diffusion renders, and `.glb` / `.obj` 3D files.
- `furnforge_hf_cache`: Named volume caching Hugging Face model weights (`runwayml/stable-diffusion-v1-5`, `lllyasviel/sd-controlnet-scribble`, `stabilityai/TripoSR`). **This prevents re-downloading multi-gigabyte models on container restarts.**

---

## 2. Dynamic Backend URL Switching (Zero-Redeploy Presentation Setup)

When presenting FurnForge AI to evaluators or professors, the frontend is typically hosted on **Vercel** (`https://furn-forge-ai.vercel.app`), but generating photoreal renders and 3D models requires an active GPU.

You do **not** need to redeploy the frontend every time your backend IP or tunnel changes.

### Method A: One-Click URL Query Parameter (Recommended for Sharing)

Append `?backend=<tunnel_url>` to your Vercel URL:

```
https://furn-forge-ai.vercel.app?backend=https://your-tunnel.trycloudflare.com
```

When loaded:
1. The frontend extracts the query parameter.
2. Sanitizes and saves it into browser `localStorage` under `furnforge_dynamic_backend`.
3. All subsequent API calls, asset URLs, and 3D model loaders immediately route to that backend.

### Method B: Interactive UI Configurator

1. In the top navigation bar, click either:
   - The status pill (e.g. `Engine: CV/CNN Active` or `Backend Offline`).
   - The **Settings / Gear icon** right inside the status badge.
2. A configuration modal will open displaying:
   - Current active backend endpoint.
   - An input field for your new URL or tunnel.
   - Quick presets: **Localhost (http://localhost:8005)** and **Reset to Default**.
   - **Test Connection** button: Pings `/api/health`, measures roundtrip latency, and confirms device (`cuda` vs `cpu`).
   - Click **Save & Connect** to instantly activate the backend without page reload.

### Creating a Free Public Tunnel in 10 Seconds

Run any of the following tools on the machine hosting your backend:

```bash
# Option 1: Cloudflare Quick Tunnel (Free, no account needed)
cloudflared tunnel --url http://localhost:8005

# Option 2: Localtunnel
npx localtunnel --port 8005

# Option 3: ngrok
ngrok http 8005
```

Copy the generated HTTPS URL (e.g. `https://random-words.trycloudflare.com`) and paste it into the UI configurator or append as `?backend=...`.

---

## 3. Production Serverless GPU Deployment

For production deployments where a GPU is only billed while actively generating images or 3D models:

### A. RunPod Serverless / Pods

1. **Deploying a GPU Pod:**
   - Base Image: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
   - Recommended GPU: **NVIDIA RTX 3090, RTX 4090, or A4000** (>= 12GB VRAM).
   - Port: Expose HTTP `8005`.
   - Network Volume: Attach a network volume mounted at `/app/backend/.hf_cache` to persist downloaded weights across pod reboots.

2. **Clone & Run:**
   ```bash
   git clone https://github.com/whysooraj/FurnForgeAI.git
   cd FurnForgeAI/backend
   pip install -r requirements.txt
   uvicorn app.main:app --host 0.0.0.0 --port 8005
   ```

3. Enable RunPod's HTTP Proxy port forwarding to receive a public HTTPS endpoint.

### B. Modal Serverless Deployment (`modal_app.py`)

Modal allows zero-idle-cost GPU deployment with cold-start caching:

```python
import modal

app = modal.App("furnforge-ai-backend")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "libgomp1", "git")
    .pip_install_from_requirements("backend/requirements.txt")
)

volume = modal.Volume.from_name("furnforge-weights", create_if_missing=True)

@app.function(
    gpu="A10G",
    image=image,
    volumes={"/app/backend/.hf_cache": volume},
    timeout=600,
)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.append("/app/backend")
    from app.main import app as web_app
    return web_app
```

Deploy with:
```bash
modal deploy modal_app.py
```
Modal provides an instant, production-grade HTTPS endpoint with automatic scale-to-zero.

---

## 4. Persistent Cloud Object Storage (Cloudflare R2 / AWS S3)

To persist generated 2D photoreal renders and 3D GLB models indefinitely without exhausting container disk space:

### Cloudflare R2 Configuration

Cloudflare R2 is S3-compatible and features **zero egress fees**, making it ideal for serving 3D `.glb` assets and high-res renders.

1. Create a bucket in Cloudflare R2 (e.g. `furnforge-assets`).
2. Generate an S3 API Token with Read/Write access.
3. Configure the following environment variables in `backend/.env` or Docker:

```env
S3_ENABLED=true
S3_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com
S3_ACCESS_KEY_ID=<your_r2_access_key>
S3_SECRET_ACCESS_KEY=<your_r2_secret_key>
S3_BUCKET_NAME=furnforge-assets
S3_PUBLIC_DOMAIN=https://pub-<hash>.r2.dev
```

### Storage Integration Pattern

When S3/R2 is configured, the backend automatically uploads output images and GLB files:

```python
import boto3
from app.config import S3_ENDPOINT_URL, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_BUCKET_NAME, S3_PUBLIC_DOMAIN

s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT_URL,
    aws_access_key_id=S3_ACCESS_KEY_ID,
    aws_secret_access_key=S3_SECRET_ACCESS_KEY,
)

def upload_artifact(local_file_path: str, object_key: str, content_type: str) -> str:
    s3_client.upload_file(
        local_file_path,
        S3_BUCKET_NAME,
        object_key,
        ExtraArgs={"ContentType": content_type}
    )
    return f"{S3_PUBLIC_DOMAIN}/{object_key}"
```

---

## 5. Health Monitoring & Troubleshooting

### Diagnostic Endpoints

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/api/health` | `GET` | Returns service status, GPU device, CNN readiness, and 3D mesh engine state. |
| `/api/3d/requirements` | `GET` | Reports CUDA GPU VRAM, PyTorch version, and 3D reconstruction availability. |
| `/api/samples` | `GET` | Returns preloaded furniture sketches. |

### Common Operational Issues

1. **CORS Errors When Using Tunnels:**
   - The FastAPI backend includes `CORSMiddleware` configured with `allow_origins=["*"]`. If CORS errors persist, ensure your tunnel provider (e.g. ngrok or Cloudflare) is not injecting an interstitial warning page. For Cloudflare tunnels, pass `--url http://localhost:8005`.
2. **CUDA Out of Memory (OOM):**
   - In `backend/app/config.py`, enable `ENABLE_LCM=true` (Latent Consistency Model) to reduce sampling steps from 25 down to 4–8 iterations.
   - For 3D reconstruction with TripoSR, a minimum of 6GB VRAM is recommended. On machines with limited VRAM, TripoSR automatically optimizes chunk sizes.
3. **Frontend Asset 404s Behind Reverse Proxy:**
   - When using Nginx, ensure trailing slashes match between `location /outputs/` and `proxy_pass http://backend:8005/outputs/`.
