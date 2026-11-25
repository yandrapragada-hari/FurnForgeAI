# FurnForge AI – Sketch-to-Photoreal Furniture & 3D Studio 🛋️✨

[![FurnForge AI CI](https://github.com/whysooraj/FurnForgeAI/actions/workflows/ci.yml/badge.svg)](https://github.com/whysooraj/FurnForgeAI/actions/workflows/ci.yml)
[![Vercel Deployment](https://img.shields.io/badge/Vercel-Live%20App-black?logo=vercel)](https://frontend-blush-five-74.vercel.app)
[![Cloudflare Pages](https://img.shields.io/badge/Cloudflare%20Pages-Live%20App-F38020?logo=cloudflare)](https://furnforge-ai.pages.dev)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![React 19](https://img.shields.io/badge/React-19-61DAFB.svg?logo=react)](https://react.dev/)
[![Three.js](https://img.shields.io/badge/Three.js-WebGL-black.svg?logo=three.js)](https://threejs.org/)

> 📖 **Comprehensive Guides & Deep Technical Documentation:**
> - [EVALUATION_REPORT.md](./EVALUATION_REPORT.md) — Academic Evaluation Report, Edge-IoU Benchmarks, Chamfer Distances & Ablation Studies.
> - [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) — Production Deployment Manual, Docker Compose, Modal Serverless GPU & Cloudflare Named Tunnels.
> - [PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md) — Complete in-depth technical blueprint, mathematical formulations, and benchmarks.
> - [WORKFLOW_ARCHITECTURE.md](./WORKFLOW_ARCHITECTURE.md) — Detailed step-by-step workflow pipeline, dataflows, and component architecture.
> - [LEARN.md](./LEARN.md) — Generative AI, Computer Vision, Diffusion Mechanics, and 3D Neural Reconstruction learning manual.

**FurnForge AI** is an advanced Generative AI, Computer Vision, and 3D Neural Reconstruction platform that transforms hand-drawn furniture sketches into photorealistic, showroom-ready furniture renders and watertight interactive 3D assets.

Designed for university capstone showcases and industrial product design demonstrations, FurnForge AI overcomes the spatial hallucination and geometric distortion of naive text-to-image foundation models by enforcing structural constraint guidance via **Classical Computer Vision (OpenCV)**, **Deep Feature Extraction (ResNet-50 + Grad-CAM)**, **Spatial ControlNet (Canny/Scribble)**, **Latent Diffusion (SD 1.5 + LCM-LoRA)**, and **Neural 3D Mesh Generation (TripoSR / Hunyuan3D)**.

---

## 🌐 Live Production Deployments

The client application is deployed across global edge networks with transparent reverse-proxy API integration:

- **Primary Vercel Production:** [https://frontend-blush-five-74.vercel.app](https://frontend-blush-five-74.vercel.app)
- **Cloudflare Pages Production:** [https://furnforge-ai.pages.dev](https://furnforge-ai.pages.dev)
- **Permanent Backend Gateway:** `https://furnforge-api.sooraj.fun`

---

## 🏛️ End-to-End Multimodal Pipeline & Workflow

```
+───────────────────────────────────────────────────────────────────────────────────────────+
│                                  FURNFORGE AI ARCHITECTURE                                │
+───────────────────────────────────────────────────────────────────────────────────────────+
  1. Sketch Ingestion (3 Modalities)
     ├── Option 1: Curated Presets (Chair, Sofa, Table, Bed, Cabinet)
     ├── Option 2: File Upload (PNG, JPG, WebP up to 10MB)
     └── Option 3: Interactive Whiteboard Studio (SketchCanvas.jsx)
          ├── Real-time Marker Pen & Eraser (Round joint stroke smoothing)
          ├── 1:1 Screen Coordinate Mapping (Mouse pointer = drawing marker nib)
          ├── True Fullscreen Canvas & Responsive Layout
          └── Auto-Bounding Box Segmentation & 512x512 Centered HD Export
          │
          ▼
  2. OpenCV Vision Preprocessing (cv_preprocessor.py)
     ├── Flat-Field Background Division (Eliminates Paper Shadows)
     ├── Bilateral Filtering (Smoothes Paper Grain, Preserves Pencil Edges)
     ├── CLAHE Local Dynamic Histogram Equalization
     ├── Canny Edge Detection & Scribble Inversion
     └── Aspect-Ratio Preserved Letterbox Padding (512x512)
          │
          ├───► 3. Deep Topology CNN (ResNet-50 Backbone)
          │        ├── 2048-dim Latent Feature Vector (layer4)
          │        ├── Topological Symmetry Score (L1 Horizontal Flip)
          │        ├── Aspect Ratio & Hu Invariant Moments (h1 - h7)
          │        ├── Furniture Category Prior Probabilities & Auto-Alignment
          │        └── Grad-CAM 262,144-pixel Thermal Attention Heatmap
          │
          ▼
  4. ControlNet Spatial Guidance (lllyasviel/control_v11p_sd15_canny)
     ├── Zero-convolutions lock exact sketch outlines and joinery
     └── Injected directly into U-Net skip-connections
          │
          ▼
  5. Latent Consistency Diffusion (SD 1.5 + LCM-LoRA)
     ├── PBR Material Synthesis (Walnut, Brushed Brass, Aniline Leather, Bouclé)
     ├── 4 to 8 Denoising Steps via Consistency Distillation Trajectory
     └── Architectural Studio Lighting, Ambient Occlusion & Contact Shadows
          │
          ├───► 6. Photorealistic 2D Render (Gallery, Comparison Slider, Variations)
          │
          ▼
  7. 3D Neural Reconstruction (mesh3d_engine.py & stabilityai/TripoSR)
     ├── Vision Transformer Triplane Density Evaluation
     ├── Marching Cubes Isosurface Mesh Extraction (160x160x160 Grid)
     ├── Rembg Alpha Matte Background Removal
     └── Interactive 3D WebGL Orbit Canvas (.GLB / .OBJ Download)
```

---

## 📊 Academic Evaluation & Quantitative Benchmarks

FurnForge AI incorporates an automated mathematical evaluation suite located in `backend/evaluation/`. Full mathematical equations, LaTeX tables, and empirical proofs are documented in [EVALUATION_REPORT.md](./EVALUATION_REPORT.md).

### Ablation Study Summary (Proving Why Every Component Exists)

| Configuration | ControlNet | OpenCV Clean | ResNet Prompt | Edge-IoU ($\tau=2$px) | Chamfer Edge Dist | CLIP Score | GPU Latency | Watertightness % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config 1: Vanilla SD 1.5 (Text Only)** | ✗ | ✗ | ✗ | **0.084** | 48.6 px | 0.742 | 3.4s | 62.0% |
| **Config 2: Standard Img2Img (VAE Noise)** | ✗ | ✗ | ✗ | **0.291** | 29.4 px | 0.685 | 3.8s | 68.5% |
| **Config 3: Raw Sketch + ControlNet** | ✓ | ✗ | ✗ | **0.518** | 16.8 px | 0.768 | 5.1s | 81.0% |
| **Config 4: FurnForge Full Pipeline** | ✓ | ✓ | ✓ | **0.785** | **6.2 px** | **0.884** | **5.6s** | **98.5%** |

- **Edge-IoU ($\tau=2$px):** Morphological intersection-over-union between user sketch contours and generated render edges ($+834\%$ improvement over vanilla text-to-image).
- **Chamfer Distance ($d_{CD}$):** Sub-10px Euclidean alignment between input pencil curves and output object borders.
- **3D Topology:** Generates closed 2-manifold surfaces with Euler characteristic $\chi = 2$ ($g = 0$ sphere topology) suitable for 3D printing.

---

## 🔌 Production Architecture: Transparent Reverse Proxy

In production web applications, the client **never hardcodes an external backend URL or uses query parameters**. FurnForge AI utilizes an enterprise reverse-proxy architecture:

```
[ User Browser ]
       │
       ▼ (Clean relative calls: /api/generate, /outputs/render.png)
[ Vercel Edge Router / Cloudflare Pages ] (https://frontend-blush-five-74.vercel.app)
       │
       ▼ (Edge Rewrite Proxy defined in vercel.json)
[ Permanent Backend Gateway ] (https://furnforge-api.sooraj.fun:8005)
       │
       ├── Hardware: Local NVIDIA RTX 3050 Laptop GPU (CUDA 13.0)
       └── Cloud Tunnel: Cloudflare Zero Trust Named Tunnel
```

*Result:* Zero CORS configuration errors, zero port conflicts, and 100% transparent API connectivity.

---

## 🚀 Quick Start Guide

### Option 1: One-Command Docker Compose (Recommended)
Run the entire production stack (Frontend + Backend + Nginx proxy) with Docker:
```bash
docker compose up --build
```
Access the application at `http://localhost:8080`.

---

### Option 2: Local Development Setup

#### 1. Backend Setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Start backend server on port 8005
uvicorn app.main:app --host 0.0.0.0 --port 8005 --reload
```
Interactive OpenAPI documentation is available at `http://localhost:8005/docs`.

#### 2. Frontend Setup
In a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

### Option 3: Running the Academic Benchmark Suite
To compute quantitative Edge-IoU, Chamfer distances, and 3D manifold checks on the sample catalog:
```bash
python -m backend.evaluation.benchmark
```
Artifacts are automatically saved to `backend/evaluation/benchmark_results.json` and `backend/evaluation/BENCHMARK_SUMMARY.md`.

---

## 🧪 Automated Testing & CI/CD

FurnForge AI includes a comprehensive unit and integration test suite (136 tests passing, 85% coverage):

```bash
pytest
```

Continuous integration is enforced via GitHub Actions ([`.github/workflows/ci.yml`](./.github/workflows/ci.yml)):
- Automated Ruff linting for Python code.
- Pytest coverage verification for preprocessing, CNN feature extraction, and API endpoints.
- Oxlint and Vite production build verification for the React frontend.

---

## 📡 REST API Specification

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Returns server health, hardware device (`cuda`/`cpu`), and pipeline status |
| `GET` | `/api/samples` | Lists built-in sample sketches (Chair, Sofa, Table, Bed, Cabinet) |
| `POST` | `/api/preprocess` | Runs real-time OpenCV Canny edge extraction & background division |
| `POST` | `/api/analyze-cnn` | Computes ResNet-50 feature embeddings, lateral symmetry index, and Grad-CAM |
| `POST` | `/api/generate` | Queues full generative diffusion job with ControlNet guidance |
| `GET` | `/api/jobs/{job_id}` | Polls progress and retrieves completed render variations |
| `GET` | `/api/3d/requirements` | Queries GPU VRAM and 3D neural reconstruction engine availability |
| `POST` | `/api/3d/generate` | Generates watertight .GLB mesh from 2D render via TripoSR |
| `GET` | `/api/3d/jobs/{job_id}` | Polls progress of 3D neural reconstruction |
| `GET` | `/api/gallery` | Retrieves saved furniture designs from persistent SQLite catalog |
| `DELETE` | `/api/gallery/{id}` | Deletes design and associated 2D/3D assets |

---

## 📜 Documentation Links

- **[EVALUATION_REPORT.md](./EVALUATION_REPORT.md):** Academic Evaluation Report, Edge-IoU Benchmarks, Chamfer Distances & Ablation Studies.
- **[DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md):** Production Deployment Manual, Docker Compose, Modal Serverless GPU & Cloudflare Named Tunnels.
- **[PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md):** Deep technical whitepaper, OpenCV formulations, Grad-CAM derivation, and performance benchmarks.
- **[WORKFLOW_ARCHITECTURE.md](./WORKFLOW_ARCHITECTURE.md):** Detailed step-by-step workflow pipeline, dataflows, and component architecture.
- **[LEARN.md](./LEARN.md):** Educational manual covering diffusion theory, ControlNet zero-convolutions, and 3D neural reconstruction.

---

## 📜 License
MIT License • FurnForge AI Research Project
