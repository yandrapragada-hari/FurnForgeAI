# FurnForge AI – Sketch-to-Photoreal Furniture & 3D Studio 🛋️✨

> 📖 **Project Documentation Suite:**
> - **[WORKFLOW_ARCHITECTURE.md](./WORKFLOW_ARCHITECTURE.md)** — End-to-end system architecture, Mermaid dataflows, and component interactions.
> - **[PROJECT_DOCUMENTATION.md](./PROJECT_DOCUMENTATION.md)** — Technical whitepaper, mathematical formulations, and benchmarks.
> - **[LEARN.md](./LEARN.md)** — Deep learning, computer vision, diffusion mechanics, and 3D neural reconstruction manual.
> - **[frontend/README.md](./frontend/README.md)** — Frontend React 19, Vite, Three.js, and WebXR architecture guide.

**FurnForge AI** (`v2.0.0`) is an end-to-end Generative AI, Computer Vision, and 3D Neural Reconstruction platform that transforms hand-drawn furniture sketches and digital whiteboard drawings into photorealistic, showroom-ready furniture renders, interactive 3D meshes (`.glb`, `.gltf`, `.obj`), and live WebXR Augmented Reality room previews.

Designed to eliminate the spatial hallucination and geometric distortion of naive text-to-image models, FurnForge AI enforces structural fidelity through **Classical Computer Vision (OpenCV)**, **Deep Feature Extraction (ResNet-50 + Grad-CAM)**, **Spatial ControlNet (Scribble & Canny)**, **Latent Consistency Diffusion (SD 1.5 + LCM-LoRA)**, and **Feed-Forward Neural 3D Reconstruction (TripoSR with Hunyuan3D-2mini fallback)**.

---

## 🏛️ End-to-End Pipeline & Architecture

```
+───────────────────────────────────────────────────────────────────────────────────────────+
│                                  FURNFORGE AI ARCHITECTURE                                │
+───────────────────────────────────────────────────────────────────────────────────────────+
  1. Multi-Modal Sketch Ingestion (3 Input Modes)
     ├── Option 1: Curated Presets (Chair, Sofa, Table, Bed, Cabinet)
     ├── Option 2: File Upload (PNG, JPG, WebP up to 10MB)
     └── Option 3: Interactive Whiteboard Studio (SketchCanvas.jsx)
          ├── Real-time Pen & Eraser (4 pen widths, 4 eraser presets + 4–80px slider)
          ├── 1:1 Subpixel Screen Mapping (Mouse pointer = SVG marker nib at (0,0))
          ├── True Entire-Screen Fullscreen Canvas (Zero scrollbars)
          └── Auto-Bounding Box Segmentation & 512×512 Centered HD Export
          │
          ▼
  2. OpenCV Vision Preprocessing (cv_preprocessor.py)
     ├── Flat-Field Background Division (Eliminates paper shadows & uneven lighting)
     ├── Bilateral Filtering (Smooths paper grain while preserving pencil edges)
     ├── CLAHE Local Dynamic Histogram Equalization
     ├── Scribble / Canny Edge Extraction & Inversion
     └── Aspect-Ratio Preserved Letterbox Padding (512×512)
          │
          ├───► 3. Deep Topology CNN (ResNet-50 Backbone - cnn_extractor.py)
          │        ├── 2048-dim Latent Feature Vector (layer4)
          │        ├── Topological Symmetry Score (L1 Horizontal Flip)
          │        ├── Aspect Ratio & Hu Invariant Moments (h1 – h7)
          │        ├── Furniture Category Classification & Auto-Alignment
          │        └── Grad-CAM Thermal Attention Heatmap
          │
          ▼
  4. ControlNet Spatial Guidance (Scribble / Canny)
     ├── Zero-convolutions lock exact sketch silhouettes and joinery
     └── Injected directly into U-Net skip-connections
          │
          ▼
  5. Latent Consistency Diffusion (SD 1.5 + LCM-LoRA - diffusion_engine.py)
     ├── PBR Material Synthesis (Wood, Metal, Glass, Leather, Fabric)
     ├── 4–8 Denoising Steps via LCM-LoRA Consistency Distillation
     └── Instant <30ms BRDF Spec Preview Mode (photoreal_renderer.py)
          │
          ├───► 6. Photorealistic 2D Render (Before/After Slider, Gallery, Variations)
          │
          ▼
  7. 3D Neural Reconstruction & WebXR AR (mesh3d_engine.py & Model3DViewer.jsx)
     ├── Priority 1: stabilityai/TripoSR (Triplane NeRF + Marching Cubes 160³)
     │    ├── Headless EGL / X11 UV Texture Baking (1024×1024)
     │    └── Automatic Triplane Vertex-Color Fallback (Zero display dependency)
     ├── Priority 2 Fallback: tencent/Hunyuan3D-2mini (CUDA >= 6GB VRAM)
     ├── Trimesh GLB / GLTF / OBJ Export & Normalization
     └── Interactive Three.js Orbit Viewer + WebXR AR Floor Placement
```

---

## ✨ Key Features

### 🎨 1. Three Sketch Input Modalities
- **Curated Sample Presets:** Instant one-click loading of standardized furniture sketches across **5 categories**: **Chair**, **Sofa**, **Table**, **Bed**, and **Cabinet**.
- **Drag-and-Drop File Upload:** Supports digital tablet exports, paper scans, and camera photos (`.png`, `.jpg`, `.webp`, up to 10MB).
- **Interactive Studio Whiteboard (`SketchCanvas.jsx`):**
  - **1:1 Subpixel Cursor Alignment:** `devicePixelRatio` canvas scaling with a custom SVG marker nib anchored at `(0, 0)` directly under the pointer.
  - **Independent Pen & Eraser Tools:** Switching tools remembers each tool's thickness independently.
  - **Dedicated Eraser Adjustment Panel:** 4 quick presets (*Fine 8px*, *Medium 20px*, *Large 38px*, *Wide 68px*), a smooth **4px–80px range slider** with live pixel readout, and a dynamic dashed hover boundary ring.
  - **Pen Styling & Palette:** 4 stroke widths (*Fine 2px*, *Pen 4px*, *Marker 8px*, *Broad 16px*), 4 designer swatches + custom color picker.
  - **History & Shortcuts:** 30-step Undo/Redo stack (`Ctrl+Z`, `Ctrl+Y`), tool hotkeys (`P`, `E`, `[`, `]`, `Esc`), and grid toggle.
  - **True Entire-Screen Fullscreen:** Expands edge-to-edge (`fixed inset-0 w-full h-full`) with body scroll locking.
  - **Auto-Framed HD Export:** Detects the bounding box of the drawn furniture and centers it with a 14% margin on a 512×512 canvas.

### 🎛️ 2. Architectural Customizer & Instant Spec Preview
- **5 Furniture Categories:** Chair, Table, Sofa, Bed, Cabinet.
- **5 Primary Materials:** Natural Wood (*Solid Oak, American Walnut, Teak, Bleached Ash*), Architectural Metal (*Polished Brass, Matte Black Steel, Brushed Aluminum, Satin Nickel*), Structural Glass (*Float, Frosted, Fluted*), Italian Leather (*Full-Grain Aniline, Cognac Saddle, Nubuck*), and Designer Textile (*Bouclé, Belgian Linen, Silk Velvet*).
- **12 Designer Color Palettes:** Warm Walnut, Charcoal Gray, Deep Emerald, Cognac Brown, Cream Bouclé, Matte Black, Champagne Gold, Navy Blue, Terracotta, Scandinavian Birch, Sage Green, White.
- **6 Design Styles & 4 Surface Finishes:** Modern, Minimalist, Traditional, Luxury Haute Couture, Scandinavian, Industrial Loft × Matte, Glossy, Polished, Brushed.
- **Instant Spec Renderer (`/api/spec-render`):** Synthesizes physically inspired BRDF material previews in `<30ms` while full diffusion jobs run asynchronously.

### 🧊 3. 3D Reconstruction, Interactive Viewer & WebXR AR
- **Dual-Engine 3D Pipeline:**
  - **Priority 1 (`stabilityai/TripoSR`):** Feed-forward triplane NeRF transformer + Marching Cubes isosurface extraction. Supports headless **EGL** GPU texture baking (`xatlas` + `moderngl`) and automatically falls back to direct triplane vertex coloring on headless Linux/Colab environments without an X11 display.
  - **Priority 2 Fallback (`tencent/Hunyuan3D-2mini`):** High-VRAM CUDA fallback (`>= 6GB VRAM`, Octree 380) via `hy3dgen`.
- **Memory-Safe Execution:** Automatically unloads 2D diffusion pipelines and flushes CUDA cache before 3D reconstruction to prevent OOM errors on 15GB T4 Colab GPUs.
- **Interactive 3D & AR Studio:** Three.js / React Three Fiber 360° orbit viewer with studio lighting, multi-format download (`.glb`, `.gltf`, `.obj`), and **WebXR Augmented Reality** (`ViewInRoomModal.jsx`) for 1:1 scale room floor placement.

---

## 🤖 Machine Learning Models & Hugging Face Registry

| Component | Model / Repository | Architecture / Specs | Role |
| :--- | :--- | :--- | :--- |
| **Base Diffusion** | [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | Latent Diffusion U-Net (860M) | Photorealistic 2D furniture synthesis |
| **ControlNet (Default)** | [`lllyasviel/control_v11p_sd15_scribble`](https://huggingface.co/lllyasviel/control_v11p_sd15_scribble) | ControlNet Zero-Convolutions | Freehand sketch & contour guidance |
| **ControlNet (Hard Edge)** | [`lllyasviel/control_v11p_sd15_canny`](https://huggingface.co/lllyasviel/control_v11p_sd15_canny) | ControlNet Zero-Convolutions | Precision Canny edge conditioning |
| **LCM-LoRA** | [`latent-consistency/lcm-lora-sdv1-5`](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) | Consistency LoRA | Fast 4–8 step diffusion inference |
| **Topology CNN** | `torchvision.models.resnet50(weights=DEFAULT)` | ResNet-50 (25.6M) | Feature extraction, symmetry, Grad-CAM |
| **Primary 3D Engine** | [`stabilityai/TripoSR`](https://huggingface.co/stabilityai/TripoSR) | Triplane Transformer + Marching Cubes | Sub-second single-image 3D mesh + textures |
| **Fallback 3D Engine** | [`tencent/Hunyuan3D-2mini`](https://huggingface.co/tencent/Hunyuan3D-2mini) | 3D DiT (`hunyuan3d-dit-v2-mini`) | High-VRAM CUDA fallback 3D generator |

---

## 💻 Technology Stack

- **Backend:** Python 3.10+, FastAPI, Uvicorn, Pydantic V2, PyTorch, Torchvision, Diffusers, Transformers, Accelerate, PEFT, OpenCV, Pillow, NumPy, Trimesh, scikit-image, rembg (`onnxruntime`), ModernGL, SQLite3.
- **Frontend:** React 19 (`^19.3.1`), Vite 8 (`^8.3.6`), Tailwind CSS v4 (`@tailwindcss/vite`), Three.js (`^0.187.0`), `@react-three/fiber`, `@react-three/drei`, WebXR Device API, Lucide React, Canvas Confetti.
- **Deployment & Tooling:** `concurrently`, Windows `run.bat`, Google Colab (`FurnForge_AI_Colab.ipynb`), Vercel (`frontend/vercel.json`), Pytest.

---

## 📁 Repository Structure

```
FurnForge AI/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI REST endpoints & static file serving
│   │   ├── config.py                # Paths, HuggingFace cache, model & device configs
│   │   ├── jobs.py                  # Async background job queue & progress tracking
│   │   ├── models/schemas.py        # Pydantic V2 request/response schemas
│   │   ├── database/db.py           # SQLite gallery repository (furnforge.db)
│   │   └── pipeline/
│   │       ├── cv_preprocessor.py   # OpenCV flat-field, bilateral filter, Canny/scribble
│   │       ├── cnn_extractor.py     # ResNet-50 topology, Hu moments & Grad-CAM
│   │       ├── prompt_builder.py    # Architectural PBR prompt compiler
│   │       ├── diffusion_engine.py  # SD 1.5 + ControlNet + LCM-LoRA inference
│   │       ├── photoreal_renderer.py# Instant <30ms BRDF specification previewer
│   │       ├── mesh3d_engine.py     # TripoSR (Priority 1) & Hunyuan3D-2mini (Fallback)
│   │       └── tsr/                 # TripoSR neural network & headless EGL texture baking
│   ├── sample_sketches/             # Built-in presets (bed, cabinet, chair, sofa, table)
│   ├── outputs/                     # Generated renders, edge maps & 3D meshes (models3d/)
│   ├── tests/                       # Pytest verification suite
│   ├── requirements.txt             # Python dependencies
│   └── .env.example                 # Backend environment template
├── frontend/
│   ├── src/
│   │   ├── App.jsx                  # Main application shell & navigation
│   │   ├── components/
│   │   │   ├── Hero.jsx             # Landing hero & feature highlights
│   │   │   ├── BeforeAfterCompare.jsx # Interactive split-screen render slider
│   │   │   ├── Studio.jsx           # Main AI studio orchestrator
│   │   │   ├── SketchCanvas.jsx     # Interactive whiteboard (Pen, Eraser, Fullscreen)
│   │   │   ├── SketchUploader.jsx   # 3-way input switcher & OpenCV edge preview
│   │   │   ├── CustomizerPanel.jsx  # Category, material, color, style & finish controls
│   │   │   ├── ResultsViewer.jsx    # Generated variations & 3D launch controls
│   │   │   ├── Model3DViewer.jsx    # Three.js / R3F interactive 3D orbit viewport
│   │   │   ├── ViewInRoomModal.jsx  # WebXR AR & simulated room placement modal
│   │   │   ├── AIInspectorModal.jsx # CNN metrics, Hu moments & Grad-CAM viewer
│   │   │   ├── TechStackModal.jsx   # Interactive system architecture modal
│   │   │   ├── Gallery.jsx          # Persistent SQLite furniture catalog
│   │   │   └── ar/                  # WebXRARScene.jsx & PreviewARScene.jsx
│   │   ├── services/api.js          # Backend REST client
│   │   └── utils/presets.js         # Categories, materials, colors, styles & samples
│   ├── package.json                 # Frontend dependencies
│   ├── vite.config.js               # Vite dev server & /api proxy to port 8005
│   ├── vercel.json                  # Vercel SPA rewrite rules
│   └── .env.example                 # Frontend VITE_BACKEND_URL template
├── docs/                            # Implementation plans & design notes
├── FurnForge_AI_Colab.ipynb         # 3-step Google Colab GPU backend + tunnel notebook
├── PROJECT_DOCUMENTATION.md         # Engineering blueprint & mathematical formulations
├── WORKFLOW_ARCHITECTURE.md         # Detailed pipeline & dataflow diagrams
├── LEARN.md                         # Deep learning & computer vision learning guide
├── generate_furnforge_doc.py        # Automated DOCX report generator
├── package.json                     # Root concurrently dev runner
├── run.bat                          # Windows one-click launcher
└── README.md                        # Repository overview
```

---

## 🚀 Quick Start Guide

### Prerequisites
- **Node.js** `v18+`
- **Python** `v3.10+`
- *(Optional)* NVIDIA CUDA GPU for accelerated diffusion & 3D reconstruction (CPU mode is supported via LCM-LoRA and TripoSR).

### Option A: One-Click Launch (Windows)
Double-click `run.bat` or run from the repository root:
```bash
npm install
npm run dev
```
This launches the FastAPI backend on `http://localhost:8005` and the Vite frontend on `http://localhost:5173`.

### Option B: Manual Local Setup

1. **Backend Setup**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8005 --reload
```
Interactive Swagger API docs are available at `http://localhost:8005/docs`.

2. **Frontend Setup**
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

### Option C: Google Colab GPU Backend + Vercel Frontend
1. Open **[FurnForge_AI_Colab.ipynb](./FurnForge_AI_Colab.ipynb)** in Google Colab with a GPU runtime (e.g., T4).
2. Run the 3 setup cells to cache weights, start Uvicorn on port `8005`, and expose a public tunnel URL.
3. Set `VITE_BACKEND_URL=<your-tunnel-url>` in `frontend/.env` or your Vercel project environment variables.

---

## 🧪 Running Tests

Run the backend integration test suite from the repository root:
```bash
python -m pytest backend/tests -v
```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Server health, compute device (`cuda`/`xpu`/`mps`/`cpu`), and model load state |
| `POST` | `/api/model/load` | Trigger background loading of diffusion and ControlNet models |
| `GET` | `/api/samples` | List built-in sample sketches (`chair`, `sofa`, `table`, `bed`, `cabinet`) |
| `POST` | `/api/preprocess` | Run OpenCV preprocessing and ResNet-50 CNN topology + Grad-CAM analysis |
| `POST` | `/api/spec-render` | Instant `<30ms` BRDF specification preview for selected material/color/finish |
| `POST` | `/api/generate` | Queue async ControlNet + SD 1.5 generation job (returns `202` + `job_id`) |
| `GET` | `/api/jobs/{job_id}` | Poll 2D generation job progress and retrieve rendered outputs |
| `GET` | `/api/3d/requirements` | Probe CUDA/VRAM, TripoSR, headless OpenGL, and Hunyuan3D-2mini readiness |
| `GET` | `/api/3d/status/{generation_id}` | Check whether 3D assets already exist for a given generation |
| `POST` | `/api/3d/generate` | Queue async 3D reconstruction job (TripoSR primary, Hunyuan3D-2mini fallback) |
| `GET` | `/api/3d/jobs/{job_id}` | Poll 3D reconstruction progress and mesh statistics |
| `GET` | `/api/3d/download/{generation_id}/{fmt}` | Download reconstructed 3D mesh (`glb`, `gltf`, or `obj`) |
| `GET` | `/api/gallery` | Retrieve saved furniture designs from SQLite (`furnforge.db`) |
| `GET` | `/api/gallery/{item_id}` | Retrieve a single saved design by ID |
| `DELETE` | `/api/gallery/{item_id}` | Delete a saved design and its associated output files |

---

## 📜 License
MIT License • FurnForge AI Research Project
