import os
import sys

# Ensure current directory is in Python module path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Hugging Face ZeroGPU registration
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False
    class spaces:
        @staticmethod
        def GPU(duration=60):
            def decorator(fn):
                return fn
            return decorator

@spaces.GPU(duration=20)
def generate_preview(prompt: str):
    """ZeroGPU entrypoint for dynamic A100 GPU allocation."""
    return f"FurnForge AI Generation Engine active for: '{prompt}'"

import gradio as gr
import uvicorn
from app.main import app as fastapi_app

# Gradio interface for status and interactive testing
with gr.Blocks(title="FurnForge AI — Backend Service") as demo:
    gr.Markdown("# 🛋️ FurnForge AI — Generative Vision & 3D Backend")
    gr.Markdown(
        """
        ### Generative Vision & 3D Neural Reconstruction API
        This Hugging Face Space powers the **FurnForge AI** client application on Vercel.
        
        - **Status:** Active & Ready (ZeroGPU Accelerated)
        - **API Documentation:** [Swagger UI](/docs) | [ReDoc](/redoc)
        - **Health Check:** [/api/health](/api/health)
        """
    )
    with gr.Row():
        test_input = gr.Textbox(
            label="Test Pipeline Prompt",
            value="Modern Scandinavian armchair in white boucle",
        )
        test_btn = gr.Button("Test Pipeline Status", variant="primary")
    test_output = gr.Textbox(label="Result")
    test_btn.click(fn=generate_preview, inputs=[test_input], outputs=[test_output])

# Mount Gradio onto the existing FastAPI app
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
