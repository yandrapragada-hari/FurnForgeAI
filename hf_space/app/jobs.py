"""
Background job queue for generation.

Diffusion takes seconds to minutes, so /api/generate enqueues a job and returns
immediately; the frontend polls /api/jobs/{id} and shows the real denoising-step
progress reported by the diffusers callback. A single worker thread processes jobs
in order because one model instance cannot run two generations at once.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
import uuid
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("FurnForgeAI.Jobs")


class Job:
    def __init__(self, fn: Callable[["Job"], Dict[str, Any]], total_units: int):
        self.id = f"job_{uuid.uuid4().hex[:12]}"
        self.fn = fn
        self.status = "queued"          # queued | running | completed | failed
        self.message = "Waiting in queue"
        self.total_units = max(1, total_units)   # e.g. number of variations
        self.unit = 0
        self.step = 0
        self.total_steps = 0
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

    def report(self, unit: int, step: int, total_steps: int, message: str) -> None:
        self.unit, self.step, self.total_steps, self.message = unit, step, total_steps, message

    @property
    def progress(self) -> float:
        if self.status == "completed":
            return 1.0
        per_unit = (self.step / self.total_steps) if self.total_steps else 0.0
        return round(min(0.99, (self.unit + per_unit) / self.total_units), 4)

    def to_dict(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "job_id": self.id,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "variation": self.unit + 1 if self.status == "running" else None,
            "num_variations": self.total_units,
            "step": self.step,
            "total_steps": self.total_steps,
            "elapsed_seconds": round((self.finished_at or now) - (self.started_at or now), 1),
            "result": self.result,
            "error": self.error,
        }


class JobManager:
    def __init__(self, default_message: str = "Starting generation") -> None:
        self._jobs: Dict[str, Job] = {}
        self._queue: "queue.Queue[Job]" = queue.Queue()
        self._default_message = default_message
        threading.Thread(target=self._worker, name="generation-worker", daemon=True).start()

    def submit(self, fn: Callable[[Job], Dict[str, Any]], total_units: int = 1) -> Job:
        job = Job(fn, total_units)
        self._jobs[job.id] = job
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            job.status, job.started_at, job.message = "running", time.time(), self._default_message
            try:
                job.result = job.fn(job)
                job.status, job.message = "completed", "Done"
            except Exception as exc:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = "Generation failed"
                logger.error("Job %s failed:\n%s", job.id, traceback.format_exc())
            finally:
                job.finished_at = time.time()
                self._queue.task_done()

