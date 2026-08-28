"""
workers/queue_client.py

Used by the API process to enqueue jobs and check status. Deliberately
does NOT import workers.tasks at module load time in a way that loads
the model - only the worker process needs the model loaded. The API
just needs to talk to Redis to push/check jobs.
"""
import os

from redis import Redis
from rq import Queue
from rq.job import Job
from rq.exceptions import NoSuchJobError

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

_redis_conn = Redis.from_url(REDIS_URL)
_queue = Queue("batch_predictions", connection=_redis_conn)


def enqueue_batch_job(csv_bytes: bytes) -> str:
    """Enqueues the job, returns a job_id immediately (does not wait for
    it to run). The actual work function lives in workers/tasks.py and
    is imported by name here so the worker process resolves it -
    this way the API process itself never needs the model loaded."""
    job = _queue.enqueue("workers.tasks.process_batch", csv_bytes, job_timeout=600)
    return job.id


def get_job_status(job_id: str) -> dict:
    try:
        job = Job.fetch(job_id, connection=_redis_conn)
    except NoSuchJobError:
        return {"status": "not_found"}

    if job.is_finished:
        return {"status": "completed", "result": job.result}
    elif job.is_failed:
        return {"status": "failed", "error": str(job.exc_info)[-500:] if job.exc_info else "Unknown error"}
    elif job.is_started:
        return {"status": "running"}
    else:
        return {"status": "queued"}
