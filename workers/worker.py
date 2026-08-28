"""
workers/worker.py

Entrypoint for a worker process. Run this in a SEPARATE terminal from
the API - it's a different process that pulls jobs off the Redis queue
and executes them, which is the whole point of async processing: the
API returns instantly, this process does the slow work in the background.

Run with:
    python -m workers.worker

You can run MULTIPLE copies of this command in multiple terminals to get
multiple workers processing the queue in parallel (this is what Stage 5
in the original plan meant by "producer-consumer" and "concurrency").
"""
import logging
import os
import sys

from redis import Redis
from rq import Queue

logging.basicConfig(level=logging.INFO)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

if __name__ == "__main__":
    conn = Redis.from_url(REDIS_URL)
    queue = Queue("batch_predictions", connection=conn)

    # RQ's default Worker uses os.fork() to isolate each job in its own
    # process - this is a Unix-only syscall and does not exist on Windows.
    # SimpleWorker runs jobs in the same process instead (no fork needed),
    # which works identically on Windows, macOS, and Linux. This is a
    # deliberate, documented trade-off: slightly less process isolation
    # per job, in exchange for cross-platform compatibility.
    if sys.platform == "win32":
        from rq.worker import SimpleWorker
        worker = SimpleWorker([queue], connection=conn)
    else:
        from rq import Worker
        worker = Worker([queue], connection=conn)

    print(f"Worker started, listening on queue 'batch_predictions' at {REDIS_URL}")
    worker.work()
