#!/usr/bin/env python3
"""
Multi-GPU Series Episode Batch Orchestrator
===========================================
Distributes anime TV series episodes across multiple GPUs using `upscale_video.py`
and `mkv_finalize.py`. Automatically applies 3x upscaling for <=720p sources
(e.g., Evangelion Episode 16) and 2x upscaling for 1080p sources.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue


def get_ep_number(filename: str) -> int:
    match = re.search(r"(?:[Ee]|-\s*)(\d{2,3})", filename)
    return int(match.group(1)) if match else 999


def get_video_height(filepath: Path) -> int:
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=height",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(filepath),
        ]
        res = subprocess.check_output(cmd, text=True).strip()
        return int(res)
    except Exception:
        return 1080


def worker_thread(worker_id: int, gpu_id: int, task_queue: Queue, failures: Queue, args, log):
    log(f"[Worker {worker_id}] Started on GPU {gpu_id}. Temp root: {args.temp_dir}")
    script_path = Path(__file__).resolve().parent / "upscale_video.py"

    while not task_queue.empty():
        try:
            ep_num, in_file = task_queue.get_nowait()
        except Exception:
            break

        out_final = args.output_dir / f"{args.series_title} - S{args.season:02d}E{ep_num:02d} [2160p HEVC Dual-Audio].mkv"
        out_tmp = args.output_dir / f".tmp_S{args.season:02d}E{ep_num:02d}.mkv"
        worker_temp = args.temp_dir / f"episode_{ep_num:02d}"

        if out_final.exists() and out_final.stat().st_size > 100_000_000:
            log(f"[Worker {worker_id}] Episode {ep_num:02d} already completed. Skipping.")
            task_queue.task_done()
            continue

        height = get_video_height(in_file)
        scale = 3 if height <= 720 else 2
        log(f"[Worker {worker_id}] Episode {ep_num:02d} detected as {height}p -> Applying {scale}x upscale.")

        cmd = [
            sys.executable,
            str(script_path),
            "-i", str(in_file),
            "-o", str(out_tmp),
            "--temp-dir", str(worker_temp),
            "--gpu", str(gpu_id),
            "--threads", args.threads,
            "--model", args.model,
            "--noise", str(args.noise),
            "--scale", str(scale),
            "--cq", str(args.cq),
        ]

        t0 = time.time()
        log(f"[Worker {worker_id} | GPU {gpu_id}] >>> STARTING Ep {ep_num:02d}: {in_file.name}")

        try:
            res = subprocess.run(cmd)
            if res.returncode == 0 and out_tmp.exists() and out_tmp.stat().st_size > 100_000_000:
                out_tmp.rename(out_final)
                elapsed = time.time() - t0
                log(f"[Worker {worker_id} | GPU {gpu_id}] [✓] COMPLETED Ep {ep_num:02d} in {elapsed/60:.2f} mins -> {out_final.name}")
                if worker_temp.exists():
                    shutil.rmtree(worker_temp, ignore_errors=True)
            else:
                failures.put(ep_num)
                log(f"[Worker {worker_id} | GPU {gpu_id}] [!] FAILED Ep {ep_num:02d}. Return code: {res.returncode}")
        except Exception as e:
            failures.put(ep_num)
            log(f"[Worker {worker_id}] Exception on Ep {ep_num:02d}: {e}")
        finally:
            task_queue.task_done()

    log(f"[Worker {worker_id}] All assigned tasks finished.")


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Anime Series 4K Batch Upscaler")
    parser.add_argument("--input-dir", type=Path, default=Path(os.environ.get("INPUT_DIR", "/input")))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("OUTPUT_DIR", "/output")))
    parser.add_argument("--temp-dir", type=Path, default=Path(os.environ.get("TEMP_DIR", "/workspace/tmp_series")))
    parser.add_argument("--log-file", type=Path, default=Path(os.environ.get("LOG_FILE", "/workspace/batch_series.log")))
    parser.add_argument("--series-title", default=os.environ.get("SERIES_TITLE", "Neon Genesis Evangelion"))
    parser.add_argument("--season", type=int, default=int(os.environ.get("SERIES_SEASON", "1")))
    parser.add_argument("--gpus", default=os.environ.get("CUGAN_GPUS", "0,1"))
    parser.add_argument("--threads", default=os.environ.get("CUGAN_THREADS", "2:2:2"))
    parser.add_argument("--model", default=os.environ.get("CUGAN_MODEL", "models-pro"))
    parser.add_argument("--noise", type=int, default=int(os.environ.get("CUGAN_NOISE", "-1")))
    parser.add_argument("--cq", type=int, default=int(os.environ.get("VIDEO_CQ", "16")))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        try:
            args.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(args.log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    files = sorted(list(args.input_dir.glob("*.mkv")), key=lambda p: get_ep_number(p.name))
    if not files:
        log(f"[!] No input .mkv files found in {args.input_dir}")
        return

    q = Queue()
    for f in files:
        q.put((get_ep_number(f.name), f))

    gpus = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    log("============================================================")
    log(f"Starting Multi-GPU 4K Series Batch Orchestrator ({args.series_title})")
    log(f"Total Episodes: {q.qsize()} | GPUs: {gpus}")
    log("============================================================")

    failures = Queue()
    workers = []
    for idx, gpu_id in enumerate(gpus):
        t = threading.Thread(target=worker_thread, args=(idx, gpu_id, q, failures, args, log))
        t.start()
        workers.append(t)

    for t in workers:
        t.join()

    if not failures.empty() or not q.empty():
        log(f"[!] Batch incomplete: {failures.qsize()} failed, {q.qsize()} pending.")
        raise SystemExit(1)
    log("[🎉] All queued episodes processed successfully!")


if __name__ == "__main__":
    main()
