#!/usr/bin/env python3
"""
Studio Ghibli / Anime 4K Chunked Streaming Upscaler & Finalizer
===============================================================
Processes full-length feature films and anime episodes in real-time frame chunks:
1. Extracts chunk N of frames via fast/accurate FFmpeg seek.
2. Upscales frames in-place using Real-CUGAN (ncnn Vulkan) on the assigned GPU.
3. Immediately encodes the upscaled chunk to a 10-bit HEVC MKV segment.
4. Deletes temporary PNG frames after every chunk to keep scratch disk usage minimal (< 25 GB).
5. Concatenates all encoded segments and losslessly remuxes original multi-channel audio,
   ASS/PGS subtitles, font attachments, and chapters using `mkvmerge`.
6. Optionally triggers a Jellyfin library scan upon publication.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from queue import Queue


def get_cugan_paths(model_name: str):
    cugan_dir = Path(os.environ.get("REALCUGAN_DIR", "/opt/realcugan"))
    if not cugan_dir.exists():
        cugan_dir = Path(__file__).resolve().parent.parent / "realcugan-ncnn-vulkan-20220728-ubuntu"
    cugan_bin = Path(os.environ.get("REALCUGAN_BIN", str(cugan_dir / "realcugan-ncnn-vulkan")))
    model_dir = cugan_dir / model_name
    return cugan_bin, model_dir


class Logger:
    def __init__(self, log_file: Path = None):
        self.log_file = log_file
        self.lock = threading.Lock()

    def __call__(self, msg: str):
        timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
        line = f"{timestamp} {msg}"
        with self.lock:
            print(line, flush=True)
            if self.log_file:
                try:
                    self.log_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                except Exception:
                    pass


def run_cmd(cmd, log, desc=None):
    if desc:
        log(f"==> {desc}")
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        cmd_str = " ".join(str(x) for x in cmd)
        log(f"[ERROR CMD]: {cmd_str}\n{res.stderr[-2000:]}")
        raise RuntimeError(f"Command failed ({res.returncode}): {cmd_str}")


def probe_video_metadata(in_file: Path):
    """Automatically detect width, height, fps, and total_frames via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames:format=duration",
        "-of", "json",
        str(in_file),
    ]
    out = json.loads(subprocess.check_output(cmd, text=True))
    stream = out["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    fps_str = stream.get("r_frame_rate", "24000/1001")
    num, den = (int(x) for x in fps_str.split("/")) if "/" in fps_str else (float(fps_str), 1)
    fps_float = num / den

    nb_frames = stream.get("nb_frames")
    if nb_frames and str(nb_frames).isdigit():
        total_frames = int(nb_frames)
    else:
        duration = float(out.get("format", {}).get("duration", 0))
        total_frames = int(math.ceil(duration * fps_float))

    return width, height, fps_str, fps_float, total_frames


def encode_segment(frames_dir: Path, out_segment_path: Path, fps: str, count: int, settings: dict, log):
    encoder = settings.get("video_encoder", "hevc_nvenc")
    cq = str(settings.get("cq", 16))
    preset = settings.get("preset", "p7")

    cmd = [
        "ffmpeg", "-y",
        "-r", fps,
        "-i", str(frames_dir / "frame_%08d.png"),
        "-vframes", str(count),
    ]

    if encoder == "hevc_nvenc":
        cmd += [
            "-c:v", "hevc_nvenc",
            "-preset", preset,
            "-profile:v", "main10",
            "-pix_fmt", "p010le",
            "-rc:v", "vbr",
            "-cq:v", cq,
            "-b:v", "0",
        ]
    elif encoder == "libx265":
        cpu_preset = preset if preset in ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow") else "slow"
        cmd += [
            "-c:v", "libx265",
            "-preset", cpu_preset,
            "-profile:v", "main10",
            "-pix_fmt", "yuv420p10le",
            "-crf", cq,
        ]
    elif encoder == "hevc_vaapi":
        cmd += [
            "-vaapi_device", settings.get("vaapi_device", "/dev/dri/renderD128"),
            "-vf", "format=p010,hwupload",
            "-c:v", "hevc_vaapi",
            "- profile:v", "main10",
            "-qp", cq,
        ]
    else:
        raise ValueError(f"Unsupported video_encoder: {encoder}")

    cmd += ["-an", str(out_segment_path)]
    run_cmd(cmd, log, f"Encoding segment {out_segment_path.name} ({count} frames via {encoder})")


def trigger_jellyfin_scan(settings: dict, log):
    jf_url = os.environ.get("JELLYFIN_URL") or settings.get("jellyfin_url")
    jf_key = os.environ.get("JELLYFIN_API_KEY") or settings.get("jellyfin_api_key")
    if not jf_url or not jf_key:
        return
    try:
        req = urllib.request.Request(
            f"{jf_url.rstrip('/')}/Library/Refresh",
            headers={"Authorization": f'MediaBrowser Token="{jf_key}"'},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            log(f"[Jellyfin] Triggered library refresh (HTTP {resp.status})")
    except Exception as e:
        log(f"[Jellyfin] Warning: Failed to trigger library scan: {e}")


def finalize_full_movie(movie: dict, seg_dir: Path, in_source: Path, final_output: Path, settings: dict, log):
    m_id = movie["id"]
    log(f"[{m_id}] Concatenating all video segments into full video track...")
    concat_txt = seg_dir / "concat_list.txt"
    seg_files = sorted(list(seg_dir.glob("seg_*.mkv")))
    if not seg_files:
        raise RuntimeError(f"[{m_id}] No segment files found in {seg_dir}")

    with open(concat_txt, "w", encoding="utf-8") as f:
        for sf in seg_files:
            p_str = str(sf.resolve()).replace("'", "'\\''")
            f.write(f"file '{p_str}'\n")

    full_video_mkv = seg_dir / "full_video_4k.mkv"
    cmd_concat = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_txt),
        "-c", "copy",
        str(full_video_mkv),
    ]
    run_cmd(cmd_concat, log, f"[{m_id}] Merging {len(seg_files)} segments")

    log(f"[{m_id}] Remuxing with original audio, subtitles, attachments, and chapters using mkvmerge...")
    tmp_final = final_output.parent / f".tmp_{final_output.name}"

    extra_track_args = []
    for track_cfg in movie.get("track_overrides", []):
        tid = track_cfg["track_id"]
        if "language" in track_cfg:
            extra_track_args += ["--language", f"{tid}:{track_cfg['language']}"]
        if "name" in track_cfg:
            extra_track_args += ["--track-name", f"{tid}:{track_cfg['name']}"]
        if "default" in track_cfg:
            extra_track_args += ["--default-track-flag", f"{tid}:{'yes' if track_cfg['default'] else 'no'}"]

    video_lang = movie.get("video_language", "jpn")
    video_track_name = movie.get("video_track_name", "4K HEVC 10-bit (Real-CUGAN Pro)")
    title = movie.get("title", final_output.stem)

    cmd_mkvmerge = [
        "mkvmerge",
        "-o", str(tmp_final),
        "--title", title,
        "--language", f"0:{video_lang}",
        "--track-name", f"0:{video_track_name}",
        "--video-tracks", "0",
        "--no-audio", "--no-subtitles", "--no-attachments", "--no-chapters",
        str(full_video_mkv),
        "--no-video",
        *extra_track_args,
        str(in_source),
    ]
    run_cmd(cmd_mkvmerge, log, f"[{m_id}] Final remux with mkvmerge")

    min_bytes = int(settings.get("min_output_size_mb", 100)) * 1_000_000
    if tmp_final.exists() and tmp_final.stat().st_size > min_bytes:
        tmp_final.rename(final_output)
        full_video_mkv.unlink(missing_ok=True)
        if settings.get("cleanup_segments_on_success", True):
            shutil.rmtree(seg_dir, ignore_errors=True)
        log(f"[{m_id}] [🎉] FINISHED AND PUBLISHED: {final_output.name} ({final_output.stat().st_size / 1e9:.2f} GB)")
        trigger_jellyfin_scan(settings, log)
    else:
        raise RuntimeError(f"[{m_id}] Final remux failed or output file smaller than {min_bytes} bytes")


def process_movie(movie: dict, gpu_id: int, settings: dict, log, failures: Queue):
    m_id = movie["id"]
    input_dir = Path(settings["input_dir"])
    output_dir = Path(settings["output_dir"])
    base_temp_dir = Path(settings["temp_dir"])

    in_file = input_dir / movie["filename"]
    out_final = output_dir / movie["output_name"]
    seg_dir = output_dir / ".segments" / m_id
    seg_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not in_file.exists():
            raise FileNotFoundError(f"Input file not found: {in_file}")

        min_bytes = int(settings.get("min_output_size_mb", 100)) * 1_000_000
        if out_final.exists() and out_final.stat().st_size > min_bytes:
            log(f"[{m_id}] Already completed ({out_final.name}). Skipping.")
            return

        width, height, fps_str, fps_float, probed_frames = probe_video_metadata(in_file)
        total_frames = int(movie.get("total_frames") or probed_frames)
        fps = movie.get("fps") or fps_str
        scale = int(movie.get("scale") or settings.get("scale", 2))
        noise = int(movie.get("noise") if "noise" in movie else settings.get("noise", -1))
        model_name = movie.get("model") or settings.get("model", "models-pro")
        threads = movie.get("threads") or settings.get("threads", "2:2:2")
        chunk_size = int(movie.get("chunk_size") or settings.get("chunk_size", 14385))

        cugan_bin, model_dir = get_cugan_paths(model_name)
        if not cugan_bin.exists():
            raise FileNotFoundError(f"Real-CUGAN binary not found at {cugan_bin}")

        work_dir = base_temp_dir / m_id
        frames_in = work_dir / "frames_in"
        frames_out = work_dir / "frames_out"
        work_dir.mkdir(parents=True, exist_ok=True)

        start_offset = 0
        seg0 = seg_dir / "seg_0000.mkv"
        if seg0.exists() and seg0.stat().st_size > 10_000_000:
            cmd_pr = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames",
                "-of", "default=noprint_wrappers=1:nokey=1", str(seg0),
            ]
            out_pr = subprocess.check_output(cmd_pr, text=True).strip()
            if out_pr.isdigit():
                start_offset = int(out_pr)
                log(f"[{m_id} | GPU {gpu_id}] Resuming after seg_0000 ({start_offset} frames)")

        chunks = []
        curr = start_offset
        chunk_idx = 1 if start_offset > 0 else 0
        while curr < total_frames:
            count = min(chunk_size, total_frames - curr)
            chunks.append((chunk_idx, curr, count))
            curr += count
            chunk_idx += 1

        log(
            f"[{m_id} | GPU {gpu_id}] Input: {width}x{height} -> {width*scale}x{height*scale} "
            f"| Total frames: {total_frames} | Chunks: {len(chunks)} (chunk_size={chunk_size})"
        )

        for c_idx, start_frame, count in chunks:
            seg_file = seg_dir / f"seg_{c_idx:04d}.mkv"
            if seg_file.exists() and seg_file.stat().st_size > 5_000_000:
                log(f"[{m_id} | GPU {gpu_id}] Segment {c_idx:04d} already exists. Skipping.")
                continue

            log(f"[{m_id} | GPU {gpu_id}] >>> STARTING CHUNK {c_idx} (frames {start_frame+1} to {start_frame+count})")
            shutil.rmtree(frames_in, ignore_errors=True)
            shutil.rmtree(frames_out, ignore_errors=True)
            frames_in.mkdir(parents=True, exist_ok=True)
            frames_out.mkdir(parents=True, exist_ok=True)

            t_c0 = time.time()
            start_sec = start_frame / fps_float
            dur_sec = (count + 10) / fps_float
            cmd_extract = [
                "ffmpeg", "-y",
                "-ss", f"{start_sec:.4f}",
                "-i", str(in_file),
                "-t", f"{dur_sec:.4f}",
                "-vframes", str(count),
                "-qscale:v", "1",
                "-qmin", "1",
                str(frames_in / "frame_%08d.png"),
            ]
            run_cmd(cmd_extract, log, f"[{m_id} | GPU {gpu_id}] Extracting {count} frames for chunk {c_idx}")

            extracted_count = len(list(frames_in.glob("frame_*.png")))
            if extracted_count == 0:
                raise RuntimeError(f"Extraction produced 0 frames for chunk {c_idx}")

            cmd_cugan = [
                str(cugan_bin),
                "-i", str(frames_in),
                "-o", str(frames_out),
                "-s", str(scale),
                "-m", str(model_dir),
                "-n", str(noise),
                "-g", str(gpu_id),
                "-j", str(threads),
                "-f", "png",
            ]
            if settings.get("tile_size"):
                cmd_cugan += ["-t", str(settings["tile_size"])]
            if settings.get("tta", False):
                cmd_cugan += ["-x"]

            run_cmd(cmd_cugan, log, f"[{m_id} | GPU {gpu_id}] Upscaling chunk {c_idx} ({extracted_count} frames)")

            upscaled_count = len(list(frames_out.glob("frame_*.png")))
            if upscaled_count != extracted_count:
                raise RuntimeError(f"Upscaling frame count mismatch: {upscaled_count} out vs {extracted_count} in")

            encode_segment(frames_out, seg_file, fps, upscaled_count, settings, log)

            shutil.rmtree(frames_in, ignore_errors=True)
            shutil.rmtree(frames_out, ignore_errors=True)

            elapsed = time.time() - t_c0
            log(
                f"[{m_id} | GPU {gpu_id}] [✓] Chunk {c_idx} finished in {elapsed/60:.2f} mins "
                f"-> {seg_file.name} ({seg_file.stat().st_size / 1e6:.1f} MB)"
            )

        finalize_full_movie(movie, seg_dir, in_file, out_final, settings, log)
        shutil.rmtree(work_dir, ignore_errors=True)

    except Exception as e:
        log(f"[{m_id} | GPU {gpu_id}] [!] ERROR: {e}")
        failures.put(m_id)


def worker_loop(gpu_id: int, job_queue: Queue, settings: dict, log, failures: Queue):
    while not job_queue.empty():
        try:
            movie = job_queue.get_nowait()
        except Exception:
            break
        target_gpu = movie.get("gpu", gpu_id)
        process_movie(movie, target_gpu, settings, log, failures)
        job_queue.task_done()


def load_configuration(config_path: Path = None):
    cfg = {}
    if config_path and config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))

    settings = cfg.get("settings", {})
    settings["input_dir"] = os.environ.get("INPUT_DIR", settings.get("input_dir", "/input"))
    settings["output_dir"] = os.environ.get("OUTPUT_DIR", settings.get("output_dir", "/output"))
    settings["temp_dir"] = os.environ.get("TEMP_DIR", settings.get("temp_dir", "/workspace/tmp"))
    settings["log_file"] = os.environ.get("LOG_FILE", settings.get("log_file", "/workspace/batch_upscale.log"))
    settings["chunk_size"] = int(os.environ.get("CHUNK_SIZE", settings.get("chunk_size", 14385)))
    settings["scale"] = int(os.environ.get("CUGAN_SCALE", settings.get("scale", 2)))
    settings["noise"] = int(os.environ.get("CUGAN_NOISE", settings.get("noise", -1)))
    settings["model"] = os.environ.get("CUGAN_MODEL", settings.get("model", "models-pro"))
    settings["threads"] = os.environ.get("CUGAN_THREADS", settings.get("threads", "2:2:2"))
    settings["video_encoder"] = os.environ.get("VIDEO_ENCODER", settings.get("video_encoder", "hevc_nvenc"))
    settings["preset"] = os.environ.get("VIDEO_PRESET", settings.get("preset", "p7"))
    settings["cq"] = int(os.environ.get("VIDEO_CQ", settings.get("cq", 16)))

    gpus_env = os.environ.get("CUGAN_GPUS")
    if gpus_env:
        settings["gpus"] = [int(x.strip()) for x in gpus_env.split(",") if x.strip()]
    elif "gpus" not in settings:
        settings["gpus"] = [0]

    items = cfg.get("items", [])
    if not items:
        in_dir = Path(settings["input_dir"])
        for idx, mkv in enumerate(sorted(in_dir.glob("*.mkv"))):
            items.append({
                "id": mkv.stem.lower().replace(" ", "_")[:32],
                "title": mkv.stem,
                "filename": mkv.name,
                "output_name": f"{mkv.stem} [2160p 4K HEVC].mkv",
            })

    return settings, items


def main():
    parser = argparse.ArgumentParser(description="Studio Ghibli / Anime 4K Chunked Multi-GPU Upscaler")
    parser.add_argument("-c", "--config", type=Path, default=Path(os.environ.get("PIPELINE_CONFIG", "/config/pipeline.json")),
                        help="Path to pipeline JSON configuration file")
    args = parser.parse_args()

    settings, items = load_configuration(args.config if args.config.exists() else None)
    Path(settings["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(settings["temp_dir"]).mkdir(parents=True, exist_ok=True)

    log = Logger(Path(settings["log_file"]) if settings.get("log_file") else None)

    log("============================================================")
    log("Starting 4K Chunked Multi-GPU Anime/Movie Upscaler")
    log(f"GPUs Configured : {settings['gpus']}")
    log(f"Chunk Size      : {settings['chunk_size']} frames (~{settings['chunk_size']/23.976/60:.1f} mins/chunk)")
    log(f"Real-CUGAN      : model={settings['model']}, scale={settings['scale']}x, noise={settings['noise']}, threads={settings['threads']}")
    log(f"Video Encoder   : {settings['video_encoder']} (preset={settings['preset']}, cq={settings['cq']})")
    log(f"Input Directory : {settings['input_dir']}")
    log(f"Output Directory: {settings['output_dir']}")
    log(f"Total Jobs      : {len(items)}")
    log("============================================================")

    if not items:
        log("[!] No movies or episodes found to process.")
        return

    job_queue = Queue()
    for item in items:
        job_queue.put(item)

    failures = Queue()
    threads = []
    for gpu_id in settings["gpus"]:
        t = threading.Thread(target=worker_loop, args=(gpu_id, job_queue, settings, log, failures))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    if not failures.empty():
        log(f"[!] {failures.qsize()} job(s) failed during processing.")
        raise SystemExit(1)

    log("[🎉] All movies/episodes successfully upscaled and finalized!")


if __name__ == "__main__":
    main()
