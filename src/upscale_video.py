#!/usr/bin/env python3
import argparse
import os
import json
import struct
import tempfile
import shutil
import subprocess
import sys
import time
from pathlib import Path
from mkv_finalize import finalize_mkv, probe

def run_cmd(cmd, desc=None):
    if desc:
        print(f"\n==> {desc}")
    print(f"[CMD] {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {' '.join(cmd)}")

def get_video_info(video_path):
    videos = [stream for stream in probe(video_path)["streams"]
              if stream.get("codec_type") == "video" and not stream.get("disposition", {}).get("attached_pic")]
    if not videos:
        raise RuntimeError("Input contains no moving video track")
    video = videos[0]
    return video["width"], video["height"], video.get("r_frame_rate", "24000/1001")


def is_valid_png(path: Path, expected_size=None) -> bool:
    try:
        with path.open("rb") as stream:
            header = stream.read(24)
            if len(header) != 24 or header[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDR":
                return False
            width, height = struct.unpack(">II", header[16:24])
            if width == 0 or height == 0 or (expected_size and (width, height) != expected_size):
                return False
            stream.seek(-12, os.SEEK_END)
            return stream.read(12) == b"\x00\x00\x00\x00IEND\xaeB\x60\x82"
    except (OSError, ValueError):
        return False


def workspace_identity(input_file, model_type="models-pro", noise_level=-1, scale=2, tta=False):
    source = Path(input_file).resolve()
    stat = source.stat()
    return {"input": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "model": model_type, "noise": noise_level, "scale": scale, "tta": bool(tta)}


def adopt_workspace(temp_dir, identity):
    """Explicitly bind a verified legacy workspace to its source; never delete frames."""
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    manifest = temp_dir / ".resume-source.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise RuntimeError(f"Workspace belongs to a different input or configuration: {temp_dir}")
        return
    staging = manifest.with_suffix(".json.tmp")
    staging.write_text(json.dumps(identity, indent=2) + "\n")
    os.replace(staging, manifest)


def prepare_workspace(temp_dir, identity):
    temp_dir = Path(temp_dir)
    if not (temp_dir / ".resume-source.json").exists() and temp_dir.exists():
        if any(temp_dir.iterdir()):
            raise RuntimeError(f"Unidentified legacy frames in {temp_dir}; verify their source before explicit adoption")
    adopt_workspace(temp_dir, identity)


def _stream_flags(stream, default=None):
    flags = {key for key, value in stream.get("disposition", {}).items() if value}
    if default is not None:
        flags.discard("default")
        if default:
            flags.add("default")
    return "+".join(sorted(flags)) or "0"


def upscale_video(input_file, output_file, temp_dir, gpu_id="0,1", threads="2:2,2:2",
                  model_type="models-pro", noise_level=-1, scale=2, tile_size=None,
                  tta=False, cq=16, max_frames=None):
    input_file, output_file, temp_dir = Path(input_file).resolve(), Path(output_file).resolve(), Path(temp_dir).resolve()
    for command in ("ffmpeg", "ffprobe", "mkvmerge"):
        if shutil.which(command) is None:
            raise RuntimeError(f"Required command is missing: {command}")
    if input_file == output_file:
        raise ValueError("Input and output must be different files")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    identity = workspace_identity(input_file, model_type, noise_level, scale, tta)
    # Include preview length: a short extraction must not be reused for a full job.
    if max_frames is not None:
        identity["max_frames"] = max_frames
    prepare_workspace(temp_dir, identity)
    stage_root = Path(os.environ.get("ANIME_STAGING_DIR", str(output_file.parent / ".staging")))
    stage_root.mkdir(parents=True, exist_ok=True)
    if stage_root.stat().st_dev != output_file.parent.stat().st_dev:
        raise RuntimeError("ANIME_STAGING_DIR must be on the output filesystem for atomic publication")
    staging = Path(tempfile.mkdtemp(prefix="encode-", dir=stage_root))
    encoded = staging / "encoded.mkv"
    try:
        _upscale_video(input_file, encoded, temp_dir, gpu_id, threads, model_type,
                       noise_level, scale, tile_size, tta, cq, max_frames)
        report = finalize_mkv(encoded, output_file, os.environ.get("ANIME_QA_SEEK_ENGINE", "host"))
    except BaseException:
        print(f"[!] Not published. Encoded staging and frames retained: {staging}, {temp_dir}", file=sys.stderr)
        raise
    shutil.rmtree(temp_dir)
    shutil.rmtree(staging)
    print(f"[✓] Remux, media integrity and {len(report['seeks'])} seek checks passed; published {output_file.name}")


def _upscale_video(
    input_file: Path,
    output_file: Path,
    temp_dir: Path,
    gpu_id: str = "0,1",
    threads: str = "2:2,2:2",
    model_type: str = "models-pro",
    noise_level: int = -1,
    scale: int = 2,
    tile_size: str = None,
    tta: bool = False,
    cq: int = 16,
    max_frames: int = None
):
    start_time = time.time()
    input_file = input_file.resolve()
    output_file = output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_dir.resolve()
    frames_in = temp_dir / "frames_in"
    frames_out = temp_dir / "frames_out"
    frames_todo = temp_dir / "frames_todo"
    extract_done_marker = temp_dir / ".extract_done"

    temp_dir.mkdir(parents=True, exist_ok=True)
    frames_in.mkdir(parents=True, exist_ok=True)
    frames_out.mkdir(parents=True, exist_ok=True)
    if frames_todo.exists():
        shutil.rmtree(frames_todo, ignore_errors=True)

    cugan_dir = Path(os.environ.get("REALCUGAN_DIR", "/opt/realcugan"))
    if not cugan_dir.exists():
        cugan_dir = Path(__file__).resolve().parent.parent / "realcugan-ncnn-vulkan-20220728-ubuntu"
    cugan_bin = Path(os.environ.get("REALCUGAN_BIN", str(cugan_dir / "realcugan-ncnn-vulkan")))
    model_path = cugan_dir / model_type

    width, height, fps = get_video_info(input_file)
    source_info = probe(input_file)
    source_video = next(stream for stream in source_info["streams"] if stream.get("codec_type") == "video" and not stream.get("disposition", {}).get("attached_pic"))
    print(f"[*] Input: {input_file.name} ({width}x{height} @ {fps} fps)")
    print(f"[*] Target resolution: {width*scale}x{height*scale} (4K UHD Master for OLED)")

    # 1. Extract frames in PNG (lossless) if not already done
    has_extracted = extract_done_marker.exists() and len(list(frames_in.glob("frame_*.png"))) > 0
    if not has_extracted:
        print("[*] Extracting frames from input video...")
        extract_cmd = ["ffmpeg", "-y", "-i", str(input_file), "-map", f"0:{source_video['index']}"]
        if max_frames:
            extract_cmd += ["-vframes", str(max_frames)]
        extract_cmd += [
            "-qscale:v", "1",
            "-qmin", "1",
            str(frames_in / "frame_%08d.png")
        ]
        run_cmd(extract_cmd, f"1/4: Extracting frames from {input_file.name}")
        extract_done_marker.touch()
    else:
        print(f"[*] Reusing previously extracted frames in {frames_in}")

    in_frames = sorted(list(frames_in.glob("frame_*.png")))
    num_frames = len(in_frames)
    if num_frames == 0:
        raise RuntimeError("Frame extraction produced no frames")
    print(f"[*] Total frames to process: {num_frames}")

    # Format multi-gpu tile size if needed
    gpu_count = len(gpu_id.split(","))
    if tile_size is None:
        formatted_tile = ",".join(["0"] * gpu_count)
    else:
        formatted_tile = tile_size

    # 2. Check for existing upscaled frames (Smart Resume)
    todo_frames = []
    for f in in_frames:
        out_f = frames_out / f.name
        if out_f.exists() and is_valid_png(out_f, (width * scale, height * scale)):
            continue
        if out_f.exists():
            out_f.unlink(missing_ok=True)
        todo_frames.append(f)

    done_count = num_frames - len(todo_frames)
    if done_count > 0:
        pct = (done_count / num_frames * 100) if num_frames > 0 else 0
        print(f"[*] RESUME DETECTED: {done_count}/{num_frames} frames ({pct:.1f}%) already completed.")

    if len(todo_frames) > 0:
        if done_count > 0:
            print(f"[*] Preparing {len(todo_frames)} missing frames for Real-CUGAN...")
            frames_todo.mkdir(parents=True, exist_ok=True)
            for f in todo_frames:
                os.link(f, frames_todo / f.name)
            input_dir_for_cugan = frames_todo
        else:
            input_dir_for_cugan = frames_in

        cugan_cmd = [
            str(cugan_bin),
            "-i", str(input_dir_for_cugan),
            "-o", str(frames_out),
            "-s", str(scale),
            "-m", str(model_path),
            "-n", str(noise_level),
            "-g", str(gpu_id),
            "-j", str(threads),
            "-t", str(formatted_tile),
            "-f", "png"
        ]
        if tta:
            cugan_cmd.append("-x")
        run_cmd(cugan_cmd, f"2/4: Upscaling {len(todo_frames)} frames (Real-CUGAN Pro, noise={noise_level}, TTA={tta}, GPU={gpu_id})")

        if frames_todo.exists():
            shutil.rmtree(frames_todo, ignore_errors=True)
    else:
        print("[✓] All frames are already upscaled in frames_out!")

    # image2 can silently stop at a gap while ffmpeg exits successfully. Refuse
    # encoding unless every expected frame is present, complete and the right size.
    expected_names = {frame.name for frame in in_frames}
    actual_names = {frame.name for frame in frames_out.glob("frame_*.png")}
    if actual_names != expected_names or any(
        not is_valid_png(frames_out / frame.name, (width * scale, height * scale)) for frame in in_frames
    ):
        raise RuntimeError("Upscaled frames are missing, extra, incomplete or have incorrect dimensions")

    # Preserve every soundtrack, subtitle, font/attachment and chapter. Prefer
    # English by language metadata without assuming a fixed number/order of tracks.
    audios = [stream for stream in source_info["streams"] if stream.get("codec_type") == "audio"]
    def is_english(stream):
        language = stream.get("tags", {}).get("language", "").lower()
        return language in ("en", "eng") or language.startswith("en-")
    audios.sort(key=lambda stream: not is_english(stream))
    english = bool(audios and is_english(audios[0]))
    audio_maps = [argument for stream in audios for argument in ("-map", f"1:{stream['index']}")]
    dispositions = []
    dispositions += ["-disposition:v:0", _stream_flags(source_video)]
    covers = [stream for stream in source_info["streams"] if stream.get("codec_type") == "video" and stream.get("disposition", {}).get("attached_pic")]
    cover_maps = [argument for stream in covers for argument in ("-map", f"1:{stream['index']}")]
    for number, stream in enumerate(covers, 1):
        dispositions += [f"-disposition:v:{number}", _stream_flags(stream)]
    for number, stream in enumerate(audios):
        dispositions += [f"-disposition:a:{number}", _stream_flags(stream, number == 0 if english else None)]
    subtitles = [stream for stream in source_info["streams"] if stream.get("codec_type") == "subtitle"]
    for number, stream in enumerate(subtitles):
        dispositions += [f"-disposition:s:{number}", _stream_flags(stream)]
    attachments = [stream for stream in source_info["streams"] if stream.get("codec_type") == "attachment"]
    # One explicit stream metadata mapping disables FFmpeg's automatic copying
    # for ALL streams. Map every output track, including font MIME/filename tags.
    stream_metadata = []
    for kind, streams in (("v", [source_video, *covers]), ("a", audios),
                          ("s", subtitles), ("t", attachments)):
        for number, stream in enumerate(streams):
            stream_metadata += [f"-map_metadata:s:{kind}:{number}", f"1:s:{stream['index']}"]

    # 3. Encode video once; every other mapped stream is copied unchanged.
    mux_cmd = [
        "ffmpeg", "-y",
        "-r", str(fps),
        "-i", str(frames_out / "frame_%08d.png"),
        "-i", str(input_file),
        "-map", "0:v:0",
        *cover_maps,
        *audio_maps,
        "-map", "1:s?",
        "-map", "1:t?",
        "-map_metadata", "1",
        *stream_metadata,
        "-map_chapters", "1",
        "-c", "copy",
        "-c:v:0", "hevc_nvenc",
        "-preset:v:0", "p7",
        "-profile:v:0", "main10",
        "-pix_fmt:v:0", "p010le",
        "-rc:v:0", "vbr",
        "-cq:v:0", str(cq),
        "-b:v:0", "0",
        "-c:a", "copy",
        "-c:s", "copy",
        "-c:t", "copy",
        *dispositions,
        str(output_file)
    ]
    run_cmd(mux_cmd, f"3/4: Master Encoding (HEVC 10-bit OLED CQ={cq}) -> {output_file.name}")

    elapsed = time.time() - start_time
    print(f"\n[✓] Finished {output_file.name} in {elapsed:.1f}s ({elapsed/60:.2f}m)!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OLED-Grade Anime Upscaling Pipeline")
    parser.add_argument("-i", "--input", required=True, type=Path, help="Input video file")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output video file")
    parser.add_argument("--temp-dir", default=Path("/workspace/upscale_tmp"), type=Path)
    parser.add_argument("--gpu", default="0,1", help="GPU IDs (e.g. 0 or 1 or 0,1)")
    parser.add_argument("--threads", default="2:2,2:2", help="Threads load:proc:save")
    parser.add_argument("--model", default="models-pro", choices=["models-pro", "models-se", "models-nose"])
    parser.add_argument("--noise", default=-1, type=int, help="-1=conservative (default), 0=none, 1/2/3=denoise")
    parser.add_argument("--scale", default=2, type=int, choices=[2, 3, 4])
    parser.add_argument("--tile-size", default=None, type=str, help="tile size per GPU (default: 0,0...)")
    parser.add_argument("--tta", action="store_true", help="Enable 8x Test-Time Augmentation")
    parser.add_argument("--cq", default=16, type=int, help="HEVC NVENC CQ (16 is near-lossless master)")
    parser.add_argument("--max-frames", default=None, type=int, help="Limit frames for quick preview test")

    args = parser.parse_args()
    upscale_video(
        input_file=args.input,
        output_file=args.output,
        temp_dir=args.temp_dir,
        gpu_id=args.gpu,
        threads=args.threads,
        model_type=args.model,
        noise_level=args.noise,
        scale=args.scale,
        tile_size=args.tile_size,
        tta=args.tta,
        cq=args.cq,
        max_frames=args.max_frames
    )
