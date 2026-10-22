#!/usr/bin/env python3
"""Rebuild an MKV and fail closed unless its media and seeking checks pass."""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def run(command, timeout=1800):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}: {command[0]}\n{result.stderr[-6000:]}\n{result.stdout[-2000:]}")
    return result


def probe(path):
    return json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                           "-show_chapters", "-show_data_hash", "sha256", "-of", "json", str(path)]).stdout)


def compare_metadata(before, after):
    durations = [float(info.get("format", {}).get("duration", "nan")) for info in (before, after)]
    if not all(math.isfinite(value) and value > 0 for value in durations) or abs(durations[0] - durations[1]) > 0.005:
        raise RuntimeError(f"Remux changed/invalidated duration: {durations}")
    fields = ("codec_type", "codec_name", "profile", "width", "height", "pix_fmt", "level",
              "sample_rate", "channels", "channel_layout", "bits_per_raw_sample", "extradata_hash",
              "color_range", "color_space", "color_transfer", "color_primaries", "sample_aspect_ratio")
    a, b = before["streams"], after["streams"]
    if len(a) != len(b):
        raise RuntimeError("Remux changed the stream count")
    for index, (left, right) in enumerate(zip(a, b)):
        for field in fields:
            if left.get(field) != right.get(field):
                raise RuntimeError(f"Stream {index} changed {field}: {left.get(field)!r} -> {right.get(field)!r}")
        if left.get("disposition", {}) != right.get("disposition", {}):
            raise RuntimeError(f"Stream {index} changed track flags")
        for key in ("language", "title", "filename", "mimetype"):
            if left.get("tags", {}).get(key) != right.get("tags", {}).get(key):
                raise RuntimeError(f"Stream {index} changed tag {key}")
    if before.get("format", {}).get("tags", {}).get("title") != after.get("format", {}).get("tags", {}).get("title"):
        raise RuntimeError("Remux changed the container title")
    a, b = before.get("chapters", []), after.get("chapters", [])
    if len(a) != len(b):
        raise RuntimeError("Remux changed the chapter count")
    for left, right in zip(a, b):
        for key in ("start_time", "end_time"):
            if abs(float(left[key]) - float(right[key])) > 0.005:
                raise RuntimeError(f"Remux changed chapter {key}")
        if left.get("tags", {}) != right.get("tags", {}):
            raise RuntimeError("Remux changed chapter tags")


def media_hashes(path, info):
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v?", "-map", "0:a?", "-map", "0:s?", "-c", "copy"]
    videos = [s for s in info["streams"] if s.get("codec_type") == "video"]
    for number, stream in enumerate(videos):
        if stream.get("codec_name") == "hevc":
            # mkvmerge repeats VPS/SPS/PPS at keyframes. Their content is checked
            # separately by the mandatory extradata_hash metadata comparison.
            command += [f"-bsf:v:{number}", "filter_units=remove_types=32|33|34"]
    command += ["-f", "streamhash", "-hash", "sha256", "-"]
    hashes = [line.strip() for line in run(command).stdout.splitlines() if ",SHA256=" in line]
    expected = sum(s.get("codec_type") in ("video", "audio", "subtitle") for s in info["streams"])
    if len(hashes) != expected:
        raise RuntimeError(f"Expected {expected} media hashes, got {len(hashes)}")
    return hashes


def interleaving(path, info, maximum_lead=2.0):
    indices = {s["index"] for s in info["streams"] if s.get("codec_type") in ("video", "audio")
               and not s.get("disposition", {}).get("attached_pic")}
    if len(indices) < 2:
        return {"max_lead_seconds": 0.0}
    result = run(["ffprobe", "-v", "error", "-show_packets", "-show_entries",
                  "packet=stream_index,pts_time,pos", "-of", "compact=p=0:nk=0", str(path)])
    events, bounds = [], {}
    for line in result.stdout.splitlines():
        parts = dict(part.split("=", 1) for part in line.split("|") if "=" in part)
        try:
            index, pts, pos = int(parts["stream_index"]), float(parts["pts_time"]), int(parts["pos"])
        except (KeyError, ValueError):
            continue
        if index not in indices or not math.isfinite(pts):
            continue
        events.append((pos, index, pts))
        lo, hi = bounds.get(index, (pts, pts))
        bounds[index] = min(lo, pts), max(hi, pts)
    if set(bounds) != indices:
        raise RuntimeError("An audio/video stream has no timed packets")
    # Compare only the timeline shared by all tracks; a shorter soundtrack must
    # not make a valid ending look like bad interleaving.
    lower = max(value[0] for value in bounds.values())
    upper = min(value[1] for value in bounds.values())
    if upper <= lower:
        raise RuntimeError("Audio and video have no overlapping timeline")
    latest, maximum = {}, 0.0
    for _, index, pts in sorted(events):
        latest[index] = max(latest.get(index, lower), min(upper, max(lower, pts)))
        if len(latest) == len(indices):
            maximum = max(maximum, max(latest.values()) - min(latest.values()))
    if maximum > maximum_lead:
        raise RuntimeError(f"Bad physical A/V interleaving: {maximum:.3f}s lead > {maximum_lead}s")
    return {"max_lead_seconds": round(maximum, 6), "packets": len(events),
            "bounds": {str(index): list(value) for index, value in bounds.items()}}


@contextmanager
def seek_target(path, engine):
    if engine == "host":
        yield ["ffmpeg"], str(path)
        return
    container = os.environ.get("JELLYFIN_CONTAINER", "jellyfin")
    mounts = json.loads(run(["docker", "inspect", container, "--format", "{{json .Mounts}}"], timeout=30).stdout)
    resolved = path.resolve()
    matches = []
    for mount in mounts:
        try:
            relative = resolved.relative_to(Path(mount["Source"]).resolve())
        except ValueError:
            continue
        matches.append((len(mount["Source"]), str(Path(mount["Destination"]) / relative)))
    prefix = ["docker", "exec", container, "/usr/lib/jellyfin-ffmpeg/ffmpeg"]
    if matches:
        yield prefix, max(matches)[1]
        return
    cache = next((m for m in mounts if m["Destination"] == "/cache" and m.get("RW")), None)
    if cache is None:
        raise RuntimeError(f"QA path is not mounted in Jellyfin and no writable /cache mount exists: {path}")
    # Repair staging can be outside the media mount. A temporary hard link lets
    # Jellyfin's exact FFmpeg read it, without changing container mounts.
    with tempfile.TemporaryDirectory(prefix=".mkv-qa-", dir=cache["Source"]) as work:
        linked = Path(work) / "candidate.mkv"
        try:
            os.link(resolved, linked)
        except OSError:
            shutil.copy2(resolved, linked)
        yield prefix, str(Path("/cache") / Path(work).name / linked.name)


def seek_checks(path, info, engine="jellyfin"):
    audios = [s for s in info["streams"] if s.get("codec_type") == "audio"]
    if not audios:
        return []
    duration = float(info["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("Invalid media duration")
    with seek_target(path, engine) as (prefix, visible):
        return _seek_checks(prefix, visible, audios, duration)


def _seek_checks(prefix, visible, audios, duration):
    checks = []
    for fraction in (0.2, 0.5, 0.8):
        seek = duration * fraction
        for audio in audios:
            command = prefix + ["-hide_banner", "-nostdin", "-v", "verbose", "-debug_ts",
                       "-analyzeduration", "200M", "-probesize", "1G", "-ss", f"{seek:.6f}", "-i", visible,
                       "-map", f"0:{audio['index']}", "-c:a", "copy", "-bsf:a",
                       f"noise=drop='lt(pts*tb,{seek:.6f})'", "-copyts", "-frames:a", "1", "-f", "null", "-"]
            result = run(command, timeout=120)
            times = [float(match.group(1)) for match in re.finditer(r"muxer <-[^\n]*?\b(?:pkt_)?pts_time:([-+0-9.eE]+)", result.stderr)]
            if not times:
                raise RuntimeError(f"Seek {seek:.3f}s produced no audio packet on stream {audio['index']}")
            error = times[0] - seek
            if error < -0.005 or error > 0.1:
                raise RuntimeError(f"Seek {seek:.3f}s found audio at {times[0]:.3f}s on stream {audio['index']}")
            checks.append({"stream": audio["index"], "seek": seek, "first_pts": times[0], "error": error})
    return checks


def validate_mkv(source, candidate, seek_engine="jellyfin"):
    source, candidate = Path(source), Path(candidate)
    before, after = probe(source), probe(candidate)
    compare_metadata(before, after)
    original_hashes, remuxed_hashes = media_hashes(source, before), media_hashes(candidate, after)
    if original_hashes != remuxed_hashes:
        raise RuntimeError("Remux changed media payload hashes")
    original_layout = interleaving(source, before, maximum_lead=float("inf"))
    layout = interleaving(candidate, after)
    for index, bounds in original_layout.get("bounds", {}).items():
        if any(abs(left - right) > 0.005 for left, right in zip(bounds, layout["bounds"][index])):
            raise RuntimeError(f"Remux changed stream {index} timeline boundaries")
    return {"source": str(source), "candidate": str(candidate), "media_hashes": remuxed_hashes,
            "interleaving": layout,
            "seeks": seek_checks(candidate, after, seek_engine), "seek_engine": seek_engine,
            "streams": len(after["streams"]), "chapters": len(after.get("chapters", []))}


def remux_and_validate(source, candidate, seek_engine="jellyfin"):
    source, candidate = Path(source), Path(candidate)
    if candidate.exists():
        raise FileExistsError(candidate)
    run(["mkvmerge", "-o", str(candidate), str(source)])
    if not candidate.is_file() or candidate.stat().st_size == 0:
        raise RuntimeError("mkvmerge produced no output")
    return validate_mkv(source, candidate, seek_engine)


def finalize_mkv(source, destination, seek_engine="jellyfin"):
    source, destination = Path(source), Path(destination)
    if source.parent.stat().st_dev != destination.parent.stat().st_dev:
        raise RuntimeError("Private staging must share the destination filesystem")
    with tempfile.TemporaryDirectory(prefix=".mkv-qa-", dir=source.parent) as staging:
        candidate = Path(staging) / "validated.mkv"
        report = remux_and_validate(source, candidate, seek_engine)
        shutil.copymode(source, candidate)
        os.replace(candidate, destination)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--seek-engine", choices=("jellyfin", "host"), default="jellyfin")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    operation = validate_mkv if args.validate_only else remux_and_validate
    report = operation(args.source, args.candidate, args.seek_engine)
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.write_text(text + "\n")
    print(text)
