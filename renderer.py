"""
MCEYS Cut Studio - highlight merging and single-pass FFmpeg rendering.

Why a single pass: cutting every segment into its own file and joining them
with `concat -c copy` accumulates AAC priming gaps and timestamp breaks, which
shows up as audio drifting away from the picture after a few cuts. Here every
segment is cut with trim/atrim inside ONE filter graph and joined with the
concat filter; segment borders are snapped to the video frame grid so video
and audio segments always have identical length.
"""
from __future__ import annotations

import subprocess

PRE_ROLL = 3.5       # seconds kept before the first kill of a fight
POST_ROLL = 2.5      # seconds kept after the last kill of a fight
MERGE_GAP = 7.0      # kills closer than this belong to the same fight -> one clip
MIN_SEGMENT = 1.0
X264_PRESET = 'superfast'
FFMPEG_THREADS = 2


def build_segments(kill_times, duration, fps, pre=PRE_ROLL, post=POST_ROLL, gap=MERGE_GAP):
    """Group kills into fights and return frame-aligned [start, end] pairs."""
    if not kill_times:
        return []
    groups = []
    for t in sorted(kill_times):
        if groups and t - groups[-1][-1] <= gap:
            groups[-1].append(t)
        else:
            groups.append([t])
    segs = []
    for g in groups:
        s, e = max(0.0, g[0] - pre), min(duration, g[-1] + post) if duration > 0 else g[-1] + post
        if segs and s <= segs[-1][1]:                 # overlapping fights -> extend
            segs[-1][1] = max(segs[-1][1], e)
        else:
            segs.append([s, e])
    snapped = []
    last = int(duration * fps) / fps if duration > 0 else None
    for s, e in segs:
        s, e = round(s * fps) / fps, round(e * fps) / fps   # frame grid
        if last is not None:
            e = min(e, last)
        if e - s >= MIN_SEGMENT:
            snapped.append((s, e))
    return snapped


def _even(expr: str) -> str:
    return f'trunc(({expr})/2)*2'


def layout_filter(fmt: str, game: str, kf_box, fps: float) -> str:
    """Filter chain from [vc] (concatenated cut) to [outv]."""
    fps_s = f'{fps:.6f}'.rstrip('0').rstrip('.')
    if fmt == '16-9':
        # original framing and resolution (only forced to even dimensions)
        return f'[vc]scale={_even("iw")}:{_even("ih")}:flags=lanczos,fps={fps_s},format=yuv420p[outv]'

    if kf_box:
        fx0, fy0, fx1, fy1 = kf_box
        kf_crop = f'crop=iw*{fx1 - fx0:.4f}:ih*{fy1 - fy0:.4f}:iw*{fx0:.4f}:ih*{fy0:.4f}'
    elif game == 'cs2':     # HUD-relative fallback: killfeed width ~0.43 x frame height
        kf_crop = 'crop=min(iw*0.45\\,ih*0.44):ih*0.17:iw-min(iw*0.45\\,ih*0.44):ih*0.005'
    else:
        kf_crop = 'crop=min(iw*0.5\\,ih*0.60):ih*0.22:iw-min(iw*0.5\\,ih*0.60):ih*0.02'

    if game == 'cs2':
        # blurred backdrop + centred 1080x1400 gameplay + killfeed card top-right
        return (
            '[vc]split=3[b][m][k];'
            '[b]scale=1080:1920:force_original_aspect_ratio=increase:flags=bicubic,crop=1080:1920,gblur=sigma=20[bg];'
            '[m]crop=min(iw\\,ih*1080/1400):ih:(iw-min(iw\\,ih*1080/1400))/2:0,scale=1080:1400:flags=lanczos[main];'
            f'[k]{kf_crop},scale=480:-2:flags=lanczos[kf];'
            '[bg][main]overlay=0:(H-h)/2[mid];'
            f'[mid][kf]overlay=W-w-30:80,fps={fps_s},format=yuv420p[outv]'
        )
    # valorant: natural centre focus + enlarged killfeed on top
    return (
        '[vc]split=2[m][k];'
        '[m]scale=-2:1920:flags=lanczos,crop=1080:1920:(iw-1080)/2:0[main];'
        f'[k]{kf_crop},scale=900:-2:flags=lanczos[kf];'
        f'[main][kf]overlay=(W-w)/2:140,fps={fps_s},format=yuv420p[outv]'
    )


def build_command(src, dst, segments, info, fmt, game, kf_box):
    n = len(segments)
    parts = [f'[0:v]split={n}' + ''.join(f'[sv{i}]' for i in range(n)) + ';']
    if info.has_audio:
        parts.append(f'[0:a]asplit={n}' + ''.join(f'[sa{i}]' for i in range(n)) + ';')
    for i, (s, e) in enumerate(segments):
        parts.append(f'[sv{i}]trim=start={s:.6f}:end={e:.6f},setpts=PTS-STARTPTS[v{i}];')
        if info.has_audio:
            parts.append(f'[sa{i}]atrim=start={s:.6f}:end={e:.6f},asetpts=PTS-STARTPTS[a{i}];')
    if info.has_audio:
        parts.append(''.join(f'[v{i}][a{i}]' for i in range(n)) + f'concat=n={n}:v=1:a=1[vc][ac];')
        parts.append('[ac]aresample=48000:async=1:first_pts=0[outa];')
    else:
        parts.append(''.join(f'[v{i}]' for i in range(n)) + f'concat=n={n}:v=1:a=0[vc];')
    parts.append(layout_filter(fmt, game, kf_box, info.fps))

    # Tuned for a 2 vCPU / 4 GB host: 2 threads each for decode, filter graph and x264, a short
    # lookahead and a bounded mux queue (measured: 351 -> 271 MB peak RSS, slightly faster).
    cmd = ['ffmpeg', '-y', '-v', 'error', '-nostdin', '-threads', str(FFMPEG_THREADS), '-i', src,
           '-filter_complex_threads', str(FFMPEG_THREADS), '-filter_complex', ''.join(parts), '-map', '[outv]']
    if info.has_audio:
        cmd += ['-map', '[outa]', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000']
    cmd += ['-c:v', 'libx264', '-preset', X264_PRESET, '-crf', '20', '-pix_fmt', 'yuv420p',
            '-threads', str(FFMPEG_THREADS), '-rc-lookahead', '10',
            '-max_muxing_queue_size', '512', '-movflags', '+faststart', dst]
    return cmd


def render(src, dst, segments, info, fmt, game, kf_box=None, timeout=1800):
    cmd = build_command(src, dst, segments, info, fmt, game, kf_box)
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError('ffmpeg failed: ' + res.stderr.strip()[-600:])
    return dst
