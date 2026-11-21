"""
MCEYS Cut Studio - killfeed detection engine (CS2 + Valorant).

Pipeline
--------
1. ``probe()`` reads resolution / fps / duration / audio presence with ffprobe.
2. ``iter_roi_frames()`` lets ffmpeg decode the video, sample it at SAMPLE_FPS,
   crop the top-right killfeed region and rescale it so that the *frame height*
   maps to 1080 px. Every threshold below is therefore expressed in "1080p
   units" and works for 720p / 1440p / 4K and for 4:3 or 4:3-stretched footage.
3. A per-game banner detector turns each ROI frame into a list of candidate
   killfeed banners (multi-stage: colour mask -> geometry -> structure ->
   glyph contrast).
4. ``BannerTracker`` follows banners from frame to frame by their text/icon
   "glyph signature". A banner only becomes a kill after it was seen in
   CONFIRM_N consecutive samples (temporal debounce), and a banner that briefly
   disappears (flashbang, UI flicker) is not counted twice.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field

import cv2
import numpy as np

SAMPLE_FPS = 10          # analysis rate
CONFIRM_N = 3            # consecutive samples a banner must be visible (0.3 s)
MAX_MISS = 2             # a confirmed banner may vanish this many samples before it is retired
MEMORY_S = 3.0           # a retired banner that re-appears within this window is the same kill
SIG_SIM = 0.85           # glyph-signature correlation needed to call two banners "the same"
NORM_H = 1080            # all geometry below is in 1080p units

# Killfeed search regions as fractions of the frame (x0, y0, x1, y1)
ROI = {
    'valorant': (0.50, 0.00, 1.00, 0.40),
    'cs2': (0.55, 0.00, 1.00, 0.35),
}

# ------------------------------------------------------------------ colour ranges (OpenCV HSV: H 0-180)
WHITE = ((0, 0, 205), (180, 60, 255))

VAL_TEAL = ((70, 60, 120), (95, 215, 255))              # ally side of a Valorant banner
VAL_RED = (((0, 100, 150), (9, 235, 255)),              # enemy side
           ((171, 100, 150), (180, 235, 255)))
VAL_YELLOW = ((22, 60, 160), (40, 215, 255))            # local-player highlight border

CS2_RED = (((0, 130, 130), (10, 255, 255)),             # local-player kill frame
           ((170, 130, 130), (180, 255, 255)))

VAL_H = (20, 66)      # banner height range (1080p units) -> HUD scale ~0.55x-1.7x
CS2_H = (16, 60)


# ================================================================== video access
@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool
    start_time: float = 0.0


def _ratio(txt: str | None) -> float:
    try:
        n, d = (txt or '0/0').split('/')
        return float(n) / float(d) if float(d) else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe(path: str) -> VideoInfo:
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries',
         'stream=codec_type,width,height,avg_frame_rate,r_frame_rate:format=duration,start_time',
         '-of', 'json', path],
        capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise ValueError('ffprobe failed: ' + out.stderr.strip()[:300])
    data = json.loads(out.stdout or '{}')
    streams = data.get('streams', [])
    video = next((s for s in streams if s.get('codec_type') == 'video'), None)
    if not video or not video.get('width'):
        raise ValueError('no video stream')
    fps = _ratio(video.get('avg_frame_rate')) or _ratio(video.get('r_frame_rate'))
    if not 1 <= fps <= 240:
        fps = 30.0
    fmt = data.get('format', {})
    return VideoInfo(
        width=int(video['width']), height=int(video['height']), fps=fps,
        duration=float(fmt.get('duration') or 0.0),
        has_audio=any(s.get('codec_type') == 'audio' for s in streams),
        start_time=float(fmt.get('start_time') or 0.0),
    )


def roi_geometry(info: VideoInfo, game: str):
    """Pixel crop of the killfeed ROI and its normalized output size."""
    fx0, fy0, fx1, fy1 = ROI[game]
    x0, y0 = int(info.width * fx0), int(info.height * fy0)
    w = (int(info.width * fx1) - x0) // 2 * 2
    h = (int(info.height * fy1) - y0) // 2 * 2
    scale = NORM_H / info.height
    out_w = max(2, int(round(w * scale / 2)) * 2)
    out_h = max(2, int(round(h * scale / 2)) * 2)
    return (x0, y0, w, h), (out_w, out_h), scale


def iter_roi_frames(path: str, info: VideoInfo, game: str, sample_fps: int = SAMPLE_FPS):
    """Yield (timestamp_seconds, roi_bgr) for the killfeed region, decoded by ffmpeg."""
    (x0, y0, w, h), (out_w, out_h), _ = roi_geometry(info, game)
    vf = f'fps={sample_fps},crop={w}:{h}:{x0}:{y0},scale={out_w}:{out_h}:flags=area'
    cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-i', path, '-an', '-sn', '-dn',
           '-vf', vf, '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
    frame_bytes = out_w * out_h * 3
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        try:
            i = 0
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                yield i / sample_fps, np.frombuffer(buf, np.uint8).reshape(out_h, out_w, 3)
                i += 1
        finally:
            proc.stdout.close()
            proc.kill() if proc.poll() is None else None
            proc.wait()


# ================================================================== detection helpers
def _in(hsv, rng):
    return cv2.inRange(hsv, np.array(rng[0], np.uint8), np.array(rng[1], np.uint8))


def _in2(hsv, ranges):
    return cv2.bitwise_or(_in(hsv, ranges[0]), _in(hsv, ranges[1]))


def _frac(mask) -> float:
    return float(np.count_nonzero(mask)) / mask.size if mask.size else 0.0


def glyph_signature(white_crop, size=(96, 12)):
    """Zero-mean, unit-norm vector of the white text/icon layout of a banner.

    Built from the white glyph mask only, so it stays stable when a translucent
    banner (CS2) moves over a changing game background."""
    if white_crop.size == 0:
        return None
    v = cv2.resize(white_crop.astype(np.float32) / 255.0, size, interpolation=cv2.INTER_AREA).ravel()
    v -= v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else None


@dataclass
class Banner:
    box: tuple          # (x0, y0, x1, y1) in normalized ROI pixels
    sig: np.ndarray
    mine: bool = True
    headshot: bool = False


# ------------------------------------------------------------------ Valorant
class HeadshotMatcher:
    """Optional headshot-icon matcher; only active with a usable template."""

    def __init__(self, template_path: str | None):
        self.tpl = None
        if template_path and os.path.exists(template_path):
            img = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
            if img is not None and min(img.shape[:2]) >= 10:
                if img.ndim == 3 and img.shape[2] == 4:
                    alpha = img[:, :, 3]
                    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
                    gray[alpha < 128] = 0
                else:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                _, self.tpl = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    @property
    def available(self) -> bool:
        return self.tpl is not None

    def match(self, white_band, banner_h: int) -> bool:
        if self.tpl is None or white_band.size == 0:
            return False
        th, tw = self.tpl.shape
        target_h = max(6, int(banner_h * 0.62))           # icon is ~60% of banner height
        tpl = cv2.resize(self.tpl, (max(4, int(tw * target_h / th)), target_h))
        if tpl.shape[0] > white_band.shape[0] or tpl.shape[1] > white_band.shape[1]:
            return False
        res = cv2.matchTemplate(white_band, tpl, cv2.TM_CCOEFF_NORMED)
        return float(res.max()) >= 0.72


def _wsum(mask01, L):
    """Sliding horizontal window sums: out[:, j] = sum(mask01[:, j:j+L])."""
    c = np.cumsum(mask01, axis=1, dtype=np.int32)
    c = np.concatenate([np.zeros((c.shape[0], 1), np.int32), c], axis=1)
    return c[:, L:] - c[:, :-L]


def _row_bands(ok_rows, max_gap=1):
    """Group True rows into [start, end] bands, bridging gaps of <= max_gap rows."""
    idx = np.flatnonzero(ok_rows)
    bands = []
    for r in idx:
        if bands and r - bands[-1][1] <= max_gap + 1:
            bands[-1][1] = r
        else:
            bands.append([r, r])
    return bands


VAL_LT, VAL_LT_WIDE, VAL_LR, VAL_LR_WIDE = 36, 110, 18, 60      # window lengths in 1080p units


def detect_valorant(roi, headshot: HeadshotMatcher | None = None) -> list[Banner]:
    """Row-scan detector: a killfeed banner row is a run of ally colour (+white
    glyphs) that turns directly into a run of enemy red. Background colour that
    merely touches the banner can never produce that transition, so the banner
    cannot merge with the scenery."""
    H, W = roi.shape[:2]
    if W < VAL_LT_WIDE + VAL_LR + 2:
        return []
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    teal = _in(hsv, VAL_TEAL) > 0
    red = _in2(hsv, VAL_RED) > 0
    white = _in(hsv, WHITE) > 0
    yellow = _in(hsv, VAL_YELLOW) > 0
    tw, rw = (teal | white), (red | white)

    # Stage 1 - per-row teal->red transition
    n = W - VAL_LT_WIDE - VAL_LR + 1
    tw_s = _wsum(tw.astype(np.uint8), VAL_LT)[:, VAL_LT_WIDE - VAL_LT:VAL_LT_WIDE - VAL_LT + n]
    tt_s = _wsum(teal.astype(np.uint8), VAL_LT_WIDE)[:, :n]
    rw_s = _wsum(rw.astype(np.uint8), VAL_LR)[:, VAL_LT_WIDE:VAL_LT_WIDE + n]
    rr_s = _wsum(red.astype(np.uint8), VAL_LR)[:, VAL_LT_WIDE:VAL_LT_WIDE + n]
    red_pad = np.pad(red.astype(np.uint8), ((0, 0), (0, VAL_LR_WIDE)))
    rr_wide = _wsum(red_pad, VAL_LR_WIDE)[:, VAL_LT_WIDE:VAL_LT_WIDE + n]
    # rows crossing the victim name are mostly white glyphs -> accept a sparser red there
    red_ok = (rw_s >= 0.85 * VAL_LR) & ((rr_s >= 0.40 * VAL_LR) | (rr_wide >= 0.12 * VAL_LR_WIDE))
    cond = (tw_s >= 0.85 * VAL_LT) & (tt_s >= 0.15 * VAL_LT_WIDE) & red_ok
    row_ok = cond.any(axis=1)

    found = []
    for y0, y1 in _row_bands(row_ok, max_gap=3):
        h = y1 - y0 + 1
        if not VAL_H[0] <= h <= VAL_H[1]:
            continue
        # consensus boundary: the column window (chevron width) hit by the most rows of the
        # band. Scenery transitions are scattered; the banner edge is shared by every row.
        ww = int(1.2 * h) + 1
        bc = cond[y0:y1 + 1]
        csum = np.convolve(bc.sum(axis=0), np.ones(ww, np.int32), mode='valid')
        k0 = int(np.argmax(csum))
        win = bc[:, k0:k0 + ww]
        rows_hit = win.any(axis=1)
        if rows_hit.sum() < 0.6 * h:
            continue
        first = np.argmax(win[rows_hit], axis=1) + k0 + VAL_LT_WIDE - 1    # left-most hit per row
        bx_in, bx = int(first.min()), int(first.max())                     # chevron root .. tip
        band = slice(y0, y1 + 1)
        # Stage 2 - extent of the ally section: walk left from the chevron root while the
        # columns stay ally colour / white glyphs (stops at the killer portrait)
        colfrac = tw[band, :bx_in + 1].mean(axis=0)
        x, misses, tol = bx_in, 0, max(6, int(0.3 * h))
        while x > 0 and misses < tol:
            x -= 1
            misses = misses + 1 if colfrac[x] < 0.6 else 0
        x += misses
        w = bx - x + 1
        if not 2.0 * h <= w <= 18 * h:
            continue
        sl = (band, slice(x, bx + 1))
        if teal[sl].mean() < 0.30 or not 0.03 <= white[sl].mean() <= 0.60:   # glyph contrast
            continue
        # enemy section length
        rfrac = rw[band, bx + 1:].mean(axis=0)
        run = int(np.argmax(rfrac < 0.5)) if (rfrac < 0.5).any() else rfrac.size
        if run < 1.0 * h:
            continue
        x_end = min(W, bx + 1 + run)
        # Stage 3 - local-player highlight (yellow frame / left chevron)
        yb = max(2, int(0.3 * h))
        top = yellow[max(0, y0 - yb):y0, x:x_end]
        bot = yellow[y1 + 1:min(H, y1 + 1 + yb), x:x_end]
        left = yellow[max(0, y0 - yb):min(H, y1 + 1 + yb), max(0, x - int(1.8 * h)):x + int(0.6 * h)]
        y_frame = min(top.mean() if top.size else 0, bot.mean() if bot.size else 0)
        y_left = left.mean() if left.size else 0
        mine = y_frame >= 0.10 or y_left >= 0.12
        sig = glyph_signature(white[band, x:x_end].astype(np.uint8) * 255)
        if sig is None:
            continue
        hs = headshot.match(white[sl].astype(np.uint8) * 255, h) if headshot else False
        found.append(Banner((max(0, x - int(1.5 * h)), max(0, y0 - yb), x_end, min(H, y1 + 1 + yb)), sig, mine, hs))
    return found


# ------------------------------------------------------------------ CS2
def _edge_cover(m):
    """Best single-row coverage of a horizontal strip (rows x cols)."""
    return float(m.mean(axis=1).max()) if m.size else 0.0


def _cs2_candidate(red, white, H, x, y, w, h):
    if not (CS2_H[0] <= h <= CS2_H[1] and 3 * h <= w <= 24 * h):
        return None
    inner = white[y + 3:y + h - 3, x + 3:x + w - 3]
    if not 0.02 <= _frac(inner) <= 0.60:                       # killfeed text inside the frame
        return None
    o = 3                                                      # crisp outer edge
    if (_frac(red[max(0, y - o - 1):max(0, y - 1), x:x + w]) > 0.35 or
            _frac(red[min(H, y + h + 1):min(H, y + h + 1 + o), x:x + w]) > 0.35):
        return None
    sig = glyph_signature(inner)
    return Banner((x, y, x + w, y + h), sig, True, False) if sig is not None else None


def detect_cs2(roi) -> list[Banner]:
    H, W = roi.shape[:2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red = _in2(hsv, CS2_RED)
    white = _in(hsv, WHITE)
    # split red into thick blobs (smoke, walls, filled entries) and thin strokes (frames)
    thick = cv2.morphologyEx(red, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    thin = cv2.bitwise_and(red, cv2.bitwise_not(cv2.dilate(thick, np.ones((3, 3), np.uint8))))
    found: list[Banner] = []

    # A) outlined entry: four thin red edges
    tc = cv2.morphologyEx(thin, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    for c in cv2.findContours(tc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, h = cv2.boundingRect(c)
        if not (CS2_H[0] <= h <= CS2_H[1] and 3 * h <= w <= 24 * h):
            continue
        t = max(3, int(round(h * 0.15)))
        m = tc > 0
        edges = sorted([_edge_cover(m[y:y + t, x:x + w]), _edge_cover(m[y + h - t:y + h, x:x + w]),
                        _edge_cover(m[y:y + h, x:x + t].T), _edge_cover(m[y:y + h, x + w - t:x + w].T)])
        if edges[1] < 0.75 or edges[0] < 0.35:        # 3 strong edges, 4th may be clipped by scenery
            continue
        b = _cs2_candidate(red, white, H, x, y, w, h)
        if b:
            found.append(b)

    # B) filled entry: solid red rectangle carrying text
    for c in cv2.findContours(thick, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, h = cv2.boundingRect(c)
        if not (CS2_H[0] <= h <= CS2_H[1] and 3 * h <= w <= 24 * h):
            continue
        if _frac(cv2.bitwise_or(red, white)[y:y + h, x:x + w]) < 0.85:
            continue
        b = _cs2_candidate(red, white, H, x, y, w, h)
        if b and not any(_iou(b.box, f.box) > 0.5 for f in found):
            found.append(b)
    return found


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


# ================================================================== temporal tracking
@dataclass
class Track:
    sig: np.ndarray
    first_t: float
    last_t: float
    hits: int = 1
    miss: int = 0
    confirmed: bool = False
    mine_votes: int = 0
    hs_votes: int = 0
    boxes: list = field(default_factory=list)


@dataclass
class KillEvent:
    t: float
    mine: bool = True
    headshot: bool = False


class BannerTracker:
    def __init__(self, confirm_n=CONFIRM_N, max_miss=MAX_MISS, memory_s=MEMORY_S, sim=SIG_SIM):
        self.confirm_n, self.max_miss, self.memory_s, self.sim = confirm_n, max_miss, memory_s, sim
        self.tracks: list[Track] = []
        self.recent: list[tuple[np.ndarray, float]] = []
        self.events: list[KillEvent] = []
        self.boxes: list[tuple] = []

    def update(self, t: float, banners: list[Banner]):
        # one-to-one greedy matching by signature similarity
        pairs = sorted(((float(np.dot(tr.sig, b.sig)), i, j)
                        for i, tr in enumerate(self.tracks) for j, b in enumerate(banners)), reverse=True)
        used_t, used_b = set(), set()
        for score, i, j in pairs:
            if score < self.sim:
                break
            if i in used_t or j in used_b:
                continue
            used_t.add(i)
            used_b.add(j)
            tr, b = self.tracks[i], banners[j]
            tr.hits += 1
            tr.miss = 0
            tr.last_t = t
            tr.sig = b.sig
            tr.mine_votes += b.mine
            tr.hs_votes += b.headshot
            tr.boxes.append(b.box)
        n_old = len(self.tracks)
        for i in range(n_old):
            if i not in used_t:
                self.tracks[i].miss += 1
        for j, b in enumerate(banners):
            if j not in used_b:
                self.tracks.append(Track(b.sig, t, t, mine_votes=int(b.mine), hs_votes=int(b.headshot), boxes=[b.box]))

        for tr in self.tracks:
            if not tr.confirmed and tr.miss == 0 and tr.hits >= self.confirm_n:
                tr.confirmed = True
                if not self._seen_recently(tr.sig, t):
                    self.events.append(KillEvent(tr.first_t, tr.mine_votes * 2 >= tr.hits, tr.hs_votes * 2 >= tr.hits))
                    self.boxes.extend(tr.boxes)

        alive = []
        for tr in self.tracks:
            if tr.miss == 0 or (tr.confirmed and tr.miss <= self.max_miss):
                alive.append(tr)
            elif tr.confirmed:
                self.recent.append((tr.sig, tr.last_t))
            # unconfirmed tracks that miss a single sample are dropped -> consecutive-frame rule
        self.tracks = alive
        self.recent = [(s, rt) for s, rt in self.recent if t - rt <= self.memory_s]

    def _seen_recently(self, sig, t) -> bool:
        for k, (s, rt) in enumerate(self.recent):
            if t - rt <= self.memory_s and float(np.dot(s, sig)) >= self.sim:
                del self.recent[k]
                return True
        return False


# ================================================================== public API
@dataclass
class ScanResult:
    kills: list[KillEvent]
    killfeed_box: tuple | None   # (fx0, fy0, fx1, fy1) as fractions of the full frame
    info: VideoInfo
    headshot_supported: bool = False


def scan_video(path: str, game: str, info: VideoInfo | None = None,
               headshot_template: str | None = None) -> ScanResult:
    if game not in ROI:
        raise ValueError(f'unsupported game: {game}')
    info = info or probe(path)
    hs = HeadshotMatcher(headshot_template) if game == 'valorant' else None
    tracker = BannerTracker()
    for t, roi in iter_roi_frames(path, info, game):
        banners = detect_valorant(roi, hs) if game == 'valorant' else detect_cs2(roi)
        tracker.update(t, banners)
    return ScanResult(tracker.events, _killfeed_box(tracker.boxes, info, game), info,
                      bool(hs and hs.available))


def _killfeed_box(boxes, info: VideoInfo, game: str):
    """Union of every confirmed banner, mapped back to full-frame fractions."""
    if not boxes:
        return None
    (x0, y0, w, h), _, scale = roi_geometry(info, game)
    bx0 = min(b[0] for b in boxes) / scale + x0
    by0 = min(b[1] for b in boxes) / scale + y0
    bx1 = max(b[2] for b in boxes) / scale + x0
    by1 = max(b[3] for b in boxes) / scale + y0
    pad_x, pad_y = (bx1 - bx0) * 0.04, (by1 - by0) * 0.08
    return (max(0.0, (bx0 - pad_x) / info.width), max(0.0, (by0 - pad_y) / info.height),
            min(1.0, (bx1 + pad_x) / info.width), min(1.0, (by1 + pad_y) / info.height))


MODES = {
    'valorant': {'team', 'solo', 'headshot', 'multikill'},
    'cs2': {'solo', 'multikill'},
}
MODE_ALIASES = {'all': 'team', 'mine': 'solo', 'multi': 'multikill', 'clutch': 'multikill'}
DEFAULT_MODE = {'valorant': 'team', 'cs2': 'solo'}


def normalize_mode(game: str, mode: str | None) -> str | None:
    m = (mode or DEFAULT_MODE[game]).strip().lower()
    m = MODE_ALIASES.get(m, m)
    return m if m in MODES[game] else None


def cluster(times: list[float], gap: float) -> list[list[float]]:
    out: list[list[float]] = []
    for t in sorted(times):
        if out and t - out[-1][-1] <= gap:
            out[-1].append(t)
        else:
            out.append([t])
    return out


def select_kills(kills: list[KillEvent], game: str, mode: str, merge_gap: float) -> list[float]:
    """Apply the requested kill filter and return the kept timestamps."""
    if game == 'valorant':
        if mode == 'team':
            pool = kills
        elif mode == 'headshot':
            pool = [k for k in kills if k.headshot]
        else:                                   # solo / multikill -> local player's kills
            pool = [k for k in kills if k.mine]
    else:
        pool = kills                            # CS2 frame = local player's kills only
    times = [k.t for k in pool]
    if mode == 'multikill':
        times = [t for grp in cluster(times, merge_gap) if len(grp) >= 2 for t in grp]
    return sorted(times)
