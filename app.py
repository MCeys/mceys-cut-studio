"""
MCEYS Cut Studio - backend API.

POST /api/process   (multipart/form-data)
    video   file      required
    game    cs2 | valorant                      required
    format  9-16 | 16-9                         default 9-16
    mode    valorant: team | solo | headshot | multikill   (default team)
            cs2:      solo | multikill                     (default solo)
-> 200 video/mp4 attachment
-> 4xx/5xx {"error": "..."}  (shown to the user by the frontend)
"""
import logging
import os
import threading
import uuid

from flask import Flask, jsonify, request, send_file

import kill_engine as ke
import renderer

try:                                  # optional; frontend calls the API same-origin
    from flask_cors import CORS
except ImportError:                   # pragma: no cover
    CORS = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
HEADSHOT_TEMPLATE = os.path.join(TEMPLATES_DIR, 'headshot.png')
for d in (UPLOAD_DIR, OUTPUT_DIR, TEMPLATES_DIR):
    os.makedirs(d, exist_ok=True)

# One heavy scan+render at a time per process (gunicorn: 1 worker, 2 threads). The second
# thread stays free for fast responses; extra heavy jobs wait in line instead of running
# in parallel and exhausting the 4 GB host.
HEAVY_JOB = threading.BoundedSemaphore(1)
QUEUE_WAIT_S = 1200

ALLOWED_EXT = {'.mp4', '.mov', '.mkv', '.webm', '.m4v'}
FORMATS = {'9-16', '16-9'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024      # matches nginx client_max_body_size
if CORS:
    CORS(app)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('cutstudio')


def error(msg, code):
    return jsonify({'error': msg}), code


def _remove(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except OSError:
            log.warning('could not remove %s', p)


@app.route('/api/process', methods=['POST'])
def handle_process():
    file = request.files.get('video')
    if file is None:
        return error('Video dosyası bulunamadı', 400)
    if not file.filename:
        return error('Dosya seçilmedi', 400)

    game = (request.form.get('game') or '').strip().lower()
    if game not in ke.ROI:
        return error("Geçersiz veya eksik 'game' parametresi (cs2 veya valorant olmalı)", 400)
    fmt = (request.form.get('format') or '9-16').strip()
    if fmt not in FORMATS:
        return error("Geçersiz 'format' parametresi (9-16 veya 16-9 olmalı)", 400)
    mode = ke.normalize_mode(game, request.form.get('mode'))
    if mode is None:
        return error(f"Geçersiz 'mode' parametresi. {game.upper()} için: "
                     + ', '.join(sorted(ke.MODES[game])), 400)

    uid = uuid.uuid4().hex[:10]
    ext = os.path.splitext(file.filename)[1].lower()
    input_path = os.path.join(UPLOAD_DIR, f'input_{uid}{ext if ext in ALLOWED_EXT else ".mp4"}')
    output_path = os.path.join(OUTPUT_DIR, f'montage_{uid}.mp4')
    file.save(input_path)
    log.info('[%s] queued game=%s format=%s mode=%s file=%s', uid, game, fmt, mode, file.filename)

    if not HEAVY_JOB.acquire(timeout=QUEUE_WAIT_S):
        _remove(input_path)
        return error('Sunucu şu an başka bir videoyu işliyor, lütfen birkaç dakika sonra tekrar deneyin.', 503)
    log.info('[%s] job start', uid)
    try:
        try:
            info = ke.probe(input_path)
        except ValueError:
            return error('Video okunamadı. Lütfen geçerli bir MP4/MOV/MKV/WebM dosyası yükleyin.', 400)

        scan = ke.scan_video(input_path, game, info, HEADSHOT_TEMPLATE)
        if mode == 'headshot' and not scan.headshot_supported:
            return error('Headshot modu için templates/headshot.png şablonu eksik veya geçersiz.', 400)
        times = ke.select_kills(scan.kills, game, mode, renderer.MERGE_GAP)
        log.info('[%s] %d banner kills, %d kept for mode=%s: %s', uid, len(scan.kills), len(times), mode,
                 ', '.join(f'{t:.1f}s' for t in times))
        segments = renderer.build_segments(times, info.duration, info.fps)
        if not segments:
            return error(f'{game.upper()} videosunda seçilen moda ({mode}) uygun kill bulunamadı.', 422)

        log.info('[%s] segments: %s', uid, ', '.join(f'{s:.2f}-{e:.2f}' for s, e in segments))
        renderer.render(input_path, output_path, segments, info, fmt, game, scan.killfeed_box)
    except Exception:
        log.exception('[%s] processing failed', uid)
        _remove(output_path)
        return error(f'{game.upper()} videosu işlenirken bir sorun oluştu', 500)
    finally:
        HEAVY_JOB.release()
        _remove(input_path)

    resp = send_file(output_path, mimetype='video/mp4', as_attachment=True,
                     download_name=f'{game}_{fmt}_{uid}.mp4')
    resp.call_on_close(lambda: _remove(output_path))
    log.info('[%s] done', uid)
    return resp


@app.errorhandler(413)
def too_large(_):
    return error('Dosya çok büyük (en fazla 1 GB).', 413)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 5001)), debug=False)
