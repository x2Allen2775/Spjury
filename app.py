#!/usr/bin/env python3
"""
SPJURY - AI Powered Sports Injury Predictor
Flask Backend Application
"""

from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import uuid
import threading
import traceback
import subprocess
import shutil


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max upload

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# In-memory task store
tasks = {}


def reencode_h264(input_path):
    """
    Re-encode video to H.264 so browsers can play it.
    OpenCV saves as mp4v (MPEG-4 Part 2) which no browser supports.
    Returns path to the browser-compatible file (replaces original).
    """
    if not os.path.exists(input_path):
        return input_path

    tmp_path = input_path.replace('.mp4', '_h264.mp4')
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-movflags', '+faststart',   # enables streaming / instant play in browser
            '-an',                        # no audio track needed
            tmp_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode == 0 and os.path.exists(tmp_path):
            os.replace(tmp_path, input_path)   # swap in place
            print(f"[ffmpeg] Re-encoded to H.264: {input_path}")
        else:
            print(f"[ffmpeg] Re-encode failed: {result.stderr.decode()[:300]}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        print(f"[ffmpeg] Exception: {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
    return input_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    sport = request.form.get('sport', '').lower()
    if sport not in ('bowling', 'batting', 'tennis'):
        return jsonify({'error': f'Unknown sport: {sport}'}), 400

    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400

    task_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower() or '.mp4'
    video_path = os.path.join(UPLOAD_DIR, f'{task_id}{ext}')
    file.save(video_path)

    task_output_dir = os.path.join(OUTPUT_DIR, task_id)
    os.makedirs(task_output_dir, exist_ok=True)

    tasks[task_id] = {
        'status': 'processing',
        'progress': 0,
        'sport': sport,
        'results': None,
        'error': None
    }

    thread = threading.Thread(
        target=_run_analysis,
        args=(task_id, sport, video_path, task_output_dir),
        daemon=True
    )
    thread.start()

    return jsonify({'task_id': task_id})


def _run_analysis(task_id, sport, video_path, output_dir):
    def progress_cb(p):
        if task_id in tasks:
            tasks[task_id]['progress'] = min(int(p), 99)

    try:
        if sport == 'bowling':
            from analyzers.bowling_module import BowlingAnalysisModule
            results = BowlingAnalysisModule().analyze(video_path, output_dir, progress_cb)
        elif sport == 'batting':
            from analyzers.batting_module import BattingAnalysisModule
            results = BattingAnalysisModule().analyze(video_path, output_dir, progress_cb)
        elif sport == 'tennis':
            from analyzers.tennis_module import TennisAnalysisModule
            results = TennisAnalysisModule().analyze(video_path, output_dir, progress_cb)
        else:
            raise ValueError(f"Unknown sport: {sport}")

        # Re-encode output video to H.264 so the browser can play it
        if 'video_path' in results and results['video_path']:
            tasks[task_id]['progress'] = 95
            reencode_h264(results['video_path'])

        tasks[task_id]['status'] = 'complete'
        tasks[task_id]['progress'] = 100
        tasks[task_id]['results'] = results

        # Clean up uploaded video
        try:
            os.remove(video_path)
        except:
            pass

    except Exception as e:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['error'] = str(e)
        tasks[task_id]['traceback'] = traceback.format_exc()
        print(f"[ERROR] task {task_id}: {e}")
        traceback.print_exc()


@app.route('/api/status/<task_id>')
def status(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    t = tasks[task_id]
    return jsonify({
        'status': t['status'],
        'progress': t['progress'],
        'sport': t['sport'],
        'error': t.get('error')
    })


@app.route('/api/results/<task_id>')
def results(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    t = tasks[task_id]
    if t['status'] != 'complete':
        return jsonify({'error': f"Not ready. Status: {t['status']}"}), 400

    # Convert file paths to URLs
    res = dict(t['results'])
    task_output_dir = os.path.join(OUTPUT_DIR, task_id)

    # Build URL prefix
    res['task_id'] = task_id
    if 'graphs' in res and res['graphs']:
        for g in res['graphs']:
            g['url'] = f'/outputs/{task_id}/{g["url"]}'
    if 'video_url' in res:
        res['video_url'] = f'/outputs/{task_id}/{res["video_url"]}'

    return jsonify(res)


@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)




@app.route('/api/chat/status')
def chat_status():
    """Check if GROQ_API_KEY is configured on the server."""
    key = os.environ.get('GROQ_API_KEY', '').strip()
    if key:
        return jsonify({'enabled': True})
    return jsonify({'enabled': False, 'reason': 'GROQ_API_KEY not set. Get a free key at console.groq.com then run: export GROQ_API_KEY=gsk_...'}), 503

@app.route('/api/chat', methods=['POST'])
def chat_proxy():
    """Proxy chat requests to Groq using the server-side API key."""
    try:
        api_key = os.environ.get('GROQ_API_KEY', '').strip()
        if not api_key:
            return jsonify({'error': 'GROQ_API_KEY not set. Run: export GROQ_API_KEY=gsk_... then restart.'}), 503

        body = request.get_json(force=True, silent=True) or {}
        messages = body.get('messages', [])
        if not messages:
            return jsonify({'error': 'No messages provided.'}), 400

        import urllib.request as _ur
        import json as _js
        payload = _js.dumps({
            'model': 'llama-3.1-8b-instant',
            'messages': messages,
            'max_tokens': 600,
            'temperature': 0.7
        }).encode('utf-8')

        req = _ur.Request(
            'https://api.groq.com/openai/v1/chat/completions',
            data=payload,
            headers={
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + api_key,
                'User-Agent': 'Mozilla/5.0 (compatible; SPJURY/1.0)',
                'Accept': 'application/json',
            },
            method='POST'
        )
        try:
            with _ur.urlopen(req, timeout=30) as r:
                result = _js.loads(r.read().decode('utf-8'))
            answer = result['choices'][0]['message']['content'].strip()
            return jsonify({'answer': answer})
        except _ur.HTTPError as he:
            body_bytes = he.read()
            try:
                err_msg = _js.loads(body_bytes).get('error', {}).get('message', body_bytes.decode())
            except Exception:
                err_msg = body_bytes.decode()
            print(f'[CHAT] Groq HTTP {he.code}: {err_msg}')
            return jsonify({'error': f'Groq {he.code}: {err_msg}'}), 502

    except Exception as e:
        import traceback as _tb
        _tb.print_exc()
        return jsonify({'error': f'{type(e).__name__}: {str(e)}'}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("  SPJURY - AI Powered Sports Injury Predictor")
    print("  Presented by Team SPJURY")
    print("=" * 60)
    print(f"  Open: http://localhost:5000")
    # Check for Groq key and print helpful message
    groq_key = os.environ.get('GROQ_API_KEY', '').strip()
    if groq_key:
        print(f"  AI Chat: GROQ_API_KEY found ✓ ({groq_key[:8]}...)")
    else:
        print("  AI Chat: GROQ_API_KEY not set — chat will be disabled")
        print("           Get free key at console.groq.com then run:")
        print("           export GROQ_API_KEY=gsk_...")
    print("=" * 60)
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
