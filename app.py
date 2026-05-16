"""
EleGuard Dashboard - Multi-Modal Wildlife Monitoring System

This Streamlit application monitors multiple video feeds, performs motion detection,
analyzes audio for significant noise, and uses a local LLM (Gemma) for inference 
to detect the presence of elephants or other threats.
"""
import cv2
import time
import base64
import queue
import threading
import requests
import html
import streamlit as st
import librosa
import numpy as np
from moviepy import VideoFileClip

st.set_page_config(page_title="EleGuard Dashboard", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

html, body, [data-testid="stApp"] {
    background: #080c18 !important;
    font-family: 'Inter', sans-serif !important;
}
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stHeader"] { display: none !important; }

.main .block-container { padding-top: 0rem !important; padding-bottom: 1.2rem !important; padding-left: 2rem !important; padding-right: 2rem !important; max-width: 100% !important; }

/* ── Page header ── */
.eleguard-header {
    display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; gap: 8px;
    padding: 18px 28px;
    background: linear-gradient(135deg, #0f1629 0%, #1a2040 100%);
    border: 1px solid #2a3560;
    border-radius: 16px;
    margin-bottom: 20px;
    margin-top: 0px;
}
.eleguard-logo { font-size: 40px; }
.eleguard-title { font-size: 26px; font-weight: 800; color: #e2e8f0; letter-spacing: -0.5px; }
.eleguard-sub { font-size: 13px; color: #64748b; font-weight: 400; margin-top: 2px; }

/* ── Camera card ── */
.cam-card {
    background: linear-gradient(160deg, #0f1629 0%, #131c35 100%);
    border: 1px solid #1e2d50;
    border-radius: 14px;
    padding: 14px 14px 10px;
    margin-bottom: 10px;
}
.cam-card-title {
    font-size: 11px; font-weight: 700; letter-spacing: 2px;
    color: #4a6fa5; text-transform: uppercase; margin-bottom: 10px;
    display: flex; align-items: center; gap: 8px;
}
.cam-dot { width: 8px; height: 8px; border-radius: 50%; background: #22c55e;
    box-shadow: 0 0 8px #22c55e; animation: blink 1.8s infinite; display: inline-block; }
.cam-dot-off { background: #374151; box-shadow: none; animation: none; }

/* ── Status badges ── */
.badge {
    padding: 10px 14px; border-radius: 10px; text-align: center;
    font-size: 13px; font-weight: 700; letter-spacing: 0.5px;
    color: white; margin: 6px 0;
}
.badge-safe {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    box-shadow: 0 0 12px rgba(16,185,129,0.2);
    color: #6ee7b7;
}
.badge-alert {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    box-shadow: 0 0 18px rgba(239,68,68,0.35);
    color: #fca5a5;
    animation: pulse-alert 1.2s infinite;
}
.badge-pending {
    background: linear-gradient(135deg, #1e293b, #273046);
    border: 1px solid #475569;
    color: #94a3b8;
}
.badge-motion {
    background: linear-gradient(135deg, #7c1d1d, #9b2c2c);
    border: 1px solid #f87171;
    color: #fecaca;
    animation: pulse-alert 1s infinite;
}
.badge-scanning {
    background: linear-gradient(135deg, #0c2240, #1e3a5f);
    border: 1px solid #3b82f6;
    color: #93c5fd;
}
.badge-standby {
    background: #111827; border: 1px solid #1f2937; color: #374151;
}

/* ── Reasoning panel ── */
.reason-box {
    height: 200px; overflow-y: auto; padding: 12px 14px;
    border-radius: 10px;
    background: #060a14;
    border: 1px solid #1e2d50;
    color: #94a3b8;
    font-size: 12px;
    line-height: 1.7;
    font-family: 'JetBrains Mono', monospace;
}
.reason-box b { color: #60a5fa; }

/* ── Debug strip label ── */
.debug-label {
    font-size: 10px; color: #334155; text-align: center;
    font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase;
    margin-top: 2px;
}

/* ── Start button override ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 12px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.35) !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(37,99,235,0.5) !important;
}

[data-testid="stInfoBox"] {
    background: #0f1629 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 10px !important;
    color: #64748b !important;
}

/* ── Divider ── */
hr { border-color: #1e2d50 !important; }

/* ── Video Player Clean Look ── */
video {
    pointer-events: none;
    border-radius: 12px;
}
video::-webkit-media-controls {
    display: none !important;
}
video::-webkit-media-controls-enclosure {
    display: none !important;
}

@keyframes blink {
    0%, 100% { opacity: 1; } 50% { opacity: 0.3; }
}
@keyframes pulse-alert {
    0%, 100% { box-shadow: 0 0 14px rgba(239,68,68,0.35); }
    50%       { box-shadow: 0 0 28px rgba(239,68,68,0.7); }
}
</style>
""", unsafe_allow_html=True)


PROMPT = (
    "Analyze the image for elephants, including partially visible elephants, "
    "dark silhouettes, nighttime elephant movement, or large elephant-like shapes. "

    "Provide very short reasoning inside <think> tags. "

    "If there is moderate or strong evidence of an elephant, the final label MUST be ALERT. "

    "End with line containing SAFE or ALERT."
)

VIDEO_PATHS = ["test/test1.mp4", "test/test2.mp4"]
N_CAMS = len(VIDEO_PATHS)

# ── Shared detection tuning ──────────────────────────────────────────────────
INFERENCE_INTERVAL   = 5.0    # minimum seconds between queuing a frame per camera
MIN_OBJECT_AREA_FRACTION = 0.010
PERSISTENCE_REQUIRED = 5
MAX_FG_RATIO         = 0.35
MAX_SCATTER_CONTOURS = 5
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
AUDIO_THRESHOLD      = 0.15
ALERT_COOLDOWN       = 30.0

AUDIO_PROMPT = (
    """
    You are EleGuard Ranger.

    This image is a mel-spectrogram of audio.

    Look for strong low-frequency bands, repeated rumble-like patterns, or sudden energetic bursts that may suggest elephant vocalizations.

    Provide very short reasoning inside <think> tags. You must start your response with <think>.

    End with SAFE or ALERT.
    """
)


# ── Inference helpers ────────────────────────────────────────────────────────

def run_inference(image, mode="video"):
    """
    Sends an image or spectrogram to the local Gemma model for inference.
    
    Args:
        image: The image array to be analyzed (BGR or RGB).
        mode: Either "video" (for camera frames) or "audio" (for spectrograms).
        
    Returns:
        The text response generated by the Gemma model.
    """
    # Shrink to 128x128 to drastically reduce image tokens and speed up CPU inference
    small = cv2.resize(image, (128, 128))
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
    b64 = base64.b64encode(buf).decode()

    prompt_text = AUDIO_PROMPT if mode == "audio" else PROMPT

    payload = {
        "model": "gemma",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }],
        "max_tokens": 64,
        "temperature": 0.1
    }

    response = requests.post(
        "http://127.0.0.1:8080/v1/chat/completions",
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def inference_worker(job_queue: queue.Queue, results: dict, stop_event: threading.Event):
    """
    Background daemon thread that processes inference jobs.
    
    Reads (cam_idx, image, mode) tuples from job_queue, calls the local LLM via run_inference,
    parses the output to determine SAFE/ALERT status, and writes results back into the shared `results` dict.
    
    Args:
        job_queue: Queue containing images to process.
        results: Shared dictionary to store the status and reasoning for each camera.
        stop_event: Event to signal the thread to terminate.
    """
    while not stop_event.is_set():
        try:
            cam_idx, data, mode = job_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        # [NEW] Discard redundant frames if camera is already in ALERT cooldown
        res = results.get(cam_idx, {})
        if res.get("status") == "ALERT":
            if time.time() - res.get("alert_time", 0) < ALERT_COOLDOWN:
                results[cam_idx]["pending"] = False
                job_queue.task_done()
                continue

        try:
            image_to_process = data
            
            # If data is raw audio, generate spectrogram in this background thread
            if mode == "audio":
                # S = librosa.feature.melspectrogram(y=data, sr=sr_audio, n_mels=128, fmax=8000)
                S = librosa.feature.melspectrogram(y=data, sr=22050, n_mels=128, fmax=8000)
                S_dB = librosa.power_to_db(S, ref=np.max)
                db_range = S_dB.max() - S_dB.min()
                if db_range > 0:
                    img_array = ((S_dB - S_dB.min()) / db_range * 255).astype(np.uint8)
                    img_array = np.flipud(img_array)
                    image_to_process = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                else:
                    image_to_process = np.zeros((128, 128, 3), dtype=np.uint8)

            out   = run_inference(image_to_process, mode).strip()
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            status = "SAFE"
            for line in reversed(lines):
                u = line.upper()
                if "ALERT" in u:
                    status = "ALERT"
                    break
                elif "SAFE" in u:
                    status = "SAFE"
                    break
            reasoning = "\n".join(lines) or "No reasoning generated."
        except Exception as e:
            status    = "SAFE"
            reasoning = f"Inference Error:\n\n{e}"

        old_res = results.get(cam_idx, {})
        alert_time = old_res.get("alert_time", 0.0)
        if status == "ALERT":
            alert_time = time.time()
        elif status == "SAFE":
            alert_time = 0.0

        results[cam_idx] = {
            "status": status, 
            "reasoning": reasoning, 
            "pending": False,
            "alert_time": alert_time
        }
        job_queue.task_done()


# ── Motion detection per frame ───────────────────────────────────────────────

def process_single_frame(frame, fgbg, state, min_area):
    """
    Processes a single video frame to detect significant motion.
    
    Uses background subtraction and morphological operations to find moving objects.
    Filters out camera shake and scatter noise. If a large enough object persists 
    across multiple frames, it triggers a motion detected state.
    
    Args:
        frame: The current video frame.
        fgbg: The background subtractor instance (e.g., MOG2).
        state: Dictionary holding the camera's motion detection state and persistence.
        min_area: Minimum contour area to qualify as a significant object.
        
    Returns:
        A visualization frame combining the motion mask and the annotated original frame.
    """
    fgmask = fgbg.apply(frame)
    mask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, MORPH_KERNEL)
    mask = cv2.erode(mask, MORPH_KERNEL, iterations=1)
    mask = cv2.dilate(mask, MORPH_KERNEL, iterations=2)

    frame_px = mask.shape[0] * mask.shape[1]
    fg_ratio = cv2.countNonZero(mask) / frame_px
    camera_shake = fg_ratio > MAX_FG_RATIO

    debug_frame = frame.copy()
    large_object_found = False

    if camera_shake:
        cv2.putText(debug_frame, f"CAMERA SHAKE ({fg_ratio:.0%}) — SUPPRESSED",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        state["persistence"] = max(0, state["persistence"] - 2)
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        qualifying = [c for c in contours if cv2.contourArea(c) > min_area]
        scatter = len(qualifying) > MAX_SCATTER_CONTOURS

        if scatter:
            cv2.putText(debug_frame, f"SCATTER NOISE ({len(qualifying)} blobs) — SUPPRESSED",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
            state["persistence"] = max(0, state["persistence"] - 1)
        if not scatter:
            x_min, y_min = frame.shape[1], frame.shape[0]
            x_max, y_max = 0, 0
            
            for cnt in qualifying:
                large_object_found = True
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)

            if large_object_found:
                pad = min(w, h) // 8
                H, W = frame.shape[:2]
                state["last_bbox"] = (
                    max(0, x_min - pad),
                    max(0, y_min - pad),
                    min(W, x_max + pad),
                    min(H, y_max + pad)
                )

        if not scatter:
            if large_object_found:
                state["persistence"] = min(PERSISTENCE_REQUIRED + 5, state["persistence"] + 1)
            else:
                state["persistence"] = max(0, state["persistence"] - 1)

    was_detected = state.get("motion_detected", False)
    state["motion_detected"] = state["persistence"] >= PERSISTENCE_REQUIRED
    state["just_triggered"] = state["motion_detected"] and not was_detected

    dh, dw = 160, 240
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    debug_viz = cv2.hconcat([cv2.resize(mask_rgb, (dw, dh)), cv2.resize(debug_frame, (dw, dh))])
    return debug_viz


# ── Page header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="eleguard-header">
  <div class="eleguard-logo">🐘</div>
  <div>
    <div class="eleguard-title">EleGuard Ranger</div>
    <div class="eleguard-sub">AI-Powered Wildlife Monitoring · 3-Camera Network · Offline Inference</div>
  </div>
</div>
""", unsafe_allow_html=True)

if "run" not in st.session_state:
    st.session_state.run = False

if st.button(
    "⏹  Stop Monitoring" if st.session_state.run else "▶  Start Monitoring",
    use_container_width=True
):
    st.session_state.run = not st.session_state.run
    if st.session_state.run:
        st.session_state.monitoring_start_time = time.time()
    st.rerun()

if not st.session_state.run:
    st.markdown('<p style="color:#334155;text-align:center;margin-top:40px;font-size:14px;">⬆ Press Start Monitoring to begin</p>', unsafe_allow_html=True)
    st.stop()

from streamlit.runtime.scriptrunner import add_script_run_ctx

# ── Start background inference worker ────────────────────────────────────────
if "infer_queue" not in st.session_state:
    st.session_state.infer_queue  = queue.Queue(maxsize=N_CAMS * 2)  # bounded: drop old jobs
    st.session_state.infer_results = {i: {"status": "SAFE", "reasoning": "", "pending": False, "alert_time": 0.0}
                                       for i in range(N_CAMS)}
    st.session_state.stop_event   = threading.Event()

if "worker" not in st.session_state or not st.session_state.worker.is_alive():
    st.session_state.worker = threading.Thread(
        target=inference_worker,
        args=(st.session_state.infer_queue,
              st.session_state.infer_results,
              st.session_state.stop_event),
        daemon=True,
        name="EleGuard-InferWorker"
    )
    add_script_run_ctx(st.session_state.worker)
    st.session_state.worker.start()

infer_queue   = st.session_state.infer_queue
infer_results = st.session_state.infer_results

# ── Build per-camera UI placeholders ─────────────────────────────────────────
ui = []

for i in range(N_CAMS):
    with st.container():
        st.markdown(
            f'<div class="cam-card">'
            f'<div class="cam-card-title">'
            f'<span class="cam-dot" id="dot-{i}"></span>'
            f'CAM {i + 1} &nbsp;·&nbsp; {VIDEO_PATHS[i]}'
            f'</div></div>',
            unsafe_allow_html=True
        )
        
        # Horizontal layout per camera
        c1, c2, c3 = st.columns([3, 2, 2], gap="medium")
        
        with c1:
            st.video(VIDEO_PATHS[i], loop=True, autoplay=True, muted=True)
            feed_ph = st.empty() # Still exists if needed for overlays, but unused now
            
        with c2:
            debug_ph = st.empty()
            st.markdown('<div style="font-weight:600; font-size:12px; color:#94a3b8; margin-bottom:8px; margin-top:8px;">🔊 AUDIO LEVEL</div>', unsafe_allow_html=True)
            vol_ph = st.empty()
            
        with c3:
            motion_ph = st.empty()
            status_ph = st.empty()
            reason_ph = st.empty()
            
        ui.append({
            "feed":   feed_ph,
            "debug":  debug_ph,
            "motion": motion_ph,
            "status": status_ph,
            "reason": reason_ph,
            "vol":    vol_ph,
        })
        st.markdown("<br>", unsafe_allow_html=True)

# ── Open all captures and Preload Audio ───────────────────────────────────────
caps, fgbgs, states = [], [], []
audios, sr_audio = [], 22050
for i, path in enumerate(VIDEO_PATHS):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        st.error(f"Cannot open '{path}' — make sure it exists in C:\\EleGuard")
        st.stop()
        
    try:
        clip = VideoFileClip(path)
        if clip.audio is not None:
            y = clip.audio.to_soundarray(fps=sr_audio)
            if y.ndim > 1:
                y = y.mean(axis=1)
            audios.append(y)
        else:
            audios.append(np.zeros(0))
        clip.close()
    except Exception as e:
        audios.append(np.zeros(0))
        st.warning(f"Could not load audio for {path}: {e}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    min_area     = int(fw * fh * MIN_OBJECT_AREA_FRACTION)

    caps.append(cap)
    fgbgs.append(cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=80, detectShadows=True))
    states.append({
        "fps":             fps,
        "total_frames":    total_frames,
        "skip_interval":   max(1, int(fps)),
        "min_area":        min_area,
        "f_count":         0,
        "persistence":     0,
        "motion_detected": False,
        "last_infer_time": 0.0,
        "alive":           True,
    })

# ── Main round-robin monitoring loop ─────────────────────────────────────────
while st.session_state.run:

    any_alive = False
    any_pending = False

    for i in range(N_CAMS):
        state = states[i]
        pls   = ui[i]
        result = infer_results[i]
        
        if result["pending"]:
            any_pending = True

        if not state["alive"]:
            # Video ended, but we must still update the UI in case inference just finished
            pass
        else:
            cap   = caps[i]
            fgbg  = fgbgs[i]
            
            # ── Synchronization Logic: Determine current playback position ──────────
            elapsed = time.time() - st.session_state.monitoring_start_time
            # Total duration to calculate modulo
            duration = state["total_frames"] / state["fps"] if state["total_frames"] > 0 else 1
            t = elapsed % duration
            target_f = int(t * state["fps"])
            state["f_count"] = target_f

            # Only read and process a frame at specific intervals to save CPU/Network
            # (Motion detection and Debug viz run at 1 FPS)
            last_target_f = state.get("last_target_f", -1)
            state["last_target_f"] = target_f
            
            should_analyze = False
            if last_target_f == -1:
                should_analyze = True
            else:
                prev_interval = last_target_f // state["skip_interval"]
                curr_interval = target_f // state["skip_interval"]
                if curr_interval > prev_interval or target_f < last_target_f:
                    should_analyze = True

            if should_analyze:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
                ret, frame = cap.read()
                
                if not ret:
                    # If read fails, try resetting
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                
                if not ret:
                    state["alive"] = False
                    pls["feed"].error(f"📼 Error reading {VIDEO_PATHS[i]}")
                else:
                    any_alive = True
                    # Sync f_count with actual video position
                    state["f_count"] = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                    # ── Motion detection (non-blocking, 1 fps) ────────────────────────
                    debug_viz = process_single_frame(frame, fgbg, state, state["min_area"])

                    # ── Debug strip ───────────────────────────────────────────────────
                    pls["debug"].image(debug_viz, caption="Mask | Annotated", use_container_width=True)

                    # ── Motion badge ──────────────────────────────────────────────────
                    p  = state["persistence"]
                    md = state["motion_detected"]
                    
                    if state.get("last_p") != p or state.get("last_md") != md:
                        state["last_p"] = p
                        state["last_md"] = md
                        if md:
                            mbadge, mlabel = "badge-motion", "🚨 &nbsp;LARGE OBJECT DETECTED"
                        elif p > 0:
                            mbadge, mlabel = "badge-scanning", f"🔍 &nbsp;SCANNING &nbsp;({p}/{PERSISTENCE_REQUIRED})"
                        else:
                            mbadge, mlabel = "badge-standby", "⬤ &nbsp;STANDBY"
                        pls["motion"].markdown(
                            f'<div class="badge {mbadge}">{mlabel}</div>',
                            unsafe_allow_html=True
                        )

                    # ── Audio processing (Synced with OpenCV frame) ──────────────────────
                    now = time.time()
                    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    audio_y = audios[i]
                    audio_triggered = False
                    
                    if len(audio_y) > 0:
                        # Use wall-clock time 't' for audio lookup
                        start_idx = int(max(0, t - 0.5) * sr_audio)
                        end_idx = int(min(len(audio_y), (t + 0.5) * sr_audio))
                        chunk = audio_y[start_idx:end_idx]
                        
                        if len(chunk) > 0:
                            rms = np.sqrt(np.mean(chunk**2))
                            # Render volume meter
                            vol_pct = int(min(100, (rms / (AUDIO_THRESHOLD * 2)) * 100))
                            color = "#ef4444" if rms > AUDIO_THRESHOLD else "#22c55e" if rms > AUDIO_THRESHOLD/2 else "#3b82f6"
                            pls["vol"].markdown(f'''
                            <div style="width:100%; height:12px; background:#1e293b; border-radius:6px; overflow:hidden;">
                                <div style="width:{vol_pct}%; height:100%; background:{color}; transition: width 0.2s;"></div>
                            </div>
                            <div style="font-size:10px; color:#64748b; margin-top:4px; text-align:right;">RMS: {rms:.3f}</div>
                            ''', unsafe_allow_html=True)
                            
                            # Audio Trigger
                            cooldown = ALERT_COOLDOWN if result["status"] == "ALERT" else INFERENCE_INTERVAL
                            if rms > AUDIO_THRESHOLD and not result["pending"] and (now - state["last_infer_time"] > cooldown):
                                audio_triggered = True
                                s_idx = int(max(0, t - 1.0) * sr_audio)
                                e_idx = int(min(len(audio_y), (t + 1.0) * sr_audio))
                                spec_chunk = audio_y[s_idx:e_idx]
                                try:
                                    infer_queue.put_nowait((i, spec_chunk, "audio"))
                                    infer_results[i]["pending"] = True
                                    any_pending = True
                                    state["last_infer_time"] = now
                                    pls["motion"].markdown('<div class="badge badge-motion">🔊 &nbsp;LOUD NOISE DETECTED</div>', unsafe_allow_html=True)
                                except queue.Full:
                                    pass

                    # ── Enqueue frame for inference (non-blocking) ────────────────────
                    cooldown = ALERT_COOLDOWN if result["status"] == "ALERT" else INFERENCE_INTERVAL
                    if not audio_triggered and state.get("motion_detected", False) and not result["pending"] and (now - state["last_infer_time"] > cooldown):
                        try:
                            crop_frame = frame.copy()
                            if "last_bbox" in state:
                                x1, y1, x2, y2 = state["last_bbox"]
                                if x2 > x1 and y2 > y1:
                                    crop_frame = frame[y1:y2, x1:x2].copy()
                            infer_queue.put_nowait((i, crop_frame, "video"))
                            infer_results[i]["pending"] = True
                            any_pending = True
                            state["last_infer_time"] = now
                        except queue.Full:
                            pass
            else:
                # Still count as alive if we are just skipping frames for performance
                any_alive = True

        # ── Render latest inference result from worker ────────────────────
        # (This runs even if the video has ended, so we catch late AI responses)
        status    = result["status"]
        reasoning = result["reasoning"]
        pending   = result["pending"]

        if (state.get("last_status") != status or 
            state.get("last_reasoning") != reasoning or 
            state.get("last_pending") != pending):
            
            state["last_status"] = status
            state["last_reasoning"] = reasoning
            state["last_pending"] = pending

            if status == "ALERT":
                pls["status"].markdown(
                    '<div class="badge badge-alert">⚠️ &nbsp;ELEPHANT ALERT</div>',
                    unsafe_allow_html=True
                )
            elif pending:
                pls["status"].markdown(
                    '<div class="badge badge-pending">⏳ &nbsp;ANALYZING…</div>',
                    unsafe_allow_html=True
                )
            else:
                pls["status"].markdown(
                    '<div class="badge badge-safe">✅ &nbsp;AREA SAFE</div>',
                    unsafe_allow_html=True
                )

            if reasoning:
                # Escape HTML to prevent the browser from hiding raw <tags> output by Gemma
                fmt = html.escape(reasoning)
                # Replace specific tags with styled UI elements
                fmt = fmt.replace("&lt;think&gt;", "<b>🧠 Thinking</b><br><br>")
                fmt = fmt.replace("&lt;/think&gt;", "<br><br><b>📋 Final Analysis</b><br><br>")
                fmt = fmt.replace("\n", "<br>")
                
                pls["reason"].markdown(
                    f'<div class="reason-box">{fmt}</div>',
                    unsafe_allow_html=True
                )
            else:
                pls["reason"].markdown(
                    f'<div class="reason-box" style="display:flex;align-items:center;justify-content:center;color:#475569;">Monitoring for movement...</div>',
                    unsafe_allow_html=True
                )

    if not any_alive and not any_pending:
        st.info("All video streams have ended and all inferences are complete.")
        break

    time.sleep(0.005)

for cap in caps:
    cap.release()
