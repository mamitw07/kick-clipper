"""
FastAPI Server for Kick Stream Highlight & Raw Clip Generator.
Includes Hardware Lock (HWID) Licensing Protection, Time-Range Slicing, Turbo Multi-Clip Engine,
and Viral Title & Hashtag Generator.
"""

import os
import sys
import uuid
import time
import socket
import webbrowser
import zipfile
import shutil
import subprocess
import logging
import json
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger("app")

# Permanently silence Windows socket shutdown error (WinError 10054) in asyncio ProactorEventLoop
if sys.platform == "win32":
    import asyncio
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
        def _safe_call_connection_lost(self, exc=None):
            try:
                if getattr(self, '_called_connection_lost', False):
                    return
                try:
                    if hasattr(self, '_protocol') and self._protocol:
                        self._protocol.connection_lost(exc)
                finally:
                    if hasattr(self, '_sock') and self._sock is not None:
                        if hasattr(self._sock, 'shutdown') and self._sock.fileno() != -1:
                            try:
                                self._sock.shutdown(socket.SHUT_RDWR)
                            except (ConnectionResetError, ConnectionAbortedError, OSError):
                                pass
                        try:
                            self._sock.close()
                        except (ConnectionResetError, ConnectionAbortedError, OSError):
                            pass
                        self._sock = None
                    server = getattr(self, '_server', None)
                    if server is not None:
                        server._detach(self)
                        self._server = None
                    self._called_connection_lost = True
            except Exception:
                pass
        _ProactorBasePipeTransport._call_connection_lost = _safe_call_connection_lost
    except Exception:
        pass

from urllib.parse import quote, unquote
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, Response, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Suppress repetitive HTTP polling access logs from terminal
logging.getLogger("uvicorn.access").disabled = True

from extractor import KickHighlightExtractor, get_cdn_session
import license_manager

app = FastAPI(title="Kick Highlight Clipper", version="2.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_ngrok_skip_warning_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_clips")
TEMP_DIR = os.path.join(BASE_DIR, "temp_work")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/clips_media", StaticFiles(directory=OUTPUT_DIR), name="clips_media")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

TASKS: Dict[str, Dict[str, Any]] = {}

class LicenseActivateRequest(BaseModel):
    key: str

class ChannelRequest(BaseModel):
    url: str

class GDriveRequest(BaseModel):
    url: str

class LiveStreamRequest(BaseModel):
    url: str

class ExtractLiveClipRequest(BaseModel):
    playback_url: str
    channel: str = "CanliYayin"
    title: Optional[str] = "Canlı Yayın Klibi"
    duration: int = 30
    mode: str = "buffer" # "buffer" or "live"
    video_format: str = "vertical_9_16"
    shorts_layout: str = "blur"
    cam_position: str = "top_right"
    enable_chat_overlay: bool = False
    normalize_audio: bool = True
    enable_hook_banner: bool = False
    hook_text: Optional[str] = None
    video_quality: str = "1080p60"

class StartExtractRequest(BaseModel):
    source: str
    title: str
    uploader: str
    duration: int
    num_clips: int = 10
    clip_duration: int = 35
    min_distance: int = 60
    distribution_mode: str = "smart_hybrid"
    quality_sensitivity: str = "balanced"
    filter_gambling: bool = True
    auto_download: bool = True
    start_sec: float = 0.0
    end_sec: Optional[float] = None
    video_format: str = "raw_16_9"
    shorts_layout: str = "blur"
    cam_position: str = "top_right"
    enable_chat_overlay: bool = False
    normalize_audio: bool = True
    enable_teaser_hook: bool = False
    enable_hook_banner: bool = False
    video_quality: str = "1080p60"

class ExtractSingleClipRequest(BaseModel):
    source: str
    uploader: str
    clip: Dict[str, Any]
    video_format: str = "raw_16_9"
    shorts_layout: str = "blur"
    cam_position: str = "top_right"
    enable_chat_overlay: bool = False
    normalize_audio: bool = True
    enable_teaser_hook: bool = False
    enable_hook_banner: bool = False
    video_quality: str = "1080p60"

class ExtractSelectedClipsRequest(BaseModel):
    source: str
    uploader: str
    clips: List[Dict[str, Any]]
    video_format: str = "raw_16_9"
    shorts_layout: str = "blur"
    cam_position: str = "top_right"
    enable_chat_overlay: bool = False
    normalize_audio: bool = True
    enable_teaser_hook: bool = False
    enable_hook_banner: bool = False
    video_quality: str = "1080p60"

class ConvertClipRequest(BaseModel):
    filename: str
    shorts_layout: str = "blur"
    cam_position: str = "top_right"
    enable_chat_overlay: bool = False
    moment_type: Optional[str] = "funny"
    normalize_audio: bool = True
    enable_teaser_hook: bool = False
    enable_hook_banner: bool = False
    hook_text: Optional[str] = None
    peak_time_rel: Optional[float] = None
    clip_duration: Optional[float] = None

class GenerateThumbnailRequest(BaseModel):
    filename: str
    title_text: Optional[str] = None
    streamer_name: Optional[str] = "Kick"
    moment_type: Optional[str] = "funny"
    hype_score: Optional[int] = 95
    target_format: Optional[str] = "916"
    frame_sec: Optional[float] = None
    badge_text: Optional[str] = None
    style: Optional[str] = "kick_neon"
    focus_box: Optional[Dict[str, float]] = None
    draw_ring: Optional[bool] = False
    ring_color: Optional[str] = "red"
    crop_to_focus: Optional[bool] = False

class GeminiThumbnailAnalyzeRequest(BaseModel):
    filename: str
    streamer: Optional[str] = "Kick"
    stream_title: Optional[str] = "Canlı Yayın"
    transcript: Optional[str] = ""
    moment_type: Optional[str] = "funny"
    duration: Optional[float] = 30.0
    peak_time_rel: Optional[float] = None

class GeminiAiImageRequest(BaseModel):
    filename: Optional[str] = ""
    streamer: Optional[str] = "Kick"
    stream_title: Optional[str] = "Canlı Yayın"
    transcript: Optional[str] = ""
    moment_type: Optional[str] = "funny"
    aspect_ratio: Optional[str] = "16:9"
    title_text: Optional[str] = None
    badge_text: Optional[str] = None
    style: Optional[str] = "kick_neon"
    variation_index: int = 0
    custom_prompt: Optional[str] = None
    include_text_overlay: bool = True
    focus_box: Optional[Dict[str, float]] = None
    draw_ring: Optional[bool] = False
    ring_color: Optional[str] = "red"
    crop_to_focus: Optional[bool] = False


class StudioRenderRequest(BaseModel):
    filename: str
    layout: str = "split_screen" # split_screen, blur, fit_crop
    cam_position: str = "top_right" # top_right, top_left, bottom_right, bottom_left, center
    cam_pan_x: float = 0.5 # 0.0 to 1.0 (horizontal position)
    cam_pan_y: float = 0.5 # 0.0 to 1.0 (vertical position)
    cam_zoom: float = 1.0 # 1.0 to 3.0
    game_pan_x: float = 0.5
    game_pan_y: float = 0.5
    game_zoom: float = 1.0
    fit_pan_x: float = 0.5
    blur_amount: int = 5
    trim_start: float = 0.0
    trim_end: Optional[float] = None
    speed: float = 1.0
    volume: float = 1.0
    normalize_audio: bool = True
    header_text: Optional[str] = ""
    header_color: str = "#ffe600"
    header_start: float = 0.0
    header_end: Optional[float] = None
    header_x: float = 0.5
    header_y: float = 0.14
    header_scale: float = 1.0
    header_theme: str = "yellow_pop"
    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    video_quality: str = "1080p60"
    encoder_choice: str = "auto"
    overlay_image: Optional[str] = None
    overlay_x: float = 0.5
    overlay_y: float = 0.5
    overlay_scale: float = 0.25
    overlay_opacity: float = 1.0
    subtitles_data: Optional[List[Dict[str, Any]]] = None
    subtitles_style: str = "yellow"
    subtitles_pos: str = "bottom"

class StudioTranscribeRequest(BaseModel):
    filename: str
    model: str = "small"
    language: str = "tr"

class AISceneAnalyzeRequest(BaseModel):
    streamer: Optional[str] = ""
    stream_title: Optional[str] = ""
    target_category: str = "all"
    sentences: Optional[List[Dict[str, Any]]] = None
    transcript_text: Optional[str] = None

def require_license():
    status = license_manager.check_license_status()
    if not status.get("is_licensed", False):
        msg = status.get("message", "Cihaz lisanslanmamış!")
        if status.get("is_expired", False):
            msg = f"1 Aylık Lisans Süreniz Doldu ({status.get('expiry_formatted')})!"
        raise HTTPException(
            status_code=403,
            detail=f"{msg} Lütfen yöneticinizden lisans anahtarı temin edin. HWID: {status.get('hwid')}"
        )
    return True

def deplevel_licensed(is_valid: bool) -> bool:
    return is_valid



@app.get("/api/license-status")
def get_license_status():
    status = license_manager.check_license_status()
    return status

@app.post("/api/activate")
def activate_device(req: LicenseActivateRequest):
    key = req.key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="Lütfen geçerli bir lisans anahtarı girin.")
    
    success, msg = license_manager.activate_license(key)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    
    status = license_manager.check_license_status()
    return {
        "status": "success",
        "message": msg,
        "license_info": status
    }

@app.post("/api/fetch-videos", dependencies=[Depends(require_license)])
def fetch_channel_videos_endpoint(req: ChannelRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="Lütfen geçerli bir Kick kanal veya video linki girin.")
    
    extractor = KickHighlightExtractor(output_dir=OUTPUT_DIR, temp_dir=TEMP_DIR)
    try:
        data = extractor.fetch_channel_videos(url, max_pages=3)
        return {"status": "success", **data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/fetch-gdrive", dependencies=[Depends(require_license)])
def fetch_gdrive_endpoint(req: GDriveRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="Lütfen geçerli bir Google Drive video veya klasör linki girin.")
    
    extractor = KickHighlightExtractor(output_dir=OUTPUT_DIR, temp_dir=TEMP_DIR)
    try:
        data = extractor.fetch_gdrive_videos(url)
        return {"status": "success", **data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/fetch-live-info", dependencies=[Depends(require_license)])
def fetch_live_info_endpoint(req: LiveStreamRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="Lütfen geçerli bir canlı yayın veya kanal linki girin.")
    extractor = KickHighlightExtractor(output_dir=OUTPUT_DIR, temp_dir=TEMP_DIR)
    try:
        data = extractor.fetch_live_stream_info(url)
        return {"status": "success", "stream": data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/extract-live-clip", dependencies=[Depends(require_license)])
def extract_live_clip_endpoint(req: ExtractLiveClipRequest):
    playback_url = req.playback_url.strip()
    if not playback_url:
        raise HTTPException(status_code=400, detail="Geçerli bir canlı yayın akışı bulunamadı.")
    
    extractor = KickHighlightExtractor(output_dir=OUTPUT_DIR, temp_dir=TEMP_DIR)
    try:
        clip_data = extractor.extract_live_clip(
            playback_url=playback_url,
            channel=req.channel,
            title=req.title,
            duration=req.duration,
            mode=req.mode,
            video_format=req.video_format,
            shorts_layout=req.shorts_layout,
            cam_position=req.cam_position,
            enable_chat_overlay=req.enable_chat_overlay,
            normalize_audio=req.normalize_audio,
            enable_hook_banner=req.enable_hook_banner,
            hook_text=req.hook_text,
            video_quality=req.video_quality
        )
        return {"status": "success", "clip": clip_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def open_windows_file_dialog() -> str:
    """Opens a native Windows Open File Dialog directly on the user's desktop."""
    tk_code = """
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', 1)
path = filedialog.askopenfilename(
    title='Video Dosyasi Sec (0 Saniye - Yukleme Yok)',
    filetypes=[
        ('Video Dosyalari', '*.mp4 *.mkv *.avi *.mov *.flv *.ts *.webm *.m4v'),
        ('Tum Dosyalar', '*.*')
    ]
)
root.destroy()
if path:
    print('SELECTED:' + path)
else:
    print('CANCELLED')
"""
    try:
        res = subprocess.run(
            [sys.executable, "-c", tk_code],
            capture_output=True,
            text=True,
            timeout=300
        )
        for line in res.stdout.strip().splitlines():
            if line.startswith("SELECTED:"):
                selected = line.replace("SELECTED:", "").strip()
                if os.path.exists(selected):
                    return os.path.normpath(selected)
            elif line.startswith("CANCELLED"):
                return ""
    except Exception as e:
        logger.warning(f"Tkinter file dialog error: {e}")

    # Fallback to PowerShell
    ps_code = """
    Add-Type -AssemblyName System.Windows.Forms
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = 'Video Dosyasi Sec (0 Saniye - Yukleme Yok)'
    $dialog.Filter = 'Video Dosyalari (*.mp4;*.mkv;*.avi;*.mov;*.flv;*.ts;*.webm)|*.mp4;*.mkv;*.avi;*.mov;*.flv;*.ts;*.webm|Tum Dosyalar (*.*)|*.*'
    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        Write-Output ("SELECTED:" + $dialog.FileName)
    } else {
        Write-Output "CANCELLED"
    }
    """
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_code],
            capture_output=True,
            text=True,
            timeout=300
        )
        for line in res.stdout.strip().splitlines():
            if line.startswith("SELECTED:"):
                selected = line.replace("SELECTED:", "").strip()
                if os.path.exists(selected):
                    return os.path.normpath(selected)
    except Exception as e:
        logger.warning(f"PowerShell file dialog error: {e}")

    return ""

def probe_video_duration(raw_path: str) -> int:
    if not os.path.exists(raw_path):
        return 28800

    ffprobe = shutil.which('ffprobe') or 'ffprobe'

    # 1. Format and stream duration via ffprobe json
    try:
        cmd = [
            ffprobe, '-v', 'error',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            raw_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        if res.returncode == 0 and res.stdout:
            data = json.loads(res.stdout.decode('utf-8', errors='ignore'))
            fmt_dur = data.get('format', {}).get('duration')
            if fmt_dur and fmt_dur != 'N/A':
                d = float(fmt_dur)
                if d > 0:
                    return int(d)
            for s in data.get('streams', []):
                s_dur = s.get('duration')
                if s_dur and s_dur != 'N/A':
                    d = float(s_dur)
                    if d > 0:
                        return int(d)
                tags = s.get('tags', {})
                for tag_k in ['DURATION', 'duration', 'DURATION-eng']:
                    if tag_k in tags:
                        t_val = tags[tag_k]
                        m = re.search(r'(\d+):(\d+):(\d+\.?\d*)', t_val)
                        if m:
                            return int(int(m.group(1))*3600 + int(m.group(2))*60 + float(m.group(3)))
    except Exception as e:
        logger.warning(f"ffprobe JSON probe warning: {e}")

    # 2. ffmpeg banner probe
    try:
        from extractor import get_ffmpeg_binary
        ff_bin = get_ffmpeg_binary()
        cmd = [ff_bin, '-analyzeduration', '100M', '-probesize', '100M', '-i', raw_path]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        err_text = res.stderr.decode('utf-8', errors='ignore')
        m = re.search(r'Duration:\s*(\d+):(\d+):(\d+\.?\d*)', err_text)
        if m:
            h, mi, s = m.groups()
            dur = int(int(h)*3600 + int(mi)*60 + float(s))
            if dur > 0:
                return dur
    except Exception as e:
        logger.warning(f"ffmpeg probe warning: {e}")

    # 3. For transport streams (.ts) without headers, probe last packet pts
    try:
        cmd = [
            ffprobe, '-v', 'error',
            '-sseof', '-60',
            '-show_entries', 'packet=pts_time',
            '-of', 'csv=p=0',
            raw_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        lines = [line.strip() for line in res.stdout.decode().splitlines() if line.strip() and line.strip() != 'N/A']
        if lines:
            pts = [float(x) for x in lines if re.match(r'^\d+(\.\d+)?$', x)]
            if pts:
                return int(max(pts))
    except Exception as e:
        logger.warning(f"last packet probe warning: {e}")

    # 4. Default: 8 hours (28800) so videos are never capped to 1 hour
    return 28800

def create_local_video_info(raw_path: str, custom_duration: Optional[int] = None) -> dict:
    filename = os.path.basename(raw_path)
    if custom_duration and custom_duration > 0:
        duration = custom_duration
    else:
        duration = probe_video_duration(raw_path)
        if duration <= 0:
            duration = 28800

    v_id = f"local_{int(time.time())}"
    return {
        "id": v_id,
        "uuid": v_id,
        "slug": v_id,
        "title": os.path.splitext(filename)[0],
        "source": raw_path,
        "duration": duration,
        "duration_formatted": f"{duration // 3600:02d}:{(duration % 3600) // 60:02d}:{duration % 60:02d}",
        "thumbnail": "",
        "created_at": "Yerel Bilgisayar Videosu",
        "uploader": "Yerel Dosya"
    }

class LocalFilePathRequest(BaseModel):
    filepath: str
    custom_duration: Optional[int] = None

@app.post("/api/add-local-file-path", dependencies=[Depends(require_license)])
def add_local_file_path_endpoint(req: LocalFilePathRequest):
    raw_path = req.filepath.strip().strip('"').strip("'")
    if not raw_path or not os.path.exists(raw_path):
        raise HTTPException(status_code=400, detail="Belirtilen dosya yolu bulunamadı! Lütfen dosya yolunun doğru olduğundan emin olun.")
    
    if not os.path.isfile(raw_path):
        raise HTTPException(status_code=400, detail="Seçilen yol geçerli bir dosya değil.")

    video_info = create_local_video_info(raw_path, custom_duration=req.custom_duration)
    return {
        "status": "success",
        "video": video_info
    }

@app.post("/api/browse-local-file", dependencies=[Depends(require_license)])
def browse_local_file_endpoint():
    selected_path = open_windows_file_dialog()
    if not selected_path:
        return {"status": "cancelled", "message": "Herhangi bir dosya seçilmedi."}
    
    if not os.path.isfile(selected_path):
        raise HTTPException(status_code=400, detail="Seçilen yol geçerli bir dosya değil.")

    video_info = create_local_video_info(selected_path)
    return {
        "status": "success",
        "video": video_info
    }

@app.post("/api/upload-local-video", dependencies=[Depends(require_license)])
async def upload_local_video_endpoint(file: UploadFile = File(...)):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Lütfen bir video dosyası seçin.")
    
    filename = f"local_{int(time.time())}_{file.filename}"
    save_path = os.path.join(TEMP_DIR, filename)

    try:
        with open(save_path, "wb") as f:
            while chunk := await file.read(1024 * 1024 * 5): # 5MB chunks
                f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video kaydedilemedi: {str(e)}")

    duration = probe_video_duration(save_path)
    if duration <= 0:
        duration = 28800

    v_id = f"local_{int(time.time())}"
    video_info = {
        "id": v_id,
        "uuid": v_id,
        "slug": v_id,
        "title": os.path.splitext(file.filename)[0],
        "source": save_path,
        "duration": duration,
        "duration_formatted": f"{duration // 3600:02d}:{(duration % 3600) // 60:02d}:{duration % 60:02d}",
        "thumbnail": "",
        "created_at": "Yerel Bilgisayar Videosu",
        "uploader": "Yerel Video"
    }

    return {
        "status": "success",
        "video": video_info
    }

def run_extraction_pipeline(
    task_id: str,
    source_url: str,
    title: str,
    uploader: str,
    duration: int,
    num_clips: int,
    clip_duration: int,
    min_distance: int,
    distribution_mode: str,
    start_sec: float,
    end_sec: Optional[float],
    video_format: str = "raw_16_9",
    shorts_layout: str = "blur",
    cam_position: str = "top_right",
    enable_chat_overlay: bool = False,
    normalize_audio: bool = True,
    enable_teaser_hook: bool = False,
    enable_hook_banner: bool = False,
    quality_sensitivity: str = "balanced",
    filter_gambling: bool = True,
    auto_download: bool = True,
    video_quality: str = "1080p60"
):
    task = TASKS[task_id]
    task["status"] = "processing"
    task["progress"] = 5
    task["logs"].append(f"'{title}' yayını taranıyor (Komik & Önemli Anlar Tespiti)...")

    def update_log(msg: str, progress: int):
        task["progress"] = progress
        task["logs"].append(msg)
        print(f"[{task_id}] [{progress}%] {msg}")

    try:
        extractor = KickHighlightExtractor(output_dir=OUTPUT_DIR, temp_dir=TEMP_DIR)

        update_log("Seçilen zaman diliminin ses akışı çekiliyor...", 15)
        audio_path, time_offset, segments = extractor.download_audio_for_analysis(
            source_url=source_url,
            duration=duration,
            start_sec=start_sec,
            end_sec=end_sec,
            callback=update_log
        )

        update_log(f"AI Clipper Engine: 10 Aşamalı Viral Tespit Motoru devrede ({num_clips} adet klip)...", 50)
        clips = extractor.detect_highlights(
            audio_path=audio_path,
            time_offset=time_offset,
            num_clips=num_clips,
            clip_duration=clip_duration,
            min_distance_sec=min_distance,
            distribution_mode=distribution_mode,
            quality_sensitivity=quality_sensitivity,
            filter_gambling=filter_gambling,
            source_url=source_url,
            stream_title=title,
            streamer=uploader,
            segments=segments,
            callback=update_log
        )
        
        # Viral titles, hooks, TikTok/Shorts captions and hashtags
        clips = extractor.generate_viral_titles(
            streamer=uploader,
            stream_title=title,
            clips=clips
        )
        task["detected_clips"] = clips

        if auto_download:
            format_desc = "9:16 Shorts (Dikey)" if video_format == "vertical_9_16" else "16:9 RAW"
            update_log(f"{len(clips)} adet kesit hazırlanıyor [{format_desc} - {video_quality}]...", 65)
            
            final_clips = extractor.extract_raw_clips(
                source_url=source_url,
                clips=clips,
                title_prefix=uploader,
                video_format=video_format,
                normalize_audio=normalize_audio,
                enable_teaser_hook=enable_teaser_hook,
                enable_hook_banner=enable_hook_banner,
                shorts_layout=shorts_layout,
                cam_position=cam_position,
                enable_chat_overlay=enable_chat_overlay,
                callback=update_log,
                video_quality=video_quality
            )
            task["completed_clips"] = final_clips
            task["status"] = "completed"
            task["progress"] = 100
            update_log(f"Tebrikler! {len(final_clips)} adet kesit başarıyla klasöre indirildi.", 100)
        else:
            task["completed_clips"] = []
            task["status"] = "completed"
            task["progress"] = 100
            update_log(f"Tarama başarıyla tamamlandı! {len(clips)} kaliteli an tespit edildi. İstediğin klipleri seçip indirebilirsin.", 100)

    except Exception as e:
        task["status"] = "error"
        task["error"] = str(e)
        task["logs"].append(f"HATA: {str(e)}")
        print(f"Task {task_id} failed: {e}")
    finally:
        try:
            for f in os.listdir(TEMP_DIR):
                fp = os.path.join(TEMP_DIR, f)
                if os.path.isfile(fp) and (f.endswith('.wav') or f.endswith('.part') or f.endswith('.ts')):
                    os.remove(fp)
        except Exception:
            pass

@app.post("/api/start-extract", dependencies=[Depends(require_license)])
def start_extract_endpoint(req: StartExtractRequest, background_tasks: BackgroundTasks):
    source = req.source.strip()
    if not source:
        raise HTTPException(status_code=400, detail="Geçerli bir video kaynağı bulunamadı.")

    task_id = str(uuid.uuid4())[:8]
    TASKS[task_id] = {
        "id": task_id,
        "title": req.title,
        "uploader": req.uploader,
        "duration": req.duration,
        "source": source,
        "num_clips": req.num_clips,
        "clip_duration": req.clip_duration,
        "distribution_mode": req.distribution_mode,
        "quality_sensitivity": req.quality_sensitivity,
        "filter_gambling": req.filter_gambling,
        "auto_download": req.auto_download,
        "start_sec": req.start_sec,
        "end_sec": req.end_sec,
        "video_format": req.video_format,
        "shorts_layout": req.shorts_layout,
        "cam_position": req.cam_position,
        "enable_chat_overlay": req.enable_chat_overlay,
        "normalize_audio": req.normalize_audio,
        "enable_teaser_hook": req.enable_teaser_hook,
        "enable_hook_banner": req.enable_hook_banner,
        "status": "pending",
        "progress": 0,
        "logs": ["Tarama işlemi başlatılıyor..."],
        "detected_clips": [],
        "completed_clips": [],
        "error": None,
        "created_at": time.time()
    }

    background_tasks.add_task(
        run_extraction_pipeline,
        task_id,
        source,
        req.title,
        req.uploader,
        req.duration,
        req.num_clips,
        req.clip_duration,
        req.min_distance,
        req.distribution_mode,
        req.start_sec,
        req.end_sec,
        req.video_format,
        req.shorts_layout,
        req.cam_position,
        req.enable_chat_overlay,
        req.normalize_audio,
        req.enable_teaser_hook,
        req.enable_hook_banner,
        req.quality_sensitivity,
        req.filter_gambling,
        req.auto_download,
        req.video_quality
    )

    return {"status": "success", "task_id": task_id}

@app.post("/api/extract-single-clip", dependencies=[Depends(require_license)])
def extract_single_clip_endpoint(req: ExtractSingleClipRequest):
    source = req.source.strip()
    if not source:
        raise HTTPException(status_code=400, detail="Geçerli bir video kaynağı bulunamadı.")

    extractor = KickHighlightExtractor(output_dir=OUTPUT_DIR, temp_dir=TEMP_DIR)
    try:
        clip_data = extractor.extract_single_clip(
            source_url=source,
            clip=req.clip,
            title_prefix=req.uploader,
            video_format=req.video_format,
            normalize_audio=req.normalize_audio,
            enable_teaser_hook=req.enable_teaser_hook,
            enable_hook_banner=req.enable_hook_banner,
            shorts_layout=req.shorts_layout,
            cam_position=req.cam_position,
            enable_chat_overlay=req.enable_chat_overlay,
            video_quality=req.video_quality
        )
        return {"status": "success", "clip": clip_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extract-selected-clips", dependencies=[Depends(require_license)])
def extract_selected_clips_endpoint(req: ExtractSelectedClipsRequest):
    source = req.source.strip()
    if not source:
        raise HTTPException(status_code=400, detail="Geçerli bir video kaynağı bulunamadı.")

    if not req.clips:
        raise HTTPException(status_code=400, detail="Lütfen indirmek için en az bir klip seçin.")

    extractor = KickHighlightExtractor(output_dir=OUTPUT_DIR, temp_dir=TEMP_DIR)
    try:
        final_clips = extractor.extract_raw_clips(
            source_url=source,
            clips=req.clips,
            title_prefix=req.uploader,
            video_format=req.video_format,
            normalize_audio=req.normalize_audio,
            enable_teaser_hook=req.enable_teaser_hook,
            enable_hook_banner=req.enable_hook_banner,
            shorts_layout=req.shorts_layout,
            cam_position=req.cam_position,
            enable_chat_overlay=req.enable_chat_overlay,
            video_quality=req.video_quality
        )
        return {"status": "success", "clips": final_clips}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/convert-clip-916", dependencies=[Depends(require_license)])
def convert_clip_916_endpoint(req: ConvertClipRequest):
    filename = os.path.basename(req.filename)
    input_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Klip dosyası bulunamadı.")

    base, ext = os.path.splitext(filename)
    if base.endswith("_916"):
        out_filename = filename
        out_path = input_path
    else:
        out_filename = f"{base}_916.mp4"
        out_path = os.path.join(OUTPUT_DIR, out_filename)

    extractor = KickHighlightExtractor(output_dir=OUTPUT_DIR, temp_dir=TEMP_DIR)
    try:
        extractor.convert_to_vertical_916(
            input_filepath=input_path,
            output_filepath=out_path,
            normalize_audio=req.normalize_audio,
            enable_teaser_hook=req.enable_teaser_hook,
            peak_time_rel=req.peak_time_rel,
            clip_duration=req.clip_duration,
            enable_hook_banner=req.enable_hook_banner,
            hook_text=req.hook_text,
            shorts_layout=req.shorts_layout,
            cam_position=req.cam_position,
            enable_chat_overlay=req.enable_chat_overlay,
            moment_type=req.moment_type or "funny"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"9:16 Dönüştürme hatası: {str(e)}")

    size_mb = round(os.path.getsize(out_path) / (1024 * 1024), 2)
    return {
        "status": "success",
        "vertical_filename": out_filename,
        "vertical_size_mb": size_mb,
        "url": f"/clips_media/{out_filename}"
    }

@app.post("/api/generate-thumbnail", dependencies=[Depends(require_license)])
def generate_thumbnail_endpoint(req: GenerateThumbnailRequest):
    filename = os.path.basename(unquote(req.filename))
    input_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Klip dosyası bulunamadı.")

    base, _ = os.path.splitext(filename)
    fmt = "916" if req.target_format == "916" else "169"
    out_thumb_name = f"{base}_thumb_{fmt}.jpg"
    out_thumb_path = os.path.join(OUTPUT_DIR, out_thumb_name)

    extractor = KickHighlightExtractor(output_dir=OUTPUT_DIR, temp_dir=TEMP_DIR)
    try:
        extractor.generate_clip_thumbnail(
            video_filepath=input_path,
            output_thumb_path=out_thumb_path,
            title_text=req.title_text or "YAYININ EN İYİ ANI",
            streamer_name=req.streamer_name or "Kick",
            moment_type=req.moment_type or "funny",
            hype_score=req.hype_score or 95,
            target_format=fmt,
            frame_sec=req.frame_sec,
            badge_text=req.badge_text,
            style=req.style or "kick_neon",
            focus_box=req.focus_box,
            draw_ring=req.draw_ring or False,
            ring_color=req.ring_color or "red",
            crop_to_focus=req.crop_to_focus or False
        )
        return {
            "status": "success",
            "thumbnail_filename": out_thumb_name,
            "url": f"/clips_media/{quote(out_thumb_name)}?v={int(time.time() * 1000)}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kapak resmi oluşturulamadı: {str(e)}")

@app.post("/api/gemini-thumbnail/analyze", dependencies=[Depends(require_license)])
def gemini_thumbnail_analyze_endpoint(req: GeminiThumbnailAnalyzeRequest):
    filename = os.path.basename(unquote(req.filename))
    input_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Klip dosyası bulunamadı.")

    from ai_headline_service import analyze_clip_for_ai_thumbnail
    analysis = analyze_clip_for_ai_thumbnail(
        streamer=req.streamer or "Kick",
        stream_title=req.stream_title or "Canlı Yayın",
        transcript=req.transcript or "",
        moment_type=req.moment_type or "funny",
        duration=req.duration or 30.0,
        peak_time_rel=req.peak_time_rel
    )

    base, _ = os.path.splitext(filename)
    out_916_name = f"{base}_thumb_916.jpg"
    out_169_name = f"{base}_thumb_169.jpg"
    out_916_path = os.path.join(OUTPUT_DIR, out_916_name)
    out_169_path = os.path.join(OUTPUT_DIR, out_169_name)

    extractor = KickHighlightExtractor(output_dir=OUTPUT_DIR, temp_dir=TEMP_DIR)
    try:
        # Pre-render 9:16
        extractor.generate_clip_thumbnail(
            video_filepath=input_path,
            output_thumb_path=out_916_path,
            title_text=analysis.get("viral_title", "YAYININ EN İYİ ANI"),
            streamer_name=req.streamer or "Kick",
            moment_type=req.moment_type or "funny",
            hype_score=95,
            target_format="916",
            frame_sec=analysis.get("best_frame_sec", 1.5),
            badge_text=analysis.get("badge_text", "ŞOK AN"),
            style=analysis.get("style", "kick_neon")
        )
        # Pre-render 16:9
        extractor.generate_clip_thumbnail(
            video_filepath=input_path,
            output_thumb_path=out_169_path,
            title_text=analysis.get("viral_title", "YAYININ EN İYİ ANI"),
            streamer_name=req.streamer or "Kick",
            moment_type=req.moment_type or "funny",
            hype_score=95,
            target_format="169",
            frame_sec=analysis.get("best_frame_sec", 1.5),
            badge_text=analysis.get("badge_text", "ŞOK AN"),
            style=analysis.get("style", "kick_neon")
        )

        return {
            "status": "success",
            "analysis": analysis,
            "thumbnails": {
                "thumb_916": f"/clips_media/{quote(out_916_name)}?v={int(time.time() * 1000)}",
                "thumb_169": f"/clips_media/{quote(out_169_name)}?v={int(time.time() * 1000)}",
                "filename_916": out_916_name,
                "filename_169": out_169_name
            }
        }
    except Exception as e:
        logger.error(f"Thumbnail pre-render error: {e}")
        return {
            "status": "partial",
            "analysis": analysis,
            "detail": str(e)
        }

@app.post("/api/gemini-thumbnail/generate-ai-image", dependencies=[Depends(require_license)])
def gemini_thumbnail_ai_image_endpoint(req: GeminiAiImageRequest):
    from ai_headline_service import get_gemini_api_key, craft_ai_thumbnail_prompt, generate_gemini_ai_image
    
    api_key = get_gemini_api_key()
    if not api_key:
        raise HTTPException(
            status_code=400, 
            detail="Gemini API anahtarı bulunamadı. Lütfen Ayarlar bölümünden Gemini API anahtarınızı kaydedin."
        )
    
    # 1. Craft or use prompt (with AI expansion if custom)
    prompt = craft_ai_thumbnail_prompt(
        streamer=req.streamer or "Kick",
        stream_title=req.stream_title or "Canlı Yayın",
        transcript=req.transcript or "",
        moment_type=req.moment_type or "funny",
        variation_index=req.variation_index,
        custom_prompt=req.custom_prompt,
        api_key=api_key
    )
    
    # 2. Call Imagen 3 model
    img_bytes, err = generate_gemini_ai_image(
        api_key=api_key,
        prompt=prompt,
        aspect_ratio=req.aspect_ratio or "16:9"
    )
    
    if not img_bytes or err:
        logger.error(f"Imagen image generation failed: {err}")
        raise HTTPException(
            status_code=500,
            detail=f"Gemini Imagen görsel oluşturamadı: {err}"
        )
        
    # 3. Save raw AI image
    timestamp = int(time.time() * 1000)
    raw_fname = f"ai_art_raw_{timestamp}.jpg"
    raw_path = os.path.join(OUTPUT_DIR, raw_fname)
    
    try:
        with open(raw_path, "wb") as f:
            f.write(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Görsel kaydedilemedi: {str(e)}")
        
    final_fname = raw_fname
    fmt = "916" if req.aspect_ratio in ["9:16", "916"] else "169"
    
    # 4. If text overlay is requested, apply thumbnail graphics via extractor
    if req.include_text_overlay:
        final_fname = f"ai_thumb_{fmt}_{timestamp}.jpg"
        final_path = os.path.join(OUTPUT_DIR, final_fname)
        
        extractor = KickHighlightExtractor(output_dir=OUTPUT_DIR, temp_dir=TEMP_DIR)
        try:
            extractor.generate_clip_thumbnail(
                video_filepath="",
                output_thumb_path=final_path,
                title_text=req.title_text or "YAYININ EN İYİ ANI",
                streamer_name=req.streamer or "Kick",
                moment_type=req.moment_type or "funny",
                hype_score=98,
                target_format=fmt,
                badge_text=req.badge_text or "YAPAY ZEKA",
                style=req.style or "kick_neon",
                base_image_path=raw_path,
                focus_box=req.focus_box,
                draw_ring=req.draw_ring or False,
                ring_color=req.ring_color or "red",
                crop_to_focus=req.crop_to_focus or False
            )
        except Exception as e:
            logger.warning(f"Failed to overlay text on AI image, falling back to raw: {e}")
            final_fname = raw_fname

    return {
        "status": "success",
        "ai_image_url": f"/clips_media/{quote(final_fname)}?v={timestamp}",
        "raw_image_url": f"/clips_media/{quote(raw_fname)}?v={timestamp}",
        "prompt_used": prompt,
        "variation_index": req.variation_index,
        "filename": final_fname
    }

@app.get("/api/studio/clips")
def get_studio_clips():
    """Returns all available video files in output_clips for the Studio editor."""
    items = []
    if os.path.exists(OUTPUT_DIR):
        for fname in os.listdir(OUTPUT_DIR):
            if fname.lower().endswith(('.mp4', '.mkv', '.mov', '.ts')):
                fpath = os.path.join(OUTPUT_DIR, fname)
                try:
                    stat = os.stat(fpath)
                    size_mb = round(stat.st_size / (1024 * 1024), 2)
                    mtime = stat.st_mtime
                    is_916 = "_916" in fname or "_studio" in fname
                    items.append({
                        "filename": fname,
                        "url": f"/clips_media/{quote(fname)}",
                        "size_mb": size_mb,
                        "mtime": mtime,
                        "is_916": is_916,
                        "name_clean": fname.replace(".mp4", "").replace("_916", "").replace("_", " ")
                    })
                except Exception:
                    pass
    # Also include root test files if available
    for t_name in ["test_sample_30s.mp4", "test_gpu_out.mp4", "test_cpu_out.mp4"]:
        t_path = os.path.join(BASE_DIR, t_name)
        if os.path.exists(t_path) and not any(it["filename"] == t_name for it in items):
            try:
                stat = os.stat(t_path)
                items.append({
                    "filename": t_name,
                    "url": f"/api/raw-file/{quote(t_name)}",
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "mtime": stat.st_mtime,
                    "is_916": False,
                    "name_clean": t_name.replace(".mp4", "")
                })
            except Exception:
                pass

    items.sort(key=lambda x: x["mtime"], reverse=True)
    return {"status": "success", "clips": items}

@app.get("/api/raw-file/{filename}")
def get_raw_file(filename: str):
    fpath = os.path.join(BASE_DIR, os.path.basename(filename))
    if not os.path.exists(fpath):
        fpath = os.path.join(OUTPUT_DIR, os.path.basename(filename))
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="Dosya bulunamadı")
    return FileResponse(fpath, media_type="video/mp4")

@app.post("/api/studio/upload-image", dependencies=[Depends(require_license)])
async def upload_studio_image_endpoint(file: UploadFile = File(...)):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Lütfen bir görsel seçin.")
    
    allowed = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Desteklenmeyen görsel formatı. PNG, JPG veya WEBP seçin.")

    overlays_dir = os.path.join(OUTPUT_DIR, "overlays")
    os.makedirs(overlays_dir, exist_ok=True)

    safe_name = f"overlay_{int(time.time())}_{os.path.basename(file.filename)}"
    save_path = os.path.join(overlays_dir, safe_name)

    try:
        with open(save_path, "wb") as f:
            while chunk := await file.read(1024 * 1024 * 2):
                f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Görsel kaydedilemedi: {str(e)}")

    return {
        "status": "success",
        "filename": safe_name,
        "url": f"/clips_media/overlays/{quote(safe_name)}"
    }

@app.post("/api/studio/render", dependencies=[Depends(require_license)])
def studio_render_endpoint(req: StudioRenderRequest):
    raw_fname = unquote(req.filename)
    if os.path.exists(raw_fname):
        input_path = raw_fname
        fname = os.path.basename(raw_fname)
    else:
        fname = os.path.basename(raw_fname)
        input_path = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(input_path):
            for chk in [TEMP_DIR, BASE_DIR]:
                p = os.path.join(chk, fname)
                if os.path.exists(p):
                    input_path = p
                    break
            else:
                raise HTTPException(status_code=404, detail="Düzenlenecek klip dosyası bulunamadı.")

    base, _ = os.path.splitext(fname)
    clean_base = re.sub(r'_(?:studio|916)+', '', base)
    timestamp = int(time.time() % 100000)
    out_filename = f"{clean_base}_studio_{req.layout[:4]}_{timestamp}.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_filename)

    extractor = KickHighlightExtractor(output_dir=OUTPUT_DIR, temp_dir=TEMP_DIR)
    try:
        t0 = time.time()
        extractor.render_studio_clip(
            input_filepath=input_path,
            output_filepath=out_path,
            layout=req.layout,
            cam_position=req.cam_position,
            cam_pan_x=req.cam_pan_x,
            cam_pan_y=req.cam_pan_y,
            cam_zoom=req.cam_zoom,
            game_pan_x=req.game_pan_x,
            game_pan_y=req.game_pan_y,
            game_zoom=req.game_zoom,
            fit_pan_x=req.fit_pan_x,
            blur_amount=req.blur_amount,
            trim_start=req.trim_start,
            trim_end=req.trim_end,
            speed=req.speed,
            volume=req.volume,
            normalize_audio=req.normalize_audio,
            header_text=req.header_text or "",
            header_color=req.header_color or "#ffe600",
            header_start=req.header_start,
            header_end=req.header_end,
            header_x=req.header_x,
            header_y=req.header_y,
            header_scale=req.header_scale,
            header_theme=req.header_theme,
            brightness=req.brightness,
            contrast=req.contrast,
            saturation=req.saturation,
            video_quality=req.video_quality,
            encoder_choice=req.encoder_choice,
            overlay_image=req.overlay_image,
            overlay_x=req.overlay_x,
            overlay_y=req.overlay_y,
            overlay_scale=req.overlay_scale,
            overlay_opacity=req.overlay_opacity,
            subtitles_data=req.subtitles_data,
            subtitles_style=req.subtitles_style,
            subtitles_pos=req.subtitles_pos
        )
        render_time = round(time.time() - t0, 2)
    except Exception as e:
        logger.error(f"Studio render error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Studio Render hatası: {str(e)}")

    size_mb = round(os.path.getsize(out_path) / (1024 * 1024), 2)
    return {
        "status": "success",
        "output_filename": out_filename,
        "size_mb": size_mb,
        "render_time_sec": render_time,
        "url": f"/clips_media/{quote(out_filename)}"
    }

@app.post("/api/studio/transcribe", dependencies=[Depends(require_license)])
def studio_transcribe_endpoint(req: StudioTranscribeRequest):
    raw_fname = unquote(req.filename)
    if os.path.exists(raw_fname):
        input_path = raw_fname
        fname = os.path.basename(raw_fname)
    else:
        fname = os.path.basename(raw_fname)
        input_path = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(input_path):
            for chk in [TEMP_DIR, BASE_DIR]:
                p = os.path.join(chk, fname)
                if os.path.exists(p):
                    input_path = p
                    break
            else:
                raise HTTPException(status_code=404, detail="Klip dosyası bulunamadı.")

    try:
        from faster_whisper import WhisperModel
        model_name = req.model if req.model in ["tiny", "base", "small"] else "small"
        
        # Whisper CPU int8 with multithreading (fast ~2-3s, 100% stable without cublas DLL errors)
        m = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=min(8, os.cpu_count() or 4))

        segments, info = m.transcribe(
            input_path,
            language=req.language or "tr",
            word_timestamps=True,
            vad_filter=True,
            beam_size=5,
            initial_prompt="Türkçe canlı yayın konuşması, komik ve heyecanlı anlar."
        )

        segments_list = list(segments)
        # Fallback if VAD filter was overly aggressive on noisy stream or quiet speech
        if not segments_list:
            logger.info("VAD filter returned 0 segments; retrying transcription with vad_filter=False...")
            fallback_segs, _ = m.transcribe(
                input_path,
                language=req.language or "tr",
                word_timestamps=True,
                vad_filter=False,
                beam_size=3,
                initial_prompt="Türkçe canlı yayın konuşması, komik ve heyecanlı anlar."
            )
            segments_list = list(fallback_segs)

        subtitles = []
        sub_id = 1
        full_text_parts = []

        for seg in segments_list:
            full_text_parts.append(seg.text.strip())
            words = seg.words if seg.words else []
            if not words:
                txt = seg.text.strip()
                if txt:
                    subtitles.append({"id": sub_id, "start": float(round(seg.start, 2)), "end": float(round(seg.end, 2)), "text": txt})
                    sub_id += 1
                continue

            chunk = []
            chunk_start = None
            for w in words:
                clean_word = w.word.strip()
                if not clean_word:
                    continue
                if chunk_start is None:
                    chunk_start = float(w.start)
                chunk.append(clean_word)
                if len(chunk) >= 4 or any(clean_word.endswith(p) for p in ['.', '!', '?', ',']):
                    subtitles.append({
                        "id": sub_id,
                        "start": float(round(chunk_start, 2)),
                        "end": float(round(w.end, 2)),
                        "text": " ".join(chunk)
                    })
                    sub_id += 1
                    chunk = []
                    chunk_start = None

            if chunk and words:
                subtitles.append({
                    "id": sub_id,
                    "start": float(round(chunk_start if chunk_start is not None else seg.start, 2)),
                    "end": float(round(words[-1].end, 2)),
                    "text": " ".join(chunk)
                })
                sub_id += 1

        return {
            "status": "success",
            "model_used": model_name,
            "subtitles": subtitles,
            "count": len(subtitles),
            "full_text": " ".join(full_text_parts)
        }
    except Exception as e:
        logger.error(f"Studio transcribe error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Altyazı çıkarılırken hata oluştu: {str(e)}")

@app.post("/api/ai/analyze-scene", dependencies=[Depends(require_license)])
def ai_analyze_scene_endpoint(req: AISceneAnalyzeRequest):
    try:
        from ai_headline_service import analyze_transcript_scenes_with_llm
        sentences = req.sentences or []
        if not sentences and req.transcript_text:
            lines = [l.strip() for l in req.transcript_text.split('\n') if l.strip()]
            for idx, line in enumerate(lines):
                sentences.append({
                    "start": idx * 4.0,
                    "end": (idx + 1) * 4.0,
                    "text": line
                })

        scenes = analyze_transcript_scenes_with_llm(
            sentences=sentences,
            streamer=req.streamer or "",
            stream_title=req.stream_title or "",
            target_category=req.target_category or "all"
        )
        return {"status": "success", "scenes": scenes}
    except Exception as e:
        logger.error(f"Scene analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Sahne analizi hatası: {str(e)}")

@app.get("/api/status/{task_id}")
def get_task_status(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Görev bulunamadı.")
    return TASKS[task_id]

@app.post("/api/open-folder", dependencies=[Depends(require_license)])
def open_output_folder():
    try:
        if sys.platform == "win32":
            os.startfile(OUTPUT_DIR)
        else:
            subprocess.run(["xdg-open", OUTPUT_DIR])
        return {"status": "success", "path": OUTPUT_DIR}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download-all/{task_id}", dependencies=[Depends(require_license)])
def download_all_zip(task_id: str, format: str = "all"):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Görev bulunamadı.")
    
    task = TASKS[task_id]
    completed_clips = task.get("completed_clips", [])
    if not completed_clips:
        raise HTTPException(status_code=400, detail="Henüz indirilmiş klip yok.")

    zip_filename = f"kick_highlights_{task_id}_{format}.zip"
    zip_filepath = os.path.join(TEMP_DIR, zip_filename)

    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for clip in completed_clips:
            if format == "916":
                fp = clip.get("vertical_filepath") or clip.get("filepath")
            elif format == "169":
                fp = clip.get("filepath")
            else:
                # all available
                fp = clip.get("filepath")
                fp_916 = clip.get("vertical_filepath")
                if fp_916 and os.path.exists(fp_916):
                    zipf.write(fp_916, arcname=os.path.basename(fp_916))

            if fp and os.path.exists(fp):
                zipf.write(fp, arcname=os.path.basename(fp))

    return FileResponse(zip_filepath, media_type="application/zip", filename=zip_filename)

@app.get("/api/download-clip/{filename}")
def download_single_clip_file(filename: str):
    safe_filename = os.path.basename(unquote(filename).strip())
    filepath = os.path.join(OUTPUT_DIR, safe_filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Klip dosyası bulunamadı.")
    return FileResponse(
        filepath,
        media_type="video/mp4",
        filename=safe_filename
    )

@app.get("/api/hls-proxy")
def hls_proxy_endpoint(url: str):
    target_url = unquote(url).strip()
    if not target_url or not target_url.startswith("http"):
        raise HTTPException(status_code=400, detail="Geçersiz stream URL.")

    s = get_cdn_session()
    try:
        r = s.get(target_url, timeout=10)
        ct = r.headers.get("Content-Type", "")
        if ".m3u8" in target_url or "mpegurl" in ct or "application/x-mpegURL" in ct:
            base_url = target_url.rsplit("/", 1)[0]
            lines = []
            for line in r.text.split("\n"):
                line_clean = line.strip()
                if line_clean and not line_clean.startswith("#"):
                    full_line_url = line_clean if line_clean.startswith("http") else f"{base_url}/{line_clean}"
                    lines.append(f"/api/hls-proxy?url={quote(full_line_url)}")
                else:
                    lines.append(line)
            return Response(content="\n".join(lines), media_type="application/vnd.apple.mpegurl")
        else:
            return Response(content=r.content, media_type=ct or "video/MP2T")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ResolveStreamRequest(BaseModel):
    source: str

@app.post("/api/resolve-preview-stream", dependencies=[Depends(require_license)])
def resolve_preview_stream_endpoint(req: ResolveStreamRequest):
    extractor = KickHighlightExtractor(output_dir=OUTPUT_DIR, temp_dir=TEMP_DIR)
    best_stream_url, segments = extractor.get_best_video_stream_and_segments(req.source)
    proxy_url = f"/api/hls-proxy?url={quote(best_stream_url)}" if best_stream_url else None
    return {
        "status": "success",
        "stream_url": best_stream_url,
        "proxy_url": proxy_url
    }

class GeminiConfigRequest(BaseModel):
    api_key: str

class RegenerateHeadlineRequest(BaseModel):
    streamer: str = ""
    stream_title: str = ""
    transcript: str = ""
    hook_text: str = ""
    payoff_text: str = ""
    moment_type: str = "funny"
    duration: float = 35.0

@app.get("/api/gemini-config")
def get_gemini_config_endpoint():
    try:
        from ai_headline_service import get_gemini_api_key
        key = get_gemini_api_key()
        has_key = bool(key)
        masked = (key[:4] + "..." + key[-4:]) if len(key) > 8 else ("aktif" if has_key else "")
        return {
            "status": "success",
            "has_key": has_key,
            "masked_key": masked
        }
    except Exception as e:
        return {"status": "error", "detail": str(e), "has_key": False}

@app.post("/api/gemini-config")
def save_gemini_config_endpoint(req: GeminiConfigRequest):
    try:
        from ai_headline_service import save_gemini_api_key
        ok = save_gemini_api_key(req.api_key)
        return {"status": "success" if ok else "error"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/api/regenerate-clip-headlines")
def regenerate_clip_headlines_endpoint(req: RegenerateHeadlineRequest):
    try:
        from ai_headline_service import generate_clip_titles_and_hooks
        res = generate_clip_titles_and_hooks(
            streamer=req.streamer,
            stream_title=req.stream_title,
            transcript=req.transcript,
            hook_text=req.hook_text,
            payoff_text=req.payoff_text,
            moment_type=req.moment_type,
            duration=req.duration
        )
        return {
            "status": "success",
            "headlines": res
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/manifest.json")
def serve_manifest():
    manifest_path = os.path.join(STATIC_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        return FileResponse(manifest_path, media_type="application/manifest+json")
    return JSONResponse({})

@app.get("/sw.js")
def serve_sw():
    sw_path = os.path.join(STATIC_DIR, "sw.js")
    if os.path.exists(sw_path):
        return FileResponse(sw_path, media_type="application/javascript")
    return Response("self.addEventListener('fetch', () => {});", media_type="application/javascript")

CURRENT_SERVER_PORT = int(os.environ.get("PORT", "7860" if (os.environ.get("SPACE_ID") or os.environ.get("CLOUD_MODE") == "1") else "8000"))

def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

@app.get("/api/network-info")
def get_network_info():
    local_ip = get_local_ip()
    return {
        "status": "success",
        "local_ip": local_ip,
        "port": CURRENT_SERVER_PORT,
        "local_url": f"http://127.0.0.1:{CURRENT_SERVER_PORT}",
        "mobile_url": f"http://{local_ip}:{CURRENT_SERVER_PORT}"
    }

@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Kick Clipper Başlatılıyor...</h1>"



@app.get("/icon-192.png")
def serve_icon_192():
    icon_path = os.path.join(STATIC_DIR, "icon-192.png")
    if os.path.exists(icon_path):
        return FileResponse(icon_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Icon not found")

@app.get("/icon-512.png")
def serve_icon_512():
    icon_path = os.path.join(STATIC_DIR, "icon-512.png")
    if os.path.exists(icon_path):
        return FileResponse(icon_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Icon not found")

def get_free_port(start_port: int = 8000, max_port: int = 8010) -> int:
    for port in range(start_port, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return port
            except OSError:
                continue
    return start_port

if __name__ == "__main__":
    import uvicorn
    is_cloud = bool(os.environ.get("SPACE_ID") or os.environ.get("CLOUD_MODE") == "1")
    default_port = int(os.environ.get("PORT", 7860 if is_cloud else 8000))
    port = default_port if is_cloud else get_free_port(default_port)
    CURRENT_SERVER_PORT = port
    local_ip = get_local_ip()
    url = f"http://127.0.0.1:{port}"
    mobile_url = f"http://{local_ip}:{port}"
    print(f"\n=======================================================")
    print(f"🎬 KICK HIGHLIGHT CLIPPER BASLATILDI (MOBIL UYUMLU)")
    print(f"💻 Bilgisayar Adresi: {url}")
    if not is_cloud:
        print(f"📱 Telefon/Mobil Adresi (Aynı Wi-Fi): {mobile_url}")
        print(f"🌐 Uzak Erişim (İnternet): Cloudflare Tunnel veya Tailscale")
    else:
        print(f"☁️ Bulut Modu Aktif (Hugging Face Spaces - Port {port})")
    print(f"=======================================================\n")
    
    if not is_cloud:
        import threading
        threading.Thread(target=lambda: (time.sleep(1), webbrowser.open(url)), daemon=True).start()
    
    # Kick Clipper FastAPI runner
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=not is_cloud, access_log=False)

