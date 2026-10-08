"""
Kick Highlight Extractor Core Logic (Viral Title AI Engine & Turbo Clipper)
Generates engaging, clickbait, and contextual titles/hashtags for YouTube Shorts, TikTok, and Reels.
"""

import os
import sys
import json
import time
import uuid
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import re
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import butter, sosfilt
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from typing import List, Dict, Any, Callable, Optional, Tuple
from curl_cffi import requests
import requests as std_requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image, ImageDraw, ImageFont
import logging

logger = logging.getLogger("extractor")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def create_robust_cdn_session():
    """High-throughput resilient session for HLS playlist & TS segment downloading (avoids WinError 10054)."""
    s = std_requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.25,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=64, pool_maxsize=64)
    s.mount('https://', adapter)
    s.mount('http://', adapter)
    s.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Connection': 'keep-alive'
    })
    return s

_cdn_session = create_robust_cdn_session()

def get_cdn_session():
    return _cdn_session

def get_ffmpeg_binary() -> str:
    # 1. Try imageio_ffmpeg first (guaranteed valid binary that avoids Windows App Control 4551 symlink blocks)
    try:
        import imageio_ffmpeg
        img_bin = imageio_ffmpeg.get_ffmpeg_exe()
        if img_bin and os.path.exists(img_bin):
            return img_bin
    except Exception:
        pass

    # 2. Check local binary in directory if exists
    local_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg.exe")
    if os.path.exists(local_bin):
        return local_bin

    # 3. Fallback to system ffmpeg
    return "ffmpeg"


_BEST_VCODEC_OPTS = None

def get_hardware_video_encoder(target_crf: int = 18) -> List[str]:
    """
    Auto-detects GPU hardware encoder (NVIDIA NVENC, Intel QSV, AMD AMF).
    RTX 5060 Ti NVENC encodes 1080p60 at 500-1000 FPS (1-2s per clip).
    """
    global _BEST_VCODEC_OPTS
    if _BEST_VCODEC_OPTS is not None:
        opts = list(_BEST_VCODEC_OPTS)
        if "-cq" in opts:
            opts[opts.index("-cq") + 1] = str(target_crf)
        elif "-crf" in opts:
            opts[opts.index("-crf") + 1] = str(target_crf)
        elif "-global_quality" in opts:
            opts[opts.index("-global_quality") + 1] = str(target_crf)
        return opts

    # 1. Test NVIDIA NVENC (RTX 5060 Ti)
    try:
        test_cmd = [
            get_ffmpeg_binary(), '-y',
            '-f', 'lavfi', '-i', 'testsrc=duration=0.5:size=256x256:rate=30',
            '-c:v', 'h264_nvenc', '-preset', 'p2', '-cq', str(target_crf), '-pix_fmt', 'yuv420p',
            '-f', 'null', '-'
        ]
        res = subprocess.run(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=4)
        if res.returncode == 0:
            _BEST_VCODEC_OPTS = ['-c:v', 'h264_nvenc', '-preset', 'p2', '-cq', str(target_crf), '-pix_fmt', 'yuv420p']
            return list(_BEST_VCODEC_OPTS)
    except Exception:
        pass

    # 2. Test Intel QSV
    try:
        test_cmd = [
            get_ffmpeg_binary(), '-y',
            '-f', 'lavfi', '-i', 'testsrc=duration=0.5:size=256x256:rate=30',
            '-c:v', 'h264_qsv', '-global_quality', str(target_crf), '-pix_fmt', 'nv12',
            '-f', 'null', '-'
        ]
        res = subprocess.run(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=4)
        if res.returncode == 0:
            _BEST_VCODEC_OPTS = ['-c:v', 'h264_qsv', '-global_quality', str(target_crf), '-pix_fmt', 'nv12']
            return list(_BEST_VCODEC_OPTS)
    except Exception:
        pass

    # 3. Fallback to CPU libx264 ultrafast multithreaded
    _BEST_VCODEC_OPTS = ['-c:v', 'libx264', '-preset', 'ultrafast', '-crf', str(target_crf), '-threads', '0', '-pix_fmt', 'yuv420p']
    return list(_BEST_VCODEC_OPTS)


class KickHighlightExtractor:
    def __init__(self, output_dir: str = "output_clips", temp_dir: str = "temp_work"):
        self.output_dir = os.path.abspath(output_dir)
        self.temp_dir = os.path.abspath(temp_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        self.session = requests.Session(impersonate='chrome124')

    def ensure_whoosh_sfx(self) -> str:
        """Guarantees existence of a 48kHz stereo cinematic whoosh SFX for flash transitions."""
        sfx_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "sfx")
        os.makedirs(sfx_dir, exist_ok=True)
        sfx_path = os.path.join(sfx_dir, "whoosh.wav")
        if not os.path.exists(sfx_path) or os.path.getsize(sfx_path) < 5000:
            try:
                sr = 48000
                dur = 0.60
                t = np.linspace(0, dur, int(sr * dur), endpoint=False)
                freq_curve = 120.0 + 430.0 * np.sin(np.pi * (t / dur))**2
                phase = 2.0 * np.pi * np.cumsum(freq_curve) / sr
                sine_wave = np.sin(phase)
                noise = np.random.normal(0, 0.35, len(t))
                env = np.sin(np.pi * (t / dur))**2.5
                whoosh_audio = (sine_wave * 0.45 + noise * 0.55) * env
                whoosh_norm = np.int16(whoosh_audio / np.max(np.abs(whoosh_audio) + 1e-6) * 28000)
                stereo = np.column_stack([whoosh_norm, whoosh_norm])
                wavfile.write(sfx_path, sr, stereo)
            except Exception as e:
                print(f"SFX synthesis notice: {e}")
        return sfx_path

    def log(self, msg: str, progress: int = 0, callback: Optional[Callable[[str, int], None]] = None):
        try:
            print(f"[{progress}%] {msg}")
        except Exception:
            pass
        if callback:
            callback(msg, progress)

    @staticmethod
    def extract_channel_and_id(url: str) -> tuple[str, Optional[str]]:
        url = url.strip()
        match = re.search(r'kick\.com/([^/?#]+)(?:/videos/([^/?#]+))?', url)
        if match:
            channel = match.group(1).lower()
            video_id = match.group(2)
            return channel, video_id
        clean = url.replace('https://', '').replace('http://', '').replace('/', '').lower()
        return clean, None

    def get_cookie_file(self) -> Optional[str]:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        user_home = os.path.expanduser("~")
        downloads_dir = os.path.join(user_home, "Downloads")
        
        candidates = []
        if os.path.exists(downloads_dir):
            try:
                for f in os.listdir(downloads_dir):
                    if "cookie" in f.lower() and f.endswith(".txt"):
                        candidates.append(os.path.join(downloads_dir, f))
            except Exception:
                pass

        candidates.extend([
            os.path.join(base_dir, 'www.youtube.com_cookies.txt'),
            os.path.join(base_dir, 'cookies.txt'),
            os.path.join(base_dir, 'cookies_fixed.txt'),
            os.path.join(os.getcwd(), 'www.youtube.com_cookies.txt'),
            os.path.join(os.getcwd(), 'cookies.txt'),
            os.path.join(os.getcwd(), 'cookies_fixed.txt')
        ])

        valid = [c for c in candidates if os.path.exists(c) and os.path.getsize(c) > 50]
        if not valid:
            return None

        # Prioritize files with LOGIN_INFO and recent timestamps
        def score_cookie_file(path: str) -> tuple:
            has_login = False
            try:
                with open(path, 'r', errors='ignore') as cf:
                    content = cf.read(4096)
                    if 'LOGIN_INFO' in content or '__Secure-3PSID' in content:
                        has_login = True
            except Exception:
                pass
            return (1 if has_login else 0, os.path.getmtime(path))

        valid.sort(key=score_cookie_file, reverse=True)
        newest = valid[0]

        local_target = os.path.join(base_dir, 'cookies.txt')
        if os.path.abspath(newest) != os.path.abspath(local_target):
            try:
                import shutil
                shutil.copy2(newest, local_target)
                return local_target
            except Exception:
                pass
        return newest

    def get_yt_dlp_opts(self, extra_opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        node_path = r"C:\Program Files\nodejs\node.exe"
        ff_bin = get_ffmpeg_binary()
        opts = {
            'quiet': True,
            'nocolor': True,
            'no_warnings': True,
            'remote_components': ['ejs:github'],
            'no_cookies_on_exit': True,
            'socket_timeout': 15,
            'retries': 2,
            'fragment_retries': 2,
            'ffmpeg_location': ff_bin,
        }
        if os.path.exists(node_path):
            opts['js_runtimes'] = {'node': {'path': node_path}}
            if r"C:\Program Files\nodejs" not in os.environ.get("PATH", ""):
                os.environ["PATH"] = r"C:\Program Files\nodejs;" + os.environ.get("PATH", "")

        cookie_f = self.get_cookie_file()
        if cookie_f:
            opts['cookiefile'] = cookie_f

        if extra_opts:
            if 'extractor_args' in extra_opts:
                yt_args = opts.setdefault('extractor_args', {}).setdefault('youtube', {})
                extra_yt = extra_opts['extractor_args'].get('youtube', {})
                yt_args.update(extra_yt)
                del extra_opts['extractor_args']
            opts.update(extra_opts)
        return opts

    def fetch_youtube_videos(self, url: str) -> Dict[str, Any]:
        import yt_dlp
        import re

        base_opts = self.get_yt_dlp_opts({
            'skip_download': True,
            'extract_flat': 'in_playlist',
            'playlistend': 15
        })

        info = None
        last_err = ""
        # 1. Try anonymous first (immune to cookie rate-limits), fallback to cookies if required
        attempts = [False, True] if base_opts.get('cookiefile') else [False]
        for use_cookies in attempts:
            ydl_opts = dict(base_opts)
            if not use_cookies:
                ydl_opts.pop('cookiefile', None)
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                if info:
                    break
            except Exception as e:
                last_err = re.sub(r'\x1b\[[0-9;]*m', '', str(e))
                if "Sign in to confirm your age" in last_err or "confirm your age" in last_err:
                    raise ValueError("⚠️ Bu YouTube videosu +18 (Yaş Kısıtlamalı) olduğu için YouTube oturum açılmasını istiyor!\n\n💡 Çözüm: Proje klasörüne 'cookies.txt' dosyanızı ekleyerek yaş kısıtlamalı videoları kesebilirsiniz.")

        if not info:
            raise ValueError(f"YouTube videosu getirilemedi: {last_err or 'Bilinmeyen hata'}")

        all_videos = []
        if 'entries' in info and info['entries']:
            for entry in info['entries']:
                if not entry: continue
                v_id = entry.get('id')
                dur = int(entry.get('duration') or 0)
                all_videos.append({
                    'id': str(v_id),
                    'uuid': str(v_id),
                    'slug': str(v_id),
                    'title': entry.get('title') or 'YouTube Video',
                    'uploader': entry.get('uploader') or info.get('uploader') or 'YouTube',
                    'duration': dur,
                    'duration_formatted': self.format_time(dur),
                    'created_at': 'YouTube Video',
                    'source': f"https://www.youtube.com/watch?v={v_id}",
                    'thumbnail': entry.get('thumbnail') or f"https://img.youtube.com/vi/{v_id}/hqdefault.jpg",
                    'views': entry.get('view_count', 0)
                })
        else:
            v_id = info.get('id')
            dur = int(info.get('duration') or 0)
            all_videos.append({
                'id': str(v_id),
                'uuid': str(v_id),
                'slug': str(v_id),
                'title': info.get('title') or 'YouTube Video',
                'uploader': info.get('uploader') or 'YouTube',
                'duration': dur,
                'duration_formatted': self.format_time(dur),
                'created_at': 'YouTube Video',
                'source': f"https://www.youtube.com/watch?v={v_id}",
                'thumbnail': info.get('thumbnail') or f"https://img.youtube.com/vi/{v_id}/hqdefault.jpg",
                'views': info.get('view_count', 0)
            })

        return {
            'channel': info.get('uploader') or info.get('title') or 'YouTube',
            'preselected_id': all_videos[0]['id'] if all_videos else None,
            'total_videos': len(all_videos),
            'videos': all_videos
        }
    def fetch_gdrive_videos(self, url: str) -> Dict[str, Any]:
        """
        Google Drive video/klasör linkinden video listesi çıkarır.
        3 katmanlı fallback:
          1) yt-dlp ile dene (meta bilgi + süre)
          2) Direkt download URL ile requests ile dene
          3) URL'den file_id çıkar, manuel video objesi oluştur (her zaman çalışır)
        """
        import re
        import yt_dlp

        clean_url = url.strip()
        file_m = re.search(r'(?:file/d/|id=|open\?id=)([a-zA-Z0-9_-]{25,})', clean_url)
        folder_m = re.search(r'folders/([a-zA-Z0-9_-]{25,})', clean_url)

        target_url = clean_url
        if folder_m:
            target_url = f"https://drive.google.com/drive/folders/{folder_m.group(1)}"
        elif file_m:
            target_url = f"https://drive.google.com/file/d/{file_m.group(1)}/view"

        # ── KATMAN 1: yt-dlp ile dene ────────────────────────────────────────────
        ydl_opts = self.get_yt_dlp_opts({
            'skip_download': True,
            'extract_flat': 'in_playlist',
            'playlistend': 50,
        })
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(target_url, download=False)
                if info:
                    all_videos = []
                    if 'entries' in info and info['entries']:
                        for entry in info['entries']:
                            if not entry:
                                continue
                            v_id  = entry.get('id')
                            dur   = int(entry.get('duration') or 0) or 28800
                            v_title = entry.get('title') or f"Drive Video {v_id}"
                            all_videos.append({
                                'id': str(v_id),
                                'uuid': str(v_id),
                                'slug': str(v_id),
                                'title': v_title,
                                'uploader': 'Google Drive',
                                'duration': dur,
                                'duration_formatted': self.format_time(dur),
                                'created_at': 'Google Drive Stok Depo',
                                'source': f"https://drive.google.com/file/d/{v_id}/view",
                                'thumbnail': entry.get('thumbnail') or "",
                                'views': 0,
                            })
                    else:
                        v_id  = info.get('id') or (file_m.group(1) if file_m else "drive_video")
                        dur   = int(info.get('duration') or 0) or 28800
                        v_title = info.get('title') or "Google Drive Video"
                        all_videos.append({
                            'id': str(v_id),
                            'uuid': str(v_id),
                            'slug': str(v_id),
                            'title': v_title,
                            'uploader': 'Google Drive',
                            'duration': dur,
                            'duration_formatted': self.format_time(dur),
                            'created_at': 'Google Drive Stok Depo',
                            'source': f"https://drive.google.com/file/d/{v_id}/view",
                            'thumbnail': info.get('thumbnail') or "",
                            'views': 0,
                        })
                    if all_videos:
                        return {
                            'channel': info.get('title') or 'Google Drive Stok Depo',
                            'preselected_id': all_videos[0]['id'],
                            'total_videos': len(all_videos),
                            'videos': all_videos,
                        }
        except Exception:
            pass  # Katman 2'ye geç

        # ── KATMAN 2: Direkt indirme URL'si üret, requests ile dosya adı al ──────
        if file_m:
            v_id = file_m.group(1)
            direct_url = f"https://drive.google.com/uc?export=download&id={v_id}"
            v_title = f"Drive_Video_{v_id[:8]}"
            dur = 28800  # varsayılan 8 saat

            try:
                import requests
                head = requests.head(direct_url, allow_redirects=True, timeout=8,
                                     headers={'User-Agent': 'Mozilla/5.0'})
                cd = head.headers.get('Content-Disposition', '')
                fn_m = re.search(r'filename\*?=["\']?(?:UTF-8\'\')?([^"\';]+)', cd)
                if fn_m:
                    raw_name = fn_m.group(1).strip()
                    v_title = re.sub(r'\.[^.]+$', '', raw_name) or v_title
            except Exception:
                pass

            single_vid = {
                'id': str(v_id),
                'uuid': str(v_id),
                'slug': str(v_id),
                'title': v_title,
                'uploader': 'Google Drive',
                'duration': dur,
                'duration_formatted': self.format_time(dur),
                'created_at': 'Google Drive Stok Depo',
                'source': f"https://drive.google.com/file/d/{v_id}/view",
                'thumbnail': "",
                'views': 0,
            }
            return {
                'channel': 'Google Drive Stok Depo',
                'preselected_id': single_vid['id'],
                'total_videos': 1,
                'videos': [single_vid],
            }

        # ── KATMAN 3: Klasör ve bilinmeyen link — son hata ───────────────────────
        raise ValueError(
            "Google Drive linki tanınamadı. Lütfen 'file/d/...' veya 'folders/...' içeren "
            "bir Drive linki girin ve dosyanın herkese açık (public) olduğundan emin olun."
        )

    def fetch_channel_videos(self, channel_or_url: str, max_pages: int = 3) -> Dict[str, Any]:
        if "youtube.com" in channel_or_url.lower() or "youtu.be" in channel_or_url.lower():
            return self.fetch_youtube_videos(channel_or_url)
        if "drive.google.com" in channel_or_url.lower() or "docs.google.com" in channel_or_url.lower():
            return self.fetch_gdrive_videos(channel_or_url)

        channel_name, preselected_id = self.extract_channel_and_id(channel_or_url)
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Referer': f'https://kick.com/{channel_name}'
        }

        all_videos = []

        # Otomatik Canlı Yayın Kontrolü (Yayıncı şu anda yayındaysa listenin en tepesine ekle)
        try:
            live_api_url = f"https://kick.com/api/v2/channels/{channel_name}"
            r_live = self.session.get(live_api_url, headers=headers, timeout=5)
            if r_live.status_code == 200:
                live_data = r_live.json()
                livestream = live_data.get('livestream')
                if livestream and livestream.get('is_live'):
                    thumb_obj = livestream.get('thumbnail') or {}
                    thumb_url = thumb_obj.get('url', '') if isinstance(thumb_obj, dict) else str(thumb_obj)
                    if not thumb_url:
                        banner = live_data.get('banner_image') or {}
                        thumb_url = banner.get('url', '') if isinstance(banner, dict) else ''
                    live_pb = live_data.get('playback_url') or ''
                    all_videos.append({
                        'id': f"live_{channel_name}",
                        'uuid': f"live_{channel_name}",
                        'slug': f"live_{channel_name}",
                        'title': f"🔴 CANLI YAYIN: {livestream.get('session_title') or channel_name}",
                        'uploader': channel_name,
                        'duration': int(livestream.get('duration', 0) / 1000) or 3600,
                        'duration_formatted': "Şu An Canlı",
                        'created_at': "CANLI YAYINDA",
                        'source': live_pb,
                        'thumbnail': thumb_url,
                        'views': livestream.get('viewer_count', 0),
                        'is_live': True
                    })
        except Exception:
            pass

        for page in range(1, max_pages + 1):
            api_url = f"https://kick.com/api/v2/channels/{channel_name}/videos?page={page}"
            try:
                r = self.session.get(api_url, headers=headers, timeout=12)
                if r.status_code != 200:
                    break
                data = r.json()
                if not isinstance(data, list) or len(data) == 0:
                    break
                for v in data:
                    thumb_obj = v.get('thumbnail', {})
                    thumb_url = thumb_obj.get('src') if isinstance(thumb_obj, dict) else ''
                    duration_sec = int(v.get('duration', 0) / 1000)
                    v_obj = v.get('video') or {}

                    all_videos.append({
                        'id': str(v.get('id')),
                        'uuid': str(v_obj.get('uuid', '')),
                        'slug': str(v.get('slug', '')),
                        'title': v.get('session_title') or f"{channel_name} Yayını",
                        'uploader': channel_name,
                        'duration': duration_sec,
                        'duration_formatted': self.format_time(duration_sec),
                        'created_at': v.get('created_at', ''),
                        'source': v.get('source'),
                        'thumbnail': thumb_url,
                        'views': v.get('views', 0)
                    })
            except Exception as e:
                print(f"Error fetching page {page} for {channel_name}: {e}")
                break

        if not all_videos:
            raise ValueError(f"'{channel_name}' kanalına ait yayın geçmişi bulunamadı.")

        return {
            'channel': channel_name,
            'preselected_id': preselected_id,
            'total_videos': len(all_videos),
            'videos': all_videos
        }

    def get_low_res_audio_stream_and_segments(self, master_url: str) -> Tuple[str, List[Dict[str, Any]]]:
        if not master_url or not master_url.startswith('http'):
            return master_url, []
        try:
            s = get_cdn_session()
            r = s.get(master_url, timeout=10)
            if r.status_code == 200:
                base = master_url.rsplit('/', 1)[0]
                lines = r.text.split('\n')
                sub_url = None
                for line in lines:
                    line_clean = line.strip()
                    if line_clean.endswith('.m3u8') and ('160p' in line_clean or '360p' in line_clean):
                        sub_url = line_clean if line_clean.startswith('http') else f"{base}/{line_clean}"
                        break
                if not sub_url:
                    for line in lines:
                        line_clean = line.strip()
                        if line_clean.endswith('.m3u8'):
                            sub_url = line_clean if line_clean.startswith('http') else f"{base}/{line_clean}"
                            break

                if sub_url:
                    r_sub = s.get(sub_url, timeout=10)
                    sub_base = sub_url.rsplit('/', 1)[0]
                    segments = []
                    current_time = 0.0
                    current_duration = 2.0

                    for s_line in r_sub.text.split('\n'):
                        s_line = s_line.strip()
                        if s_line.startswith('#EXTINF:'):
                            try:
                                dur_str = s_line.split(':')[1].split(',')[0]
                                current_duration = float(dur_str)
                            except Exception:
                                current_duration = 2.0
                        elif s_line and not s_line.startswith('#'):
                            seg_url = s_line if s_line.startswith('http') else f"{sub_base}/{s_line}"
                            segments.append({
                                'url': seg_url,
                                'start_time': current_time,
                                'duration': current_duration
                            })
                            current_time += current_duration
                    return sub_url, segments
        except Exception as e:
            print(f"Error resolving low-res stream: {e}")
        return master_url, []

    def get_best_video_stream_and_segments(self, master_url: str, video_quality: str = "1080p60") -> Tuple[str, List[Dict[str, Any]]]:
        if not master_url or not master_url.startswith('http'):
            return master_url, []

        if "drive.google.com" in master_url.lower() or "docs.google.com" in master_url.lower():
            import yt_dlp
            ydl_opts = self.get_yt_dlp_opts({'format': 'best'})
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(master_url, download=False)
                    return info.get('url', master_url), []
            except Exception:
                return master_url, []

        if "youtube.com" in master_url.lower() or "youtu.be" in master_url.lower():
            import yt_dlp
            if video_quality == "2k_60fps":
                fmt_str = 'bestvideo[height<=1440]+bestaudio/bestvideo+bestaudio/best'
            elif video_quality == "1080p60":
                fmt_str = 'bestvideo[height<=1080]+bestaudio/bestvideo+bestaudio/best'
            else:
                fmt_str = 'bestvideo+bestaudio/best'

            ydl_opts = self.get_yt_dlp_opts({'format': fmt_str})
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(master_url, download=False)
                    req = info.get('requested_formats')
                    if req and len(req) >= 2:
                        return f"{req[0]['url']}|||{req[1]['url']}", []
                    return info.get('url', master_url), []
            except Exception:
                return master_url, []

        try:
            s = get_cdn_session()
            r = s.get(master_url, timeout=10)
            if r.status_code == 200:
                base = master_url.rsplit('/', 1)[0]
                lines = [line.strip() for line in r.text.split('\n') if line.strip()]

                # If already a media playlist
                if any(line.startswith('#EXTINF:') for line in lines):
                    sub_url = master_url
                    r_sub = r
                else:
                    sub_url = None
                    for qual in ['1080p60', '1080p', 'source', 'chunked', '720p60', '720p', '480p']:
                        for line in lines:
                            if line.endswith('.m3u8') and qual in line.lower():
                                sub_url = line if line.startswith('http') else f"{base}/{line}"
                                break
                        if sub_url:
                            break

                    if not sub_url:
                        for line in lines:
                            if line.endswith('.m3u8') and not line.startswith('#'):
                                sub_url = line if line.startswith('http') else f"{base}/{line}"
                                break

                    if not sub_url:
                        return master_url, []

                    r_sub = s.get(sub_url, timeout=10)

                if r_sub.status_code == 200:
                    sub_base = sub_url.rsplit('/', 1)[0]
                    segments = []
                    current_time = 0.0
                    current_duration = 2.0

                    for s_line in r_sub.text.split('\n'):
                        s_line = s_line.strip()
                        if s_line.startswith('#EXTINF:'):
                            try:
                                dur_str = s_line.split(':')[1].split(',')[0]
                                current_duration = float(dur_str)
                            except Exception:
                                current_duration = 2.0
                        elif s_line and not s_line.startswith('#'):
                            seg_url = s_line if s_line.startswith('http') else f"{sub_base}/{s_line}"
                            segments.append({
                                'url': seg_url,
                                'start_time': current_time,
                                'duration': current_duration
                            })
                            current_time += current_duration
                    return sub_url, segments
        except Exception as e:
            print(f"Error resolving best video stream: {e}")
        return master_url, []


    def download_audio_for_analysis(
        self,
        source_url: str,
        duration: int = 0,
        start_sec: float = 0,
        end_sec: Optional[float] = None,
        callback: Optional[Callable[[str, int], None]] = None
    ) -> Tuple[str, float, List[Dict[str, Any]]]:
        self.log("Turbo analiz motoru hazırlanıyor...", 10, callback)

        target_start = max(0.0, float(start_sec))
        if end_sec and float(end_sec) > target_start:
            target_end = float(end_sec)
        elif duration > 0:
            target_end = max(target_start + 1800.0, float(duration))
        else:
            target_end = target_start + 28800.0

        if duration > 0 and target_start >= float(duration):
            self.log(f"ℹ️ Belirtilen aralık ({self.format_time(target_start)} - {self.format_time(target_end)}) taranıyor...", 14, callback)

        audio_output_path = os.path.join(self.temp_dir, f"audio_{int(time.time())}.wav")

        # 1. YouTube & Google Drive Ultra High-Speed Direct Audio Downloader via yt-dlp & FFmpeg
        is_yt = ("youtube.com" in source_url.lower() or "youtu.be" in source_url.lower())
        is_gdrive = ("drive.google.com" in source_url.lower() or "docs.google.com" in source_url.lower())
        if is_yt or is_gdrive:
            src_name = "Google Drive" if is_gdrive else "YouTube"
            self.log(f"{src_name} ses akışı çekiliyor ({self.format_time(target_start)} - {self.format_time(target_end)})...", 15, callback)
            import yt_dlp

            unique_audio_token = f"audio_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            target_duration = max(1.0, target_end - target_start) if target_end > target_start else None

            # ── 1.A. YOUTUBE LIGHTNING-FAST DIRECT STREAMING VIA FFMPEG (0.5s, no download blocks) ──
            if is_yt:
                try:
                    info_dict = None
                    direct_stream_opts = self.get_yt_dlp_opts({'quiet': True, 'extract_flat': False})
                    cookie_attempts = [True, False] if direct_stream_opts.get('cookiefile') else [False]
                    for use_cookie in cookie_attempts:
                        c_opts = dict(direct_stream_opts)
                        if not use_cookie:
                            c_opts.pop('cookiefile', None)
                        try:
                            with yt_dlp.YoutubeDL(c_opts) as ydl_stream:
                                info_dict = ydl_stream.extract_info(source_url, download=False)
                            if info_dict:
                                break
                        except Exception:
                            pass

                    if info_dict:
                        if 'entries' in info_dict and info_dict['entries']:
                            info_dict = info_dict['entries'][0]

                        formats = info_dict.get('formats') or []
                        audio_fmts = [f for f in formats if f.get('vcodec') == 'none' and f.get('acodec') != 'none' and f.get('url')]
                        if not audio_fmts:
                            audio_fmts = [f for f in formats if f.get('acodec') != 'none' and f.get('url')]

                        if audio_fmts:
                            # Prioritize formats with reliable streaming (m4a/opus)
                            def score_fmt(f):
                                abr = f.get('abr') or 0
                                ext_bonus = 25 if f.get('ext') == 'm4a' else 0
                                return abr + ext_bonus
                            audio_fmts.sort(key=score_fmt, reverse=True)
                            best_audio = audio_fmts[0]
                            stream_url = best_audio['url']

                            slice_dur = str(target_duration if target_duration else 28800.0)
                            ff_cmd = [
                                get_ffmpeg_binary(), '-y',
                                '-ss', str(target_start),
                                '-i', stream_url,
                                '-t', slice_dur,
                                '-vn', '-sn', '-dn',
                                '-acodec', 'pcm_s16le',
                                '-ar', '16000',
                                '-ac', '1',
                                audio_output_path
                            ]
                            http_headers = best_audio.get('http_headers') or info_dict.get('http_headers') or {}
                            if http_headers:
                                h_lines = "".join(f"{k}: {v}\r\n" for k, v in http_headers.items())
                                ff_cmd = [
                                    get_ffmpeg_binary(), '-y',
                                    '-headers', h_lines,
                                    '-ss', str(target_start),
                                    '-i', stream_url,
                                    '-t', slice_dur,
                                    '-vn', '-sn', '-dn',
                                    '-acodec', 'pcm_s16le',
                                    '-ar', '16000',
                                    '-ac', '1',
                                    audio_output_path
                                ]

                            res = subprocess.run(ff_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
                            if res.returncode == 0 and os.path.exists(audio_output_path) and os.path.getsize(audio_output_path) > 1000:
                                self.log(f"YouTube ses akışı başarıyla bağlandı ({self.format_time(target_start)} - {self.format_time(target_end)}) ✓", 45, callback)
                                return audio_output_path, target_start, []
                except Exception as stream_ex:
                    self.log(f"YouTube ses akışı alternatif motorla deneniyor...", 18, callback)

            # ── 1.B. YT-DLP RANGE / AUDIO DOWNLOADER (FALLBACK) ──
            temp_yt_pattern = os.path.join(self.temp_dir, f"{unique_audio_token}.%(ext)s")
            extra_dl = {}
            fmt_opt = 'ba[ext=m4a]/ba[abr<=128]/ba/b/best'
            # Ses dosyalari cok kucuk oldugu icin (~20MB) dogrudan indirme 3-4 saniyede biter.
            # download_ranges ise FFmpeg ile 2x hizda 15 dakika beklettigi icin sadece cok uzun yayinlarda (>4 saat) kullanilir.
            if target_end > target_start and (target_end - target_start) > 14400:
                extra_dl['download_ranges'] = yt_dlp.utils.download_range_func(None, [(target_start, target_end)])
                extra_dl['force_keyframes_at_cuts'] = True

            base_ydl_opts = self.get_yt_dlp_opts({
                'format': fmt_opt,
                'outtmpl': temp_yt_pattern,
                'nopart': True,
                'overwrites': True,
                'quiet': True,
                **extra_dl
            })

            ytdlp_ok = False
            last_err = ""
            for use_cookies in ([True, False] if base_ydl_opts.get('cookiefile') else [False]):
                ydl_opts = dict(base_ydl_opts)
                if not use_cookies:
                    ydl_opts.pop('cookiefile', None)
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([source_url])
                    ytdlp_ok = True
                    break
                except Exception as e:
                    last_err = re.sub(r'\x1b\[[0-9;]*m', '', str(e))
                    if "Sign in to confirm your age" in last_err or "confirm your age" in last_err:
                        raise ValueError("⚠️ Bu video +18 (Yaş Kısıtlamalı) olduğu için oturum açılması isteniyor!\n\n💡 Çözüm: Proje klasörünüze 'cookies.txt' dosyanızı ekleyerek kısıtlamayı kaldırabilirsiniz.")

            if not ytdlp_ok and is_gdrive:
                # yt-dlp Google Drive için başarısız → ffmpeg ile direkt indir
                import re as _re
                file_m_dl = _re.search(r'(?:file/d/|id=|open\?id=)([a-zA-Z0-9_-]{25,})', source_url)
                if file_m_dl:
                    v_id_dl = file_m_dl.group(1)
                    direct_dl_url = f"https://drive.google.com/uc?export=download&id={v_id_dl}&confirm=t"
                    ff_cmd = [
                        get_ffmpeg_binary(), '-y',
                        '-ss', str(target_start),
                        '-i', direct_dl_url,
                        '-t', str(max(1.0, target_end - target_start)),
                        '-vn', '-sn', '-dn',
                        '-acodec', 'pcm_s16le',
                        '-ar', '16000',
                        '-ac', '1',
                        audio_output_path
                    ]
                    result = subprocess.run(ff_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if result.returncode == 0 and os.path.exists(audio_output_path) and os.path.getsize(audio_output_path) > 1000:
                        return audio_output_path, target_start, []

            if not ytdlp_ok:
                raise ValueError(f"{src_name} ses indirme hatası: {last_err or 'Bilinmeyen bağlantı hatası'}")

            found = [os.path.join(self.temp_dir, f) for f in os.listdir(self.temp_dir) 
                     if f.startswith(unique_audio_token) 
                     and not f.endswith('.part') 
                     and not f.endswith('.ytdl')
                     and os.path.getsize(os.path.join(self.temp_dir, f)) > 1000]
            if not found:
                raise RuntimeError(f"{src_name} ses dosyası indirilemedi.")

            found.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            raw_audio_file = found[0]

            self.log("Ses frekansları çözülüyor (16 kHz Vokal Detayı)...", 45, callback)
            
            # If download_ranges was used, the file is already sliced, so slice_ss is 0.0
            range_used = 'download_ranges' in extra_dl
            slice_ss = 0.0 if (range_used or is_gdrive) else target_start
            duration_slice = max(1.0, target_end - target_start) if target_end > target_start else None
            conv_cmd = [
                get_ffmpeg_binary(), '-y',
                '-ss', str(slice_ss),
                '-i', raw_audio_file
            ]
            if duration_slice is not None:
                conv_cmd.extend(['-t', str(duration_slice)])
            conv_cmd.extend([
                '-vn', '-sn', '-dn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_output_path
            ])
            subprocess.run(conv_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            try: os.remove(raw_audio_file)
            except: pass

            return audio_output_path, target_start, []

        # 2. Kick HLS Stream or other sources
        audio_stream_url, segments = self.get_low_res_audio_stream_and_segments(source_url)

        if segments and len(segments) > 0:
            filtered_segments = []
            for s in segments:
                s_start = s['start_time']
                s_end = s_start + s['duration']
                if s_end >= target_start and s_start <= target_end:
                    filtered_segments.append(s)

            if not filtered_segments:
                filtered_segments = segments

            actual_offset = filtered_segments[0]['start_time']
            total_chunks = len(filtered_segments)
            time_window_min = round((filtered_segments[-1]['start_time'] + filtered_segments[-1]['duration'] - actual_offset) / 60, 1)

            self.log(f"Seçilen {time_window_min} dakikalık aralık taranıyor ({total_chunks} paket)...", 15, callback)

            # Download all chunks contiguously to maintain 100% time accuracy
            sampled = list(enumerate(filtered_segments))
            sample_total = len(sampled)

            results = {}
            completed_count = 0

            def fetch_single_chunk(item):
                idx, seg = item
                s = get_cdn_session()
                for attempt in range(4):
                    try:
                        res = s.get(seg['url'], timeout=(4, 8))
                        if res.status_code == 200 and len(res.content) > 0:
                            return idx, res.content
                    except Exception:
                        time.sleep(0.15 * (attempt + 1))
                return idx, b''

            with ThreadPoolExecutor(max_workers=40) as executor:
                future_to_idx = {executor.submit(fetch_single_chunk, item): item[0] for item in sampled}
                for future in as_completed(future_to_idx):
                    idx, content = future.result()
                    results[idx] = content
                    completed_count += 1

                    if completed_count % 20 == 0 or completed_count == sample_total:
                        fraction = completed_count / sample_total
                        cur_prog = int(15 + fraction * 30)
                        self.log(f"Ses indiriliyor (%{int(fraction*100)} - {completed_count}/{sample_total})...", cur_prog, callback)

            combined_ts = os.path.join(self.temp_dir, f"temp_{int(time.time())}.ts")
            with open(combined_ts, "wb") as f:
                for idx in sorted(results.keys()):
                    if results[idx]:
                        f.write(results[idx])

            self.log("Ses frekansları çözülüyor (16 kHz Vokal Detayı)...", 46, callback)
            cmd = [
                get_ffmpeg_binary(), '-y',
                '-i', combined_ts,
                '-vn', '-sn', '-dn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_output_path
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            if os.path.exists(combined_ts):
                try: os.remove(combined_ts)
                except Exception: pass

            return audio_output_path, actual_offset, filtered_segments

        else:
            actual_offset = target_start
            self.log(f"Ses akışı çekiliyor ({self.format_time(target_start)} - {self.format_time(target_end)})...", 20, callback)
            
            reconnect_opts = []
            if audio_stream_url.startswith('http'):
                reconnect_opts = ['-reconnect', '1', '-reconnect_streamed', '1', '-reconnect_delay_max', '5']

            cmd = [
                get_ffmpeg_binary(), '-y',
                *reconnect_opts,
                '-ss', str(target_start),
                '-i', audio_stream_url,
                '-t', str(target_end - target_start),
                '-vn', '-sn', '-dn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_output_path
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300, check=True)

            if not os.path.exists(audio_output_path) or os.path.getsize(audio_output_path) == 0:
                raise RuntimeError("Ses dosyası oluşturulamadı.")

            return audio_output_path, actual_offset, []

    def clean_stream_title_text(self, raw_title: str) -> str:
        """
        Extracts clean topic/game from raw stream title by removing commands and spam.
        """
        if not raw_title:
            return ""
        t = raw_title
        # Remove commands like !dc, !kick, !discord, !insta, !prime
        t = re.sub(r'![a-zA-Z0-9_-]+', '', t)
        # Remove URLs
        t = re.sub(r'https?://\S+', '', t)
        # Remove pipe, slash, brackets junk
        t = re.sub(r'[\[\]\(\)\|\/\\#_~]+', ' ', t)
        # Remove common fluff keywords
        fluff_pattern = r'\b(1080p|720p|60fps|yayındayız|yayındayım|canlı|live|yeni|dc|kick|instagram|discord)\b'
        t = re.sub(fluff_pattern, '', t, flags=re.IGNORECASE)
        # Remove excess whitespace
        t = re.sub(r'\s+', ' ', t).strip()
        if len(t) > 35:
            t = t[:32].rsplit(' ', 1)[0]
        return t.title() if t else ""

    def generate_viral_titles(self, streamer: str, stream_title: str, clips: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates clean, natural, engaging titles and hashtags tailored to each clip moment.
        """
        clean_name = (streamer or "Kick").replace('@', '').strip()
        clean_name = clean_name.capitalize() if clean_name else "Yayıncı"
        topic = self.clean_stream_title_text(stream_title)

        try:
            from ai_headline_service import generate_clip_titles_and_hooks
        except Exception:
            generate_clip_titles_and_hooks = None

        try:
            from ai_clipper_engine import turkish_upper, is_meaningful_clause, clean_profanity_and_fillers
        except Exception:
            turkish_upper = lambda s: s.upper()
            is_meaningful_clause = lambda s: len((s or "").split()) >= 2
            clean_profanity_and_fillers = lambda s: s or ""

        updated_clips = []
        for i, clip in enumerate(clips, 1):
            m_type = clip.get("moment_type", "funny")
            raw_hook = (clip.get("hook_text") or clip.get("hook_3s_quote") or "").strip().strip('"').strip("'")
            hook_txt = clean_profanity_and_fillers(raw_hook)
            payoff_txt = clean_profanity_and_fillers((clip.get("payoff_text") or clip.get("payoff") or "").strip())
            full_txt = (clip.get("transcript_text") or "").strip()

            title_options = clip.get("title_options") or []
            chosen_title = clip.get("viral_title") or ""
            capcut_headline = clip.get("capcut_hook_headline") or ""

            # Check if current titles are generic templates
            canned_phrases = [
                "KİMSE FARK ETMEDİ AMA... ⚡", 
                "BURADA GÜLME KRİZİ KOPTU 🤣", 
                "İNANILMAZ BİR OLAY YAŞANDI 😱", 
                "REKOR KIRAN O AN! 🔥",
                "BUNU KİMSE BEKLEMİYORDU! 😱",
                "YAYININ EN İYİ ANI 🔥",
                "YAYININ EN İYİ ANI! 🔥"
            ]
            is_generic = (not chosen_title or any(cp in chosen_title for cp in canned_phrases) or "Komik An #" in chosen_title or "Yayının En İyi Anı" in chosen_title)

            # Generate smart AI titles if empty or generic
            if (not title_options or is_generic) and generate_clip_titles_and_hooks:
                try:
                    service_res = generate_clip_titles_and_hooks(
                        streamer=clean_name,
                        stream_title=topic,
                        transcript=full_txt,
                        hook_text=hook_txt,
                        payoff_text=payoff_txt,
                        moment_type=m_type,
                        duration=clip.get("duration", 35.0)
                    )
                    if service_res and service_res.get("title_options"):
                        title_options = service_res["title_options"]
                        chosen_title = service_res.get("viral_title", title_options[0])
                        capcut_headline = service_res.get("capcut_hook_headline", capcut_headline or chosen_title)
                except Exception as ex:
                    print(f"generate_clip_titles_and_hooks error: {ex}")

            if not title_options:
                title_options = [
                    f'{clean_name} Canlı Yayında Şaşırttı! 🔥',
                    f'Bunu Kimse Beklemiyordu! 🤣 ({clean_name})',
                    f'{clean_name} - Yayından Çok Özel Anlar ✨'
                ]

            if not chosen_title:
                chosen_title = title_options[0]

            clean_tag = re.sub(r'[^a-zA-Z0-9_]', '', clean_name.lower())
            tags = f"#{clean_tag} #kick #kickclips #shorts #kesfet #fyp #trend"
            tiktok_caption = f"{chosen_title}\n\n👉 Takip etmeyi unutmayın! 🔥\n\n{tags}"
            shorts_caption = f"{chosen_title}\n\n{tags}"

            if not capcut_headline:
                capcut_headline = chosen_title

            clean_3s_quote = clip.get("hook_3s_quote", "")
            if not clean_3s_quote or not is_meaningful_clause(clean_3s_quote):
                clean_3s_quote = hook_txt[:45] if is_meaningful_clause(hook_txt) else chosen_title

            updated_clips.append({
                **clip,
                "viral_title": chosen_title,
                "title_options": title_options,
                "tags": tags,
                "tiktok_caption": tiktok_caption,
                "shorts_caption": shorts_caption,
                "capcut_hook_headline": capcut_headline,
                "hook_3s_quote": clean_3s_quote,
                "hook_category": clip.get("hook_category", "🎯 Merak Boşluğu"),
                "hook_retention_score": clip.get("hook_retention_score", 88)
            })

        return updated_clips

    def calculate_smart_context_window(
        self,
        peak_idx: int,
        times: np.ndarray,
        vocal_smooth: np.ndarray,
        clip_duration: float,
        moment_type: str = "funny"
    ) -> float:
        """
        Calculates dynamic hook lead-in and narrative setup window before the peak moment.
        Viral Short Narrative Pacing:
        - Climax (punchline, scream, laugh, clutch) is placed at ~68% into the clip.
        - Provides ~18-26s of narrative setup / storytelling BEFORE the peak.
        - Provides ~7-12s of reaction and payoff AFTER the peak.
        """
        peak_time = float(times[peak_idx])

        # Climax golden ratio: ~68% through the clip duration
        ideal_lead = float(np.clip(clip_duration * 0.68, 16.0, clip_duration - 6.0))
        min_lead = float(np.clip(clip_duration * 0.45, 12.0, clip_duration - 8.0))
        max_lead = float(np.clip(clip_duration * 0.82, 20.0, clip_duration - 4.0))
        lookback_sec = max_lead + 10.0

        start_search_time = max(0.0, peak_time - lookback_sec)
        mask = (times >= start_search_time) & (times <= peak_time)
        indices = np.where(mask)[0]

        if len(indices) < 5:
            return ideal_lead

        target_lead_time = peak_time - ideal_lead

        # Look for the nearest energy minimum (silence/breath pause between thoughts) in the valid lead-in range
        valid_range_mask = (times[indices] >= (peak_time - max_lead)) & (times[indices] <= (peak_time - min_lead))
        valid_indices = indices[valid_range_mask]

        if len(valid_indices) > 0:
            valid_energies = vocal_smooth[valid_indices]
            min_thresh = np.percentile(valid_energies, 30)
            pause_candidates = valid_indices[valid_energies <= min_thresh]
            if len(pause_candidates) > 0:
                best_pause_idx = pause_candidates[np.argmin(np.abs(times[pause_candidates] - target_lead_time))]
                smart_lead = peak_time - float(times[best_pause_idx])
                return float(np.clip(smart_lead, min_lead, max_lead))

        return ideal_lead

    def check_is_gambling_moment(
        self,
        source_url: str,
        peak_timestamp: float,
        stream_title: str = "",
        segments: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[bool, str]:
        """
        Universal Anti-Gambling Vision & Metadata Shield:
        Detects and blocks ALL active gambling, casino, slot, betting, and stake game types:
        1. Pragmatic Play, Hacksaw Gaming, NoLimit City, Relax, Push, Play'n GO slots.
        2. Stake Originals (Plinko, Mines, Crash, Limbo, Dice, Keno, HiLo).
        3. Live Casino & Tables (Roulette, Blackjack, Baccarat, Poker, Craps).
        4. Live Game Shows (Crazy Time, Monopoly Live, Candyland, Funky Time, Mega Wheel).
        5. Frame-level multi-category visual classifier (felt tables, reel matrix, candy neon spectrum, Stake balance UI, bottom multiplier rows).
        """
        gambling_keywords = [
            # Turkish & General Casino / Betting
            "slot", "slotlar", "kumar", "kasino", "casino", "bahis", "bet", "betting", 
            "spin", "freespin", "bonus buy", "jackpot", "max win", "tumble", "scatter", 
            "wild", "wager", "re-spin", "x5000", "x1000", "x10000", "x500", "vurgun",
            # Pragmatic Play
            "bonanza", "sweet bonanza", "sugar rush", "gates of olympus", "starlight princess", 
            "big bass", "fruit party", "dog house", "madame destiny", "buffalo king", "zeus", 
            "dede", "olympus 1000", "bonanza 1000", "sugar rush 1000", "starlight 1000", 
            "gems bonanza", "juicy fruits", "wild west gold", "cleocatra", "pragmatic",
            # Hacksaw Gaming
            "wanted dead", "rip city", "le bandit", "chaos crew", "dork unit", 
            "gladiator legends", "hand of anubis", "rotten", "2 wild 2 die", "beam boys", 
            "stack em", "hacksaw",
            # NoLimit City
            "san quentin", "tombstone", "tombstone rip", "mental", "fire in the hole", 
            "deadwood", "folsom prison", "das xboot", "serial", "disturbed", "the crypt", "nolimit",
            # Relax / Push / Play'n GO
            "money train", "razor shark", "jammin jars", "book of dead", "reactoonz", 
            "moon princess", "rise of olympus", "fire joker", "tome of madness", 
            "relax gaming", "push gaming",
            # Stake Originals
            "stake", "plinko", "mines", "limbo", "crash", "dice", "keno", "hilo", "stake.com",
            # Live Casino & Tables
            "rulet", "roulette", "blackjack", "baccarat", "poker", "craps", "lightning roulette",
            # Live Game Shows
            "crazy time", "monopoly live", "funky time", "dream catcher", "mega wheel", 
            "candyland", "crazy coin flip", "lightning storm", "evolution live"
        ]

        if stream_title:
            t_lower = stream_title.lower()
            for kw in gambling_keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', t_lower) or kw in [
                    "stake", "bonanza", "olympus", "pragmatic", "rulet", "slot", 
                    "kasino", "plinko", "mines", "limbo", "hacksaw", "nolimit", "dede"
                ]:
                    if kw in t_lower:
                        return True, f"Yayın Başlığı Kumar/Slot İçeriği ('{kw}')"

        # Frame extraction from exact TS chunk
        img = None
        if segments and len(segments) > 0:
            target_seg = None
            for s in segments:
                s_start = s['start_time']
                s_end = s_start + s['duration']
                if s_start <= peak_timestamp <= s_end:
                    target_seg = s
                    break
            if not target_seg:
                target_seg = min(segments, key=lambda s: abs(s['start_time'] - peak_timestamp))

            if target_seg:
                try:
                    s_client = get_cdn_session()
                    res = s_client.get(target_seg['url'], timeout=5)
                    if res.status_code == 200 and len(res.content) > 1000:
                        ffmpeg_bin = get_ffmpeg_binary()
                        cmd = [
                            ffmpeg_bin,
                            '-y',
                            '-i', 'pipe:0',
                            '-vframes', '1',
                            '-vf', 'scale=320:180',
                            '-f', 'image2pipe',
                            '-vcodec', 'mjpeg',
                            '-'
                        ]
                        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        out, _ = proc.communicate(input=res.content, timeout=4.0)
                        if out and len(out) > 500:
                            import io
                            img = Image.open(io.BytesIO(out))
                except Exception:
                    pass

        # Fallback to direct ffmpeg if segment not available
        if img is None and source_url and source_url.startswith("http"):
            try:
                ffmpeg_bin = get_ffmpeg_binary()
                cmd = [
                    ffmpeg_bin,
                    '-ss', str(max(0.0, float(peak_timestamp))),
                    '-i', source_url,
                    '-vframes', '1',
                    '-vf', 'scale=320:180',
                    '-f', 'image2pipe',
                    '-vcodec', 'mjpeg',
                    '-'
                ]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out, _ = proc.communicate(timeout=6.0)
                if out and len(out) > 500:
                    import io
                    img = Image.open(io.BytesIO(out))
            except Exception:
                pass

        if img is not None:
            try:
                img_rgb = img.convert('RGB').resize((320, 180))
                img_hsv = img.convert('HSV').resize((320, 180))
                rgb_arr = np.array(img_rgb)
                hsv_arr = np.array(img_hsv)

                h_arr = hsv_arr[:, :, 0]
                s_arr = hsv_arr[:, :, 1] / 255.0
                v_arr = hsv_arr[:, :, 2] / 255.0

                center_s = s_arr[20:160, 50:270]
                center_v = v_arr[20:160, 50:270]
                center_h = h_arr[20:160, 50:270]

                h_std = float(np.std(center_h))
                s_std = float(np.std(center_s))
                s_mean = float(np.mean(center_s))

                # 1. Live Casino Felt Table Detection (Green, Blue, Burgundy/Red Felts)
                green_felt = (center_h >= 55) & (center_h <= 110) & (center_s >= 0.25) & (center_v >= 0.15) & (center_v <= 0.85)
                green_ratio = float(np.mean(green_felt))

                blue_felt = (center_h >= 140) & (center_h <= 180) & (center_s >= 0.35) & (center_v >= 0.15) & (center_v <= 0.85)
                blue_felt_ratio = float(np.mean(blue_felt))

                red_felt = ((center_h <= 12) | (center_h >= 240)) & (center_s >= 0.40) & (center_v >= 0.20) & (center_v <= 0.80)
                red_felt_ratio = float(np.mean(red_felt))

                # 2. Vivid Neon Candy/Gem Colors (Sugar Rush, Starlight, Sweet Bonanza, Olympus, Candyland)
                vivid_candy = ((center_h > 195) | ((center_h >= 120) & (center_h <= 165))) & (center_s > 0.42) & (center_v > 0.38)
                candy_ratio = float(np.mean(vivid_candy))

                # 3. Bottom Multiplier / Reel Footer (Plinko rows, Game Show wheel pins, Slot win bar)
                bottom_s = s_arr[140:175, 40:280]
                bottom_h = h_arr[140:175, 40:280]
                bottom_rainbow = (float(np.std(bottom_h)) > 50.0) and (float(np.mean(bottom_s)) > 0.45)

                # 4. Stake / Casino Top Bar (Dark header with high-contrast balance pill)
                top_v = v_arr[0:25, 0:320]
                top_dark_ratio = float(np.mean(top_v < 0.25))
                top_bright_ratio = float(np.mean(top_v > 0.85))
                has_casino_header = (top_dark_ratio > 0.65) and (top_bright_ratio > 0.05) and (s_mean > 0.35)

                if green_ratio > 0.22 or blue_felt_ratio > 0.25 or red_felt_ratio > 0.28:
                    return True, "Rulet/Blackjack/Masa Kumarı Görseli"
                elif s_mean > 0.40 and h_std > 38 and s_std > 0.16:
                    return True, "Slot/Pragmatic/Hacksaw/NoLimit Oyun Grid Ekranı"
                elif candy_ratio > 0.12 and s_mean > 0.38:
                    return True, "Renkli Şeker/Mücevher Slotu (Bonanza/Sugar Rush/Starlight)"
                elif bottom_rainbow and s_mean > 0.35:
                    return True, "Plinko / Canlı Çark / Game Show Çarpanları"
                elif has_casino_header and h_std > 35:
                    return True, "Stake / Online Casino Arayüzü"

            except Exception:
                pass

        return False, ""

    def detect_highlights(
        self,
        audio_path: str,
        time_offset: float = 0.0,
        num_clips: int = 10,
        clip_duration: int = 35,
        min_distance_sec: int = 60,
        distribution_mode: str = "smart_hybrid",
        quality_sensitivity: str = "balanced",
        filter_gambling: bool = True,
        source_url: Optional[str] = None,
        stream_title: str = "",
        streamer: str = "",
        segments: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[Callable[[str, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        AI Audio Intelligence Highlight Engine:
        - Multi-band voice filtering (extracts human speech formants, isolates screams, laughs, and speech bursts).
        - Laughter & Comedy Detection (3-6 Hz envelope rhythmic modulation).
        - Hype & Drama Shock Detection (steep onset surges, emotional excitement).
        - Quality Gate (strictly eliminates dead-air, quiet pauses, and monotonic background hum).
        - Smart Setup-to-Payoff Context Windowing.
        - Anti-Gambling Vision & Metadata Filter (strictly purges slot/casino/roulette scenes).
        """
        self.log(f"AI Clipper Engine başlatılıyor (10 Aşamalı Viral Tespit - {num_clips} klip)...", 50, callback)

        sample_rate, audio_data = wavfile.read(audio_path)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0

        total_samples = len(audio_data)
        total_duration = total_samples / sample_rate

        if total_duration < 10:
            raise ValueError("Ses süresi analiz için yetersiz.")

        # ---------------------------------------------------------------------
        # 10-PASS NEXT-GEN AI CLIPPER ENGINE (Klap/Vizard/Captions grade)
        # ---------------------------------------------------------------------
        try:
            from ai_clipper_engine import AIClipperEngine
            engine = AIClipperEngine(temp_dir=self.temp_dir)
            ai_clips = engine.run_ai_clipper_pipeline(
                audio_path=audio_path,
                title=stream_title,
                streamer=streamer,
                duration=total_duration,
                time_offset=time_offset,
                num_clips=num_clips,
                clip_duration=clip_duration,
                min_distance_sec=min_distance_sec,
                callback=callback
            )

            # Anti-gambling filter if enabled
            if filter_gambling and source_url and ai_clips:
                filtered_ai = []
                for c in ai_clips:
                    p_time = c.get('peak_time', c.get('start_time', 0))
                    is_gamb, gamb_reason = self.check_is_gambling_moment(source_url, p_time, stream_title=stream_title, segments=segments)
                    if not is_gamb:
                        filtered_ai.append(c)
                    else:
                        self.log(f"🚫 Kumar/Slot sahnesi elendi ({self.format_time(p_time)} - {gamb_reason})", 62, callback)
                ai_clips = filtered_ai

            if ai_clips and len(ai_clips) > 0:
                for i, c in enumerate(ai_clips, 1):
                    c['id'] = i
                self.log(f"✨ AI Clipper Engine: {len(ai_clips)} adet en yüksek puanlı viral kesit belirlendi!", 63, callback)
                return ai_clips
        except Exception as e:
            self.log(f"AI Clipper Engine uyarısı: {e}, akustik motora dönülüyor...", 51, callback)
            print(f"AIClipperEngine fallback: {e}")

        self.log("İnsan sesi frekansları ve kahkaha modülasyonları ayrıştırılıyor...", 53, callback)

        # 1. Bandpass filters for human speech & harmonics
        nyquist = sample_rate / 2.0
        vocal_high = min(3600.0, nyquist - 50.0)
        sos_vocal = butter(2, [250.0, vocal_high], btype='bandpass', fs=sample_rate, output='sos')
        vocal_signal = sosfilt(sos_vocal, audio_data)

        # High vocal formant band (1200Hz - 3600Hz) for screams, shrieks, laughs
        sos_high = butter(2, [1200.0, vocal_high], btype='bandpass', fs=sample_rate, output='sos')
        high_signal = sosfilt(sos_high, audio_data)

        # Low bass rumble (50Hz - 220Hz) for game bass, motor noise, subwoofer
        sos_bass = butter(2, [50.0, min(220.0, nyquist - 50.0)], btype='bandpass', fs=sample_rate, output='sos')
        bass_signal = sosfilt(sos_bass, audio_data)

        # 2. Fast O(N) sliding frame RMS
        win_size = int(sample_rate * 0.40) # 400ms window
        hop_size = int(sample_rate * 0.15) # 150ms step

        def fast_rms(sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            sq = sig ** 2
            cumsum = np.pad(np.cumsum(sq, dtype=np.float64), (1, 0), 'constant')
            starts = np.arange(0, len(sig) - win_size + 1, hop_size)
            ends = starts + win_size
            return starts, np.sqrt(np.maximum((cumsum[ends] - cumsum[starts]) / win_size, 1e-12))

        starts, vocal_rms = fast_rms(vocal_signal)
        _, high_rms = fast_rms(high_signal)
        _, bass_rms = fast_rms(bass_signal)
        times = (starts + win_size / 2.0) / sample_rate

        # 3. Moving baseline and dynamic contrast (20-second rolling window)
        n_roll = int(20.0 / 0.15)
        vocal_smooth = uniform_filter1d(vocal_rms, size=5)
        local_base = uniform_filter1d(minimum_filter1d(vocal_smooth, size=n_roll), size=7) + 1e-5
        contrast = vocal_smooth / local_base

        # 4. Laughter rhythmic modulation index (2.5s window ~ 16 frames)
        mod_win = int(2.5 / 0.15)
        local_std = np.sqrt(np.maximum(0, uniform_filter1d(vocal_smooth**2, size=mod_win) - uniform_filter1d(vocal_smooth, size=mod_win)**2))
        mod_cv = local_std / (uniform_filter1d(vocal_smooth, size=mod_win) + 1e-5)
        high_ratio = high_rms / (vocal_rms + 1e-4)

        # 5. Multi-factor highlight indicators with voice prominence weighting
        laughter_raw = vocal_smooth * (1.0 + 3.0 * np.clip(mod_cv, 0, 3.5)) * (0.7 + 1.5 * high_ratio)
        hype_raw = vocal_smooth * np.clip(contrast, 1.0, 6.0) * (0.7 + 1.8 * high_ratio)

        # 6. Strict Quality Gate & Anti-Dead-Air Filter
        sens_factor = 1.35 if quality_sensitivity == "strict_quality" else (0.80 if quality_sensitivity == "broad" else 1.05)
        q_base = np.percentile(vocal_smooth, 35)
        q_std = np.std(vocal_smooth) + 1e-6
        vocal_dominance = vocal_rms / (bass_rms + 1e-4)

        # Persistence check over 1.2s to filter out accidental 0.1s mic bumps or single mouse clicks
        win_persist = max(3, int(1.2 / 0.15))
        energy_persistence = uniform_filter1d(vocal_smooth, size=win_persist)
        sustained_speech = energy_persistence > (q_base + 0.18 * q_std)

        valid_mask = (
            (vocal_smooth > (q_base + 0.28 * sens_factor * q_std)) &
            (contrast > (1.20 * sens_factor)) &
            (vocal_dominance > 0.35) &
            sustained_speech
        )

        # Normalization across valid segments
        l_max = np.percentile(laughter_raw[valid_mask], 99.5) if np.any(valid_mask) else 1.0
        h_max = np.percentile(hype_raw[valid_mask], 99.5) if np.any(valid_mask) else 1.0

        l_norm = np.clip(laughter_raw / (l_max + 1e-6), 0.0, 1.5)
        h_norm = np.clip(hype_raw / (h_max + 1e-6), 0.0, 1.5)

        # Composite score calculation
        if distribution_mode == "funny_only":
            composite_score = np.where(valid_mask, l_norm, 0.0)
        elif distribution_mode == "hype_only":
            composite_score = np.where(valid_mask, h_norm, 0.0)
        else: # smart_hybrid, distributed
            composite_score = np.where(valid_mask, 0.58 * l_norm + 0.42 * h_norm, 0.0)

        # 7. Local peak identification (minimum quality threshold)
        min_peak_thresh = 0.20 * sens_factor
        peak_indices = []
        for i in range(1, len(composite_score) - 1):
            if composite_score[i] > composite_score[i - 1] and composite_score[i] > composite_score[i + 1] and composite_score[i] >= min_peak_thresh:
                peak_indices.append(i)

        if not peak_indices:
            peak_indices = [np.argmax(composite_score)] if len(composite_score) > 0 else [0]

        sorted_peaks = sorted(peak_indices, key=lambda idx: composite_score[idx], reverse=True)

        selected_clips = []
        used_ranges = []

        def build_clip_obj(idx: int) -> Dict[str, Any]:
            p_time = float(times[idx])
            l_val = float(l_norm[idx]) * 100.0
            h_val = float(h_norm[idx]) * 100.0
            mod_val = float(mod_cv[idx])
            c_val = float(contrast[idx])
            hr_val = float(high_ratio[idx])

            if mod_val > 0.35 or (l_val > h_val * 0.85 and l_val >= 45):
                m_type = "funny"
                m_icon = "😂"
                m_label = "😂 Komik An (Kahkaha & Şaka)"
            elif c_val >= 2.6 and hr_val >= 0.60:
                m_type = "shock"
                m_icon = "😱"
                m_label = "😱 Şok / Çığlık (Jumpscare)"
            elif h_val >= 65:
                m_type = "hype"
                m_icon = "🔥"
                m_label = "🔥 Önemli An (Büyük Hype & Heyecan)"
            elif c_val >= 1.8:
                m_type = "action"
                m_icon = "🎯"
                m_label = "🎯 Kritik An (Aksiyon & Clutch)"
            else:
                m_type = "chat"
                m_icon = "💬"
                m_label = "💬 Bomba Muhabbet / İtiraf"

            lead_in = self.calculate_smart_context_window(idx, times, vocal_smooth, clip_duration, m_type)
            start_rel = max(0.0, min(total_duration - clip_duration, p_time - lead_in))
            end_rel = min(total_duration, start_rel + clip_duration)

            abs_start = time_offset + start_rel
            abs_end = time_offset + end_rel
            abs_peak = time_offset + p_time

            raw_sc = float(composite_score[idx])
            hype_pct = int(min(100, max(75, 75 + raw_sc * 25)))

            return {
                "start_time_rel": start_rel,
                "end_time_rel": end_rel,
                "peak_time": round(abs_peak, 2),
                "start_time": round(abs_start, 2),
                "end_time": round(abs_end, 2),
                "duration": round(abs_end - abs_start, 2),
                "start_formatted": self.format_time(abs_start),
                "end_formatted": self.format_time(abs_end),
                "peak_formatted": self.format_time(abs_peak),
                "moment_type": m_type,
                "moment_icon": m_icon,
                "moment_label": m_label,
                "laughter_score": int(min(100, l_val)),
                "hype_score": hype_pct,
                "lead_in_sec": round(lead_in, 1),
                "raw_score": round(raw_sc, 2)
            }

        # Slicing for distributed mode (only pick if candidate exceeds minimum quality)
        if distribution_mode == "distributed" and total_duration >= 60 and num_clips >= 2:
            slice_width = total_duration / float(num_clips)
            for slice_idx in range(num_clips):
                slice_start = slice_idx * slice_width
                slice_end = (slice_idx + 1) * slice_width

                slice_peaks = [idx for idx in sorted_peaks if slice_start <= times[idx] < slice_end and valid_mask[idx] and composite_score[idx] >= min_peak_thresh]

                for best_idx in slice_peaks:
                    p_time_abs = time_offset + float(times[best_idx])
                    if filter_gambling and source_url:
                        is_gamb, gamb_reason = self.check_is_gambling_moment(source_url, p_time_abs, stream_title=stream_title, segments=segments)
                        if is_gamb:
                            self.log(f"🚫 Kumar/Slot sahnesi elendi ({self.format_time(p_time_abs)} - {gamb_reason})", 57, callback)
                            continue

                    c_obj = build_clip_obj(best_idx)
                    s_rel, e_rel = c_obj["start_time_rel"], c_obj["end_time_rel"]

                    overlap = False
                    for u_s, u_e in used_ranges:
                        if not (e_rel + 20 <= u_s or s_rel - 20 >= u_e):
                            overlap = True
                            break
                    if not overlap:
                        used_ranges.append((s_rel, e_rel))
                        selected_clips.append(c_obj)
                        break

        # Fill remaining slots from global top ranking (strictly quality ordered)
        effective_dist = max(clip_duration + 20, min_distance_sec)
        for idx in sorted_peaks:
            if len(selected_clips) >= num_clips:
                break

            p_time_abs = time_offset + float(times[idx])
            if filter_gambling and source_url:
                is_gamb, gamb_reason = self.check_is_gambling_moment(source_url, p_time_abs, stream_title=stream_title, segments=segments)
                if is_gamb:
                    self.log(f"🚫 Kumar/Slot sahnesi elendi ({self.format_time(p_time_abs)} - {gamb_reason})", 58, callback)
                    continue

            c_obj = build_clip_obj(idx)
            s_rel, e_rel = c_obj["start_time_rel"], c_obj["end_time_rel"]

            is_overlapping = False
            for u_s, u_e in used_ranges:
                if not (e_rel + effective_dist <= u_s or s_rel - effective_dist >= u_e):
                    is_overlapping = True
            if not is_overlapping:
                used_ranges.append((s_rel, e_rel))
                selected_clips.append(c_obj)

        # Vizard.ai style sorting: Rank clips by Virality & Hype Score (Highest score first!)
        selected_clips = sorted(selected_clips, key=lambda c: (c.get('hype_score', 0), c.get('raw_score', 0)), reverse=True)
        for i, c in enumerate(selected_clips, 1):
            c['id'] = i

        funny_count = sum(1 for c in selected_clips if c.get("moment_type") == "funny")
        hype_count = sum(1 for c in selected_clips if c.get("moment_type") in ["hype", "shock", "action"])
        self.log(f"Vizard.ai AI Analizi Tamamlandı! {len(selected_clips)} adet yüksek viral skorlu kesit ({funny_count} Komik / {hype_count} Hype & Zirve an)", 60, callback)
        return selected_clips

    def convert_to_vertical_916(
        self,
        input_filepath: str,
        output_filepath: Optional[str] = None,
        normalize_audio: bool = True,
        enable_teaser_hook: bool = False,
        peak_time_rel: Optional[float] = None,
        clip_duration: Optional[float] = None,
        enable_hook_banner: bool = False,
        hook_text: Optional[str] = None,
        shorts_layout: str = "blur",
        cam_position: str = "top_right",
        enable_chat_overlay: bool = False,
        moment_type: str = "funny",
        video_quality: str = "1080p60",
        enable_sfx: bool = True
    ) -> str:
        """
        Ultra-Fast FFmpeg Rendering:
        - Cold Open Teaser (Climax preview at 0.0s)
        - Dynamic Hook Text Banner ("⚡ AZ SONRA... 🔥")
        - Cinematic Whoosh SFX at flash cut transition
        - Single-pass high-throughput NVENC hardware acceleration
        - Ultra-fast A/V sync & peak limiter
        """
        if not output_filepath:
            base, ext = os.path.splitext(input_filepath)
            output_filepath = f"{base}_916.mp4"

        # Determine if Cold Open Teaser Hook is feasible
        has_teaser = False
        t_start = 0.0
        t_end = 2.2
        t_dur = 2.2
        sfx_input_arg = []
        if enable_teaser_hook:
            c_dur = float(clip_duration) if clip_duration and clip_duration > 0 else 30.0
            if peak_time_rel is None or peak_time_rel < 3.0 or peak_time_rel > (c_dur - 2.0):
                peak_time_rel = min(c_dur - 2.5, max(3.5, c_dur * 0.55))
            t_start = max(0.0, peak_time_rel - 1.1)
            t_end = min(c_dur, t_start + 2.2)
            t_dur = round(t_end - t_start, 2)
            if t_dur >= 1.4:
                has_teaser = True

        filter_parts = []

        # 1. Video & Audio sources (Cold Open with white flash transition + SFX)
        if has_teaser:
            sfx_path = self.ensure_whoosh_sfx()
            sfx_available = enable_sfx and os.path.exists(sfx_path)
            if sfx_available:
                sfx_input_arg = ['-i', sfx_path]

            filter_parts.append(
                f"[0:v]trim=start={t_start:.2f}:end={t_end:.2f},setpts=PTS-STARTPTS,"
                f"fade=t=out:st={t_dur - 0.20:.2f}:d=0.20:color=white[tv_flash];"
                f"[0:a]atrim=start={t_start:.2f}:end={t_end:.2f},asetpts=PTS-STARTPTS,"
                f"afade=t=in:ss=0:d=0.08,afade=t=out:st={t_dur - 0.12:.2f}:d=0.12[ta_fade];"
                f"[0:v]setpts=PTS-STARTPTS[mv];"
                f"[0:a]asetpts=PTS-STARTPTS[ma];"
                f"[tv_flash][ta_fade][mv][ma]concat=n=2:v=1:a=1[cv][ca_concat]"
            )
            if sfx_available:
                sfx_delay_ms = max(0, int((t_dur - 0.25) * 1000))
                filter_parts.append(
                    f"[1:a]adelay={sfx_delay_ms}|{sfx_delay_ms}[sfx];"
                    f"[ca_concat][sfx]amix=inputs=2:duration=first:dropout_transition=0[ca]"
                )
            else:
                filter_parts.append("[ca_concat]acopy[ca]")

            v_in = "[cv]"
            a_in = "[ca]"
        else:
            v_in = "[0:v]"
            a_in = "[0:a]"

        # 2. 9:16 Visual Layout (Split Screen, Fit/Fill Crop, or Blur Background)
        target_w = 1440 if video_quality == "2k_60fps" else 1080
        target_h = 2560 if video_quality == "2k_60fps" else 1920

        if shorts_layout in ["split_screen", "split"]:
            # Split Screen (Top: Streamer/Webcam, Bottom: Gameplay)
            cam_h = int(target_h * 0.44) # 844px (1080p) or 1126px (2K)
            game_h = target_h - cam_h

            if cam_position == "top_left":
                cam_crop = "crop=w=iw*0.35:h=ih*0.48:x=0:y=0"
            elif cam_position == "bottom_right":
                cam_crop = "crop=w=iw*0.35:h=ih*0.48:x=iw*0.65:y=ih*0.52"
            elif cam_position == "bottom_left":
                cam_crop = "crop=w=iw*0.35:h=ih*0.48:x=0:y=ih*0.52"
            elif cam_position == "center":
                cam_crop = "crop=w=iw*0.50:h=ih*0.60:x=iw*0.25:y=ih*0.20"
            else: # default top_right
                cam_crop = "crop=w=iw*0.35:h=ih*0.48:x=iw*0.65:y=0"

            filter_parts.append(
                f"{v_in}{cam_crop},scale={target_w}:{cam_h}:force_original_aspect_ratio=increase,crop={target_w}:{cam_h},setpts=PTS-STARTPTS[cam];"
                f"{v_in}scale={target_w}:{game_h}:force_original_aspect_ratio=increase,crop={target_w}:{game_h},setpts=PTS-STARTPTS[game];"
                f"[cam][game]vstack=inputs=2[v_split];"
                f"[v_split]drawbox=x=0:y={cam_h-2}:w={target_w}:h=4:color=#53fc18@0.95:t=fill[v916]"
            )
        elif shorts_layout in ["fit_crop", "fill", "center_crop"]:
            # Center Fit / Fill Crop
            filter_parts.append(
                f"{v_in}scale=-2:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h}:(in_w-{target_w})/2:0,setpts=PTS-STARTPTS[v916]"
            )
        else: # default "blur"
            blur_w = 240 if video_quality == "2k_60fps" else 180
            blur_h = 426 if video_quality == "2k_60fps" else 320
            filter_parts.append(
                f"{v_in}scale={blur_w}:{blur_h}:force_original_aspect_ratio=increase,crop={blur_w}:{blur_h},boxblur=5:3,scale={target_w}:{target_h}:flags=bilinear,drawbox=color=black@0.22:t=fill,setpts=PTS-STARTPTS[bg];"
                f"{v_in}scale={target_w}:-2,setpts=PTS-STARTPTS[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v916]"
            )
        v_current = "[v916]"

        # 3. Cold Open Hook Text Badge Overlay ("⚡ AZ SONRA... 🔥")
        if has_teaser:
            font_arg = "fontfile='C\\:/Windows/Fonts/arialbd.ttf':" if sys.platform == "win32" and os.path.exists("C:/Windows/Fonts/arialbd.ttf") else ""
            teaser_label = "⚡ AZ SONRA... 🔥"
            if hook_text and len(str(hook_text).strip()) >= 3:
                safe_hook = re.sub(r'[\r\n\t]+', ' ', str(hook_text)).strip().replace('\\', '').replace("'", "").replace(':', '')
                if len(safe_hook) > 26:
                    safe_hook = safe_hook[:24] + "..."
                teaser_label = f"⚡ AZ SONRA: {safe_hook} 🔥"

            filter_parts.append(
                f"{v_current}drawtext={font_arg}text='{teaser_label}':fontsize=38:fontcolor=white:box=1:boxcolor=red@0.85:boxborderw=12:x=(w-text_w)/2:y=120:enable='between(t,0,{t_dur:.2f})'[v_teaser_badge]"
            )
            v_current = "[v_teaser_badge]"

        # 4. Dynamic Hook Banner Overlay (Modern High-CTR Market Standard)
        if enable_hook_banner and hook_text:
            safe_text = re.sub(r'[\r\n\t]+', ' ', str(hook_text)).strip()
            safe_text = safe_text.replace('\\', '').replace("'", "").replace(':', '').replace('%', '')
            if len(safe_text) > 32:
                safe_text = safe_text[:30] + "..."

            b_start = t_dur if has_teaser else 0.0
            b_end = b_start + 3.5
            font_arg = "fontfile='C\\:/Windows/Fonts/arialbd.ttf':" if sys.platform == "win32" and os.path.exists("C:/Windows/Fonts/arialbd.ttf") else ""
            y_banner = "h*0.18"
            filter_parts.append(
                f"{v_current}drawtext={font_arg}text='{safe_text}':fontsize=40:fontcolor=0xffe600:box=1:boxcolor=0x0a0a0a@0.90:boxborderw=16:x=(w-text_w)/2:y={y_banner}:enable='between(t,{b_start:.2f},{b_end:.2f})'[vbanner]"
            )
            v_current = "[vbanner]"

        # 4. Live Kick Chat Reaction Overlay with Emote Wave
        if enable_chat_overlay:
            if moment_type == "funny":
                chat_lines = [
                    ("mert_k", "KEKW 😂", 0.4),
                    ("can99", "AHAHAHA KOPTUM 💀", 1.0),
                    ("berk_x", "böyle kahkaha yok 🤣", 1.7),
                    ("selin", "gözümden yaş geldi 😭", 2.5),
                    ("burak", "chat yıkıldı wtf 😂", 3.3),
                    ("kaan", "OMEGALUL", 4.1)
                ]
            elif moment_type in ["shock", "horror"]:
                chat_lines = [
                    ("ali_v", "NEEEEE 😱", 0.4),
                    ("mert", "jumpscare yedim wtf 💀", 1.0),
                    ("ayse", "KALBİM DURDU 😨", 1.7),
                    ("serdar", "çığlığa bak ses kısıldı 🔊", 2.5),
                    ("deniz", "KORKUDAN SIÇRADIM 👻", 3.3)
                ]
            else: # hype, action, chat
                chat_lines = [
                    ("emre", "W W W 🔥", 0.4),
                    ("volkan", "O NASIL HAREKET 🎯", 1.0),
                    ("burak", "HYPEEEEE ⚡", 1.7),
                    ("kemal", "clutch geldi helal 🏆", 2.5),
                    ("can", "YIKTI GEÇTİ 🔥", 3.3)
                ]

            font_chat = "fontfile='C\\:/Windows/Fonts/segoeui.ttf':" if sys.platform == "win32" and os.path.exists("C:/Windows/Fonts/segoeui.ttf") else ""
            c_box_w = 340
            c_box_h = 210
            c_x = 24
            c_y = 1920 - c_box_h - 130 # Just above TikTok bottom bar

            filter_parts.append(
                f"{v_current}drawbox=x={c_x}:y={c_y}:w={c_box_w}:h={c_box_h}:color=black@0.65:t=fill,"
                f"drawbox=x={c_x}:y={c_y}:w={c_box_w}:h=2:color=#53fc18@0.9:t=fill,"
                f"drawtext={font_chat}text='💬 Kick Sohbet':fontsize=20:fontcolor=0x53fc18:x={c_x+12}:y={c_y+8}:enable='between(t,0,999)'[v_chat_base]"
            )
            v_current = "[v_chat_base]"

            for idx_line, (u_name, u_msg, t_s) in enumerate(chat_lines):
                y_pos = c_y + 36 + (idx_line * 28)
                if y_pos + 26 > (c_y + c_box_h):
                    break
                safe_msg = u_msg.replace("'", "").replace(":", "")
                filter_parts.append(
                    f"{v_current}drawtext={font_chat}text='{u_name}\\: {safe_msg}':fontsize=20:fontcolor=white:x={c_x+12}:y={y_pos}:enable='gte(t,{t_s:.1f})'[v_c_{idx_line}]"
                )
                v_current = f"[v_c_{idx_line}]"

        # 5. Broadcast Standard Audio Loudness & Limiter (-14 LUFS, TikTok/Shorts standard)
        if normalize_audio:
            filter_parts.append(f"{a_in}loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000:async=1000:first_pts=0[afinal]")
            a_current = "[afinal]"
        else:
            filter_parts.append(f"{a_in}aresample=48000:async=1000:first_pts=0[afinal]")
            a_current = "[afinal]"

        full_filter_complex = ";".join(filter_parts)

        crf_val = '16' if video_quality == '2k_60fps' else '18'
        bitrate_a = '256k' if video_quality == '2k_60fps' else '192k'
        v_enc = get_hardware_video_encoder(target_crf=int(crf_val))

        cmd = [
            get_ffmpeg_binary(), '-y',
            '-fflags', '+genpts+discardcorrupt',
            '-i', input_filepath,
            *sfx_input_arg,
            '-filter_complex', full_filter_complex,
            '-map', v_current,
            '-map', a_current,
            *v_enc,
            '-threads', '0',
            '-r', '60',
            '-c:a', 'aac',
            '-b:a', bitrate_a,
            '-ar', '48000',
            '-avoid_negative_ts', 'make_zero',
            '-max_muxing_queue_size', '1024',
            '-movflags', '+faststart',
            output_filepath
        ]

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=60)
        return output_filepath

    def render_studio_clip(
        self,
        input_filepath: str,
        output_filepath: str,
        layout: str = "split_screen",
        cam_position: str = "top_right",
        cam_pan_x: float = 0.5,
        cam_pan_y: float = 0.5,
        cam_zoom: float = 1.0,
        game_pan_x: float = 0.5,
        game_pan_y: float = 0.5,
        game_zoom: float = 1.0,
        fit_pan_x: float = 0.5,
        blur_amount: int = 5,
        trim_start: float = 0.0,
        trim_end: Optional[float] = None,
        speed: float = 1.0,
        volume: float = 1.0,
        normalize_audio: bool = True,
        header_text: str = "",
        header_color: str = "#ffe600",
        header_start: float = 0.0,
        header_end: Optional[float] = None,
        header_x: float = 0.5,
        header_y: float = 0.14,
        header_scale: float = 1.0,
        header_theme: str = "yellow_pop",
        brightness: float = 0.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        video_quality: str = "1080p60",
        encoder_choice: str = "auto",
        overlay_image: Optional[str] = None,
        overlay_x: float = 0.5,
        overlay_y: float = 0.5,
        overlay_scale: float = 0.25,
        overlay_opacity: float = 1.0,
        subtitles_data: Optional[List[Dict[str, Any]]] = None,
        subtitles_style: str = "yellow",
        subtitles_pos: str = "bottom"
    ) -> str:
        """
        Studio Mode High-Speed 9:16 Video Renderer.
        Supports:
        - Interactive Mouse-Driven Pan & Zoom (Webcam & Gameplay)
        - CapCut-Style Free Placement Image/Logo Overlays
        - Dynamic Blur Background & Fill/Crop
        - Interactive Trimming & Scrubbing
        - Playback Speed & Volume Adjustment
        - Color Grading (Brightness, Contrast, Saturation)
        - Viral Hook Banner with Custom Themes, Timing & Canvas Positioning
        - High-Precision Turkish Subtitle Burn-In (Whisper + ASS)
        - GPU Hardware NVENC Acceleration (1-2s render)
        """
        if not os.path.exists(input_filepath):
            raise FileNotFoundError(f"Input file not found: {input_filepath}")

        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)

        target_w = 1440 if video_quality == "2k_60fps" else 1080
        target_h = 2560 if video_quality == "2k_60fps" else 1920

        # Construct input trim args
        input_trim_args = []
        if trim_start and trim_start > 0.05:
            input_trim_args.extend(['-ss', f"{trim_start:.2f}"])
        if trim_end and trim_end > 0.05 and (trim_end > (trim_start or 0)):
            input_trim_args.extend(['-to', f"{trim_end:.2f}"])

        filter_parts = []
        v_in = "[0:v]"
        a_in = "[0:a]"

        # 1. 9:16 Layout Filter
        if layout in ["split_screen", "split"]:
            cam_h = int(target_h * 0.44) # 844px
            game_h = target_h - cam_h     # 1076px

            c_zoom = max(1.0, min(3.0, float(cam_zoom or 1.0)))
            c_pan_x = max(0.0, min(1.0, float(cam_pan_x if cam_pan_x is not None else 0.5)))
            c_pan_y = max(0.0, min(1.0, float(cam_pan_y if cam_pan_y is not None else 0.5)))

            g_zoom = max(1.0, min(3.0, float(game_zoom or 1.0)))
            g_pan_x = max(0.0, min(1.0, float(game_pan_x if game_pan_x is not None else 0.5)))
            g_pan_y = max(0.0, min(1.0, float(game_pan_y if game_pan_y is not None else 0.5)))

            cam_crop = f"crop=w='min(iw,(ih/{c_zoom:.3f})*({target_w}/{cam_h}))':h='min(ih,ih/{c_zoom:.3f})':x='(iw-out_w)*{c_pan_x:.3f}':y='(ih-out_h)*{c_pan_y:.3f}',scale={target_w}:{cam_h}"
            game_crop = f"crop=w='min(iw,(ih/{g_zoom:.3f})*({target_w}/{game_h}))':h='min(ih,ih/{g_zoom:.3f})':x='(iw-out_w)*{g_pan_x:.3f}':y='(ih-out_h)*{g_pan_y:.3f}',scale={target_w}:{game_h}"

            filter_parts.append(
                f"{v_in}{cam_crop},setpts=PTS-STARTPTS[cam];"
                f"{v_in}{game_crop},setpts=PTS-STARTPTS[game];"
                f"[cam][game]vstack=inputs=2[v_split];"
                f"[v_split]drawbox=x=0:y={cam_h-2}:w={target_w}:h=4:color=#53fc18@0.95:t=fill[v_base]"
            )
        elif layout in ["fit_crop", "fill", "center_crop"]:
            f_pan_x = max(0.0, min(1.0, float(fit_pan_x if fit_pan_x is not None else 0.5)))
            fit_crop = f"crop=w='(ih)*({target_w}/{target_h})':h='ih':x='(iw-out_w)*{f_pan_x:.3f}':y=0,scale={target_w}:{target_h}"
            filter_parts.append(
                f"{v_in}{fit_crop},setpts=PTS-STARTPTS[v_base]"
            )
        else: # "blur"
            b_val = max(2, min(15, int(blur_amount or 5)))
            blur_w = 240 if video_quality == "2k_60fps" else 180
            blur_h = 426 if video_quality == "2k_60fps" else 320
            filter_parts.append(
                f"{v_in}scale={blur_w}:{blur_h}:force_original_aspect_ratio=increase,crop={blur_w}:{blur_h},boxblur={b_val}:{max(1, b_val//2)},scale={target_w}:{target_h}:flags=bilinear,drawbox=color=black@0.22:t=fill,setpts=PTS-STARTPTS[bg];"
                f"{v_in}scale={target_w}:-2,setpts=PTS-STARTPTS[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v_base]"
            )

        v_cur = "[v_base]"

        # 2. Color Grading (Brightness, Contrast, Saturation)
        if abs(brightness) > 0.01 or abs(contrast - 1.0) > 0.01 or abs(saturation - 1.0) > 0.01:
            filter_parts.append(
                f"{v_cur}eq=brightness={brightness:.2f}:contrast={contrast:.2f}:saturation={saturation:.2f}[v_eq]"
            )
            v_cur = "[v_eq]"

        # 3. Video Speed
        if abs(speed - 1.0) > 0.02:
            pts_mult = round(1.0 / speed, 4)
            filter_parts.append(f"{v_cur}setpts={pts_mult}*PTS[v_spd]")
            v_cur = "[v_spd]"

        # 4. Header Text / Hook Banner (With Custom Themes, Timing & Canvas Positioning)
        if header_text and len(str(header_text).strip()) > 0:
            safe_text = re.sub(r'[\r\n\t]+', ' ', str(header_text)).strip()
            safe_text = safe_text.replace('\\', '').replace("'", "").replace(':', '').replace('%', '')
            if len(safe_text) > 44:
                safe_text = safe_text[:42] + "..."
            
            font_path = "C:/Windows/Fonts/arialbd.ttf"
            font_arg = f"fontfile='{font_path.replace(':', r'\:')}':" if sys.platform == "win32" and os.path.exists(font_path) else ""
            
            h_scale = max(0.5, min(2.5, float(header_scale or 1.0)))
            f_size = max(24, min(90, int(42 * h_scale)))
            h_x = max(0.05, min(0.95, float(header_x if header_x is not None else 0.5)))
            h_y = max(0.05, min(0.95, float(header_y if header_y is not None else 0.14)))
            
            # Theme styling
            theme = str(header_theme or "yellow_pop").lower()
            if "kick" in theme or "neon" in theme or header_color == "#53fc18":
                f_color = "#53fc18"
                b_color = "0x090c12@0.94"
                border_w = 4
                border_col = "0x53fc18@0.7"
            elif "red" in theme or "alarm" in theme or header_color == "#ff4757":
                f_color = "white"
                b_color = "0xee5253@0.96"
                border_w = 3
                border_col = "white"
            elif "dark" in theme or "glass" in theme or header_color == "#ffffff":
                f_color = "white"
                b_color = "0x0f172a@0.90"
                border_w = 2
                border_col = "0x475569"
            elif "fire" in theme:
                f_color = "white"
                b_color = "0xff5252@0.96"
                border_w = 3
                border_col = "0xffeaa7"
            elif "cyber" in theme or "purple" in theme or header_color == "#00f0ff":
                f_color = "#00f0ff"
                b_color = "0x1a0b2e@0.94"
                border_w = 3
                border_col = "0xa855f7"
            elif "clean" in theme or "outline" in theme:
                f_color = header_color if str(header_color).startswith("#") else "#ffe600"
                b_color = "0x000000@0.0"
                border_w = 5
                border_col = "black"
            else: # yellow_pop default
                f_color = header_color if str(header_color).startswith("#") else "#ffe600"
                b_color = "0x0c0f17@0.94"
                border_w = 4
                border_col = "0xffe600@0.7"

            # Timing relative to trimmed clip
            h_start = max(0.0, float(header_start or 0.0) - (trim_start or 0.0))
            enable_str = ""
            if header_end is not None and float(header_end) > 0.05:
                h_end = max(h_start + 0.1, float(header_end) - (trim_start or 0.0))
                enable_str = f":enable='between(t\\,{h_start:.2f}\\,{h_end:.2f})'"
            elif h_start > 0.05:
                enable_str = f":enable='gte(t\\,{h_start:.2f})'"

            box_args = f":box=1:boxcolor={b_color}:boxborderw={int(14 * h_scale)}:borderw={border_w}:bordercolor={border_col}" if "clean" not in theme else ":box=0:borderw=5:bordercolor=black"

            filter_parts.append(
                f"{v_cur}drawtext={font_arg}text='{safe_text}':fontsize={f_size}:fontcolor={f_color}{box_args}:x='({target_w}*{h_x:.3f}-tw/2)':y='({target_h}*{h_y:.3f}-th/2)'{enable_str}[v_txt]"
            )
            v_cur = "[v_txt]"

        # 5. Image Overlay (Logo, sticker, etc.)
        img_input_args = []
        if overlay_image and str(overlay_image).strip():
            img_path = str(overlay_image).strip()
            if not os.path.exists(img_path):
                cand1 = os.path.join(self.output_dir, "overlays", os.path.basename(img_path))
                cand2 = os.path.join(self.output_dir, os.path.basename(img_path))
                if os.path.exists(cand1):
                    img_path = cand1
                elif os.path.exists(cand2):
                    img_path = cand2

            if os.path.exists(img_path):
                img_input_args = ['-i', img_path]
                o_scale = max(0.05, min(2.5, float(overlay_scale or 0.25)))
                o_w = int(target_w * o_scale)
                o_opac = max(0.1, min(1.0, float(overlay_opacity or 1.0)))
                o_x = max(-0.5, min(1.5, float(overlay_x if overlay_x is not None else 0.5)))
                o_y = max(-0.5, min(1.5, float(overlay_y if overlay_y is not None else 0.5)))

                opac_filter = f",colorchannelmixer=aa={o_opac:.2f}" if o_opac < 0.99 else ""
                filter_parts.append(
                    f"[1:v]scale={o_w}:-1,format=rgba{opac_filter}[ov_scaled];"
                    f"{v_cur}[ov_scaled]overlay=x='({target_w}*{o_x:.3f}-w/2)':y='({target_h}*{o_y:.3f}-h/2)':format=auto[v_img]"
                )
                v_cur = "[v_img]"

        # 6. Burn-in Turkish Subtitles (High Quality ASS)
        if subtitles_data and len(subtitles_data) > 0:
            try:
                ass_filename = f"subs_{int(time.time()*1000)}.ass"
                ass_path = os.path.join(self.temp_dir, ass_filename)
                
                # Determine vertical margin based on position
                pos = str(subtitles_pos or "bottom").lower()
                margin_v = 280 if pos == "bottom" else (760 if pos == "middle" else 1380)

                # Determine style (supports 7 rich TikTok/Shorts styles)
                s_style = str(subtitles_style or "yellow").lower()
                if "neon" in s_style or "green" in s_style:
                    style_line = f"Style: SubKick,Arial,58,&H0018FC53,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,3,2,60,60,{margin_v},1"
                    style_name = "SubKick"
                elif "white" in s_style:
                    style_line = f"Style: SubWhite,Arial,58,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,3,2,60,60,{margin_v},1"
                    style_name = "SubWhite"
                elif "boxed" in s_style:
                    style_line = f"Style: SubBoxed,Arial,52,&H00FFFFFF,&H000000FF,&H00000000,&HCC0C0F17,-1,0,0,0,100,100,0,0,3,16,0,2,60,60,{margin_v},1"
                    style_name = "SubBoxed"
                elif "red" in s_style or "fire" in s_style:
                    style_line = f"Style: SubFire,Arial,58,&H003838FF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,3,2,60,60,{margin_v},1"
                    style_name = "SubFire"
                elif "cyan" in s_style or "blue" in s_style:
                    style_line = f"Style: SubCyan,Arial,58,&H00FFF000,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,3,2,60,60,{margin_v},1"
                    style_name = "SubCyan"
                elif "badge" in s_style:
                    style_line = f"Style: SubBadge,Arial,52,&H00000000,&H000000FF,&H00000000,&H0000E6FF,-1,0,0,0,100,100,0,0,3,14,0,2,60,60,{margin_v},1"
                    style_name = "SubBadge"
                else: # Default: CapCut Yellow Pop
                    style_line = f"Style: SubYellow,Arial,58,&H0000E6FF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,3,2,60,60,{margin_v},1"
                    style_name = "SubYellow"

                def sec_to_ass(s_val: float) -> str:
                    h = int(s_val // 3600)
                    m = int((s_val % 3600) // 60)
                    s = s_val % 60
                    return f"{h}:{m:02d}:{s:05.2f}"

                dialogue_lines = []
                t_offset = float(trim_start or 0.0)
                t_limit = float(trim_end or 999999.0)

                for item in subtitles_data:
                    raw_txt = str(item.get("text") or "").strip()
                    if not raw_txt:
                        continue
                    s_st = float(item.get("start", 0.0))
                    s_en = float(item.get("end", 0.0))

                    if s_en <= t_offset or s_st >= t_limit:
                        continue # Outside trimmed window

                    rel_st = max(0.0, s_st - t_offset)
                    rel_en = max(rel_st + 0.2, s_en - t_offset)

                    clean_txt = raw_txt.replace('\n', '\\N').replace('\r', '').replace('{', '(').replace('}', ')')
                    dialogue_lines.append(f"Dialogue: 0,{sec_to_ass(rel_st)},{sec_to_ass(rel_en)},{style_name},,0,0,0,,{clean_txt}")

                if dialogue_lines:
                    ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {target_w}
PlayResY: {target_h}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{style_line}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""" + "\n".join(dialogue_lines) + "\n"

                    with open(ass_path, "w", encoding="utf-8") as f_ass:
                        f_ass.write(ass_content)

                    clean_ass_path = ass_path.replace('\\', '/')
                    escaped_ass = clean_ass_path.replace(':', r'\:').replace("'", r"\'")
                    filter_parts.append(f"{v_cur}subtitles='{escaped_ass}'[v_sub]")
                    v_cur = "[v_sub]"
            except Exception as e_sub:
                logger.warning(f"Could not generate or attach ASS subtitles: {e_sub}")

        # 6. Audio Pipeline (Speed, Volume, Normalization, Resampling)
        a_filters = []
        if abs(speed - 1.0) > 0.02:
            a_filters.append(f"atempo={speed:.4f}")
        if abs(volume - 1.0) > 0.02:
            a_filters.append(f"volume={volume:.2f}")
        if normalize_audio:
            a_filters.append("loudnorm=I=-14:TP=-1.5:LRA=11")
        a_filters.append("aresample=48000:async=1000:first_pts=0")

        filter_parts.append(f"{a_in}{','.join(a_filters)}[afinal]")
        a_cur = "[afinal]"

        full_filter_complex = ";".join(filter_parts)

        # 7. Encoder Selection (NVENC vs CPU)
        if encoder_choice == "nvenc":
            v_enc = ['-c:v', 'h264_nvenc', '-preset', 'p2', '-cq', '18', '-pix_fmt', 'yuv420p']
        elif encoder_choice == "cpu":
            v_enc = ['-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18', '-threads', '0', '-pix_fmt', 'yuv420p']
        else:
            v_enc = get_hardware_video_encoder(target_crf=18)

        bitrate_a = '256k' if video_quality == '2k_60fps' else '192k'

        cmd = [
            get_ffmpeg_binary(), '-y',
            '-fflags', '+genpts+discardcorrupt',
            *input_trim_args,
            '-i', input_filepath,
            *img_input_args,
            '-filter_complex', full_filter_complex,
            '-map', v_cur,
            '-map', a_cur,
            *v_enc,
            '-threads', '0',
            '-r', '60',
            '-c:a', 'aac',
            '-b:a', bitrate_a,
            '-ar', '48000',
            '-avoid_negative_ts', 'make_zero',
            '-max_muxing_queue_size', '1024',
            '-movflags', '+faststart',
            output_filepath
        ]

        logger.info(f"Studio Render command: {' '.join(cmd)}")
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=90)
        return output_filepath

    def generate_clip_thumbnail(
        self,
        video_filepath: str,
        output_thumb_path: str,
        title_text: str = "EN İYİ AN",
        streamer_name: str = "Kick",
        moment_type: str = "funny",
        hype_score: int = 95,
        target_format: str = "916",
        frame_sec: Optional[float] = None,
        badge_text: Optional[str] = None,
        style: str = "kick_neon",
        base_image_path: Optional[str] = None,
        focus_box: Optional[Dict[str, float]] = None,
        draw_ring: bool = False,
        ring_color: str = "red",
        crop_to_focus: bool = False
    ) -> str:
        """
        Generates high-CTR, high-clickthrough viral thumbnail cover for YouTube Shorts & TikTok.
        Captures high-energy frame from video or uses provided AI generated base image,
        applies interactive ROI focus, viral red attention ring, dark gradient, bright bold title badge,
        streamer watermark, and hype/moment tags.
        """
        target_w, target_h = (1080, 1920) if target_format == "916" else (1280, 720)
        temp_frame = ""

        if base_image_path and os.path.exists(base_image_path) and os.path.getsize(base_image_path) > 0:
            source_img_path = base_image_path
        else:
            temp_frame = os.path.join(self.temp_dir, f"thumb_raw_{int(time.time() * 1000)}.jpg")
            ss_val = str(round(max(0.1, float(frame_sec)), 2)) if (frame_sec is not None and float(frame_sec) >= 0) else '1.5'
            
            cmd = [
                get_ffmpeg_binary(), '-y',
                '-ss', ss_val,
                '-i', video_filepath,
                '-vframes', '1',
                '-q:v', '2',
                temp_frame
            ]
            try:
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=15)
            except Exception:
                try:
                    subprocess.run([get_ffmpeg_binary(), '-y', '-i', video_filepath, '-ss', ss_val, '-vframes', '1', '-q:v', '2', temp_frame], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                except Exception:
                    pass
            source_img_path = temp_frame

        if not os.path.exists(source_img_path) or os.path.getsize(source_img_path) == 0:
            img = Image.new('RGB', (target_w, target_h), color=(15, 23, 42))
        else:
            try:
                img = Image.open(source_img_path).convert('RGB')
                orig_w, orig_h = img.width, img.height
                target_ratio = target_w / target_h

                # 1. Custom ROI / Focus Box Crop & Zoom
                if crop_to_focus and focus_box and isinstance(focus_box, dict) and focus_box.get('w', 0) > 0.05 and focus_box.get('h', 0) > 0.05:
                    bx = float(focus_box.get('x', 0.0)) * orig_w
                    by = float(focus_box.get('y', 0.0)) * orig_h
                    bw = float(focus_box.get('w', 0.3)) * orig_w
                    bh = float(focus_box.get('h', 0.3)) * orig_h
                    bcx = bx + bw / 2.0
                    bcy = by + bh / 2.0

                    # Expand slightly to give room around the subject
                    cw = max(bw * 1.35, 100)
                    ch = max(bh * 1.35, 100)
                    if cw / ch < target_ratio:
                        cw = ch * target_ratio
                    else:
                        ch = cw / target_ratio

                    # Clamp to image boundaries
                    cw = min(cw, orig_w)
                    ch = min(ch, orig_h)
                    if cw / ch < target_ratio:
                        ch = cw / target_ratio
                    else:
                        cw = ch * target_ratio

                    cl = max(0, min(orig_w - cw, bcx - cw / 2.0))
                    ct = max(0, min(orig_h - ch, bcy - ch / 2.0))
                    img = img.crop((int(cl), int(ct), int(cl + cw), int(ct + ch)))
                else:
                    img_ratio = img.width / img.height
                    if img_ratio > target_ratio:
                        new_w = int(img.height * target_ratio)
                        left = (img.width - new_w) // 2
                        img = img.crop((left, 0, left + new_w, img.height))
                    else:
                        new_h = int(img.width / target_ratio)
                        top = (img.height - new_h) // 2
                        img = img.crop((0, top, img.width, top + new_h))

                img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

                # Enhance visual punchiness: Contrast, Color Saturation & Sharpness
                try:
                    from PIL import ImageEnhance
                    img = ImageEnhance.Contrast(img).enhance(1.22)
                    img = ImageEnhance.Color(img).enhance(1.30)
                    img = ImageEnhance.Sharpness(img).enhance(1.35)
                    img = ImageEnhance.Brightness(img).enhance(1.04)
                except Exception:
                    pass
            except Exception:
                img = Image.new('RGB', (target_w, target_h), color=(15, 23, 42))

        # 2. Viral Attention Ring & Spotlight Effect
        if draw_ring and focus_box and isinstance(focus_box, dict) and focus_box.get('w', 0) > 0.02 and focus_box.get('h', 0) > 0.02:
            try:
                rx0 = int(float(focus_box.get('x', 0.2)) * target_w)
                ry0 = int(float(focus_box.get('y', 0.2)) * target_h)
                rw = int(float(focus_box.get('w', 0.3)) * target_w)
                rh = int(float(focus_box.get('h', 0.3)) * target_h)
                rx1 = rx0 + rw
                ry1 = ry0 + rh

                # Padding so ring frames the subject nicely
                px = max(10, int(rw * 0.08))
                py = max(10, int(rh * 0.08))
                rx0 = max(12, rx0 - px)
                ry0 = max(12, ry0 - py)
                rx1 = min(target_w - 12, rx1 + px)
                ry1 = min(target_h - 12, ry1 + py)

                ring_canvas = Image.new('RGBA', (target_w, target_h), (0, 0, 0, 0))
                ring_draw = ImageDraw.Draw(ring_canvas, 'RGBA')

                # Soft spotlight effect: subtly darken outer background (20% dim)
                dark_mask = Image.new('RGBA', (target_w, target_h), (0, 0, 0, 65))
                dark_draw = ImageDraw.Draw(dark_mask)
                dark_draw.ellipse([rx0, ry0, rx1, ry1], fill=(0, 0, 0, 0))
                img.paste(dark_mask, (0, 0), dark_mask)

                # Color definitions
                if ring_color == "yellow":
                    core_col = (255, 230, 0, 255)
                    glow_col = (255, 210, 0, 70)
                elif ring_color == "neon_green":
                    core_col = (83, 252, 24, 255)
                    glow_col = (83, 252, 24, 70)
                else: # red (classic viral YouTube thumbnail ring)
                    core_col = (255, 36, 36, 255)
                    glow_col = (255, 20, 20, 75)

                # Outer soft glowing rings
                for g_thick, g_alpha in [(16, 40), (12, 60), (8, 90)]:
                    g_col = (glow_col[0], glow_col[1], glow_col[2], g_alpha)
                    ring_draw.ellipse([rx0 - g_thick // 2, ry0 - g_thick // 2, rx1 + g_thick // 2, ry1 + g_thick // 2], outline=g_col, width=g_thick)

                # Main crisp high-contrast attention ring
                ring_thick = 8 if target_format == "916" else 7
                ring_draw.ellipse([rx0, ry0, rx1, ry1], outline=core_col, width=ring_thick)

                # Inner white specular rim for 3D metallic feel
                ring_draw.ellipse([rx0 + 2, ry0 + 2, rx1 - 2, ry1 - 2], outline=(255, 255, 255, 160), width=2)

                # Draw iconic Viral Pointer Arrow directed at the ring
                arrow_w = 40 if target_format == "916" else 32
                arrow_tip_x = rx1 - 4
                arrow_tip_y = ry0 + (ry1 - ry0) // 4
                arrow_start_x = min(target_w - 20, arrow_tip_x + arrow_w * 2)
                arrow_start_y = max(20, arrow_tip_y - arrow_w * 1.5)

                # Arrow shaft & head
                ring_draw.line([(arrow_start_x, arrow_start_y), (arrow_tip_x, arrow_tip_y)], fill=(0, 0, 0, 240), width=14)
                ring_draw.line([(arrow_start_x, arrow_start_y), (arrow_tip_x, arrow_tip_y)], fill=core_col, width=8)
                # Arrowhead triangle
                poly_head = [
                    (arrow_tip_x, arrow_tip_y),
                    (arrow_tip_x + 24, arrow_tip_y - 12),
                    (arrow_tip_x + 12, arrow_tip_y + 24)
                ]
                ring_draw.polygon(poly_head, fill=core_col, outline=(0, 0, 0, 255))

                img.paste(ring_canvas, (0, 0), ring_canvas)
            except Exception as ring_err:
                logger.warning(f"Error drawing attention ring: {ring_err}")


        # Vignette / Smooth Dark Gradient overlays (bottom focus for text legibility)
        draw = ImageDraw.Draw(img, 'RGBA')
        
        # Subtle top gradient for badges (height: 180px)
        for y in range(min(180, target_h)):
            alpha = int(140 * (1 - y / 180))
            draw.line([(0, y), (target_w, y)], fill=(0, 0, 0, alpha))
            
        # Rich bottom cinematic gradient (height: 520px)
        grad_h = min(520, int(target_h * 0.45))
        grad_start = target_h - grad_h
        for y in range(grad_h):
            prog = y / grad_h
            alpha = int(240 * (prog ** 1.6))
            draw.line([(0, grad_start + y), (target_w, grad_start + y)], fill=(0, 0, 0, alpha))

        # Font selection: Impact (iconic YouTube thumbnail font) -> Arial Black -> Arial Bold
        font_impact = "C:/Windows/Fonts/impact.ttf" if sys.platform == "win32" and os.path.exists("C:/Windows/Fonts/impact.ttf") else None
        font_ariblk = "C:/Windows/Fonts/ariblk.ttf" if sys.platform == "win32" and os.path.exists("C:/Windows/Fonts/ariblk.ttf") else None
        font_arialbd = "C:/Windows/Fonts/arialbd.ttf" if sys.platform == "win32" and os.path.exists("C:/Windows/Fonts/arialbd.ttf") else None
        best_font_path = font_impact or font_ariblk or font_arialbd
        
        def load_font(size, force_impact=True):
            f_path = best_font_path if force_impact else (font_ariblk or font_arialbd or best_font_path)
            if f_path:
                try: return ImageFont.truetype(f_path, size)
                except: pass
            return ImageFont.load_default()

        # Color scheme based on style
        if style == "fire":
            st_badge_bg = (255, 59, 48, 240)
            st_badge_text_col = (255, 255, 255, 255)
            mo_badge_bg = (255, 149, 0, 245)
            mo_badge_outline = (255, 214, 10, 255)
            title_fill_col = (255, 220, 0, 255)
        elif style == "shock":
            st_badge_bg = (0, 240, 255, 240)
            st_badge_text_col = (0, 0, 0, 255)
            mo_badge_bg = (255, 230, 0, 245)
            mo_badge_outline = (0, 240, 255, 255)
            title_fill_col = (255, 255, 255, 255)
        elif style == "drama":
            st_badge_bg = (220, 38, 38, 240)
            st_badge_text_col = (255, 255, 255, 255)
            mo_badge_bg = (30, 10, 15, 240)
            mo_badge_outline = (239, 68, 68, 255)
            title_fill_col = (255, 230, 0, 255)
        else: # kick_neon
            st_badge_bg = (83, 252, 24, 240)
            st_badge_text_col = (0, 0, 0, 255)
            mo_badge_bg = (10, 15, 10, 240)
            mo_badge_outline = (83, 252, 24, 255)
            title_fill_col = (255, 230, 0, 255)

        # Top-Left Modern Rounded Pill Badge (Streamer & Kick)
        badge_w, badge_h = (260, 56) if target_format == "916" else (210, 46)
        badge_x, badge_y = (36, 36) if target_format == "916" else (24, 24)
        draw.rounded_rectangle([(badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h)], radius=16, fill=st_badge_bg)
        font_badge = load_font(26 if target_format == "916" else 20, force_impact=False)
        clean_streamer = (streamer_name or "Kick").replace('@', '').capitalize()
        draw.text((badge_x + 16, badge_y + 11), f"KICK • {clean_streamer[:14]}", fill=st_badge_text_col, font=font_badge)

        # Top-Right Moment Badge (Pill with neon outline)
        if badge_text:
            m_label = badge_text.upper()
        else:
            m_label = "😂 KAHKAHA" if moment_type == "funny" else ("😱 ŞOK AN" if moment_type in ["shock", "horror"] else f"🔥 %{hype_score} HYPE")
            
        font_sub = load_font(24 if target_format == "916" else 18, force_impact=False)
        m_bbox = font_sub.getbbox(m_label) if hasattr(font_sub, 'getbbox') else (0, 0, 150, 30)
        m_w = m_bbox[2] - m_bbox[0] + 32
        m_x = target_w - m_w - (36 if target_format == "916" else 24)
        draw.rounded_rectangle([(m_x, badge_y), (m_x + m_w, badge_y + badge_h)], radius=16, fill=mo_badge_bg, outline=mo_badge_outline, width=3)
        draw.text((m_x + 16, badge_y + 11), m_label, fill=(255, 255, 255, 255), font=font_sub)

        # Main Bold Viral Title (Clean 3D Stroke Typography - No ugly black block bars)
        clean_title = (title_text or "YAYININ EN İYİ ANI").upper()
        words = clean_title.split()
        lines = []
        cur_line = []
        max_chars = 17 if target_format == "916" else 26
        for w in words:
            if len(" ".join(cur_line + [w])) <= max_chars:
                cur_line.append(w)
            else:
                lines.append(" ".join(cur_line))
                cur_line = [w]
        if cur_line:
            lines.append(" ".join(cur_line))
        lines = lines[:3]

        font_title_size = 64 if target_format == "916" else 48
        font_title = load_font(font_title_size, force_impact=True)

        y_text_start = target_h - (190 if target_format == "916" else 110) - (len(lines) * (font_title_size + 14))
        for line in lines:
            bbox = font_title.getbbox(line) if hasattr(font_title, 'getbbox') else (0, 0, 300, 50)
            lw = bbox[2] - bbox[0]
            lx = (target_w - lw) // 2

            # 1. 3D Drop Shadow (Offset depth)
            for sx, sy in [(3, 5), (4, 6), (5, 7), (2, 4), (4, 4)]:
                draw.text((lx + sx, y_text_start + sy), line, fill=(0, 0, 0, 230), font=font_title)

            # 2. Heavy Black Border Stroke (Circled outline)
            outline_range = 5 if target_format == "916" else 4
            for ox in range(-outline_range, outline_range + 1):
                for oy in range(-outline_range, outline_range + 1):
                    if ox * ox + oy * oy <= (outline_range * outline_range):
                        draw.text((lx + ox, y_text_start + oy), line, fill=(0, 0, 0, 255), font=font_title)

            # 3. Vibrant Front Color Fill
            draw.text((lx, y_text_start), line, fill=title_fill_col, font=font_title)
            y_text_start += font_title_size + 18

        os.makedirs(os.path.dirname(output_thumb_path), exist_ok=True)
        img.save(output_thumb_path, format="JPEG", quality=92)

        if os.path.exists(temp_frame):
            try: os.remove(temp_frame)
            except: pass

        return output_thumb_path

    def ensure_perfect_av_sync(self, filepath: str) -> bool:
        """
        Guarantees 100% sample-accurate, lip-sync-locked audio and video.
        Inspects container streams using ffprobe. If audio start_time deviates from video start_time
        by more than 35ms (e.g. video padding before audio starts), it seeks to the exact alignment point
        and re-encodes via hardware NVENC / libx264 so both streams start at 0.000000s.
        """
        if not filepath or not os.path.exists(filepath) or os.path.getsize(filepath) < 1000:
            return False

        try:
            probe_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', filepath]
            proc = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=12)
            if proc.returncode != 0:
                return False
            data = json.loads(proc.stdout.decode('utf-8', errors='ignore'))
            v = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'), None)
            a = next((s for s in data.get('streams', []) if s.get('codec_type') == 'audio'), None)
            if not v or not a:
                return False

            v_st = float(v.get('start_time', 0.0) or 0.0)
            a_st = float(a.get('start_time', 0.0) or 0.0)
            diff = a_st - v_st

            if abs(diff) < 0.035 and abs(v_st) < 0.035 and abs(a_st) < 0.035:
                return True

            cut_ss = max(v_st, a_st)
            temp_fixed = filepath + f".sync_{int(time.time() * 1000)}.mp4"

            v_enc = get_hardware_video_encoder(target_crf=18)
            seek_arg = ['-ss', str(cut_ss)] if cut_ss > 0.035 else []
            cmd = [
                get_ffmpeg_binary(), '-y',
                *seek_arg,
                '-i', filepath,
                '-map', '0:v:0',
                '-map', '0:a:0?',
                *v_enc,
                '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
                '-af', 'aresample=48000:async=1000:first_pts=0',
                '-fps_mode', 'cfr',
                '-avoid_negative_ts', 'make_zero',
                '-movflags', '+faststart',
                temp_fixed
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=60)
            if os.path.exists(temp_fixed) and os.path.getsize(temp_fixed) > 1000:
                os.replace(temp_fixed, filepath)
                return True
        except Exception as e:
            print(f"ensure_perfect_av_sync warning: {e}")
        return False

    def extract_single_clip(
        self,
        source_url: str,
        clip: Dict[str, Any],
        title_prefix: str = "KickClip",
        video_format: str = "raw_16_9",
        normalize_audio: bool = True,
        enable_teaser_hook: bool = False,
        enable_hook_banner: bool = False,
        shorts_layout: str = "blur",
        cam_position: str = "top_right",
        enable_chat_overlay: bool = False,
        segments: Optional[List[Dict[str, Any]]] = None,
        best_stream_url: Optional[str] = None,
        video_quality: str = "2k_60fps"
    ) -> Dict[str, Any]:
        """
        Extracts, cuts, and renders a single clip with guaranteed zero audio-video desync and generates HD thumbnails.
        """
        clean_name = "".join(c for c in title_prefix if c.isalnum() or c in (' ', '_', '-')).rstrip().replace(' ', '_')
        time_tag = clip.get('start_formatted', 'clip').replace(':', '-')
        clip_id = clip.get('id', 1)
        out_filename = f"{clean_name}_Klip_{clip_id}_{time_tag}.mp4"
        out_filepath = os.path.join(self.output_dir, out_filename)

        clip_start = float(clip['start_time'])
        clip_dur = float(clip['duration'])
        clip_end = clip_start + clip_dur

        if segments is None or best_stream_url is None:
            best_stream_url, segments = self.get_best_video_stream_and_segments(source_url, video_quality=video_quality)

        def fetch_chunk_item(item):
            idx, seg_url = item
            s = get_cdn_session()
            for attempt in range(4):
                try:
                    res = s.get(seg_url, timeout=(4, 10))
                    if res.status_code == 200 and len(res.content) > 0:
                        return idx, res.content
                except Exception:
                    time.sleep(0.15 * (attempt + 1))
            return idx, b''

        success = False

        # 1. Direct High-Bitrate YouTube & Google Drive Slicer via yt-dlp (Guaranteed true 1080p60 / 2K 60fps)
        is_youtube = ("youtube.com" in source_url.lower() or "youtu.be" in source_url.lower())
        is_gdrive = ("drive.google.com" in source_url.lower() or "docs.google.com" in source_url.lower())
        if is_youtube or is_gdrive:
            try:
                import yt_dlp
                if is_gdrive:
                    fmt_pref = 'best/bestvideo+bestaudio'
                elif video_quality == "2k_60fps":
                    fmt_pref = 'bestvideo[height<=1440]+bestaudio/best'
                elif video_quality == "1080p60":
                    fmt_pref = 'bestvideo[height<=1080]+bestaudio/best'
                else: # source_raw
                    fmt_pref = 'bestvideo+bestaudio/best'

                unique_slice_token = f"yt_slice_{clip_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
                temp_yt_pattern = os.path.join(self.temp_dir, f"{unique_slice_token}.%(ext)s")
                base_ydl_opts = self.get_yt_dlp_opts({
                    'format': fmt_pref,
                    'download_ranges': yt_dlp.utils.download_range_func(None, [(clip_start, clip_end)]),
                    'force_keyframes_at_cuts': True,
                    'outtmpl': temp_yt_pattern,
                    'nopart': True,
                    'overwrites': True,
                    'quiet': True,
                    'merge_output_format': 'mp4'
                })
                for use_cookies in ([True, False] if base_ydl_opts.get('cookiefile') else [False]):
                    ydl_opts = dict(base_ydl_opts)
                    if not use_cookies:
                        ydl_opts.pop('cookiefile', None)
                    try:
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([source_url])
                        break
                    except Exception:
                        pass

                found = [os.path.join(self.temp_dir, f) for f in os.listdir(self.temp_dir) 
                         if f.startswith(unique_slice_token) 
                         and not f.endswith('.part') 
                         and not f.endswith('.ytdl') 
                         and f.lower().endswith(('.mp4', '.mkv', '.webm', '.ts'))
                         and os.path.getsize(os.path.join(self.temp_dir, f)) > 10000]

                # Clean up any leftover tiny/empty files from this cut token
                for f in os.listdir(self.temp_dir):
                    if f.startswith(unique_slice_token) and os.path.join(self.temp_dir, f) not in found:
                        try: os.remove(os.path.join(self.temp_dir, f))
                        except Exception: pass
                if found:
                    raw_yt_clip = found[0]
                    # Check raw stream timings to prevent any audio-video desync
                    probe_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', raw_yt_clip]
                    yt_v_st = 0.0
                    yt_a_st = 0.0
                    try:
                        p_out = subprocess.check_output(probe_cmd, timeout=8).decode('utf-8', errors='ignore')
                        yt_data = json.loads(p_out)
                        yt_v = next((s for s in yt_data.get('streams', []) if s.get('codec_type') == 'video'), None)
                        yt_a = next((s for s in yt_data.get('streams', []) if s.get('codec_type') == 'audio'), None)
                        yt_v_st = float(yt_v.get('start_time', 0.0) or 0.0) if yt_v else 0.0
                        yt_a_st = float(yt_a.get('start_time', 0.0) or 0.0) if yt_a else 0.0
                    except Exception:
                        pass

                    yt_diff = yt_a_st - yt_v_st
                    copied = False

                    # Only direct copy if stream starts perfectly at 0.000s without offset and audio normalization is not requested
                    if not normalize_audio and (video_format in ["raw_16_9", "vertical_9_16"] or video_quality == "source_raw") and abs(yt_diff) < 0.035 and abs(yt_v_st) < 0.035 and abs(yt_a_st) < 0.035:
                        fast_cmd = [
                            get_ffmpeg_binary(), '-y',
                            '-i', raw_yt_clip,
                            '-map', '0:v:0',
                            '-map', '0:a:0?',
                            '-c', 'copy',
                            '-movflags', '+faststart',
                            '-avoid_negative_ts', 'make_zero',
                            out_filepath
                        ]
                        try:
                            subprocess.run(fast_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=15)
                            if os.path.exists(out_filepath) and os.path.getsize(out_filepath) > 1000:
                                copied = True
                                success = True
                        except Exception:
                            pass

                    if not copied:
                        scale_opt = []
                        if video_quality == "2k_60fps":
                            scale_opt = ['-vf', 'scale=2560:1440:force_original_aspect_ratio=decrease,pad=2560:1440:(ow-iw)/2:(oh-ih)/2', '-r', '60']
                        elif video_quality == "1080p60":
                            scale_opt = ['-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2', '-r', '60']

                        v_enc = get_hardware_video_encoder(target_crf=18)
                        cut_lead = max(yt_v_st, yt_a_st)
                        seek_arg = ['-ss', str(cut_lead)] if cut_lead > 0.035 else []
                        af_filter = 'loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000:async=1000:first_pts=0' if normalize_audio else 'aresample=48000:async=1000:first_pts=0'
                        trans_cmd = [
                            get_ffmpeg_binary(), '-y',
                            *seek_arg,
                            '-i', raw_yt_clip,
                            '-map', '0:v:0',
                            '-map', '0:a:0?',
                            *scale_opt,
                            *v_enc,
                            '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
                            '-af', af_filter,
                            '-fps_mode', 'cfr',
                            '-movflags', '+faststart',
                            '-avoid_negative_ts', 'make_zero',
                            out_filepath
                        ]
                        subprocess.run(trans_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=60)
                        if os.path.exists(out_filepath) and os.path.getsize(out_filepath) > 1000:
                            success = True
                    try: os.remove(raw_yt_clip)
                    except: pass
            except Exception as e:
                print(f"Direct YouTube/GDrive slice error, trying fallback: {e}")
                # Google Drive için ffmpeg ile direkt indirme fallback
                if is_gdrive and not success:
                    import re as _re2
                    file_m_v = _re2.search(r'(?:file/d/|id=|open\?id=)([a-zA-Z0-9_-]{25,})', source_url)
                    if file_m_v:
                        try:
                            v_id_v = file_m_v.group(1)
                            direct_v_url = f"https://drive.google.com/uc?export=download&id={v_id_v}&confirm=t"
                            tmp_raw = os.path.join(self.temp_dir, f"gdrive_raw_{clip_id}_{int(time.time())}.mp4")
                            ff_dl_cmd = [
                                get_ffmpeg_binary(), '-y',
                                '-ss', str(clip_start),
                                '-i', direct_v_url,
                                '-t', str(clip_dur),
                                '-c', 'copy',
                                '-movflags', '+faststart',
                                tmp_raw
                            ]
                            res_dl = subprocess.run(ff_dl_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
                            if res_dl.returncode == 0 and os.path.exists(tmp_raw) and os.path.getsize(tmp_raw) > 10000:
                                import shutil
                                shutil.move(tmp_raw, out_filepath)
                                success = True
                        except Exception as gd_err:
                            print(f"GDrive ffmpeg fallback failed: {gd_err}")

        # 2. Turbo segment-based extraction (for Kick HLS Streams)
        if not success and segments and len(segments) > 0:
            matching_segments = []
            for seg in segments:
                s_start = seg['start_time']
                s_end = s_start + seg['duration']
                if s_end >= max(0.0, clip_start - 2.0) and s_start <= (clip_end + 2.0):
                    matching_segments.append(seg)

            if matching_segments:
                first_seg_start = matching_segments[0]['start_time']
                relative_start = max(0.0, clip_start - first_seg_start)

                indexed_chunks = list(enumerate([s['url'] for s in matching_segments]))
                chunk_results = {}
                with ThreadPoolExecutor(max_workers=min(30, len(indexed_chunks) or 1)) as executor:
                    futures = {executor.submit(fetch_chunk_item, item): item[0] for item in indexed_chunks}
                    for future in as_completed(futures):
                        idx, data = future.result()
                        chunk_results[idx] = data

                temp_ts = os.path.join(self.temp_dir, f"clip_temp_{clip_id}_{int(time.time())}.ts")
                with open(temp_ts, "wb") as f:
                    for idx in sorted(chunk_results.keys()):
                        if chunk_results[idx]:
                            f.write(chunk_results[idx])

                if os.path.exists(temp_ts) and os.path.getsize(temp_ts) > 0:
                    v_enc = get_hardware_video_encoder(target_crf=18)
                    af_filter = 'loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000:async=1000:first_pts=0' if normalize_audio else 'aresample=48000:async=1000:first_pts=0'
                    cmd = [
                        get_ffmpeg_binary(), '-y',
                        '-fflags', '+genpts+discardcorrupt',
                        '-ss', str(relative_start),
                        '-i', temp_ts,
                        '-t', str(clip_dur),
                        '-map', '0:v:0',
                        '-map', '0:a:0?',
                        *v_enc,
                        '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
                        '-af', af_filter,
                        '-fps_mode', 'cfr',
                        '-avoid_negative_ts', 'make_zero',
                        '-max_muxing_queue_size', '2048',
                        '-movflags', '+faststart',
                        out_filepath
                    ]
                    try:
                        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=45)
                        if os.path.exists(out_filepath) and os.path.getsize(out_filepath) > 1000:
                            success = True
                    except Exception as e:
                        print(f"Kick TS transcode error: {e}")

                if os.path.exists(temp_ts):
                    try: os.remove(temp_ts)
                    except Exception: pass

        # 3. Fallback for non-segment sources (Local files, direct Kick streams)
        if not success:
            target_url = best_stream_url if best_stream_url else source_url
            is_split_stream = False
            v_input = target_url
            a_input = target_url
            if "|||" in target_url:
                is_split_stream = True
                v_input, a_input = target_url.split("|||", 1)

            if video_quality == "2k_60fps":
                scale_vf = "scale=2560:1440:force_original_aspect_ratio=decrease,pad=2560:1440:(ow-iw)/2:(oh-ih)/2"
                fps_val = "60"
                crf_val = "16"
            elif video_quality == "1080p60":
                scale_vf = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
                fps_val = "60"
                crf_val = "17"
            else: # source_raw
                scale_vf = None
                fps_val = None
                crf_val = "17"

            cmd = [get_ffmpeg_binary(), '-y']
            if is_split_stream:
                cmd.extend(['-ss', str(clip_start), '-i', v_input])
                cmd.extend(['-ss', str(clip_start), '-i', a_input])
                cmd.extend(['-map', '0:v:0', '-map', '1:a:0'])
            else:
                cmd.extend(['-ss', str(clip_start), '-i', target_url])
                cmd.extend(['-map', '0:v:0', '-map', '0:a:0?'])

            cmd.extend(['-t', str(clip_dur)])
            if scale_vf:
                cmd.extend(['-vf', scale_vf])
            if fps_val:
                cmd.extend(['-r', fps_val])

            v_enc = get_hardware_video_encoder(target_crf=int(crf_val))
            af_filter = 'loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000:async=1000:first_pts=0' if normalize_audio else 'aresample=48000:async=1000:first_pts=0'
            cmd.extend([
                *v_enc,
                '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
                '-af', af_filter,
                '-fps_mode', 'cfr',
                '-avoid_negative_ts', 'make_zero',
                '-max_muxing_queue_size', '2048',
                '-movflags', '+faststart',
                out_filepath
            ])
            try:
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=90)
                if os.path.exists(out_filepath) and os.path.getsize(out_filepath) > 0:
                    success = True
            except Exception as ex:
                print(f"Encode error: {ex}")

        if success and os.path.exists(out_filepath) and os.path.getsize(out_filepath) > 0:
            # Bulletproof Audio/Video Sync Assurance
            self.ensure_perfect_av_sync(out_filepath)
            clip_data = {
                **clip,
                "filename": out_filename,
                "filepath": out_filepath,
                "size_mb": round(os.path.getsize(out_filepath) / (1024 * 1024), 2),
                "status": "success",
                "has_916": False,
                "vertical_filename": None,
                "vertical_size_mb": None,
                "thumbnail_916": None,
                "thumbnail_169": None
            }

            # Generate 9:16 Shorts format if requested
            if video_format in ["vertical_9_16", "both"]:
                try:
                    vert_filename = f"{clean_name}_Klip_{clip_id}_{time_tag}_916.mp4"
                    vert_filepath = os.path.join(self.output_dir, vert_filename)

                    peak_rel = float(clip['peak_time']) - float(clip['start_time']) if 'peak_time' in clip else None
                    m_type = clip.get('moment_type', 'funny')
                    l_score = clip.get('laughter_score', 0)
                    h_score = clip.get('hype_score', 90)

                    if clip.get('hook_line'):
                        hook_banner_text = clip['hook_line']
                    elif m_type == "funny" or l_score >= 50:
                        hook_banner_text = "BURADA KAHKAHA KOPTU 😂"
                    elif m_type in ["shock", "horror"]:
                        hook_banner_text = "NE OLDUĞUNA İNANAMAYACAKSIN 😱"
                    elif m_type == "action":
                        hook_banner_text = "BU HAREKET İMKANSIZ 🎯"
                    elif m_type == "hype" or h_score >= 88:
                        hook_banner_text = "SONUNA KADAR İZLE 🔥"
                    else:
                        hook_banner_text = "BURAYA DİKKAT ⚡"

                    self.convert_to_vertical_916(
                        input_filepath=out_filepath,
                        output_filepath=vert_filepath,
                        normalize_audio=normalize_audio,
                        enable_teaser_hook=enable_teaser_hook,
                        peak_time_rel=peak_rel,
                        clip_duration=clip_dur,
                        enable_hook_banner=enable_hook_banner,
                        hook_text=hook_banner_text,
                        shorts_layout=shorts_layout,
                        cam_position=cam_position,
                        enable_chat_overlay=enable_chat_overlay,
                        moment_type=clip.get('moment_type', 'funny'),
                        video_quality=video_quality
                    )
                    if os.path.exists(vert_filepath) and os.path.getsize(vert_filepath) > 0:
                        clip_data["has_916"] = True
                        clip_data["vertical_filename"] = vert_filename
                        clip_data["vertical_filepath"] = vert_filepath
                        clip_data["vertical_size_mb"] = round(os.path.getsize(vert_filepath) / (1024 * 1024), 2)
                except Exception:
                    pass

            return clip_data
        else:
            raise RuntimeError(f"Klip #{clip_id} kesilemedi.")

    def extract_raw_clips(
        self,
        source_url: str,
        clips: List[Dict[str, Any]],
        title_prefix: str = "KickClip",
        video_format: str = "raw_16_9",
        normalize_audio: bool = False,
        enable_teaser_hook: bool = False,
        enable_hook_banner: bool = False,
        shorts_layout: str = "blur",
        cam_position: str = "top_right",
        enable_chat_overlay: bool = False,
        callback: Optional[Callable[[str, int], None]] = None,
        video_quality: str = "2k_60fps"
    ) -> List[Dict[str, Any]]:
        results = []
        total_clips = len(clips)

        self.log("En yüksek kalitede video akışı çözümleniyor...", 62, callback)
        best_stream_url, segments = self.get_best_video_stream_and_segments(source_url, video_quality=video_quality)

        max_workers = min(3, total_clips) if total_clips > 1 else 1
        self.log(f"🚀 Turbo Çok Çekirdekli Motor devrede ({max_workers} paralel kanal, GPU NVENC aktif)...", 64, callback)

        completed_count = 0
        progress_lock = threading.Lock()

        def process_one_clip(item):
            nonlocal completed_count
            idx, clip = item
            with progress_lock:
                start_pct = int(64 + ((idx - 0.6) / total_clips) * 33)
                self.log(f"Klip {idx}/{total_clips} video verisi çekiliyor ({clip.get('start_formatted', '')} - {clip.get('end_formatted', '')})...", start_pct, callback)
            try:
                clip_data = self.extract_single_clip(
                    source_url=source_url,
                    clip=clip,
                    title_prefix=title_prefix,
                    video_format=video_format,
                    normalize_audio=normalize_audio,
                    enable_teaser_hook=enable_teaser_hook,
                    enable_hook_banner=enable_hook_banner,
                    shorts_layout=shorts_layout,
                    cam_position=cam_position,
                    enable_chat_overlay=enable_chat_overlay,
                    segments=segments,
                    best_stream_url=best_stream_url,
                    video_quality=video_quality
                )
                with progress_lock:
                    completed_count += 1
                    prog = int(65 + (completed_count / total_clips) * 33)
                    self.log(f"Klip {completed_count}/{total_clips} başarıyla tamamlandı ✓ ({clip.get('start_formatted', '')} - {clip.get('end_formatted', '')})", prog, callback)
                return idx, clip_data
            except Exception as err:
                with progress_lock:
                    completed_count += 1
                    prog = int(65 + (completed_count / total_clips) * 33)
                    self.log(f"Uyarı: Klip #{idx} oluşturulamadı: {err}", prog, callback)
                return idx, None

        indexed_items = list(enumerate(clips, 1))
        collected_map = {}

        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_one_clip, it) for it in indexed_items]
                for f in as_completed(futures):
                    idx, res = f.result()
                    if res:
                        collected_map[idx] = res
        else:
            for it in indexed_items:
                idx, res = process_one_clip(it)
                if res:
                    collected_map[idx] = res

        for idx in sorted(collected_map.keys()):
            results.append(collected_map[idx])

        self.log(f"Tebrikler! {len(results)} adet kesit başarıyla hazırlandı.", 100, callback)
        return results

    @staticmethod
    def format_time(seconds: float) -> str:
        sec = int(seconds)
        hrs = sec // 3600
        mins = (sec % 3600) // 60
        s = sec % 60
        if hrs > 0:
            return f"{hrs:02d}:{mins:02d}:{s:02d}"
        return f"{mins:02d}:{s:02d}"

    def fetch_live_stream_info(self, url: str) -> Dict[str, Any]:
        """
        Resolves active live stream metadata and master/direct HLS URL for Kick and YouTube.
        """
        clean_url = url.strip()
        if not clean_url:
            raise ValueError("Lütfen geçerli bir canlı yayın veya kanal linki girin.")

        is_youtube = ("youtube.com" in clean_url.lower() or "youtu.be" in clean_url.lower())

        if is_youtube:
            import yt_dlp
            ydl_opts = self.get_yt_dlp_opts({
                'skip_download': True,
                'extract_flat': False
            })
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(clean_url, download=False)
                    if not info:
                        raise ValueError("YouTube yayın bilgisi alınamadı.")

                    if 'entries' in info and info['entries']:
                        live_entries = [e for e in info['entries'] if e and (e.get('is_live') or e.get('live_status') in ['is_live', 'live'])]
                        info = live_entries[0] if live_entries else info['entries'][0]

                    is_live = bool(info.get('is_live') or info.get('live_status') in ['is_live', 'live'])
                    if not is_live:
                        raise ValueError("Bu YouTube linki şu anda aktif bir CANLI YAYIN değil veya yayın sona ermiş.")

                    playback_url = info.get('url')
                    formats = info.get('formats') or []
                    m3u8_fmts = [f for f in formats if f.get('protocol') in ['m3u8', 'm3u8_native'] or '.m3u8' in f.get('url', '')]
                    if m3u8_fmts:
                        m3u8_fmts.sort(key=lambda x: (x.get('height') or 0, x.get('tbr') or 0), reverse=True)
                        playback_url = m3u8_fmts[0].get('url') or playback_url

                    video_id = info.get('id', '')
                    channel_name = info.get('uploader') or info.get('channel') or 'YouTube Kanalı'
                    title = info.get('title') or 'YouTube Canlı Yayını'
                    thumb = info.get('thumbnail') or (f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg" if video_id else "")
                    viewers = info.get('concurrent_view_count') or 0

                    return {
                        'platform': 'youtube',
                        'channel': channel_name,
                        'title': title,
                        'viewers': viewers,
                        'thumbnail': thumb,
                        'category': 'Canlı Yayın',
                        'playback_url': playback_url or clean_url,
                        'video_id': video_id,
                        'embed_url': f"https://www.youtube.com/embed/{video_id}?autoplay=1&mute=1" if video_id else "",
                        'is_live': True
                    }
            except Exception as e:
                err_str = str(e)
                if "Bu YouTube linki" in err_str:
                    raise
                raise ValueError(f"YouTube canlı yayını bağlanamadı: {err_str}")

        else:
            # Kick Live Stream
            channel_name, _ = self.extract_channel_and_id(clean_url)
            headers = {
                'Accept': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                'Referer': f'https://kick.com/{channel_name}'
            }
            api_url = f"https://kick.com/api/v2/channels/{channel_name}"
            try:
                r = self.session.get(api_url, headers=headers, timeout=12)
                if r.status_code != 200:
                    raise ValueError(f"'{channel_name}' kanalı Kick'te bulunamadı (HTTP {r.status_code}).")
                data = r.json()
                livestream = data.get('livestream')
                if not livestream or not livestream.get('is_live'):
                    raise ValueError(f"'{channel_name}' kanalı şu anda Kick'te CANLI YAYINDA DEĞİL.")

                playback_url = data.get('playback_url')
                if not playback_url:
                    raise ValueError(f"'{channel_name}' canlı yayın video akışı (playback_url) alınamadı.")

                thumb_obj = livestream.get('thumbnail') or {}
                thumb_url = thumb_obj.get('url', '') if isinstance(thumb_obj, dict) else str(thumb_obj)
                if not thumb_url:
                    banner = data.get('banner_image') or {}
                    thumb_url = banner.get('url', '') if isinstance(banner, dict) else ''

                cat_list = livestream.get('categories') or []
                cat_name = cat_list[0].get('name', 'Genel') if cat_list and isinstance(cat_list[0], dict) else 'Canlı Yayın'

                return {
                    'platform': 'kick',
                    'channel': channel_name,
                    'title': livestream.get('session_title') or f"{channel_name} Canlı Yayını",
                    'viewers': livestream.get('viewer_count') or 0,
                    'thumbnail': thumb_url,
                    'category': cat_name,
                    'playback_url': playback_url,
                    'channel_slug': channel_name,
                    'embed_url': f"https://player.kick.com/{channel_name}?autoplay=true&muted=true",
                    'is_live': True
                }
            except Exception as e:
                err_str = str(e)
                if "CANLI YAYINDA DEĞİL" in err_str or "bulunamadı" in err_str:
                    raise
                raise ValueError(f"Kick canlı yayını bağlanamadı: {err_str}")

    def extract_live_clip(
        self,
        playback_url: str,
        channel: str = "CanliYayin",
        title: Optional[str] = "Canlı Yayın Klibi",
        duration: int = 30,
        mode: str = "buffer",
        video_format: str = "vertical_9_16",
        shorts_layout: str = "blur",
        cam_position: str = "top_right",
        enable_chat_overlay: bool = False,
        normalize_audio: bool = True,
        enable_hook_banner: bool = False,
        hook_text: Optional[str] = None,
        video_quality: str = "1080p60",
        callback: Optional[Callable[[str, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Extracts an instant clip from a live stream either from the immediate past HLS buffer
        or by recording the live stream for the specified duration.
        """
        clean_channel = "".join(c for c in channel if c.isalnum() or c in (' ', '_', '-')).rstrip().replace(' ', '_')
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        time_tag = time.strftime("%H-%M-%S")
        clip_uid = uuid.uuid4().hex[:6]

        raw_temp_name = f"live_raw_{clip_uid}_{int(time.time())}.mp4"
        raw_temp_path = os.path.join(self.temp_dir, raw_temp_name)

        out_filename = f"{clean_channel}_CANLI_{time_tag}.mp4"
        out_filepath = os.path.join(self.output_dir, out_filename)

        if callback: callback("Canlı yayın akışı yakalanıyor...", 15)

        captured = False

        # Attempt buffer extraction if requested
        if mode == "buffer":
            try:
                s = get_cdn_session()
                r_m = s.get(playback_url, timeout=6)
                if r_m.status_code == 200:
                    lines = r_m.text.splitlines()
                    sub_url = None
                    for i, l in enumerate(lines):
                        if 'RESOLUTION=1920x1080' in l:
                            sub_url = lines[i+1].strip()
                            break
                    if not sub_url:
                        for l in lines:
                            if l.startswith('http'):
                                sub_url = l
                                break
                    if not sub_url and playback_url.endswith('.m3u8'):
                        sub_url = playback_url

                    if sub_url:
                        r_s = s.get(sub_url, timeout=6)
                        if r_s.status_code == 200:
                            sub_lines = r_s.text.splitlines()
                            segments = []
                            cur_dur = 2.0
                            for l in sub_lines:
                                if l.startswith('#EXTINF:'):
                                    try: cur_dur = float(l.split(':')[1].split(',')[0])
                                    except: cur_dur = 2.0
                                elif l.startswith('http'):
                                    segments.append((l, cur_dur))

                            if segments:
                                needed_segs = []
                                acc = 0.0
                                for seg_url, seg_d in reversed(segments):
                                    needed_segs.append(seg_url)
                                    acc += seg_d
                                    if acc >= float(duration):
                                        break
                                needed_segs.reverse()

                                if callback: callback(f"{len(needed_segs)} adet canlı video parçası indiriliyor...", 35)

                                def fetch_item(item):
                                    idx, u = item
                                    for _ in range(3):
                                        try:
                                            res = s.get(u, timeout=(3, 8))
                                            if res.status_code == 200 and len(res.content) > 0:
                                                return idx, res.content
                                        except Exception:
                                            time.sleep(0.1)
                                    return idx, b''

                                with ThreadPoolExecutor(max_workers=min(20, len(needed_segs))) as executor:
                                    results = dict(executor.map(fetch_item, enumerate(needed_segs)))

                                temp_ts = os.path.join(self.temp_dir, f"live_buf_{clip_uid}.ts")
                                with open(temp_ts, "wb") as f_out:
                                    for i in sorted(results.keys()):
                                        if results[i]:
                                            f_out.write(results[i])

                                if os.path.exists(temp_ts) and os.path.getsize(temp_ts) > 5000:
                                    remux_cmd = [get_ffmpeg_binary(), '-y', '-i', temp_ts, '-c', 'copy', raw_temp_path]
                                    res_rm = subprocess.run(remux_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
                                    if res_rm.returncode == 0 and os.path.exists(raw_temp_path) and os.path.getsize(raw_temp_path) > 1000:
                                        captured = True
                                    try: os.remove(temp_ts)
                                    except: pass
            except Exception as e:
                print(f"HLS buffer direct slice notice: {e}")

            # Fallback to ffmpeg sseof
            if not captured:
                if callback: callback("FFmpeg canlı arabelleğinden kesiliyor...", 30)
                cmd_buf = [
                    get_ffmpeg_binary(), '-y',
                    '-sseof', f"-{duration}",
                    '-i', playback_url,
                    '-t', str(duration),
                    '-c', 'copy',
                    raw_temp_path
                ]
                res_buf = subprocess.run(cmd_buf, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=max(30, duration + 15))
                if res_buf.returncode == 0 and os.path.exists(raw_temp_path) and os.path.getsize(raw_temp_path) > 1000:
                    captured = True

        # Mode == "live" or buffer fallback
        if not captured:
            if callback: callback(f"Canlı yayın {duration} saniye boyunca kaydediliyor...", 40)
            cmd_rec = [
                get_ffmpeg_binary(), '-y',
                '-i', playback_url,
                '-t', str(duration),
                '-c', 'copy',
                raw_temp_path
            ]
            res_rec = subprocess.run(cmd_rec, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=duration + 30)
            if res_rec.returncode == 0 and os.path.exists(raw_temp_path) and os.path.getsize(raw_temp_path) > 1000:
                captured = True

        if not captured or not os.path.exists(raw_temp_path) or os.path.getsize(raw_temp_path) == 0:
            raise ValueError("Canlı yayından görüntü akışı yakalanamadı. Yayın bağlantısını ve durumunu kontrol edin.")

        if callback: callback("GPU Donanım Hızlandırma ile video işleniyor...", 70)

        v_enc = get_hardware_video_encoder(target_crf=18)
        af_filter = 'loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000:async=1000:first_pts=0' if normalize_audio else 'aresample=48000:async=1000:first_pts=0'

        tc_cmd = [
            get_ffmpeg_binary(), '-y',
            '-i', raw_temp_path,
            '-map', '0:v:0',
            '-map', '0:a:0?',
            *v_enc,
            '-c:a', 'aac', '-b:a', '192k', '-ar', '48000',
            '-af', af_filter,
            '-fps_mode', 'cfr',
            '-avoid_negative_ts', 'make_zero',
            '-movflags', '+faststart',
            out_filepath
        ]
        subprocess.run(tc_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=90)
        try: os.remove(raw_temp_path)
        except: pass

        if not os.path.exists(out_filepath) or os.path.getsize(out_filepath) == 0:
            raise ValueError("Canlı klip kaydedilirken dönüştürme hatası oluştu.")

        self.ensure_perfect_av_sync(out_filepath)

        size_mb = round(os.path.getsize(out_filepath) / (1024 * 1024), 2)
        clip_data = {
            "id": int(time.time()),
            "title": title or f"{clean_channel} Canlı Klip",
            "filename": out_filename,
            "filepath": out_filepath,
            "duration": duration,
            "duration_formatted": self.format_time(duration),
            "size_mb": size_mb,
            "uploader": clean_channel,
            "has_916": False,
            "vertical_filename": None,
            "vertical_filepath": None,
            "vertical_size_mb": None,
            "thumbnail": None,
            "created_at": time.strftime("%d.%m.%Y %H:%M:%S")
        }

        # Vertical 9:16 Shorts Conversion
        if video_format in ["vertical_9_16", "both"]:
            if callback: callback("📱 9:16 Shorts formatı hazırlanıyor...", 85)
            try:
                vert_filename = f"{clean_channel}_CANLI_{time_tag}_916.mp4"
                vert_filepath = os.path.join(self.output_dir, vert_filename)
                banner_txt = hook_text or "CANLI YAYINDA OLAY AN 🔥"

                self.convert_to_vertical_916(
                    input_filepath=out_filepath,
                    output_filepath=vert_filepath,
                    normalize_audio=normalize_audio,
                    enable_teaser_hook=False,
                    clip_duration=float(duration),
                    enable_hook_banner=enable_hook_banner,
                    hook_text=banner_txt,
                    shorts_layout=shorts_layout,
                    cam_position=cam_position,
                    enable_chat_overlay=enable_chat_overlay,
                    moment_type="hype",
                    video_quality=video_quality
                )
                if os.path.exists(vert_filepath) and os.path.getsize(vert_filepath) > 0:
                    clip_data["has_916"] = True
                    clip_data["vertical_filename"] = vert_filename
                    clip_data["vertical_filepath"] = vert_filepath
                    clip_data["vertical_size_mb"] = round(os.path.getsize(vert_filepath) / (1024 * 1024), 2)
            except Exception as e:
                print(f"Live clip 9:16 conversion notice: {e}")

        # Thumbnail generation
        thumb_name = f"thumb_live_{clip_uid}.jpg"
        thumb_path = os.path.join(self.output_dir, thumb_name)
        try:
            self.generate_clip_thumbnail(
                video_filepath=vert_filepath if clip_data.get('has_916') else out_filepath,
                output_thumb_path=thumb_path,
                title_text="CANLI YAYIN ANI",
                streamer_name=clean_channel,
                moment_type="hype",
                target_format="916" if clip_data.get('has_916') else "169",
                frame_sec=max(1.0, duration / 2.0)
            )
            if os.path.exists(thumb_path):
                clip_data["thumbnail"] = thumb_name
        except Exception:
            pass

        if callback: callback("🎉 Canlı klip başarıyla oluşturuldu!", 100)
        return clip_data

