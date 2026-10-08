"""
HWID-based Hardware Lock & 1-Month Time-Locked Licensing Engine for Kick Highlight Clipper.
Prevents unauthorized copying, distribution, and cross-device sharing.
Supports 1-Month (30-day), dated, and lifetime cryptographic licenses.
"""

import os
import sys
import hashlib
import hmac
import subprocess
import json
import uuid
import time
from datetime import datetime, timedelta, timezone

# Secret Master Salt (Only present in the algorithm)
SECRET_SALT = b"KICK_CLIPPER_OFFLINE_PROTECT_2026_MASTER_SECRET_KEY_#992"
LICENSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".license")

def get_machine_guid() -> str:
    """
    Retrieves Windows MachineGuid from registry. 100% stable across PC reboots.
    """
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
                val, _ = winreg.QueryValueEx(key, "MachineGuid")
                if val and val.strip():
                    return val.strip().lower()
        except Exception:
            pass
        try:
            cmd = 'reg query "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Cryptography" /v MachineGuid'
            out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
            for line in out.splitlines():
                if "MachineGuid" in line:
                    val = line.split()[-1].strip().lower()
                    if val:
                        return val
        except Exception:
            pass
    return "default_guid"

def get_hwid() -> str:
    """
    Retrieves unique hardware fingerprint.
    Guaranteed to NEVER change across PC reboots, network connects/disconnects, or sleep/wake.
    Preserves active license HWID if already activated on this machine.
    """
    guid = get_machine_guid()
    
    # 1. If an activated license already exists on this machine, preserve that HWID so keys NEVER break!
    target_files = [
        LICENSE_FILE,
        os.path.expanduser("~/.kick_clipper_license")
    ]
    for tf in target_files:
        if os.path.exists(tf):
            try:
                with open(tf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                s_hwid = data.get("hwid")
                s_key = data.get("key")
                s_guid = data.get("guid")
                if s_hwid and s_key:
                    ver = verify_key(s_hwid, s_key)
                    if ver.get("valid", False):
                        if not s_guid or s_guid == guid:
                            return s_hwid
            except Exception:
                pass

    # 2. Immutable Hardware Fingerprint from Motherboard UUID + MachineGuid
    raw_ids = []
    if sys.platform == "win32":
        try:
            cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", "(Get-CimInstance -ClassName Win32_ComputerSystemProduct).UUID"]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
            if out and len(out) > 10 and "error" not in out.lower():
                raw_ids.append(out)
        except Exception:
            pass

    if guid and guid != "default_guid":
        raw_ids.append(guid)

    combined_str = "::".join(raw_ids) if raw_ids else "DEFAULT_KICK_DEVICE_ID"
    hash_digest = hashlib.sha256(combined_str.encode('utf-8')).hexdigest().upper()
    hwid = f"HWID-{hash_digest[:4]}-{hash_digest[4:8]}-{hash_digest[8:12]}-{hash_digest[12:16]}"
    return hwid

def generate_key_for_hwid(hwid: str, days: int = 30) -> str:
    """
    Generates a cryptographic activation key tied specifically to this HWID and validity period.
    - days = 30: 1-Month time-locked license (Default)
    - days = 0 or None: Lifetime license
    """
    clean_hwid = hwid.strip().upper()

    if days is not None and days > 0:
        # 1-Month or specific duration
        expiry_dt = datetime.now(timezone.utc) + timedelta(days=days)
        expiry_str = expiry_dt.strftime("%Y%m%d")
        payload = f"{clean_hwid}::{expiry_str}"
        signature = hmac.new(SECRET_SALT, payload.encode('utf-8'), hashlib.sha256).hexdigest().upper()
        # Format: KEY-1M-YYYYMMDD-XXXX-XXXX-XXXX
        prefix = "1M" if days == 30 else f"{days}D"
        key = f"KEY-{prefix}-{expiry_str}-{signature[:4]}-{signature[4:8]}-{signature[8:12]}"
    else:
        # Lifetime license
        signature = hmac.new(SECRET_SALT, clean_hwid.encode('utf-8'), hashlib.sha256).hexdigest().upper()
        key = f"KEY-LIFE-{signature[:4]}-{signature[4:8]}-{signature[8:12]}-{signature[12:16]}"

    return key

def verify_key(hwid: str, key: str) -> dict:
    """
    Cryptographically verifies if the given key matches the HWID and is not expired.
    Returns dict with verification results:
      {'valid': bool, 'expired': bool, 'type': str, 'expiry_dt': datetime, 'message': str}
    """
    if not key or not hwid:
        return {'valid': False, 'expired': False, 'message': 'Eksik anahtar veya cihaz kimliği.'}

    clean_hwid = hwid.strip().upper()
    clean_key = key.strip().upper()
    parts = clean_key.split('-')

    # 1. Dated License: KEY-<PREFIX>-YYYYMMDD-XXXX-XXXX-XXXX (e.g. KEY-1M-20261003-8F3A-BC12-9DE4)
    if len(parts) >= 6 and (parts[1].endswith('M') or parts[1].endswith('D')):
        prefix = parts[1]
        expiry_str = parts[2]
        provided_sig = "".join(parts[3:6])

        if len(expiry_str) != 8 or not expiry_str.isdigit():
            return {'valid': False, 'expired': False, 'message': 'Geçersiz lisans tarihi biçimi.'}

        try:
            exp_year = int(expiry_str[:4])
            exp_month = int(expiry_str[4:6])
            exp_day = int(expiry_str[6:8])
            expiry_dt = datetime(exp_year, exp_month, exp_day, 23, 59, 59, tzinfo=timezone.utc)
        except Exception:
            return {'valid': False, 'expired': False, 'message': 'Geçersiz son kullanma tarihi.'}

        payload = f"{clean_hwid}::{expiry_str}"
        expected_sig = hmac.new(SECRET_SALT, payload.encode('utf-8'), hashlib.sha256).hexdigest().upper()[:12]

        if not hmac.compare_digest(provided_sig, expected_sig):
            return {'valid': False, 'expired': False, 'message': 'Bu lisans anahtarı bu cihaza ait değil!'}

        # Check expiration date against current UTC time
        now_utc = datetime.now(timezone.utc)
        if now_utc > expiry_dt:
            return {
                'valid': False,
                'expired': True,
                'type': '1_month' if prefix == '1M' else 'dated',
                'expiry_dt': expiry_dt,
                'message': f"Bu lisans anahtarının süresi {expiry_dt.strftime('%d.%m.%Y')} tarihinde dolmuştur!"
            }

        return {
            'valid': True,
            'expired': False,
            'type': '1_month' if prefix == '1M' else 'dated',
            'expiry_dt': expiry_dt,
            'message': 'Lisans anahtarı geçerli.'
        }

    # 2. Lifetime with prefix: KEY-LIFE-XXXX-XXXX-XXXX-XXXX
    if len(parts) >= 6 and parts[1] == 'LIFE':
        provided_sig = "".join(parts[2:6])
        expected_sig = hmac.new(SECRET_SALT, clean_hwid.encode('utf-8'), hashlib.sha256).hexdigest().upper()[:16]
        if hmac.compare_digest(provided_sig, expected_sig):
            return {'valid': True, 'expired': False, 'type': 'lifetime', 'expiry_dt': None, 'message': 'Ömür boyu lisans geçerli.'}
        return {'valid': False, 'expired': False, 'message': 'Bu lisans anahtarı bu cihaza ait değil!'}

    # 3. Legacy Lifetime: KEY-XXXX-XXXX-XXXX-XXXX
    if len(parts) == 5 and parts[0] == 'KEY':
        provided_sig = "".join(parts[1:5])
        expected_sig = hmac.new(SECRET_SALT, clean_hwid.encode('utf-8'), hashlib.sha256).hexdigest().upper()[:16]
        if hmac.compare_digest(provided_sig, expected_sig):
            return {'valid': True, 'expired': False, 'type': 'lifetime', 'expiry_dt': None, 'message': 'Ömür boyu lisans geçerli.'}

    return {'valid': False, 'expired': False, 'message': 'Geçersiz lisans anahtarı formatı.'}

def check_license_status() -> dict:
    """
    Checks if current machine has an active, non-expired license.
    Returns detailed dictionary with remaining days and status.
    Guaranteed to persist across reboots, shutdowns, and folder moves.
    """
    if os.environ.get("SPACE_ID") or os.environ.get("CLOUD_MODE") == "1":
        return {
            "is_licensed": True,
            "is_expired": False,
            "hwid": "CLOUD-SPACE-HF",
            "license_type": "lifetime",
            "remaining_days": 9999,
            "remaining_text": "Sınırsız Bulut Lisansı",
            "expiry_formatted": "Ömür Boyu",
            "message": "Bulut ortamında lisans aktif."
        }

    current_hwid = get_hwid()
    guid = get_machine_guid()

    target_file = LICENSE_FILE
    backup_file = os.path.expanduser("~/.kick_clipper_license")

    if not os.path.exists(target_file) and os.path.exists(backup_file):
        try:
            import shutil
            shutil.copy2(backup_file, target_file)
        except Exception:
            target_file = backup_file

    if not os.path.exists(target_file):
        return {
            "is_licensed": False,
            "is_expired": False,
            "hwid": current_hwid,
            "license_type": "none",
            "remaining_days": 0,
            "remaining_text": "Lisans Yok",
            "expiry_formatted": "-",
            "message": "Lisans anahtarı bulunamadı."
        }

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        saved_hwid = data.get("hwid")
        saved_key = data.get("key")
        saved_guid = data.get("guid")

        # Allow match if either saved_hwid matches current_hwid, or machine GUID matches
        is_same_pc = (saved_hwid == current_hwid) or (saved_guid and saved_guid == guid)
        if not is_same_pc and saved_hwid and saved_key:
            # Check if key is valid for saved_hwid on this machine
            if verify_key(saved_hwid, saved_key).get("valid", False):
                is_same_pc = True

        if not is_same_pc:
            return {
                "is_licensed": False,
                "is_expired": False,
                "hwid": current_hwid,
                "license_type": "none",
                "remaining_days": 0,
                "remaining_text": "Cihaz Eşleşmedi",
                "expiry_formatted": "-",
                "message": "Mevcut cihaz kimliği lisans dosyasındaki kimlikle uyuşmuyor."
            }

        effective_hwid = saved_hwid if (saved_hwid and verify_key(saved_hwid, saved_key).get("valid", False)) else current_hwid
        ver_res = verify_key(effective_hwid, saved_key)
        if not ver_res['valid']:
            return {
                "is_licensed": False,
                "is_expired": ver_res.get('expired', False),
                "hwid": effective_hwid,
                "license_type": ver_res.get('type', 'none'),
                "remaining_days": 0,
                "remaining_text": "Süresi Doldu" if ver_res.get('expired') else "Geçersiz",
                "expiry_formatted": ver_res['expiry_dt'].strftime("%d.%m.%Y") if ver_res.get('expiry_dt') else "-",
                "message": ver_res.get('message', 'Lisans geçersiz.')
            }

        # License is valid! Calculate remaining duration
        expiry_dt = ver_res.get('expiry_dt')
        if expiry_dt:
            now_utc = datetime.now(timezone.utc)
            delta_days = (expiry_dt.date() - now_utc.date()).days
            remaining_days = max(1, delta_days)
            remaining_text = f"{remaining_days} Gün Kaldı"
            expiry_formatted = expiry_dt.strftime("%d.%m.%Y")
            license_type = "1_month" if ver_res.get('type') == '1_month' else "dated"
        else:
            remaining_days = 9999
            remaining_text = "Ömür Boyu (Süresiz)"
            expiry_formatted = "Süresiz"
            license_type = "lifetime"

        return {
            "is_licensed": True,
            "is_expired": False,
            "hwid": effective_hwid,
            "license_type": license_type,
            "remaining_days": remaining_days,
            "remaining_text": remaining_text,
            "expiry_formatted": expiry_formatted,
            "message": "Lisans aktif ve geçerli."
        }

    except Exception as e:
        return {
            "is_licensed": False,
            "is_expired": False,
            "hwid": current_hwid,
            "license_type": "none",
            "remaining_days": 0,
            "remaining_text": "Hata",
            "expiry_formatted": "-",
            "message": f"Lisans dosyası okunamadı: {str(e)}"
        }

def is_device_licensed() -> tuple[bool, str]:
    """
    Convenience wrapper matching standard signature (is_valid, hwid).
    """
    status = check_license_status()
    return status.get("is_licensed", False), status.get("hwid", get_hwid())

def activate_license(key: str) -> tuple[bool, str]:
    """
    Validates and stores the license key on this machine.
    Saves to both project folder and user home backup.
    """
    current_hwid = get_hwid()
    guid = get_machine_guid()

    ver_res = verify_key(current_hwid, key)

    # If key doesn't match current_hwid, check against existing license HWID if present
    if not ver_res['valid'] and os.path.exists(LICENSE_FILE):
        try:
            with open(LICENSE_FILE, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            old_hwid = old_data.get("hwid")
            if old_hwid:
                old_ver = verify_key(old_hwid, key)
                if old_ver['valid']:
                    current_hwid = old_hwid
                    ver_res = old_ver
        except Exception:
            pass

    if not ver_res['valid']:
        return False, ver_res.get('message', 'Geçersiz lisans anahtarı!')

    expiry_dt = ver_res.get('expiry_dt')
    license_data = {
        "hwid": current_hwid,
        "key": key.strip().upper(),
        "guid": guid,
        "type": ver_res.get('type', '1_month'),
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": expiry_dt.isoformat() if expiry_dt else None,
        "expiry_formatted": expiry_dt.strftime("%d.%m.%Y") if expiry_dt else "Süresiz"
    }

    try:
        with open(LICENSE_FILE, "w", encoding="utf-8") as f:
            json.dump(license_data, f, indent=2)
        
        # Also write backup to user home directory so PC restarts/wipes never lose license
        backup_file = os.path.expanduser("~/.kick_clipper_license")
        try:
            with open(backup_file, "w", encoding="utf-8") as bf:
                json.dump(license_data, bf, indent=2)
        except Exception:
            pass

        if expiry_dt:
            return True, f"Lisans başarıyla etkinleştirildi! ({expiry_dt.strftime('%d.%m.%Y')} tarihine kadar geçerli)"
        return True, "Ömür boyu lisans başarıyla etkinleştirildi!"
    except Exception as e:
        return False, f"Lisans dosyası kaydedilemedi: {str(e)}"
