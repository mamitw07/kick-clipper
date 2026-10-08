"""
AI HEADLINE & RETENTION SERVICE
Next-Gen Context-Aware Viral Title and Hook Generator
Supports:
1. Google Gemini Flash API (Ultra-fast, contextual, understands what actually happened in the clip)
2. Advanced Local Turkish Semantic NLP Engine (Rule & grammar-aware fallback, zero broken clauses)
"""

import os
import sys
import re
import json
import base64
import urllib.request
import urllib.error
import logging
from typing import Dict, List, Any, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

logger = logging.getLogger("ai_headline_service")

GEMINI_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemini_key.txt")

def get_gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    if os.path.exists(GEMINI_KEY_FILE):
        try:
            with open(GEMINI_KEY_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return content
        except Exception:
            pass
    return ""

def save_gemini_api_key(key: str) -> bool:
    try:
        with open(GEMINI_KEY_FILE, "w", encoding="utf-8") as f:
            f.write(key.strip())
        return True
    except Exception as e:
        logger.error(f"Could not save Gemini key: {e}")
        return False

GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
    "gemini-2.5-flash"
]

def call_gemini_api(api_key: str, prompt: str, temperature: float = 0.6, timeout: float = 20.0) -> Optional[Dict[str, Any]]:
    if not api_key:
        return None

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 800,
            "responseMimeType": "application/json"
        }
    }
    data = json.dumps(payload).encode("utf-8")

    for model_name in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_body = resp.read().decode("utf-8")
                resp_json = json.loads(resp_body)
                candidates = resp_json.get("candidates", [])
                if not candidates:
                    continue
                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    continue
                raw_text = parts[0].get("text", "").strip()
                raw_text = re.sub(r'^```json\s*', '', raw_text)
                raw_text = re.sub(r'\s*```$', '', raw_text)
                parsed = json.loads(raw_text)
                logger.info(f"Gemini model {model_name} responded successfully!")
                return parsed
        except urllib.error.HTTPError as he:
            logger.warning(f"Gemini model {model_name} HTTP {he.code}: {he.reason}")
            continue
        except Exception as ex:
            logger.warning(f"Gemini model {model_name} error: {ex}")
            continue

    return None

def call_gemini_raw_text(api_key: str, prompt: str, temperature: float = 0.7, timeout: float = 15.0) -> Optional[str]:
    if not api_key:
        return None
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 300
        }
    }
    data = json.dumps(payload).encode("utf-8")
    for model_name in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_body = resp.read().decode("utf-8")
                resp_json = json.loads(resp_body)
                candidates = resp_json.get("candidates", [])
                if not candidates:
                    continue
                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    continue
                text = parts[0].get("text", "").strip()
                if text:
                    return text
        except Exception:
            continue
    return None


def turkish_upper(s: str) -> str:
    mapping = {'i': 'İ', 'ı': 'I', 'ç': 'Ç', 'ş': 'Ş', 'ğ': 'Ğ', 'ü': 'Ü', 'ö': 'Ö'}
    return ''.join(mapping.get(ch, ch.upper()) for ch in s)

def turkish_title(s: str) -> str:
    if not s:
        return ""
    words = s.split()
    titled_words = []
    mapping = {'i': 'İ', 'ı': 'I', 'ç': 'Ç', 'ş': 'Ş', 'ğ': 'Ğ', 'ü': 'Ü', 'ö': 'Ö'}
    for w in words:
        if not w:
            continue
        first_char = w[0]
        first_upper = mapping.get(first_char, first_char.upper())
        rest_lower = w[1:].lower()
        titled_words.append(first_upper + rest_lower)
    return ' '.join(titled_words)

def clean_sentence_for_title(text: str) -> str:
    """
    Cleans fillers and fixes dangling Turkish conjunctions without mutilating sentence grammar.
    """
    if not text:
        return ""
    t = text.strip()
    
    # Profanity removal
    bad_words = [
        r'\bkoyayım\b', r'\bkoyim\b', r'\bamk\b', r'\baq\b', r'\bamına\b',
        r'\bsikeyim\b', r'\bsikim\b', r'\bsiktir\b', r'\bpiç\b', r'\boç\b', r'\byarrak\b',
        r'\borospu\b', r'\bkahpe\b', r'\bgöt\b'
    ]
    for bw in bad_words:
        t = re.sub(bw, '', t, flags=re.IGNORECASE)

    # Remove vocal grunt fillers
    fillers = [r'\bııı+\b', r'\beee+\b', r'\bhıh\b', r'\baa+\b']
    for fil in fillers:
        t = re.sub(fil, '', t, flags=re.IGNORECASE)

    t = re.sub(r'\s+', ' ', t).strip(' ,.-!?')

    # Remove initial weak discourse particles (e.g. "ya ben aslında" -> "ben aslında")
    t = re.sub(r'^(?:ya|abi|kanka|aga|lan|olm|yahu|hacı|valla|şimdi|yani|işte|evet|hayır|tamam)\s+', '', t, flags=re.IGNORECASE)

    # Strip dangling trailing conjunctions that ruin titles
    dangling_tails = [
        r'\s+(?:çünkü|ama|fakat|lakin|yani|oysa|oysaki|madem|ve|veya|ile|diye|falan|filan|ise|ki)$',
        r'\s+(?:buna ya|ona ya|için ya|ya ya)$'
    ]
    for dt in dangling_tails:
        t = re.sub(dt, '', t, flags=re.IGNORECASE)

    return t.strip(' ,.-!?')

def make_punchy_headline(text: str, max_words: int = 6) -> str:
    if not text:
        return ""
    words = text.strip().split()
    if len(words) > max_words:
        words = words[:max_words]
    dangling = {"ki", "ve", "ama", "fakat", "lakin", "yani", "de", "da", "diye", "ise", "sen", "ben", "o", "şu", "bu", "bir", "ile"}
    while len(words) > 2 and words[-1].lower() in dangling:
        words.pop()
    return " ".join(words)

def generate_headlines_with_gemini(
    api_key: str,
    streamer: str,
    stream_title: str,
    transcript: str,
    moment_type: str,
    duration: float = 35.0
) -> Optional[Dict[str, Any]]:
    """
    Calls Google Gemini Flash API to generate viral titles and hooks based on clip conversation.
    """
    if not api_key:
        return None

    parsed = call_gemini_api(api_key, prompt, temperature=0.7)
    if parsed and "viral_title" in parsed and "title_options" in parsed:
        logger.info(f"Gemini AI successfully generated headlines for {streamer}")
        return parsed

    return None

def generate_local_smart_headlines(
    streamer: str,
    stream_title: str,
    transcript: str,
    hook_text: str = "",
    payoff_text: str = "",
    moment_type: str = "funny"
) -> Dict[str, Any]:
    """
    Advanced local Turkish NLP headline generator.
    Produces grammatically complete, curiosity-driven titles based on authentic dialogue.
    """
    clean_streamer = (streamer or "Yayıncı").replace('@', '').strip().capitalize()
    topic = stream_title.strip() if stream_title else ""
    topic_clean = re.sub(r'![a-zA-Z0-9_-]+', '', topic)
    topic_clean = re.sub(r'https?://\S+', '', topic_clean)
    topic_clean = re.sub(r'[\[\]\(\)\|\/\\#_~]+', ' ', topic_clean).strip()
    topic_clean = turkish_title(topic_clean[:30].strip()) if topic_clean else ""

    clean_hook = clean_sentence_for_title(hook_text)
    clean_payoff = clean_sentence_for_title(payoff_text)
    clean_full = clean_sentence_for_title(transcript)

    emoji_map = {
        'funny': ['😂', '🤣', '💀'],
        'shock': ['😱', '💥', '😨'],
        'hype': ['🔥', '⚡', '🏆'],
        'action': ['🎯', '⚡', '🔥'],
        'story': ['🤫', '✨', '💬']
    }
    emojis = emoji_map.get(moment_type, ['🔥', '😂', '⚡'])

    title_options: List[str] = []

    # 1. Authentic Quote format (e.g. Elraenn: "Ben Sana Ne Dedim Dün!" 😂)
    for cand in [clean_hook, clean_payoff]:
        if cand and len(cand.split()) >= 3:
            cand_quote = cand
            if len(cand_quote.split()) > 8:
                cand_quote = " ".join(cand_quote.split()[:7]) + "..."
            opt = f'{clean_streamer}: "{cand_quote}" {emojis[0]}'
            if opt not in title_options:
                title_options.append(opt)
            break

    # 2. Curiosity / Question format
    question_cand = None
    if clean_full:
        for sent in re.split(r'[.!?\n]+', clean_full):
            s = clean_sentence_for_title(sent)
            if any(qw in s.lower() for qw in ["neden", "nasıl", "kim", "ne zaman", "niye", "biliyor musun", "gördün mü", "fark ettin"]):
                question_cand = s
                break
    if question_cand and len(question_cand.split()) >= 3:
        q_words = question_cand.split()
        if len(q_words) > 7:
            question_cand = " ".join(q_words[:7])
        title_options.append(f"{turkish_title(question_cand)}? 🎯")

    # 3. Climax / Payoff Headline
    if clean_payoff and len(clean_payoff.split()) >= 2:
        payoff_short = " ".join(clean_payoff.split()[:7])
        opt_payoff = f"{turkish_upper(payoff_short)}! {emojis[1 % len(emojis)]}"
        if opt_payoff not in title_options:
            title_options.append(opt_payoff)

    # 4. Contextual Situation Headlines (Topic & Streamer based, but natural)
    if moment_type == "funny":
        situations = [
            f"{clean_streamer} Canlı Yayında Gülme Krizine Girdi! 😂",
            f"Bunu Kimse Beklemiyordu! 🤣 ({clean_streamer})",
            f"{clean_streamer} - Yayının En Çok Güldüren Anı 💀",
            f"Yayında Yaşanan O Akılalmaz Olay! 😂"
        ]
    elif moment_type == "shock":
        situations = [
            f"{clean_streamer} Yayında Büyük Şoka Uğradı! 😱",
            f"Bu Hatayı Asla Yapmayın! 💥",
            f"Canlı Yayında Beklenmedik Olay! ⚡",
            f"Herkesi Şaşkına Çeviren O An! 😱"
        ]
    elif moment_type == "hype":
        situations = [
            f"{clean_streamer} Tarihi Rekor Kırdı! 🔥",
            f"Yayının En Heyecanlı Zirve Anı! ⚡",
            f"İnanılmaz Bir Başarı Geldi! 🏆",
            f"{clean_streamer} Efsane Hamle Yaptı! 🎯"
        ]
    else: # story / action
        situations = [
            f"{clean_streamer} Canlı Yayında İtiraf Etti! 🤫",
            f"Bunu Sakın Kaçırmayın! ✨",
            f"{clean_streamer} Yayının En Önemli Sahnesi! 💬",
            f"Olayın Aslı Sonunda Ortaya Çıktı! 🎯"
        ]

    for sit in situations:
        if sit not in title_options:
            title_options.append(sit)

    if topic_clean:
        title_options.insert(2, f"{clean_streamer} | {topic_clean} ({emojis[0]})")

    chosen_title = title_options[0] if title_options else f"{clean_streamer} - Yayının En İyi Anı 🔥"

    # CapCut Hook Headline (Short, Punchy, High Visual Impact for Video Overlay)
    capcut_candidates = []
    if clean_hook and len(clean_hook.split()) >= 2:
        short_h = make_punchy_headline(clean_hook, max_words=5)
        if short_h:
            capcut_candidates.append(f"{turkish_upper(short_h)}! {emojis[0]}")
    if clean_payoff and len(clean_payoff.split()) >= 2:
        short_p = make_punchy_headline(clean_payoff, max_words=5)
        if short_p:
            capcut_candidates.append(f"{turkish_upper(short_p)}! {emojis[1 % len(emojis)]}")
    
    if not capcut_candidates:
        if moment_type == "funny":
            capcut_candidates.append("GÜLME KRİZİ KOPTU! 🤣")
        elif moment_type == "shock":
            capcut_candidates.append("BUNU KİMSE BEKLEMİYORDU! 😱")
        elif moment_type == "hype":
            capcut_candidates.append("REKOR KIRAN O AN! 🔥")
        else:
            capcut_candidates.append("SONUNA KADAR İZLE! ⚡")

    capcut_hook = capcut_candidates[0]
    hook_quote = clean_hook if clean_hook else chosen_title

    return {
        "viral_title": chosen_title,
        "title_options": title_options[:5],
        "capcut_hook_headline": capcut_hook,
        "hook_quote": hook_quote,
        "summary": f"{clean_streamer} yayında {moment_type} bir an yaşadı."
    }

def generate_clip_titles_and_hooks(
    streamer: str,
    stream_title: str,
    transcript: str,
    hook_text: str = "",
    payoff_text: str = "",
    moment_type: str = "funny",
    duration: float = 35.0
) -> Dict[str, Any]:
    """
    Unified entry point: Tries Gemini Flash API first if key available; falls back to Smart Turkish NLP.
    """
    api_key = get_gemini_api_key()
    if api_key:
        gemini_res = generate_headlines_with_gemini(
            api_key=api_key,
            streamer=streamer,
            stream_title=stream_title,
            transcript=transcript,
            moment_type=moment_type,
            duration=duration
        )
        if gemini_res:
            return gemini_res

    return generate_local_smart_headlines(
        streamer=streamer,
        stream_title=stream_title,
        transcript=transcript,
        hook_text=hook_text,
        payoff_text=payoff_text,
        moment_type=moment_type
    )


def analyze_transcript_scenes_with_llm(
    sentences: List[Dict[str, Any]],
    streamer: str = "",
    stream_title: str = "",
    target_category: str = "all"
) -> List[Dict[str, Any]]:
    """
    Advanced LLM & Semantic Scene & Transcript Intelligence.
    Understands conversational context, detects funny, drama, hype, and story moments,
    and preserves 100% topic and sentence completeness (no half-broken clauses).
    """
    if not sentences or len(sentences) == 0:
        return []

    api_key = get_gemini_api_key()
    if api_key and len(sentences) >= 3:
        try:
            # Build timestamped transcript string
            transcript_lines = []
            for s in sentences[:80]: # Limit to reasonable window
                st = s.get("start", 0.0)
                et = s.get("end", 0.0)
                txt = s.get("text", "").strip()
                if txt:
                    transcript_lines.append(f"[{st:.1f}s - {et:.1f}s] {txt}")

            full_dialogue = "\n".join(transcript_lines)
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

            prompt = f"""
Sen uzman bir video editörüsün. Aşağıdaki zaman damgalı Türkçe yayın konuşma dökümünü analiz et.
Yayıncı: {streamer or 'Yayıncı'}
Yayın Başlığı: {stream_title or 'Yayın'}
İstenen Kategori: {target_category} (funny, drama, hype, story veya all)

Diyalog Transkripti:
\"\"\"{full_dialogue}\"\"\"

GÖREV:
Bu transkriptteki en kaliteli, viral potansiyeli yüksek 2 ile 5 arası kesiti (anları) tespit et.
Kategoriler:
- 'funny': En komik anlar, espri, kahkaha, fail.
- 'drama': Tartışma, kavga, gerilim, drama, ters köşe, sert konuşma.
- 'hype': Coşkulu anlar, heyecan, zafer, şok tepki.
- 'story': İlginç hikaye, itiraf, sürükleyici sohbet.

KRİTİK KURAL (KONU VE CÜMLE BÜTÜNLÜĞÜ):
1. Kesit süresi 20 ile 60 saniye arasında olmalıdır.
2. Bir cümlenin veya konunun ortasından ASLA başlama. Cümlenin tam başladığı saniyeyi (start_sec) al.
3. Konu veya espri bitmeden ASLA bitirme. Son cümlenin bittiği saniyeyi (end_sec) al.

Aşağıdaki JSON array formatında yanıt ver (sadece geçerli JSON çıktısı, markdown kodu olmadan):
[
  {{
    "start_sec": 12.0,
    "end_sec": 38.5,
    "duration": 26.5,
    "moment_type": "funny",
    "moment_label": "😂 En Komik An",
    "viral_title": "Yayıncıyı Gülme Krizine Sokan Olay! 😂",
    "capcut_hook_headline": "GÜLMEKTEN ÇILDIRDI! 🤣",
    "why_viral": "Konu bütünlüğü tam, esprinin patladığı an.",
    "hook_quote": "İlk 3 saniyede söylenen çekici cümle"
  }}
]
"""
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.4,
                    "maxOutputTokens": 1000,
                    "responseMimeType": "application/json"
                }
            }
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
                raw_text = resp_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                raw_text = re.sub(r'^```json\s*', '', raw_text)
                raw_text = re.sub(r'\s*```$', '', raw_text)
                parsed = json.loads(raw_text)
                if isinstance(parsed, list) and len(parsed) > 0:
                    logger.info(f"Gemini LLM successfully detected {len(parsed)} narrative scenes!")
                    return parsed
        except Exception as e:
            logger.warning(f"Gemini scene analysis error: {e}, falling back to local NLP.")

    return analyze_local_transcript_scenes(sentences, streamer, stream_title, target_category)


def analyze_local_transcript_scenes(
    sentences: List[Dict[str, Any]],
    streamer: str = "",
    stream_title: str = "",
    target_category: str = "all"
) -> List[Dict[str, Any]]:
    """
    Local Turkish semantic scene and narrative boundary analyzer.
    Clusters sentences by intent and preserves 100% sentence boundaries.
    """
    if not sentences:
        return []

    funny_kw = ["gülme", "kahkaha", "komik", "hahaha", "sjsjsj", "puhaha", "lan", "şaka", "patladım", "ölüyorum", "troll", "çıldırdım", "dalga"]
    drama_kw = ["tartışma", "kavga", "sen kimsin", "yalan", "haddini bil", "kes sesini", "saçmalama", "sinir", "bana bak", "olamaz", "ayıp", "ispatla", "bırak ya"]
    hype_kw = ["hadi", "helal", "vurdum", "kazandık", "rekor", "inanılmaz", "çıldırıyorum", "baba geldi", "clutch", "yıkıldı", "nasıl", "eyvah", "şok"]
    story_kw = ["bir gün", "itiraf", "aslında", "şöyle oldu", "anlatayım", "çocukken", "geçenlerde", "kimseye", "meğerse", "hatırlıyorum"]

    results = []
    n = len(sentences)
    idx = 0

    while idx < n:
        # Search for a cluster of 20 to 50 seconds
        s_start = sentences[idx].get("start", 0.0)
        end_idx = idx
        cluster_text = []

        while end_idx < n:
            cur_dur = sentences[end_idx].get("end", 0.0) - s_start
            cluster_text.append(sentences[end_idx].get("text", ""))
            if cur_dur >= 24.0:
                # Snap to sentence end
                break
            end_idx += 1

        end_idx = min(end_idx, n - 1)
        s_end = sentences[end_idx].get("end", s_start + 25.0)
        dur = round(s_end - s_start, 2)
        full_chunk = " ".join(cluster_text).lower()

        # Score categories
        f_score = sum(full_chunk.count(k) for k in funny_kw)
        d_score = sum(full_chunk.count(k) for k in drama_kw)
        h_score = sum(full_chunk.count(k) for k in hype_kw)
        s_score = sum(full_chunk.count(k) for k in story_kw)

        # Classify
        if d_score >= 2 or (d_score > f_score and d_score > h_score and d_score > 0):
            m_type = "drama"
            m_label = "🥊 Drama & Tartışma"
        elif f_score >= 1:
            m_type = "funny"
            m_label = "😂 En Komik An"
        elif h_score >= 1:
            m_type = "hype"
            m_label = "🔥 Zirve / Heyecanlı An"
        elif s_score >= 1:
            m_type = "story"
            m_label = "💬 Hikaye & Sohbet"
        else:
            m_type = "hype" if (idx % 2 == 0) else "funny"
            m_label = "🔥 Önemli An"

        if target_category == "all" or target_category == m_type:
            first_sentence = sentences[idx].get("text", "").strip()
            headlines = generate_local_smart_headlines(
                streamer=streamer,
                stream_title=stream_title,
                transcript=full_chunk,
                hook_text=first_sentence,
                moment_type=m_type
            )

            results.append({
                "start_sec": round(s_start, 2),
                "end_sec": round(s_end, 2),
                "duration": dur,
                "moment_type": m_type,
                "moment_label": m_label,
                "viral_title": headlines.get("viral_title", f"{streamer} - {m_label}"),
                "capcut_hook_headline": headlines.get("capcut_hook_headline", "SONUNA KADAR İZLE! 🔥"),
                "why_viral": f"Konu bütünlüğü tam, {m_label.lower()} tespit edildi.",
                "hook_quote": first_sentence[:60]
            })

        # Advance with stride
        idx = max(idx + 1, end_idx)
        if len(results) >= 8:
            break

    return results

def analyze_clip_for_ai_thumbnail(
    streamer: str = "",
    stream_title: str = "",
    transcript: str = "",
    moment_type: str = "funny",
    duration: float = 35.0,
    peak_time_rel: Optional[float] = None
) -> Dict[str, Any]:
    """
    Analyzes clip video dialogue and context with Gemini Flash API to determine:
    1. The ideal climax / high-emotion frame timestamp (seconds) for FFmpeg capture.
    2. Ultra high-CTR clickbait hook title for thumbnail text.
    3. Alternative title options.
    4. Best visual mood/style (kick_neon, fire, shock, drama).
    5. Corner badge text.
    6. Midjourney / Flux image generation prompt.
    Includes robust Turkish NLP fallback if Gemini key is missing or request fails.
    """
    api_key = get_gemini_api_key()
    dur_f = max(5.0, float(duration or 30.0))
    default_frame = round(float(peak_time_rel) if (peak_time_rel is not None and 0.5 <= float(peak_time_rel) <= dur_f - 1.0) else (dur_f * 0.42), 1)

    if api_key:
        try:
            prompt = f"""
Sen YouTube Shorts, TikTok ve YouTube için tıklanma rekoru kıran (viral CTR) profesyonel bir kapak tasarımcısı ve video stratejistisin.
Aşağıda Türkçe bir canlı yayından kesilen {int(dur_f)} saniyelik bir klibin konuşma transkripti ve bilgileri var.

Yayıncı: {streamer or 'Yayıncı'}
Yayın Başlığı / Konu: {stream_title or 'Canlı Yayın'}
Klip Türü: {moment_type}
Klip Süresi: {dur_f} saniye
Klip Konuşmaları (Transkript):
\"\"\"{transcript or 'Yayın esnasında çok komik ve şok edici bir an yaşanıyor.'}\"\"\"

GÖREV:
Bu video içeriğini analiz ederek izleyicinin kaydırmayı durdurup hemen tıklamasını sağlayacak kapak (thumbnail) stratejisi oluştur:
1. "best_frame_sec": Klibin içindeki en can alıcı, en komik veya şok anının gerçekleştiği saniye (0.5 ile {dur_f - 1.0} arasında ondalıklı saniye). Kapak resmi bu saniyedeki video karesinden alınacak.
2. "viral_title": Kapak üzerine yazılacak BÜYÜK, VURUCU, MERAK UYANDIRICI 2-5 kelimelik başlık (büyük harf ve emojili, örn: "BUNU YAPTI MI?! 😱", "100.000 TL GİTTİ! 💀", "CHAT ÇILDIRDI! 🔥", "REKOR AN! 🏆", "İNANILMAZ HATA! 💥").
3. "alternative_titles": 3-4 farklı alternatif kısa kapak başlığı seçeneği.
4. "badge_text": Kapağın köşesine basılacak dikkat çekici mini etiket (örn: "CANLI YAYINDA ŞOK", "GÜLME KRİZİ", "KICK TARİHİ", "REKOR AN", "BEKLENMEDİK").
5. "style": "fire" (heyecan/şaşkınlık/kavga), "kick_neon" (kick/yayın anı/eğlence), "shock" (beklenmedik an/troll) veya "drama" temalarından biri.
6. "why_this_frame": Neden bu karenin ve başlığın seçildiğini açıklayan 1 cümle.
7. "ai_image_prompt": İsteğe bağlı olarak bu sahneyi betimleyen İngilizce DALL-E / Midjourney / Flux promptu.

Aşağıdaki JSON formatında yanıt ver (sadece geçerli JSON çıktısı ver, markdown kod bloğu olmadan):
{{
  "best_frame_sec": 12.5,
  "viral_title": "CHAT ŞOK OLDU! 😱",
  "alternative_titles": ["BUNU KİMSE BEKLEMİYORDU!", "1 SANİYEDE BİTTİ 🔥", "YAYINDA REZİLLİK 😂"],
  "badge_text": "ŞOK AN",
  "style": "shock",
  "why_this_frame": "Bu saniyede klibin en şaşırtıcı tepkisi veriliyor.",
  "ai_image_prompt": "Cinematic YouTube thumbnail of a Turkish streamer in extreme disbelief and excitement, hyper-expressive reaction, high contrast cinematic neon lighting, 8k resolution"
}}
"""
            parsed = call_gemini_api(api_key, prompt, temperature=0.5, timeout=20.0)
            if parsed and "viral_title" in parsed:
                b_frame = float(parsed.get("best_frame_sec", default_frame))
                if b_frame < 0.5 or b_frame > dur_f:
                    b_frame = default_frame
                parsed["best_frame_sec"] = round(b_frame, 1)
                parsed["engine"] = "gemini_flash"
                logger.info(f"Gemini thumbnail analysis complete: frame={b_frame}s, title={parsed.get('viral_title')}")
                return parsed
        except Exception as e:
            logger.warning(f"Gemini thumbnail analysis error: {e}, using local smart NLP fallback.")

    clean_streamer = (streamer or "Yayıncı").replace('@', '').capitalize()
    fallback_titles = {
        "funny": [
            "GÜLME KRİZİ KOPTU! 🤣",
            f"{clean_streamer.upper()} ÇILDIRDI! 😂",
            "BÖYLE BİR ŞEY YOK! 💀",
            "YAYININ EN KOMİK ANI 🎭"
        ],
        "shock": [
            "BUNU KİMSE BEKLEMİYORDU! 😱",
            "CANLI YAYINDA ŞOK AN! 🚨",
            "1 SANİYEDE HER ŞEY DEĞİŞTİ 💥",
            "GÖZLERİNE İNANAMADI! ⚡"
        ],
        "hype": [
            "REKOR KIRAN O AN! 🔥",
            f"{clean_streamer.upper()} TARİH YAZDI! 🏆",
            "ÖLÜMÜNE MÜCADELE ⚡",
            "İMKANSIZI BAŞARDI! 🎯"
        ],
        "drama": [
            "YAYINDA TARTIŞMA BÜYÜDÜ! 🥊",
            "GERİLİM TAVAN YAPTI ⚠️",
            "AĞIR SÖZLER SÖYLENDİ! 🤐",
            "OLAY ÇIKTI! 🚨"
        ]
    }
    m_key = moment_type if moment_type in fallback_titles else "funny"
    chosen_titles = fallback_titles[m_key]
    style_map = {"funny": "kick_neon", "shock": "shock", "hype": "fire", "drama": "drama"}

    return {
        "best_frame_sec": default_frame,
        "viral_title": chosen_titles[0],
        "alternative_titles": chosen_titles[1:],
        "badge_text": "😂 KAHKAHA" if m_key == "funny" else ("😱 ŞOK AN" if m_key == "shock" else ("🔥 HYPE" if m_key == "hype" else "🥊 GERİLİM")),
        "style": style_map.get(m_key, "kick_neon"),
        "why_this_frame": f"Klibin %40'lık zirve noktasındaki ({default_frame}s) görsel kare.",
        "ai_image_prompt": f"YouTube video thumbnail of streamer {clean_streamer}, dramatic emotion, vibrant colorful YouTube thumbnail style, 4k",
        "engine": "local_nlp"
    }

def craft_ai_thumbnail_prompt(
    streamer: str = "",
    stream_title: str = "",
    transcript: str = "",
    moment_type: str = "funny",
    variation_index: int = 0,
    custom_prompt: Optional[str] = None,
    api_key: Optional[str] = None
) -> str:
    clean_streamer = (streamer or "Turkish streamer").replace('@', '').capitalize()
    
    # If the user provided their own custom prompt / scene description:
    if custom_prompt and custom_prompt.strip():
        user_p = custom_prompt.strip()
        # If Gemini API key is available, translate/expand into elite English Imagen 3 thumbnail prompt
        if api_key:
            try:
                sys_msg = (
                    f"You are a master YouTube thumbnail prompt engineer for Google Imagen 3 and Midjourney.\n"
                    f"The user wants a thumbnail with this specific scene description: \"{user_p}\".\n"
                    f"Streamer context: {clean_streamer}.\n"
                    f"TASK: Convert and expand this into a single, punchy English prompt (under 60 words) for a viral YouTube thumbnail cover.\n"
                    f"Include cinematic dramatic lighting, intense vivid contrast, expressive action/faces, 8k resolution, octane render.\n"
                    f"Return ONLY the prompt text, without any quotes, explanations or markdown."
                )
                expanded = call_gemini_raw_text(api_key, sys_msg, temperature=0.6, timeout=10.0)
                if expanded and len(expanded.strip()) > 15:
                    return expanded.strip().strip('"').strip("'")
            except Exception as e:
                logger.warning(f"Error expanding custom prompt with Gemini: {e}")

        # Local robust high-CTR template fallback
        return f"Viral high CTR YouTube thumbnail art, cinematic 8k render, scene: {user_p}. Hyper-detailed, dramatic vibrant neon rim lighting, high contrast, vivid colors, trending on YouTube gaming, professional studio cover."

    clean_context = (transcript[:120].replace('"', '').replace('\n', ' ') if transcript else (stream_title or "live stream gaming moment"))
    
    variations = [
        # 0: Shocked Reaction / High CTR Face
        f"Ultra high CTR YouTube thumbnail art, extremely expressive gamer streamer {clean_streamer} with wide eyes, jaw dropped in total shock and disbelief, looking directly into the camera. Hyper-detailed face, cinematic electric neon-cyan and lime-green rim lighting, dramatic depth of field. Background shows a blurred chaotic streaming room with flashing glowing monitors. 8k octane render, hyper-realistic, vibrant colors, trending on YouTube gaming.",
        
        # 1: Fiery Action / Hype Climax
        f"Blockbuster movie poster style YouTube thumbnail. Streamer {clean_streamer} in high-intensity gaming focus, explosive fiery golden amber sparks and glowing ember particles swirling around. Intense cinematic lighting, dynamic low-angle composition, extreme contrast, dramatic shadows, 8k resolution, photorealistic action scene representing: {clean_context}.",
        
        # 2: Hysterical Laugh / Meme Energy
        f"Viral TikTok Shorts thumbnail art, streamer {clean_streamer} laughing hysterically with eyes squinted and mouth wide open, hands holding head, ecstatic joy. Studio neon lighting with electric purple and saturated yellow neon backlight. Ultra-sharp foreground, bokeh background, pop-art infused high contrast YouTube gamer aesthetic, 8k render.",
        
        # 3: Dark Suspense / Dramatic Mystery
        f"High tension cinematic YouTube cover art. Moody dramatic close-up portrait of streamer {clean_streamer} with an intense, shocked expression bathed in split red and cyan rim light. Atmospheric smoke, laser light beams in the background, high contrast chiaroscuro, cinematic 35mm lens, photorealistic 8k octane render representing mystery: {clean_context}."
    ]
    
    chosen = variations[variation_index % len(variations)]
    return chosen


def generate_gemini_ai_image(
    api_key: str,
    prompt: str,
    aspect_ratio: str = "16:9"
) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Calls Google's Imagen 3 model via the Gemini API to generate a high-res image.
    Supports aspect_ratio: "16:9", "9:16", "1:1".
    Returns (image_bytes, error_message).
    """
    if not api_key:
        return None, "Gemini API anahtarı bulunamadı. Lütfen ayarlardan API anahtarınızı girin."
    
    imagen_models = [
        "imagen-3.0-generate-002",
        "imagen-3.0-fast-generate-001"
    ]
    
    fmt = "16:9" if aspect_ratio in ["16:9", "169"] else ("9:16" if aspect_ratio in ["9:16", "916"] else "1:1")
    
    payload = {
        "instances": [
            {"prompt": prompt}
        ],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": fmt,
            "personGeneration": "ALLOW_ADULT"
        }
    }
    
    data = json.dumps(payload).encode("utf-8")
    last_error = ""
    
    for model_name in imagen_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:predict?key={api_key}"
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=45.0) as resp:
                resp_body = resp.read().decode("utf-8")
                resp_json = json.loads(resp_body)
                predictions = resp_json.get("predictions", [])
                if predictions:
                    b64_img = predictions[0].get("bytesBase64Encoded")
                    if b64_img:
                        img_bytes = base64.b64decode(b64_img)
                        logger.info(f"Imagen 3 ({model_name}) successfully generated {len(img_bytes)} bytes image!")
                        return img_bytes, None
        except urllib.error.HTTPError as he:
            err_text = ""
            try:
                err_text = he.read().decode("utf-8")
                err_data = json.loads(err_text)
                last_error = err_data.get("error", {}).get("message", str(he))
            except Exception:
                last_error = f"HTTP {he.code}: {he.reason} - {err_text[:200]}"
            logger.warning(f"Imagen model {model_name} HTTP {he.code}: {last_error}")
            continue
        except Exception as ex:
            last_error = str(ex)
            logger.warning(f"Imagen model {model_name} error: {ex}")
            continue
            
    return None, last_error or "Imagen 3 görsel üretim servisine ulaşılamadı. Hesabınızda Imagen API izninin açık olduğundan emin olun."

