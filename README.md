---
title: Kick Clipper
emoji: 🎬
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🎬 Kick Highlight Clipper

Kick yayınlarından ve yayın geçmişlerinden (VOD) **en heyecanlı / dikkat çekici anları (audio spikes & peak volume)** otomatik olarak tespit edip **orijinal kalitede kesen** profesyonel klip otomasyonu.

---

## ⚡ Özellikler

1. **⚡ Akıllı Heyecan Tespiti:** Ses frekansları ve desibel patlamalarından reaksiyonları ve heyecanlı anları anında bulur.
2. **🎬 Kayıpsız Ham Kesit (RAW):** Video yeniden encode edilmeden saniyeler içinde orijinal çözünürlük ve kalitede kırpılır.
3. **⏱️ Hızlı Zaman Aralığı Seçimi:** Uzun yayınların yalnızca istenen saat/dakika aralığını hızlıca tarayabilme.
4. **✨ Viral Başlık ve Hashtag Motoru:** TikTok, Shorts ve Reels için yapay zeka destekli başlık ve etiket önerileri.
5. **🛡️ HWID Donanım Koruması:** Lisans anahtarı ile cihaz bazlı kilit ve aktivasyon.

---

## 🚀 Hızlı Başlangıç

### Web Paneli ile Çalıştırma
`start.bat` dosyasına çift tıklayın veya terminalde:
```bash
python app.py
```
Tarayıcınızda açılan adrese gidin: `http://127.0.0.1:8000`

1. Kick yayın linkini yapıştırın ve **"Yayınları Getir"** butonuna basın.
2. İstediğiniz yayını ve zaman dilimini seçin.
3. **"Klipleri Çıkar"** butonuna basın.
4. Klipleri tarayıcıda izleyin, MP4 veya ZIP olarak indirin!

---

## 📱 Mobil Cihazlardan Kullanım (Sen ve Arkadaşların İçin)

Uygulama artık tam **Mobil ve PWA (Progressive Web App)** uyumludur! Telefonunuzdan sanki App Store'dan indirilmiş gibi tam ekran çalışır.

### 1. Evde / Aynı Wi-Fi Üzerinde:
1. `start.bat` dosyasını çalıştırın.
2. Web panelinin sağ üstündeki **"📱 Telefona Bağla"** butonuna tıklayın.
3. Çıkan QR kodu telefonunuzun kamerasıyla tarayın (veya ekrandaki `http://192.168.1.X:8000` adresine gidin).
4. **Telefona Uygulama Olarak Ekleme:**
   - **iPhone (Safari):** Paylaş butonuna basıp **"Ana Ekrana Ekle"** deyin.
   - **Android (Chrome):** Sağ üstteki menüden **"Uygulamayı Yükle"** deyin.

### 2. Arkadaşların Kendi Evindeyken (Dış İnternet Erişimi):
1. `INTERNETTEN_PAYLAS.bat` dosyasına çift tıklayın (Cloudflare güvenli tünelini otomatik başlatır).
2. Terminalde çıkan `https://xxxx.trycloudflare.com` bağlantısını kopyalayıp arkadaşlarınıza atın.
3. Arkadaşlarınız dünyanın herhangi bir yerinden telefonlarıyla girip klip çıkarabilir ve doğrudan telefonlarına indirebilir!
