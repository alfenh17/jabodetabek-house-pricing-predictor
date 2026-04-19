"""
scrape_rumah123_FINAL.py  —  v6 PRODUCTION
===========================================
Perubahan dari v5:
  - TIDAK buka detail page → 3-4x lebih cepat
  - Koordinat dari geopy geocoding (lokasi string → lat/lon)
  - Multi-city otomatis dalam satu run
  - Auto-retry kalau satu halaman gagal
  - Robust: tidak akan crash di tengah jalan

Target: ~2000 baris dari 5 kota (400 per kota)
Estimasi waktu: 30-45 menit (tergantung koneksi)

Setup:
  pip install playwright patchright pandas tqdm geopy
  playwright install chromium

Jalankan:
  # Semua 5 kota sekaligus (recommended):
  python scrape_rumah123_FINAL.py --mode multi

  # Satu kota saja:
  python scrape_rumah123_FINAL.py --city jakarta --pages 17

  # Tanpa geocoding (lebih cepat, isi koordinat manual nanti):
  python scrape_rumah123_FINAL.py --mode multi --no-geocode
"""

import argparse
import re
import time
import random
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

try:
    from patchright.sync_api import sync_playwright
    ENGINE = "patchright"
except ImportError:
    from playwright.sync_api import sync_playwright
    ENGINE = "playwright"

try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False

print(f"✓ {ENGINE} loaded")
print(f"✓ geopy {'available' if GEOPY_AVAILABLE else 'NOT installed — run: pip install geopy'}")


# ─── Config ───────────────────────────────────────────────────────────────────

USER_DATA_DIR = str(Path.home() / "playwright_rumah123_v6")

# Target: 400 listing per kota → ~17 halaman (±24 listing/halaman)
MULTI_CITY_CONFIG = [
    {"city": "jakarta",   "slug": "dki-jakarta",        "pages": 17, "out": "data/raw/jakarta.csv"},
    {"city": "bogor",     "slug": "bogor",               "pages": 17, "out": "data/raw/bogor.csv"},
    {"city": "depok",     "slug": "depok",               "pages": 17, "out": "data/raw/depok.csv"},
    {"city": "tangerang", "slug": "tangerang",           "pages": 17, "out": "data/raw/tangerang.csv"},
    {"city": "bekasi",    "slug": "bekasi",              "pages": 17, "out": "data/raw/bekasi.csv"},
]

CITY_SLUGS = {
    "jakarta"           : "dki-jakarta",
    "jakarta selatan"   : "jakarta-selatan",
    "jakarta barat"     : "jakarta-barat",
    "jakarta timur"     : "jakarta-timur",
    "jakarta utara"     : "jakarta-utara",
    "jakarta pusat"     : "jakarta-pusat",
    "bogor"             : "bogor",
    "depok"             : "depok",
    "tangerang"         : "tangerang",
    "tangerang selatan" : "tangerang-selatan",
    "bekasi"            : "bekasi",
}

# Hanya blokir font & media — image HARUS dibiarkan (trigger lazy-load card)
BLOCK_TYPES = {"font", "media"}


# ─── Geocoder ─────────────────────────────────────────────────────────────────

class Geocoder:
    """
    Geocode lokasi string ke lat/lon menggunakan Nominatim (OpenStreetMap).
    Cache hasil supaya tidak query ulang lokasi yang sama.
    Rate limit: 1 request/detik (sesuai ToS Nominatim).
    """
    def __init__(self):
        if GEOPY_AVAILABLE:
            self.geolocator = Nominatim(
                user_agent="jakarta_house_price_scraper_portfolio_2026"
            )
        self._cache: dict[str, tuple] = {}

    def geocode(self, lokasi: str, city: str) -> tuple[float | None, float | None]:
        if not GEOPY_AVAILABLE:
            return None, None

        # Build query: "Sawah Besar, Jakarta Pusat, Indonesia"
        query = f"{lokasi}, {city}, Indonesia" if lokasi else f"{city}, Indonesia"

        if query in self._cache:
            return self._cache[query]

        try:
            time.sleep(1.1)   # Nominatim rate limit: max 1 req/s
            location = self.geolocator.geocode(query, timeout=10)
            if location:
                result = (location.latitude, location.longitude)
            else:
                # Fallback: coba hanya city
                time.sleep(1.1)
                location = self.geolocator.geocode(f"{city}, Indonesia", timeout=10)
                result = (location.latitude, location.longitude) if location else (None, None)

            self._cache[query] = result
            return result

        except (GeocoderTimedOut, GeocoderServiceError):
            self._cache[query] = (None, None)
            return None, None
        except Exception:
            self._cache[query] = (None, None)
            return None, None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def parse_price(raw: str) -> float | None:
    if not raw:
        return None
    s = raw.upper().replace("RP", "").replace("\xa0", "").replace(" ", "").strip()
    try:
        if "MILIAR" in s or (s.endswith("M") and "JT" not in s):
            n = re.sub(r"[^\d,.]", "", s).replace(",", ".")
            return float(n) * 1e9
        if "JUTA" in s or "JT" in s:
            n = re.sub(r"[^\d,.]", "", s).replace(",", ".")
            return float(n) * 1e6
        n = re.sub(r"[^\d]", "", s)
        return float(n) if n else None
    except ValueError:
        return None


def parse_area(raw: str) -> float | None:
    """Handle "LT: \n2102 m²" — strip newline lalu ambil angka."""
    if not raw:
        return None
    cleaned = re.sub(r"\s+", " ", raw).strip()
    m = re.search(r"[\d]+(?:[.,]\d+)?", cleaned)
    return float(m.group().replace(",", ".")) if m else None


def parse_rooms(raw: str) -> int | None:
    """Handle "3+1" → 4, "2 + 10" → 12."""
    if not raw:
        return None
    raw = re.sub(r"\s+", "", str(raw))
    if "+" in raw:
        parts = re.findall(r"\d+", raw)
        return sum(int(p) for p in parts) if parts else None
    m = re.search(r"\d+", raw)
    return int(m.group()) if m else None


def split_lokasi_city(raw: str) -> tuple[str | None, str | None]:
    """
    "Sawah Besar, Jakarta Pusat" → ("Sawah Besar", "Jakarta Pusat")
    "Bekasi Selatan, Bekasi"     → ("Bekasi Selatan", "Bekasi")
    """
    if not raw:
        return None, None
    parts = [p.strip() for p in raw.split(",")]
    return (parts[0], parts[-1]) if len(parts) >= 2 else (raw.strip(), raw.strip())


# ─── Parse satu card ──────────────────────────────────────────────────────────

def parse_card(card, geocoder: Geocoder, use_geocode: bool) -> dict | None:
    try:
        rec = {}

        # ── Harga ─────────────────────────────────────────────────────────────
        price_el = card.locator("[data-testid='ldp-text-price']").first
        rec["harga"] = parse_price(
            price_el.inner_text(timeout=2_000) if price_el.count() > 0 else ""
        )
        if not rec["harga"]:
            return None

        # ── Judul ─────────────────────────────────────────────────────────────
        title_el = card.locator("h2.text-accent.font-medium").first
        rec["title"] = (
            title_el.inner_text(timeout=2_000).strip()
            if title_el.count() > 0 else None
        )

        # ── URL ───────────────────────────────────────────────────────────────
        link_el = card.locator("a[href*='/properti/']").first
        if link_el.count() > 0:
            href = link_el.get_attribute("href", timeout=2_000) or ""
            rec["url"] = (
                f"https://www.rumah123.com{href}"
                if href.startswith("/") else href
            )
        else:
            rec["url"] = None

        # ── Lokasi & City ─────────────────────────────────────────────────────
        loc_el = card.locator("p.text-greyText.text-sm.truncate").first
        lokasi_raw = (
            loc_el.inner_text(timeout=2_000).strip()
            if loc_el.count() > 0 else ""
        )
        rec["lokasi"], rec["city"] = split_lokasi_city(lokasi_raw)

        # ── Koordinat via geocoding ────────────────────────────────────────────
        if use_geocode and rec["lokasi"] and rec["city"]:
            rec["latitude"], rec["longitude"] = geocoder.geocode(
                rec["lokasi"], rec["city"]
            )
        else:
            rec["latitude"]  = None
            rec["longitude"] = None

        # ── Spesifikasi: init ──────────────────────────────────────────────────
        rec["jumlah_kamar_tidur"] = None
        rec["jumlah_kamar_mandi"] = None
        rec["carport"]            = None
        rec["luas_tanah_m2"]      = None
        rec["luas_bangunan_m2"]   = None

        # ── Scan semua span ────────────────────────────────────────────────────
        for span in card.locator("span").all():
            try:
                cls  = span.get_attribute("class") or ""
                html = span.inner_html(timeout=1_500)
                text = re.sub(r"\s+", " ", span.inner_text(timeout=1_500)).strip()

                if not text:
                    continue

                if "bedroom-icon" in html:
                    rec["jumlah_kamar_tidur"] = parse_rooms(
                        re.sub(r"[^\d+\s]", "", text).strip()
                    )
                elif "bathroom-icon" in html:
                    rec["jumlah_kamar_mandi"] = parse_rooms(
                        re.sub(r"[^\d+\s]", "", text).strip()
                    )
                elif "carports-icon" in html:
                    rec["carport"] = parse_rooms(
                        re.sub(r"[^\d+\s]", "", text).strip()
                    )
                elif (
                    "text-greyText" not in cls
                    and "LT" in text.upper()
                    and "m²" in text.lower()
                    and rec["luas_tanah_m2"] is None
                ):
                    rec["luas_tanah_m2"] = parse_area(text)
                elif (
                    "text-greyText" not in cls
                    and "LB" in text.upper()
                    and "m²" in text.lower()
                    and rec["luas_bangunan_m2"] is None
                ):
                    rec["luas_bangunan_m2"] = parse_area(text)

            except Exception:
                continue

        # ── Sertifikat & Kondisi ───────────────────────────────────────────────
        try:
            full = card.inner_text(timeout=3_000).upper()
            rec["sertifikat"] = (
                "SHM"   if "SHM"   in full else
                "HGB"   if "HGB"   in full else
                "Girik" if "GIRIK" in full else None
            )
            rec["kondisi"] = (
                "Baru"           if "BARU"      in full else
                "Siap Huni"      if "SIAP HUNI" in full else
                "Butuh Renovasi" if "RENOVASI"  in full else
                "Bagus"          if "BAGUS"     in full else None
            )
        except Exception:
            rec["sertifikat"] = None
            rec["kondisi"]    = None

        rec["tahun_dibangun"] = None
        rec["source"]         = "rumah123_playwright"
        rec["scraped_at"]     = datetime.now().isoformat()

        return rec

    except Exception:
        return None


# ─── Scrape satu halaman ──────────────────────────────────────────────────────

def scrape_page(
    page,
    url: str,
    geocoder: Geocoder,
    use_geocode: bool,
    retry: int = 2,
) -> list[dict]:
    """
    Scrape satu halaman. Auto-retry sampai `retry` kali kalau gagal.
    """
    for attempt in range(1, retry + 2):
        records = []
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30_000)

            # Scroll untuk trigger lazy-load card content
            time.sleep(2.0)   # beri waktu JS render setelah domcontentloaded
            for _ in range(5):
                page.mouse.wheel(0, 600)
                time.sleep(0.5)

            # Tunggu card muncul
            try:
                page.wait_for_selector(
                    "div[class*='flex-col gap-y-2']",
                    timeout=15_000
                )
            except Exception:
                if attempt <= retry:
                    print(f"    ⚠ Card tidak muncul, retry {attempt}/{retry}...")
                    time.sleep(3)
                    continue
                else:
                    page.screenshot(path=f"debug_{int(time.time())}.png")
                    print("    ✗ Gagal setelah semua retry — skip halaman ini")
                    return []

            # Scroll sekali lagi sampai bawah
            for _ in range(3):
                page.mouse.wheel(0, 800)
                time.sleep(0.3)
            time.sleep(0.8)

            cards = page.locator("div[class*='flex-col gap-y-2']").all()

            for card in cards:
                rec = parse_card(card, geocoder, use_geocode)
                if rec:
                    records.append(rec)

            return records   # sukses → return langsung

        except Exception as e:
            if attempt <= retry:
                print(f"    ⚠ Error ({e}), retry {attempt}/{retry}...")
                time.sleep(4)
            else:
                print(f"    ✗ Skip halaman setelah {retry} retry: {e}")
                return []

    return records


# ─── Scrape satu kota ─────────────────────────────────────────────────────────

def scrape_city(
    page,
    city_name : str,
    city_slug : str,
    max_pages : int,
    output_csv: str,
    geocoder  : Geocoder,
    use_geocode: bool,
) -> pd.DataFrame:

    BASE_URL = f"https://www.rumah123.com/jual/{city_slug}/rumah/?page={{page}}"
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    all_records = []

    print(f"\n{'─'*55}")
    print(f"  🏙  {city_name.upper()} ({city_slug})")
    print(f"  Target: ~{max_pages * 24} listings dari {max_pages} halaman")
    print(f"{'─'*55}")

    for page_num in tqdm(range(1, max_pages + 1), desc=f"  {city_name}"):
        url     = BASE_URL.format(page=page_num)
        records = scrape_page(page, url, geocoder, use_geocode)
        all_records.extend(records)

        if records:
            print(f"    ✓ Halaman {page_num}: +{len(records)} | Total: {len(all_records)}")
        else:
            print(f"    ✗ Halaman {page_num}: 0 records")

        # Auto-save setiap 5 halaman
        if page_num % 5 == 0 and all_records:
            pd.DataFrame(all_records).to_csv(output_csv, index=False)
            print(f"    💾 Auto-saved {len(all_records)} rows → {output_csv}")

        # Delay antar halaman
        delay = random.uniform(2.5, 5.0)
        time.sleep(delay)

    # Final save kota ini
    if all_records:
        df = pd.DataFrame(all_records)

        COL_ORDER = [
            "city", "luas_bangunan_m2", "luas_tanah_m2",
            "jumlah_kamar_tidur", "jumlah_kamar_mandi", "carport",
            "tahun_dibangun", "sertifikat", "kondisi",
            "latitude", "longitude", "harga",
            "title", "lokasi", "url", "source", "scraped_at",
        ]
        df = df[[c for c in COL_ORDER if c in df.columns]]
        df.to_csv(output_csv, index=False)

        total = len(df)
        print(f"\n  ✅ {city_name}: {total} listings saved → {output_csv}")
        print(f"     LT: {df['luas_tanah_m2'].notna().sum()} | "
              f"LB: {df['luas_bangunan_m2'].notna().sum()} | "
              f"KT: {df['jumlah_kamar_tidur'].notna().sum()} | "
              f"Lat: {df['latitude'].notna().sum()}")
        return df
    else:
        print(f"\n  ⚠ {city_name}: tidak ada data")
        return pd.DataFrame()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(
    mode       : str  = "multi",
    city       : str  = "jakarta",
    pages      : int  = 17,
    out        : str  = "data/raw/scrape_rumah123.csv",
    use_geocode: bool = True,
    headless   : bool = False,
):
    geocoder = Geocoder()

    cities_to_scrape = (
        MULTI_CITY_CONFIG if mode == "multi"
        else [{
            "city" : city,
            "slug" : CITY_SLUGS.get(city.lower(), city),
            "pages": pages,
            "out"  : out,
        }]
    )

    print(f"\n{'='*55}")
    print(f"  Rumah123 Scraper v6 — Production")
    print(f"  Mode      : {mode}")
    print(f"  Kota      : {len(cities_to_scrape)} kota")
    print(f"  Geocoding : {'✓ aktif (Nominatim)' if use_geocode and GEOPY_AVAILABLE else '✗ nonaktif'}")
    print(f"  Engine    : {ENGINE}")
    print(f"{'='*55}")

    all_dfs = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch_persistent_context(
            user_data_dir = USER_DATA_DIR,
            channel       = "chromium",
            headless      = headless,
            no_viewport   = True,
            args          = ["--disable-blink-features=AutomationControlled"],
        )

        page = browser.new_page()

        # Blokir font & media saja — image dibiarkan untuk lazy-load
        page.route(
            "**/*",
            lambda r: r.abort()
            if r.request.resource_type in BLOCK_TYPES
            else r.continue_()
        )

        # Buka homepage → dapat cookies
        print("\n  Membuka Rumah123 homepage...")
        page.goto("https://www.rumah123.com", wait_until="domcontentloaded")
        time.sleep(random.uniform(2, 3))

        for cfg in cities_to_scrape:
            df = scrape_city(
                page        = page,
                city_name   = cfg["city"],
                city_slug   = cfg["slug"],
                max_pages   = cfg["pages"],
                output_csv  = cfg["out"],
                geocoder    = geocoder,
                use_geocode = use_geocode and GEOPY_AVAILABLE,
            )
            if not df.empty:
                all_dfs.append(df)

            # Jeda antar kota
            if cfg != cities_to_scrape[-1]:
                jeda = random.uniform(5, 10)
                print(f"\n  ⏳ Jeda {jeda:.1f}s sebelum kota berikutnya...")
                time.sleep(jeda)

        browser.close()

    # ── Gabung semua kota ──────────────────────────────────────────────────────
    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        merged_path = "data/raw/rumah123_all_cities.csv"
        Path(merged_path).parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(merged_path, index=False)

        total = len(df_all)
        print(f"\n{'='*55}")
        print(f"  🎉 SEMUA KOTA SELESAI!")
        print(f"  Total keseluruhan  : {total:,} listings")
        print(f"  Harga valid        : {df_all['harga'].notna().sum():,} ({df_all['harga'].notna().sum()/total*100:.0f}%)")
        print(f"  LT terisi          : {df_all['luas_tanah_m2'].notna().sum():,} ({df_all['luas_tanah_m2'].notna().sum()/total*100:.0f}%)")
        print(f"  LB terisi          : {df_all['luas_bangunan_m2'].notna().sum():,} ({df_all['luas_bangunan_m2'].notna().sum()/total*100:.0f}%)")
        print(f"  KT terisi          : {df_all['jumlah_kamar_tidur'].notna().sum():,} ({df_all['jumlah_kamar_tidur'].notna().sum()/total*100:.0f}%)")
        print(f"  Koordinat terisi   : {df_all['latitude'].notna().sum():,} ({df_all['latitude'].notna().sum()/total*100:.0f}%)")
        print(f"\n  Per kota:")
        for city_name, grp in df_all.groupby("city", observed=True):
            print(f"    {city_name:<20}: {len(grp):,} rows")
        print(f"\n  Merged CSV → {merged_path}")
        print(f"{'='*55}\n")

        return df_all

    print("\n⚠ Tidak ada data sama sekali.")
    return pd.DataFrame()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rumah123 Scraper v6 — Multi-city production run"
    )
    parser.add_argument(
        "--mode", default="multi", choices=["multi", "single"],
        help="'multi' = semua 5 kota (default), 'single' = satu kota"
    )
    parser.add_argument("--city",       default="jakarta",
                        help="Kota (untuk --mode single)")
    parser.add_argument("--pages",      type=int, default=17,
                        help="Jumlah halaman per kota (default: 17 ≈ 400 listing)")
    parser.add_argument("--out",        default="data/raw/scrape_rumah123.csv",
                        help="Output CSV (untuk --mode single)")
    parser.add_argument("--no-geocode", action="store_true",
                        help="Nonaktifkan geocoding koordinat")
    parser.add_argument("--headless",   action="store_true",
                        help="Jalankan tanpa buka browser window")
    args = parser.parse_args()

    main(
        mode        = args.mode,
        city        = args.city,
        pages       = args.pages,
        out         = args.out,
        use_geocode = not args.no_geocode,
        headless    = args.headless,
    )
