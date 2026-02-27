import asyncio
import csv
import re
import ssl
import sys
import time
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup

BASE_URL = "https://www.ap.ge"
SEARCH_URL = "https://www.ap.ge/az/search?&page={page}"
OUTPUT_CSV = Path(__file__).parent.parent / "data" / "data.csv"
CONCURRENCY = 8
DELAY = 0.3  # seconds between batches

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "az,en;q=0.9",
}

CSV_FIELDS = [
    "car_id", "url", "title", "body_type", "price_gel",
    "year", "city", "source", "customs_cleared", "phone",
    "mileage_km", "mileage_miles", "transmission", "engine", "steering",
    "page",
]


def parse_page(html: str, page_num: int) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.find_all("div", class_="boxCatalog2")
    results = []

    for card in cards:
        car_id_raw = card.get("id", "")
        car_id = car_id_raw.replace("car", "") if car_id_raw.startswith("car") else ""

        # Title and body type
        title_div = card.find("div", class_="titleCatalog")
        title = ""
        body_type = ""
        url = ""
        if title_div:
            a_tag = title_div.find("a")
            if a_tag:
                title = a_tag.get_text(strip=True)
                url = BASE_URL + a_tag.get("href", "")
            # body type is text after the <a> tag
            body_type = title_div.get_text(strip=True).replace(title, "").strip()

        # Price
        price_div = card.find("div", class_="priceCatalog")
        price_gel = ""
        if price_div:
            price_text = price_div.get_text(strip=True)
            price_gel = re.sub(r"[^\d\s]", "", price_text).strip().replace(" ", "")

        # Param block: year, city, source, customs, phone
        param_div = card.find("div", class_="paramCatalog")
        year = city = source = customs_cleared = phone = ""
        if param_div:
            # Get text before the definition-wrap div
            param_text = ""
            for child in param_div.children:
                if hasattr(child, "get") and child.get("class") and "definition-wrap" in child.get("class", []):
                    break
                param_text += str(child)
            param_text = BeautifulSoup(param_text, "html.parser").get_text(" ", strip=True)

            # Year
            m = re.search(r"(\d{4})\s*yil", param_text)
            if m:
                year = m.group(1)

            # City and source  e.g. "Rustavi (AUTOPAPA)"
            city_match = re.search(r"\d{4}\s*yil,\s*([^,(]+?)(?:\s*\(([^)]+)\))?,", param_text)
            if city_match:
                city = city_match.group(1).strip()
                source = city_match.group(2).strip() if city_match.group(2) else ""

            # Customs status
            if "gömrükdə rəsmiləşdirilib" in param_text:
                customs_cleared = "yes"
            elif "gömrükdə rəsmiləşdirilməyib" in param_text:
                customs_cleared = "no"

            # Phone
            phone_match = re.search(r"telefon\.\s*([^\s,<]+)", param_text)
            if phone_match:
                phone = phone_match.group(1).strip()

        # Specs inside definition-wrap
        def_wrap = card.find("div", class_="definition-wrap")
        mileage_km = mileage_miles = transmission = engine = steering = ""
        if def_wrap:
            speedo = def_wrap.find("div", class_="speedometer")
            if speedo:
                speedo_text = speedo.get_text()
                km_m = re.search(r"([\d\s]+)\s*min km", speedo_text)
                mi_m = re.search(r"([\d\s]+)\s*Miles", speedo_text)
                if km_m:
                    mileage_km = km_m.group(1).replace(" ", "").strip()
                if mi_m:
                    mileage_miles = mi_m.group(1).replace(" ", "").strip()

            trans = def_wrap.find("div", class_="transmission")
            if trans:
                transmission = trans.get_text(strip=True)

            gas = def_wrap.find("div", class_="gas")
            if gas:
                engine = gas.get_text(strip=True)

            steer = def_wrap.find("div", class_="steering")
            if steer:
                steering = steer.get_text(strip=True)

        results.append({
            "car_id": car_id,
            "url": url,
            "title": title,
            "body_type": body_type,
            "price_gel": price_gel,
            "year": year,
            "city": city,
            "source": source,
            "customs_cleared": customs_cleared,
            "phone": phone,
            "mileage_km": mileage_km,
            "mileage_miles": mileage_miles,
            "transmission": transmission,
            "engine": engine,
            "steering": steering,
            "page": page_num,
        })

    return results


def get_total_pages(html: str) -> int:
    pages = re.findall(r"page=(\d+)", html)
    if pages:
        return max(int(p) for p in pages)
    return 1


async def fetch_page(session: aiohttp.ClientSession, page: int, sem: asyncio.Semaphore) -> list[dict]:
    url = SEARCH_URL.format(page=page)
    async with sem:
        try:
            async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                html = await resp.text(encoding="utf-8", errors="ignore")
                return parse_page(html, page)
        except Exception as e:
            print(f"  [!] Page {page} error: {e}", file=sys.stderr)
            return []


async def main():
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    connector = aiohttp.TCPConnector(ssl=ssl_ctx, limit=CONCURRENCY)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession(connector=connector) as session:
        # Fetch page 1 first to get total pages
        print("Fetching page 1 to determine total pages...")
        async with session.get(
            SEARCH_URL.format(page=1), headers=HEADERS,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            first_html = await resp.text(encoding="utf-8", errors="ignore")

        total_pages = get_total_pages(first_html)
        first_page_data = parse_page(first_html, 1)
        print(f"Total pages: {total_pages} | Cars on page 1: {len(first_page_data)}")

        sem = asyncio.Semaphore(CONCURRENCY)
        all_records = list(first_page_data)

        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(first_page_data)

            # Scrape remaining pages in batches
            remaining = list(range(2, total_pages + 1))
            batch_size = CONCURRENCY * 4
            start_time = time.time()

            for i in range(0, len(remaining), batch_size):
                batch = remaining[i:i + batch_size]
                tasks = [fetch_page(session, p, sem) for p in batch]
                results = await asyncio.gather(*tasks)

                batch_records = []
                for page_records in results:
                    batch_records.extend(page_records)
                    all_records.extend(page_records)

                writer.writerows(batch_records)
                f.flush()

                done = i + len(batch) + 1  # +1 for page 1
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                remaining_pages = total_pages - done
                eta = remaining_pages / rate if rate > 0 else 0
                print(
                    f"  Pages {batch[0]}-{batch[-1]} done | "
                    f"Total records: {len(all_records)} | "
                    f"Progress: {done}/{total_pages} | "
                    f"ETA: {eta:.0f}s"
                )

                await asyncio.sleep(DELAY)

    print(f"\nDone! {len(all_records)} records saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
