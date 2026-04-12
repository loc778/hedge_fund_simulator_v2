# debug/debug_screener.py
# Run this ONCE on your laptop to check what requests gets from screener
# paste the output back

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

SCREENER_EMAIL    = os.getenv("SCREENER_EMAIL")
SCREENER_PASSWORD = os.getenv("SCREENER_PASSWORD")
BASE_URL          = "https://www.screener.in"
LOGIN_URL         = f"{BASE_URL}/login/"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": BASE_URL,
}

# ── Login ─────────────────────────────────────────────────────────────
session = requests.Session()
session.headers.update(HEADERS)

resp = session.get(LOGIN_URL, timeout=15)
soup = BeautifulSoup(resp.text, "html.parser")
csrf = soup.find("input", {"name": "csrfmiddlewaretoken"})["value"]

session.headers["Referer"] = LOGIN_URL
resp = session.post(LOGIN_URL, data={
    "csrfmiddlewaretoken": csrf,
    "username": SCREENER_EMAIL,
    "password": SCREENER_PASSWORD,
}, timeout=15)
print(f"Login status: {resp.status_code}, URL after login: {resp.url}")

# ── Fetch RELIANCE page ───────────────────────────────────────────────
resp = session.get("https://www.screener.in/company/RELIANCE/consolidated/", timeout=20)
print(f"\nRELIANCE page status: {resp.status_code}")
print(f"Content length: {len(resp.text)} chars")

soup = BeautifulSoup(resp.text, "html.parser")

# Check profit-loss section
pl = soup.find("section", {"id": "profit-loss"})
print(f"\nprofit-loss section found: {pl is not None}")

if pl:
    table = pl.find("table")
    print(f"table found: {table is not None}")
    if table:
        # Print raw first 500 chars of table HTML
        print(f"\nFirst 800 chars of table HTML:")
        print(str(table)[:800])

        tbody = table.find("tbody")
        print(f"\ntbody found: {tbody is not None}")
        if tbody:
            rows = tbody.find_all("tr")
            print(f"rows in tbody: {len(rows)}")
            if rows:
                # First row details
                cells = rows[0].find_all(["td","th"])
                print(f"\nFirst row - {len(cells)} cells")
                print(f"  Cell 0 raw HTML: {str(cells[0])[:200]}")
                print(f"  Cell 0 get_text(strip=True): {repr(cells[0].get_text(strip=True))}")
                if len(cells) > 1:
                    print(f"  Cell 1 get_text(strip=True): {repr(cells[1].get_text(strip=True))}")
                if len(cells) > 2:
                    print(f"  Cell 2 get_text(strip=True): {repr(cells[2].get_text(strip=True))}")

# Check company-ratios
ratios = soup.find(class_="company-ratios")
print(f"\ncompany-ratios found: {ratios is not None}")
if ratios:
    lis = ratios.find_all("li")
    print(f"li count: {len(lis)}")
    if lis:
        print(f"First li HTML: {str(lis[0])[:300]}")
        for li in lis[:5]:
            name  = li.find(class_="name")
            value = li.find(class_="number") or li.find(class_="value")
            print(f"  name={repr(name.get_text(strip=True) if name else None)} "
                  f"value={repr(value.get_text(strip=True) if value else None)}")