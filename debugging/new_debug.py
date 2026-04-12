# debugging/check_rownames.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.screener.in"
}
BASE_URL  = "https://www.screener.in"
LOGIN_URL = f"{BASE_URL}/login/"

session = requests.Session()
session.headers.update(HEADERS)
resp = session.get(LOGIN_URL, timeout=15)
from bs4 import BeautifulSoup
soup = BeautifulSoup(resp.text, "html.parser")
csrf = soup.find("input", {"name": "csrfmiddlewaretoken"})["value"]
session.headers["Referer"] = LOGIN_URL
session.post(LOGIN_URL, data={
    "csrfmiddlewaretoken": csrf,
    "username": os.getenv("SCREENER_EMAIL"),
    "password": os.getenv("SCREENER_PASSWORD")
}, timeout=15)
print("Logged in")

# Check these 3 tickers — one bank, one non-bank, one SHRIRAMFIN
for sym in ["RELIANCE", "HDFCBANK", "SHRIRAMFIN"]:
    url = f"{BASE_URL}/company/{sym}/consolidated/"
    soup = BeautifulSoup(session.get(url, timeout=20).text, "html.parser")
    print(f"\n{'='*50}")
    print(f"TICKER: {sym}")
    for section_id in ["profit-loss", "balance-sheet", "cash-flow"]:
        section = soup.find("section", {"id": section_id})
        if not section:
            print(f"  {section_id}: SECTION NOT FOUND")
            continue
        table = section.find("table")
        if not table:
            print(f"  {section_id}: TABLE NOT FOUND")
            continue
        rows = table.find("tbody").find_all("tr")
        row_names = [r.find_all(["td","th"])[0].get_text(strip=True) for r in rows if r.find_all(["td","th"])]
        print(f"  {section_id} rows: {row_names}")
import time; time.sleep(2)