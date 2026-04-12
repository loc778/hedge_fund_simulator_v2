# debugging/check_insurance_rows.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests, time
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

HEADERS = {"User-Agent": "Mozilla/5.0 Chrome/120.0.0.0"}
BASE_URL = "https://www.screener.in"
LOGIN_URL = f"{BASE_URL}/login/"

session = requests.Session()
session.headers.update(HEADERS)
resp = session.get(LOGIN_URL, timeout=15)
soup = BeautifulSoup(resp.text, "html.parser")
csrf = soup.find("input", {"name": "csrfmiddlewaretoken"})["value"]
session.headers["Referer"] = LOGIN_URL
session.post(LOGIN_URL, data={
    "csrfmiddlewaretoken": csrf,
    "username": os.getenv("SCREENER_EMAIL"),
    "password": os.getenv("SCREENER_PASSWORD")
}, timeout=15)
print("Logged in")

# Insurance + fintech + one failing non-bank
for sym in ["HDFCLIFE", "ICICIPRU", "BAJAJFINSV", "HCLTECH", "NESTLEIND"]:
    url = f"{BASE_URL}/company/{sym}/consolidated/"
    soup = BeautifulSoup(session.get(url, timeout=20).text, "html.parser")
    print(f"\n{'='*50}\nTICKER: {sym}")
    for section_id in ["profit-loss", "balance-sheet"]:
        section = soup.find("section", {"id": section_id})
        if not section:
            print(f"  {section_id}: NOT FOUND")
            continue
        table = section.find("table")
        if not table:
            continue
        # Also print year headers
        thead = table.find("thead")
        headers = [th.get_text(strip=True) for th in thead.find_all("th")] if thead else []
        print(f"  {section_id} years: {headers[1:5]}")  # first 4 year cols
        rows = table.find("tbody").find_all("tr")
        row_names = [r.find_all(["td","th"])[0].get_text(strip=True) for r in rows if r.find_all(["td","th"])]
        print(f"  {section_id} rows: {row_names}")
    time.sleep(2)