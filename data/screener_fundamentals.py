# data/screener_fundamentals.py — hedge_v2 (v4 final)
# ═══════════════════════════════════════════════════════════
# WHAT THIS SCRIPT DOES:
#   Scrapes annual fundamental data from Screener.in for all
#   Nifty 500 tickers and saves to nifty500_fundamentals table.
#
# COLUMNS POPULATED HERE:
#   Revenue, Gross_Profit, EBITDA, Net_Income, EPS_Basic,
#   EPS_Diluted, Total_Assets, Total_Liabilities, Total_Equity,
#   Cash, Total_Debt, Book_Value_PS, Operating_CF, Capex,
#   Free_Cash_Flow, Debt_to_Equity, ROCE, PE_Ratio,
#   Dividend_Yield, ROE, ROA
#
# COLUMNS LEFT NULL HERE (populated elsewhere):
#   FCF_Yield   → features.py  (needs Close price from OHLCV)
#   PB_Ratio    → features.py  (needs Close price from OHLCV)
#   EV_EBITDA   → not computed (needs live market cap)
#   FII_Holding → fii_dii.py
#   DII_Holding → fii_dii.py
#
# FIXES FROM v3:
#   1. BANKING_TICKERS: removed insurance/fintech (HDFCLIFE, BAJAJFINSV,
#      SBILIFE, ICICIPRU, PAYTM, POLICYBZR, JIOFIN) — they use Sales+ layout
#   2. get_years(): cleans malformed year strings like 'Mar 20169m',
#      'Mar 202415m', skips 'TTM' column
#   3. FCF null bug: when capex_calc < 0 (asset sale year), capex was
#      set to None blocking FCF computation. Fixed — FCF now computed
#      directly from op_cf + inv_cf when FCF row missing
#   4. Bank revenue: added 'Total Income', 'Net Revenue' keywords
#   5. Removed dead variable 'other_assets'
# ═══════════════════════════════════════════════════════════

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import re
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import text

from config import TABLES, TICKERS, TIER_X_EXCLUDED
from data.db import get_engine, save_to_db

load_dotenv()
engine = get_engine()

SCREENER_EMAIL    = os.getenv("SCREENER_EMAIL")
SCREENER_PASSWORD = os.getenv("SCREENER_PASSWORD")

BASE_URL  = "https://www.screener.in"
LOGIN_URL = f"{BASE_URL}/login/"
DELAY     = 3.0

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer":         BASE_URL,
}

# ── BANKING_TICKERS ───────────────────────────────────────────────────
# Only pure deposit-taking banks and pure lending NBFCs go here.
# These use 'Revenue+' row on screener P&L, 'Financing Profit' for EBITDA,
# and 'Borrowing' (no s) for debt on balance sheet.
#
# REMOVED from this set vs v3:
#   HDFCLIFE, SBILIFE, ICICIPRU  → insurance, screener shows Sales+ layout
#   BAJAJFINSV                   → holding company, shows Sales+ layout
#   PAYTM, POLICYBZR, JIOFIN     → fintech, shows Sales+ layout
#   BAJAJHFL                     → often 404 on consolidated
BANKING_TICKERS = {
    # Scheduled commercial banks
    "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "BANDHANBNK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "PNB.NS",
    "BANKBARODA.NS", "CANBK.NS", "UNIONBANK.NS", "YESBANK.NS",
    "INDUSINDBK.NS", "RBLBANK.NS", "DCBBANK.NS", "KTKBANK.NS",
    "KARURVYSYA.NS", "CUB.NS", "AUBANK.NS", "EQUITASBNK.NS",
    "UJJIVANSFB.NS", "ESAFSFB.NS",
    # Pure lending NBFCs
    "BAJFINANCE.NS", "MUTHOOTFIN.NS", "CHOLAFIN.NS", "M&MFIN.NS",
    "MANAPPURAM.NS", "RECLTD.NS", "PFC.NS", "IRFC.NS",
    "SHRIRAMFIN.NS", "LICHSGFIN.NS", "ABCAPITAL.NS", "AAVAS.NS",
    "HOMEFIRST.NS", "CANFINHOME.NS", "APTUS.NS", "CREDITACC.NS",
}


# ═══════════════════════════════════════════════════════════
# SECTION 1 — LOGIN
# ═══════════════════════════════════════════════════════════

def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)

    resp = session.get(LOGIN_URL, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    csrf_input = soup.find("input", {"name": "csrfmiddlewaretoken"})
    if not csrf_input:
        raise RuntimeError("CSRF token not found on screener.in login page")

    session.headers["Referer"] = LOGIN_URL
    payload = {
        "csrfmiddlewaretoken": csrf_input["value"],
        "username":            SCREENER_EMAIL,
        "password":            SCREENER_PASSWORD,
    }
    resp = session.post(LOGIN_URL, data=payload, timeout=15)
    resp.raise_for_status()

    if "login" in resp.url.lower():
        raise RuntimeError("Login failed — check SCREENER_EMAIL / SCREENER_PASSWORD in .env")

    print(f"  Logged in as {SCREENER_EMAIL}")
    return session


# ═══════════════════════════════════════════════════════════
# SECTION 2 — PAGE FETCH
# ═══════════════════════════════════════════════════════════

def fetch_company_page(session: requests.Session, symbol: str,
                       retries: int = 3) -> BeautifulSoup | None:
    """
    Tries consolidated page first, then standalone.
    Returns parsed BeautifulSoup if profit-loss section found, else None.
    """
    sym = symbol.replace(".NS", "").replace(".BO", "")
    urls = [
        f"{BASE_URL}/company/{sym}/consolidated/",
        f"{BASE_URL}/company/{sym}/",
    ]
    for url in urls:
        for attempt in range(1, retries + 1):
            try:
                resp = session.get(url, timeout=20)
                if resp.status_code == 404:
                    break
                if resp.status_code == 429:
                    time.sleep(30 * attempt)
                    continue
                if resp.status_code != 200:
                    time.sleep(5 * attempt)
                    continue
                soup = BeautifulSoup(resp.text, "html.parser")
                if soup.find("section", {"id": "profit-loss"}):
                    return soup
                break
            except Exception:
                if attempt < retries:
                    time.sleep(5 * attempt)
    return None


# ═══════════════════════════════════════════════════════════
# SECTION 3 — PARSING UTILITIES
# ═══════════════════════════════════════════════════════════

def clean_number(val) -> float | None:
    """Strips currency/percent symbols and converts to float."""
    if val is None:
        return None
    s = str(val).strip()
    s = s.replace(",", "").replace("%", "").replace("\u20b9", "").replace("\xa0", "")
    s = s.replace("+", "").strip()
    if s in ("", "--", "-", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_section_table(soup: BeautifulSoup, section_id: str) -> dict:
    """
    Parses a screener financial table section.
    Returns { row_name_clean: { year_str: float } }
    Strips trailing '+' from expandable row names.
    """
    section = soup.find("section", {"id": section_id})
    if not section:
        return {}
    table = section.find("table")
    if not table:
        return {}

    thead = table.find("thead")
    headers = []
    if thead:
        for th in thead.find_all("th"):
            headers.append(th.get_text(strip=True))
    if not headers:
        return {}

    year_cols = headers[1:]
    result    = {}
    tbody     = table.find("tbody")
    if not tbody:
        return {}

    for tr in tbody.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        row_name = cells[0].get_text(strip=True).rstrip("+").strip()
        if not row_name:
            continue
        values = {}
        for i, cell in enumerate(cells[1:]):
            if i < len(year_cols):
                val = clean_number(cell.get_text(strip=True))
                if val is not None:
                    values[year_cols[i]] = val
        if values:
            result[row_name] = values

    return result


def parse_company_ratios(soup: BeautifulSoup) -> dict:
    """
    Parses the .company-ratios box — current snapshot values.
    Returns { metric_name: float }
    """
    ratios    = {}
    container = soup.find(class_="company-ratios")
    if not container:
        return ratios

    for li in container.find_all("li"):
        name_el  = li.find(class_="name")
        value_el = li.find(class_="number") or li.find(class_="value")
        if not name_el or not value_el:
            spans = li.find_all("span")
            if len(spans) >= 2:
                name_el, value_el = spans[0], spans[1]
        if name_el and value_el:
            name = name_el.get_text(strip=True)
            val  = clean_number(value_el.get_text(strip=True))
            if name and val is not None:
                ratios[name] = val

    return ratios


def get_val(table_data: dict, keywords: list, year: str) -> float | None:
    """
    Finds first row matching any keyword, returns value for that year.
    Pass 1: exact match. Pass 2: substring match.
    Two-pass prevents 'Interest' from matching 'Net Interest Income'
    before 'Interest Expenses'.
    """
    for row_name, values in table_data.items():
        if any(kw.lower() == row_name.lower() for kw in keywords):
            v = values.get(year)
            if v is not None:
                return v
    for row_name, values in table_data.items():
        if any(kw.lower() in row_name.lower() for kw in keywords):
            v = values.get(year)
            if v is not None:
                return v
    return None


def get_years(tables: list) -> list:
    """
    Returns sorted list of clean 'Mon YYYY' year strings across all tables.

    Handles screener quirks:
      'Mar 20169m'  -> 'Mar 2016'  (9-month transition period label)
      'Mar 202415m' -> 'Mar 2024'  (15-month period label)
      'TTM'         -> skipped     (trailing twelve months = duplicate)

    Without this, HCLTECH (Jun year-end) and NESTLEIND (Dec year-end)
    lose 1-2 years of data silently.
    """
    years    = set()
    pattern  = re.compile(
        r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}$"
    )
    malformed = re.compile(
        r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\d*[a-zA-Z]*$"
    )
    MONTH_ORDER = {
        "Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
        "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12
    }

    for t in tables:
        for row_vals in t.values():
            for raw_y in row_vals.keys():
                y = raw_y.strip()
                if y.upper() == "TTM":
                    continue
                if pattern.match(y):
                    years.add(y)
                    continue
                m = malformed.match(y)
                if m:
                    years.add(f"{m.group(1)} {m.group(2)}")

    return sorted(
        years,
        key=lambda y: (int(y.split()[1]), MONTH_ORDER.get(y.split()[0], 0))
    )


# ═══════════════════════════════════════════════════════════
# SECTION 4 — MAIN PARSER
# ═══════════════════════════════════════════════════════════

def parse_screener_page(soup: BeautifulSoup, ticker: str) -> list[dict]:
    """
    Extracts fundamentals from a screener company page.
    Returns list of dicts — one per annual period.
    """
    is_bank = ticker in BANKING_TICKERS

    pl  = parse_section_table(soup, "profit-loss")
    bs  = parse_section_table(soup, "balance-sheet")
    cf  = parse_section_table(soup, "cash-flow")
    rat = parse_company_ratios(soup)

    if not pl and not bs:
        return []

    years = get_years([pl, bs, cf])
    if not years:
        return []

    # Current snapshot ratios — same value applied to all years for this ticker
    pe        = rat.get("Stock P/E")
    book_val  = rat.get("Book Value")
    div_yield = rat.get("Dividend Yield")
    roe_snap  = rat.get("ROE")
    roce_snap = rat.get("ROCE")

    records = []
    for year_str in years:
        try:
            period_date = datetime.strptime(year_str, "%b %Y").date()
        except ValueError:
            continue

        # ── P&L ───────────────────────────────────────────────────────
        if is_bank:
            revenue = get_val(pl, ["Revenue", "Interest Earned",
                                   "Net Interest Income", "Total Income",
                                   "Net Revenue"], year_str)
            ebitda  = get_val(pl, ["Financing Profit", "Operating Profit",
                                   "Pre-Provision", "PPOP"], year_str)
        else:
            revenue = get_val(pl, ["Sales", "Net Sales",
                                   "Revenue from Operations",
                                   "Revenue"], year_str)
            ebitda  = get_val(pl, ["Operating Profit", "EBITDA",
                                   "PBDIT"], year_str)

        net_income = get_val(pl, ["Net Profit", "PAT",
                                  "Profit after Tax"], year_str)
        eps        = get_val(pl, ["EPS in Rs", "EPS",
                                  "Earnings Per Share"], year_str)

        # ── Gross Profit (non-banks only) ─────────────────────────────
        gross_profit = None
        if not is_bank and revenue is not None:
            expenses = get_val(pl, ["Expenses"], year_str)
            if expenses is not None:
                gross_profit = round(revenue - expenses, 2)

        # ── Balance Sheet ─────────────────────────────────────────────
        eq_cap   = get_val(bs, ["Equity Capital"], year_str) or 0
        reserves = get_val(bs, ["Reserves"], year_str) or 0
        total_eq = (eq_cap + reserves) if (eq_cap or reserves) else None

        total_debt   = get_val(bs, ["Borrowings", "Borrowing",
                                    "Total Debt", "Long Term Borrowings"], year_str)
        total_liab   = get_val(bs, ["Total Liabilities"], year_str)
        total_assets = get_val(bs, ["Total Assets"], year_str)

        fixed_assets = get_val(bs, ["Fixed Assets", "Net Block"], year_str) or 0
        cwip         = get_val(bs, ["CWIP"], year_str) or 0
        investments  = get_val(bs, ["Investments"], year_str) or 0

        # ── Cash (derived) ────────────────────────────────────────────
        # Banks: use Investments as liquid asset proxy (SLR securities)
        # Non-banks: Total Assets - Fixed Assets - CWIP - Investments
        #            clamped to (0, 95% of total assets)
        cash = None
        if is_bank:
            if investments > 0:
                cash = round(investments, 2)
        else:
            if total_assets is not None and total_assets > 0:
                non_cash = fixed_assets + cwip + investments
                cash_raw = total_assets - non_cash
                if 0 < cash_raw < (total_assets * 0.95):
                    cash = round(cash_raw, 2)

        # ── Cash Flow ─────────────────────────────────────────────────
        op_cf  = get_val(cf, ["Cash from Operating Activity",
                              "Operating Activity"], year_str)
        inv_cf = get_val(cf, ["Cash from Investing Activity",
                              "Investing Activity"], year_str)
        fcf    = get_val(cf, ["Free Cash Flow"], year_str)

        # ── Capex and FCF ─────────────────────────────────────────────
        # Screener definition: FCF = Operating CF - Capex
        #
        # Case 1: FCF row exists on screener (most tickers, recent years)
        #   Use FCF directly. Capex = op_cf - fcf (can be negative = asset sale).
        #
        # Case 2: FCF row missing (older years, some tickers)
        #   FCF = op_cf + inv_cf  (inv_cf is negative when investing = outflow)
        #   Capex = abs(inv_cf)
        #
        # Case 3: Only op_cf available
        #   FCF = op_cf (conservative — no capex deduction)
        #
        # BUG FIXED vs v3: v3 set capex=None when capex_calc < 0,
        # which blocked FCF even when op_cf + inv_cf existed.

        capex = None

        if fcf is not None:
            if op_cf is not None:
                capex = round(op_cf - fcf, 2)   # negative ok in asset-sale years
        else:
            if op_cf is not None and inv_cf is not None:
                fcf   = round(op_cf + inv_cf, 2)
                capex = abs(inv_cf)
            elif op_cf is not None:
                fcf = op_cf   # lower bound when no investing CF data

        # ── Computed Ratios ───────────────────────────────────────────
        debt_to_eq = None
        if total_debt is not None and total_eq is not None and total_eq != 0:
            debt_to_eq = round(total_debt / total_eq, 4)

        roa = None
        if net_income is not None and total_assets is not None and total_assets != 0:
            roa = round(net_income / total_assets, 4)

        # ── Scale to absolute INR ─────────────────────────────────────
        # Screener values are in Crores. DB BIGINT needs absolute INR.
        def cr(val):
            return int(val * 1e7) if val is not None else None

        records.append({
            "Ticker":            ticker,
            "Period":            period_date,
            "Revenue":           cr(revenue),
            "Gross_Profit":      cr(gross_profit),
            "EBITDA":            cr(ebitda),
            "Net_Income":        cr(net_income),
            "EPS_Basic":         round(eps, 4) if eps is not None else None,
            "EPS_Diluted":       round(eps, 4) if eps is not None else None,
            "Total_Assets":      cr(total_assets),
            "Total_Liabilities": cr(total_liab),
            "Total_Equity":      cr(total_eq),
            "Cash":              cr(cash),
            "Total_Debt":        cr(total_debt),
            "Book_Value_PS":     round(book_val, 4) if book_val is not None else None,
            "Operating_CF":      cr(op_cf),
            "Capex":             cr(capex),
            "Free_Cash_Flow":    cr(fcf),
            "Debt_to_Equity":    debt_to_eq,
            "FCF_Yield":         None,   # computed in features.py
            "ROCE":              round(roce_snap / 100, 4) if roce_snap is not None else None,
            "PE_Ratio":          round(pe, 4) if pe is not None else None,
            "PB_Ratio":          None,   # computed in features.py
            "EV_EBITDA":         None,   # needs live market cap
            "Dividend_Yield":    round(div_yield / 100, 4) if div_yield is not None else None,
            "ROE":               round(roe_snap / 100, 4) if roe_snap is not None else None,
            "ROA":               roa,
            "FII_Holding":       None,   # fii_dii.py
            "DII_Holding":       None,   # fii_dii.py
        })

    return records


# ═══════════════════════════════════════════════════════════
# SECTION 5 — RESUME + DEDUP
# ═══════════════════════════════════════════════════════════

def get_already_scraped() -> set:
    """Returns set of tickers already present in fundamentals table."""
    try:
        result = pd.read_sql(
            f"SELECT DISTINCT Ticker FROM {TABLES['fundamentals']}",
            con=engine
        )
        return set(result["Ticker"].tolist())
    except Exception:
        return set()


def deduplicate_db():
    """
    Removes duplicate (Ticker, Period) rows — keeps highest id.
    Runs at start to clean up any partial previous runs.
    """
    print("  Deduplicating fundamentals table...")
    try:
        with engine.connect() as conn:
            conn.execute(text(f"""
                DELETE t1 FROM {TABLES['fundamentals']} t1
                INNER JOIN {TABLES['fundamentals']} t2
                WHERE t1.Ticker = t2.Ticker
                  AND t1.Period = t2.Period
                  AND t1.id < t2.id
            """))
            conn.commit()
        print("  Deduplication done.")
    except Exception as e:
        print(f"  Deduplication skipped (non-fatal): {e}")


# ═══════════════════════════════════════════════════════════
# SECTION 6 — MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("SCREENER.IN FUNDAMENTALS — hedge_v2 (v4 final)")
    print("=" * 60)

    if not SCREENER_EMAIL or not SCREENER_PASSWORD:
        print("Error: SCREENER_EMAIL or SCREENER_PASSWORD not set in .env")
        return

    deduplicate_db()

    target  = [t for t in TICKERS if t not in TIER_X_EXCLUDED]
    done    = get_already_scraped()
    pending = [t for t in target if t not in done]

    print(f"\nTotal   : {len(target)}")
    print(f"Done    : {len(done)}")
    print(f"Pending : {len(pending)}\n")

    if not pending:
        print("All tickers already scraped.")
        return

    print("Logging in...")
    try:
        session = create_session()
    except Exception as e:
        print(f"Error: {e}")
        return

    all_records = []
    failed      = []
    no_data     = []

    for idx, ticker in enumerate(pending, 1):
        print(f"  [{idx}/{len(pending)}] {ticker}...", end=" ", flush=True)

        try:
            soup = fetch_company_page(session, ticker)
            if soup is None:
                print("page not found")
                no_data.append(ticker)
                time.sleep(DELAY)
                continue

            records = parse_screener_page(soup, ticker)
            if not records:
                print("no data parsed")
                no_data.append(ticker)
                time.sleep(DELAY)
                continue

            all_records.extend(records)

            ref = records[-2] if len(records) > 1 else records[-1]
            print(
                f"OK {len(records)} yrs | "
                f"Rev={'Y' if ref['Revenue'] else 'N'} "
                f"GP={'Y' if ref['Gross_Profit'] else 'N'} "
                f"NI={'Y' if ref['Net_Income'] else 'N'} "
                f"FCF={'Y' if ref['Free_Cash_Flow'] else 'N'} "
                f"Cash={'Y' if ref['Cash'] else 'N'} "
                f"PE={'Y' if records[-1]['PE_Ratio'] else 'N'} "
                f"ROE={'Y' if records[-1]['ROE'] else 'N'} "
                f"ROA={'Y' if ref['ROA'] else 'N'} "
                f"D/E={'Y' if ref['Debt_to_Equity'] else 'N'}"
            )

        except Exception as e:
            print(f"Error: {e}")
            failed.append(ticker)

        if len(all_records) >= 260:
            df = pd.DataFrame(all_records)
            save_to_db(df, TABLES["fundamentals"], engine)
            all_records = []

        time.sleep(DELAY)

        if idx % 100 == 0:
            print(f"\n  Re-logging in at ticker {idx}...")
            try:
                session = create_session()
            except Exception:
                pass

    if all_records:
        df = pd.DataFrame(all_records)
        save_to_db(df, TABLES["fundamentals"], engine)

    print(f"""
{'='*60}
COMPLETE
  Scraped  : {len(pending) - len(failed) - len(no_data)}
  No data  : {len(no_data)}
  Errors   : {len(failed)}
{'='*60}""")

    if no_data:
        print(f"No data  : {no_data}")
    if failed:
        print(f"Errors   : {failed}")

    print("\nNext: python data/macro.py")


if __name__ == "__main__":
    main()