"""
daily_refresh.py
----------------
Daily data refresh pipeline for the hedge fund simulator.
Run from project root: python daily_refresh.py

Runs all incremental update scripts in order. Each script resumes
from its last run date automatically. On failure, logs the error
and continues to the next step.

Intended to be scheduled via Windows Task Scheduler after market close
(18:30 IST or later).
"""

import subprocess
import sys
import os
import logging
from datetime import datetime

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_FILE = f"logs/daily_refresh_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Pipeline definition ───────────────────────────────────────────────────────
# Each entry: (display_name, script_path, extra_args, skip_if_missing)
PIPELINE = [
    ("OHLCV Ingestion",         "data/bhavcopy_ingestion.py",              [], False),
    ("Technical Indicators",    "data/indicators.py",                      [], False),
    ("Macro Data",              "data/macro.py",                           [], False),
    ("FII/DII Data",            "data/fii_dii_stockedge.py",               [], False),
    ("HMM Regime Detection",    "ML_scripts/hmm.py",                       [], False),
    ("Feature Engineering",     "data/features.py",                        [], False),
    ("Export Features",         "data/export_features.py",                 [], False),
    ("Ensemble",                "ML_scripts/ensemble_final.py",            [], False),
]

# ── Runner ────────────────────────────────────────────────────────────────────
def run_step(name, script, args, skip_if_missing):
    if not os.path.exists(script):
        if skip_if_missing:
            log.warning(f"SKIP  [{name}] — {script} not found")
        else:
            log.error(f"FAIL  [{name}] — {script} not found (required)")
        return False

    cmd = [sys.executable, script] + args
    log.info(f"START [{name}] — {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        log.info(f"DONE  [{name}]")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"FAIL  [{name}] — exit code {e.returncode}")
        return False
    except Exception as e:
        log.error(f"FAIL  [{name}] — {e}")
        return False


def main():
    log.info("=" * 60)
    log.info(f"DAILY REFRESH — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    results = {}
    for name, script, args, skip_if_missing in PIPELINE:
        ok = run_step(name, script, args, skip_if_missing)
        results[name] = ok

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("REFRESH SUMMARY")
    log.info("=" * 60)
    passed = [n for n, ok in results.items() if ok]
    failed = [n for n, ok in results.items() if not ok]

    for n in passed:
        log.info(f"  [OK]  {n}")
    for n in failed:
        log.error(f" [NOT OK]  {n}")

    log.info(f"\nCompleted: {len(passed)}/{len(results)}  |  Log: {LOG_FILE}")

    if failed:
        log.warning("Some steps failed. Review log file for details.")
        log.warning(f"Log: {LOG_FILE}")
        sys.exit(1)


if __name__ == "__main__":
    main()