"""
setup_pipeline.py
-----------------
Full project setup for a new system.
Run from project root: python setup_pipeline.py

Runs all pipeline scripts in order. On failure, logs the error and continues.
Scripts that do not exist are skipped with a warning.
"""

import subprocess
import sys
import os
import logging
from datetime import datetime

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_FILE = f"logs/setup_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    ("Install Requirements",    None,                                      [], False),  # pip install
    ("Setup DB",                "data/setup_db.py",                        [], False),
    ("OHLCV Ingestion",         "data/bhavcopy_ingestion.py",              [], False),
    ("Technical Indicators",    "data/indicators.py",                      [], False),
    ("Screener Fundamentals",   "data/screener_fundamentals.py",           [], False),
    ("Macro Data",              "data/macro.py",                           [], False),
    ("FII/DII Data",            "data/fii_dii_stockedge.py",               [], False),
    ("RBI Macro",               "data/rbi_macro.py",                       [], False),
    ("Data Quality",            "data/data_quality.py",                    [], False),
    ("HMM Regime Detection",    "ML_scripts/hmm.py",                       [], False),
    ("Feature Engineering",     "data/features.py",                        [], False),
    ("Export Features",         "data/export_features.py",                 [], False),
    ("Ensemble",                "ML_scripts/ensemble_final.py",            [], False),
]

# ── Runner ────────────────────────────────────────────────────────────────────
def run_pip_install():
    cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    log.info(f"START [Install Requirements] — {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        log.info("DONE  [Install Requirements]")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"FAIL  [Install Requirements] — exit code {e.returncode}")
        log.error("Cannot continue without dependencies. Exiting.")
        sys.exit(1)
    except Exception as e:
        log.error(f"FAIL  [Install Requirements] — {e}")
        sys.exit(1)


def run_step(name, script, args, skip_if_missing):
    if script is None:
        return run_pip_install()

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
    log.info("SETUP PIPELINE — FULL SYSTEM SETUP")
    log.info("=" * 60)

    results = {}
    for name, script, args, skip_if_missing in PIPELINE:
        ok = run_step(name, script, args, skip_if_missing)
        results[name] = ok

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("PIPELINE SUMMARY")
    log.info("=" * 60)
    passed = [n for n, ok in results.items() if ok]
    failed = [n for n, ok in results.items() if not ok]

    for n in passed:
        log.info(f"  [OK]  {n}")
    for n in failed:
        log.error(f" [NOT OK]  {n}")

    log.info(f"\nCompleted: {len(passed)}/{len(results)}  |  Log: {LOG_FILE}")

    if failed:
        log.warning("Some steps failed. Review log, fix issues, and re-run failed steps manually.")
        sys.exit(1)


if __name__ == "__main__":
    main()