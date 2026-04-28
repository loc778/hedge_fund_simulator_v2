import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent

PIPELINE = [
    "data/setup_db.py",
    "data/bhavcopy_ingestion.py",
    "data/indicators.py",
    "data/macro.py",
    "data/rbi_macro.py",
    "data/fii_dii_stockedge.py",
    "data/screener_fundamentals.py",
    "data/data_quality.py",
    "data/hmm.py",
    "data/export_hmm_features.py",
    "data/features.py",
    "data/export_features.py"
    "data/load_regimes.py",
    "data/export_market_regimes.py",
    

    
]

LOG_DIR = ROOT / logs
LOG_DIR.mkdir(exist_ok=True)


def run_step(script):

    print(f"\nRunning {script}")

    log_path = LOG_DIR / f"{Path(script).stem}.log"

    start = time.time()

    with open(log_path, "w") as f:

        result = subprocess.run(
            [sys.executable, script],
            stdout=f,
            stderr=subprocess.STDOUT
        )

    secs = round(time.time()-start,2)

    if result.returncode != 0:
        print(f"FAILED at {script}")
        print(f"Check {log_path}")
        sys.exit(1)

    print(f"Done ({secs}s)")


if __name__=="__main__":

    print("="*60)
    print("STARTING QUANT DATA PIPELINE")
    print("="*60)

    t0 = time.time()

    for script in PIPELINE:
        run_step(script)

    total = round(time.time()-t0,2)

    print("\n" + "="*60)
    print(f"PIPELINE COMPLETE in {total} sec")
    print("="*60)