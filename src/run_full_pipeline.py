# run_full_pipeline.py
import os
import sys
import time

def run(cmd, label):
    print("\n" + "="*60)
    print(f"[{label}] Running: {cmd}")
    print("="*60)
    start = time.time()
    exit_code = os.system(cmd)
    elapsed = time.time() - start
    if exit_code != 0:
        print(f"ERROR: Command failed with exit code {exit_code}: {cmd}")
        sys.exit(exit_code)
    print(f"\n[{label}] Completed in {elapsed/60:.2f} min ({elapsed:.1f} sec)")
    print("="*60)
    return elapsed

if __name__ == "__main__":
    total_start = time.time()

    t1 = run("python src/data_prep.py", "DATA PREP")
    t2 = run("python src/generate_label_versions.py", "LABEL MAPPING")
    t3 = run("python src/plot_label_distribution.py", "EDA PLOTS")
    t4 = run("python src/run_all.py", "TRAIN & EVAL (ALL MODELS/VERSIONS)")

    total_elapsed = time.time() - total_start

    print("\n" + "="*60)
    print("FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"\nSTEP TIMES:")
    print(f"  Data prep         : {t1/60:.2f} min ({t1:.1f} sec)")
    print(f"  Label mapping     : {t2/60:.2f} min ({t2:.1f} sec)")
    print(f"  EDA/plots         : {t3/60:.2f} min ({t3:.1f} sec)")
    print(f"  Train/eval (all)  : {t4/60:.2f} min ({t4:.1f} sec)")
    print(f"\nTOTAL TIME         : {total_elapsed/60:.2f} min ({total_elapsed:.1f} sec)")
    print("="*60)
