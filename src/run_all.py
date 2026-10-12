"""Reproduce everything:  python src/run_all.py   (~20 min on a 2-core laptop)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import prepare_pools, experiments, privacy, figures

t0 = time.time()
print("[1/4] text pools");            prepare_pools.build_chat()
print("[2/4] detection experiments"); experiments.main()
print("[3/4] privacy experiments");   privacy.main()
print("[4/4] figures")
for f in (figures.fig1_pairs, figures.fig2_ladder, figures.fig3_sweeps, figures.fig4_fingerprint,
          figures.fig5_privacy, figures.fig6_ablation):
    f()
print(f"done in {(time.time() - t0) / 60:.1f} min")
