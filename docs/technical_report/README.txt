Adaptive Video Bitrate Selection under Varying Network Conditions

main.tex is the technical report based on Bobby-Zhu/abr-q-learning commit
007a5c18f7875911a9ea778a4bcb3bd0686f627c. The report uses the project's
archived policies, manifests, and evaluation results.

To edit in Overleaf, upload main.tex and the images/ directory together.
To build locally, run "tectonic main.tex" from this directory, or run
"pdflatex main.tex" twice. The compiled report is also included.

To reproduce the additional analysis, use Python 3.10 or later and point
--repo at a checkout of the commit above. Run from this directory:

  python replay_case.py --repo /path/to/abr-q-learning
  python check_splits.py --repo /path/to/abr-q-learning
  python recording_intervals.py --repo /path/to/abr-q-learning

replay_case.py prints the two selected downloads for each policy and checks
all full-episode metrics against the committed results. Costs in the body
use full precision before rounding. check_splits.py uses Git history or the
included original validation manifest in evidence/. recording_intervals.py
draws 10,000 recording-level bootstrap samples with seed 0 and reports the
nearest-rank percentile interval for the slice-weighted paired mean.
