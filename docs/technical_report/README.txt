Reproducing the integrated ABR report

The team report is in ../final_report_redline/ABR-Report.pdf. To rebuild it,
run "tectonic main.tex" in ../final_report_redline/, where the source and the
images/ directory are located.

Run the following commands with Python 3.10 or later from the repository root:

  python docs/technical_report/replay_case.py --repo .
  python docs/technical_report/check_splits.py --repo .
  python docs/technical_report/recording_intervals.py --repo .

replay_case.py prints the two selected downloads for each policy and checks
all full-episode metrics against the committed results. Costs in the body
use full precision before rounding. check_splits.py uses Git history or the
included original validation manifest in evidence/. recording_intervals.py
draws 10,000 recording-level bootstrap samples with seed 0 and reports the
nearest-rank percentile interval for the slice-weighted paired mean.
