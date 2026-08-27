# Pinned reference implementation

The official scorer (`scorer/official_bbq.py`) is cross-validated against the
version of `lm-evaluation-harness` installed in the `bias_transfer_clean`
conda env at the time this fix was built (2026-08-25):

- Package: `lm_eval` version `0.4.11`
- File: `lm_eval/tasks/bbq/utils.py`
  (`<site-packages>/lm_eval/tasks/bbq/utils.py`)
- File SHA256 (source text): `eab60fb9adef49491233a864e7462e3654fdd3b23de0833af57acec5a7ad8885`

If this conda env's `lm_eval` is ever upgraded, re-run
`tests/exhaustive_equivalence_test.py` before trusting `scorer/official_bbq.py`
again -- it re-derives the ground truth from whatever `lm_eval` is currently
installed, so a silent upstream change would silently change what "official"
means here.

Paper also cites BBQ itself: Parrish et al. 2022, ACL Findings,
https://aclanthology.org/2022.findings-acl.165/ (page 6 defines S_DIS/S_AMB).
