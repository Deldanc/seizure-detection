# Requirement verification

The final independent requirement-by-requirement audit is maintained in [`final_requirement_audit.md`](final_requirement_audit.md). This file is a short pointer to avoid maintaining a second copy of the same evidence table.

Current verified state:

- Active anticipatory model: `models/anticipatory_ga_anfis_featureaudit_v4.joblib` (schema 3).
- 345 candidates cover delta 0.5–4 Hz, theta 4–8 Hz, alpha 8–13 Hz, beta 13–30 Hz, and gamma 30–80 Hz, with absolute and relative power plus five time-domain measures.
- The saved Sugeno ANFIS converged after the L2 objective/gradient correction. Its selected feature list and model/scaler dimensions match.
- Real EDF upload, result rendering, and CSV export were exercised; invalid files and missing-model behavior were verified.
- `.venv/bin/python -m unittest discover -s tests -v`: 25 tests pass. `.venv/bin/python -m pip check`: no broken requirements.
- Held-out test event result remains 0/1 with two false alarms and no lead time. The final audit explicitly marks predictive efficacy as failed and patient-independent evaluation as blocked by the available data.

Follow the final audit for the complete checklist, actual metrics, limitations, and exact reproduction commands.
