NeuroGuard project notes

Use README.md for the current macOS setup, training, testing, and Flask commands.
The older dashboard notes below described a separate demo and referred to
artifacts/ports that are not the current app configuration.

The bundled sample/chb03_36_sample.edf is synthetic interface-test data. It is
not a CHB-MIT recording and the anticipatory workflow rejects its generic
EEG01–EEG23 channel names. Research evaluation and model training use only the
local raw CHB-MIT EDFs listed in reports/dataset_audit.md.

The current `/predict` detector uses models/ga_anfis_artifact.joblib. The
separate anticipatory `/anticipate` route uses
models/anticipatory_ga_anfis_featureaudit_v4.joblib. The anticipatory feature
catalog includes delta 0.5-4 Hz, theta 4-8 Hz, alpha 8-13 Hz, beta 13-30 Hz,
and gamma 30-80 Hz absolute and relative power. See README.md and the reports directory
for model semantics, validation results, and limitations.
