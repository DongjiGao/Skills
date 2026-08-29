# VoiceBench PEDANT portable artifacts

VoiceBench pins `qa_metrics==0.2.17`, whose `PEDANT()` constructor downloads
mutable `master` pickles on every invocation. Those pickles require the removed
private scikit-learn 1.3 `Log` class and cannot load in the universal NeMo Skills
NumPy 2.1/scikit-learn 1.7 environment.

These files are an inference-only, portable export of the exact official assets
at qa_metrics revision `26623c6ca3313e7a58c48fd6ce2b8579eb0c742e`:

- `lr_classifier`: SHA-256 `9c3dd5998745ece1cfb6cb16b953358f794e8716d63a08f3b212d8bc32a59b8b`
- `tf-idf_vectorizer`: SHA-256 `e886c9ba3ddb15a4b693b075dfe19cfad36dc74e7f8f195b8d9e98e989f2ba4f`

The source pickles were authenticated and loaded under Python 3.9,
NumPy 1.26.4, SciPy 1.11.4, scikit-learn 1.3.2, and joblib 1.3.2. The export
contains only the vocabulary, IDF vector, linear coefficients/intercept/classes,
and their fixed source-parity fixture. `voicebench.py` authenticates all eight
files and checks decision/prediction parity at every scorer initialization.

The exported model is an `SGDClassifier` with one `(1, 94056)` coefficient row,
classes `correct`/`incorrect`, and a 94,053-entry unigram vocabulary. The final
three features are PEDANT's F1, precision, and recall values. Binary prediction
uses scikit-learn's exact `decision > 0 -> classes_[1]` rule.

Do not regenerate these files from mutable URLs. Any protocol update must pin a
new source revision, authenticate both source files, repeat the exact-environment
export/parity procedure, and update every SHA-256 constant in `voicebench.py`.
