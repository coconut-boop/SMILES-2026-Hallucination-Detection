# SMILES-2026 Hallucination Detection Solution

## Reproducibility

This repository is self-contained apart from the Python packages and the
`Qwen/Qwen2.5-0.5B` model weights downloaded by Hugging Face Transformers.

Recommended environment:

- Python 3.10 or newer
- CUDA GPU if available; CPU also works, but feature extraction is much slower
- Dependencies from `requirements.txt`

Run the solution from the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python solution.py
```

On Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

The command `python solution.py` performs the complete pipeline:

1. Loads `data/dataset.csv`.
2. Builds hidden-state features from `prompt + response` with
   `Qwen/Qwen2.5-0.5B`.
3. Evaluates the probe on cross-validation splits.
4. Saves the evaluation summary to `results.json`.
5. Loads `data/test.csv`.
6. Trains the final probe and saves competition predictions to
   `predictions.csv`.

The fixed infrastructure files are left runnable as provided. The implemented
parts are contained in `aggregation.py`, `probe.py`, and `splitting.py`.

## Final Solution

### Feature aggregation

`aggregation.py` uses the final part of the generated sequence because the
answer-level hallucination signal is expected to be strongest near the model's
response tokens. For each sample, the implementation:

- keeps the real token positions according to the attention mask;
- takes the last 64 non-padding tokens;
- reads several late and middle transformer layers with offsets
  `-1`, `-2`, `-4`, `-8`, and `-12`;
- concatenates four summaries for each selected layer:
  the last token, the tail mean, the tail max, and the difference between the
  last token and the tail mean.

This gives the probe both local information from the final token and more
stable context from the response tail. The approach also avoids using every
layer and every token, keeping the feature matrix compact enough for a simple
classifier.

The optional `extract_geometric_features` function is implemented as an
additional experiment. It computes sequence length, layer-wise norms,
inter-layer cosine similarities, start-to-end representation similarity, and
tail variance. In the final configuration, `USE_GEOMETRIC` remains `False` in
`solution.py`, so these features are available but not used by default.

### Probe classifier

`probe.py` implements `HallucinationProbe` with a deterministic scikit-learn
pipeline:

- `StandardScaler` normalizes the aggregated hidden-state features;
- `LogisticRegression` with `class_weight="balanced"` handles the binary
  classification task;
- the decision threshold is tuned on the available training or validation data
  to maximize accuracy, using F1 as a tie-breaker.

Although the class still subclasses `torch.nn.Module` to match the expected
interface, the final classifier uses logistic regression. This was chosen
because the dataset is small, the feature vectors are high-dimensional, and a
linear model is less prone to overfitting than a larger neural probe.

### Splitting strategy

`splitting.py` uses a 5-fold `StratifiedKFold` split with a stratified
validation subset inside each fold. Stratification preserves the ratio of
truthful and hallucinated samples in train, validation, and held-out test
parts. The fixed random seed makes the evaluation reproducible.

## Experiments and Failed Attempts

The following ideas were tried or considered but were not kept as the default
configuration:

- A neural MLP probe was kept as a possible architecture in the class skeleton,
  but the final implementation uses logistic regression because the small
  dataset made the simpler model more stable.
- Geometric/statistical features were implemented, including norm, cosine
  drift, and variance features. They remain available behind `USE_GEOMETRIC`,
  but the default solution leaves them disabled to avoid increasing feature
  dimensionality unless the extra signal is confirmed by validation.
- Single-layer pooling was discarded in favor of multi-layer aggregation,
  because hallucination signals can appear differently across late and middle
  transformer layers.
- Using only the final token was discarded because it ignores useful response
  context. The final feature set combines final-token information with pooled
  statistics over the response tail.

## Output Files

After running:

```bash
python solution.py
```

the repository root should contain:

- `results.json` with evaluation metrics;
- `predictions.csv` with predicted labels for `data/test.csv`.

`results.json` should be committed to the repository for the application.
`predictions.csv` should also be uploaded to public cloud storage and linked in
the application form, according to the competition instructions.
