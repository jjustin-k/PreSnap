"""Run a staged validation-only search; never fit or score the test partition."""
import hashlib
import json
import platform
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "2026-11-08-random-forest-search"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/cleaned_data/cleaned_data.csv"
    df = pd.read_csv(source)
    train_i, remaining_i = next(GroupShuffleSplit(n_splits=1, train_size=.7, random_state=42).split(df, groups=df.game_id))
    train, remaining = df.iloc[train_i], df.iloc[remaining_i]
    val_i, test_i = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=42).split(remaining, groups=remaining.game_id))
    val = remaining.iloc[val_i]
    groups = [set(train.game_id), set(val.game_id), set(remaining.iloc[test_i].game_id)]
    assert all(not groups[a] & groups[b] for a, b in [(0, 1), (0, 2), (1, 2)])
    features = df.columns.drop(["game_id", "defensive_coverage"]).tolist()
    assert {"offensive_team", "defensive_team"} <= set(features)
    preprocessing = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         make_column_selector(dtype_include=["object", "string", "category"]))
    ], remainder="passthrough")
    X = preprocessing.fit_transform(train[features])
    V = preprocessing.transform(val[features])
    y, vy = train.defensive_coverage, val.defensive_coverage
    labels = sorted(set(y) | set(vy))
    runs = []
    base = dict(n_estimators=100, max_depth=None, min_samples_leaf=1,
                max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)

    def run(params, stage, reason):
        start = time.perf_counter()
        model = RandomForestClassifier(**params).fit(X, y)
        pred = model.predict(V)
        record = dict(id=len(runs)+1, stage=stage, reason=reason, parameters=params.copy(),
                      seconds=round(time.perf_counter()-start, 3),
                      validation_accuracy=accuracy_score(vy, pred),
                      validation_macro_f1=f1_score(vy, pred, labels=labels, average="macro", zero_division=0),
                      training_macro_f1=f1_score(y, model.predict(X), labels=labels, average="macro", zero_division=0),
                      classification_report=classification_report(vy, pred, labels=labels, output_dict=True, zero_division=0))
        runs.append(record)
        (OUT / "runs.json").write_text(json.dumps(runs, indent=2)+"\n")
        print(f"Run {record['id']:02}: {stage}; macro-F1={record['validation_macro_f1']:.4f}; accuracy={record['validation_accuracy']:.4f}", flush=True)
        return record

    # Controlled grid: isolate weighting and regularization, then test interactions.
    for weight, depth, leaf in product([None, "balanced", "balanced_subsample"], [None, 10, 20], [1, 3, 5, 10]):
        run(dict(base, class_weight=weight, max_depth=depth, min_samples_leaf=leaf), "weight-depth-leaf",
            "Compare unweighted and inverse-frequency weighting; depth 10/20 and larger leaves constrain memorization, with unrestricted depth/leaf 1 as controls.")
    best = max(runs, key=lambda r: (r["validation_macro_f1"], r["validation_accuracy"]))
    for features_per_split in ["log2", .5, None]:
        run(dict(best["parameters"], max_features=features_per_split), "feature-sampling",
            "Test whether considering fewer or more encoded features per split improves useful interactions versus tree diversity; sqrt is the existing control.")
    best = max(runs, key=lambda r: (r["validation_macro_f1"], r["validation_accuracy"]))
    for count in [300, 500]:
        run(dict(best["parameters"], n_estimators=count), "tree-count",
            "Check whether more trees stabilize probabilities and improve validation performance; 100 trees is the control.")
    finalists = sorted(runs, key=lambda r: (r["validation_macro_f1"], r["validation_accuracy"]), reverse=True)[:3]
    summaries = []
    for candidate in finalists:
        repeated = [candidate] + [run(dict(candidate["parameters"], random_state=seed), "seed-check",
                                    f"Check seed sensitivity of finalist run {candidate['id']} on the same validation games.") for seed in [7, 123]]
        summaries.append(dict(initial_run=candidate["id"], parameters=candidate["parameters"],
                              run_ids=[r["id"] for r in repeated],
                              mean_macro_f1=float(np.mean([r["validation_macro_f1"] for r in repeated])),
                              std_macro_f1=float(np.std([r["validation_macro_f1"] for r in repeated])),
                              mean_accuracy=float(np.mean([r["validation_accuracy"] for r in repeated]))))
    summaries.sort(key=lambda r: (r["mean_macro_f1"], r["mean_accuracy"]), reverse=True)
    majority = y.value_counts().idxmax()
    bp = np.repeat(majority, len(vy))
    summary = dict(date="2026-11-08", selection="Top three seed-42 configurations rechecked at seeds 7 and 123; choose highest mean validation macro-F1 across all three seeds, breaking ties by mean accuracy.",
                   versions=dict(python=platform.python_version(), sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__),
                   sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, Path(__file__), ROOT / "presnap/training/random_forest_trainer.py"]},
                   features=features, encoded_feature_count=X.shape[1], labels=labels,
                   splits=dict(training=dict(rows=len(train), games=len(groups[0])), validation=dict(rows=len(val), games=len(groups[1])), test=dict(rows=len(test_i), games=len(groups[2]))),
                   game_ids=dict(training=sorted(groups[0]), validation=sorted(groups[1])),
                   class_counts=dict(training=y.value_counts().to_dict(), validation=vy.value_counts().to_dict()),
                   baseline=dict(label=majority, accuracy=accuracy_score(vy,bp), macro_f1=f1_score(vy,bp,labels=labels,average="macro",zero_division=0)),
                   finalists=summaries, best=summaries[0], test_evaluated=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("BEST: " + json.dumps(summaries[0]), flush=True)


if __name__ == "__main__":
    main()
