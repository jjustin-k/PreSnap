from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('data/cleaned_data/cleaned_data.csv')

def retrieve_data_sets():
    # 70% training, 30% remaining
    split = GroupShuffleSplit(
        n_splits=1, train_size=0.70, random_state=42
    )
   
    train_idx, remaining_idx = next(
        split.split(df, groups=df['game_id'])
    )

    train_df = df.iloc[train_idx].copy().drop(columns=['game_id'])
    remaining_df = df.iloc[remaining_idx].copy()

    # Split remaining games equally: 15% validation, 15% test
    split = GroupShuffleSplit(
        n_splits=1, test_size=0.50, random_state=42
    )
    val_idx, test_idx = next(
        split.split(remaining_df, groups=remaining_df['game_id'])
    )

    val_df = remaining_df.iloc[val_idx].copy().drop(columns=['game_id'])
    test_df = remaining_df.iloc[test_idx].copy().drop(columns=['game_id'])
    return train_df, val_df, test_df


def train_model(train_df):
    X, y = train_df.drop(columns=['defensive_coverage']), train_df['defensive_coverage']
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                make_column_selector(dtype_include=["object", "string", "category"]),
            )
        ],
        remainder="passthrough",
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            # Selected by validation search; see benchmarks/2026-11-08-random-forest-search/.
            n_estimators=300,
            max_depth=20,
            max_features="log2",
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])

    model.fit(X, y)
    return model


def validation(train_df, validation_df):
    model = train_model(train_df)
    X_val, y_val = validation_df.drop(columns=['defensive_coverage']), validation_df['defensive_coverage']
    y_pred = model.predict(X_val)

    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("Macro F1:", f1_score(
        y_val, y_pred, average="macro", zero_division=0
    ))

    print(classification_report(y_val, y_pred, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(
        y_val,
        y_pred,
        normalize="true",
        xticks_rotation=90,
    )
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    train_df, val_df, test_df = retrieve_data_sets()
    validation(train_df, val_df)
