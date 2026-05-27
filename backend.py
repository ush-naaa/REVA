
# ============================================================
# PHASE 0: IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from collections import Counter

# ============================================================
# PHASE 1: LOAD & BASIC CLEANING
# ============================================================

print("\n📂 Loading dataset...")
df = pd.read_csv("house_prices.csv")

if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

df = df[
    (df["Area_in_Marla"] > 0) &
    (df["bedrooms"] > 0) &
    (df["baths"] > 0)
]

print("Dataset shape after basic cleaning:", df.shape)

# ============================================================
# PHASE 2: OUTLIER REMOVAL (IQR METHOD)
# ============================================================

def remove_outliers(data, col):
    Q1, Q3 = data[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return data[
        (data[col] >= Q1 - 1.5 * IQR) &
        (data[col] <= Q3 + 1.5 * IQR)
    ]

df = remove_outliers(df, "price")
df = remove_outliers(df, "Area_in_Marla")

print("Dataset shape after outlier removal:", df.shape)

# ============================================================
# PHASE 3: TARGET ENGINEERING (PRICE TIERS)
# ============================================================

df["price_tier"] = pd.cut(
    df["price"],
    bins=[0, 7_000_000, 22_000_000, np.inf],
    labels=["Low", "Medium", "High"]
)

# Remove ambiguous samples
df = df[
    (df["price"] < 6_500_000) |
    ((df["price"] >= 9_000_000) & (df["price"] <= 18_000_000)) |
    (df["price"] > 25_000_000)
]

df.dropna(subset=["price_tier"], inplace=True)

print("\n🎯 Target distribution (counts):")
print(df["price_tier"].value_counts())

print("\n🎯 Target distribution (percentages):")
print(df["price_tier"].value_counts(normalize=True))

# ============================================================
# PHASE 4: FEATURE ENGINEERING
# ============================================================

MINIMUM_COUNT = 150
popular_locations = df["location"].value_counts()
popular_locations = popular_locations[popular_locations >= MINIMUM_COUNT].index
df = df[df["location"].isin(popular_locations)].copy()

location_freq = df["location"].value_counts(normalize=True)
df["location_freq"] = df["location"].map(location_freq)

df["total_rooms"] = df["bedrooms"] + df["baths"]

num_features = [
    "baths",
    "bedrooms",
    "Area_in_Marla",
    "total_rooms",
    "location_freq"
]

cat_features = ["property_type", "city", "purpose"]

X = df[num_features + cat_features]
y = df["price_tier"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ============================================================
# PHASE 5: TRAIN–TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ============================================================
# PHASE 6: PREPROCESSING PIPELINE
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
    ]
)

X_train_final = preprocessor.fit_transform(X_train)
X_test_final = preprocessor.transform(X_test)

# ============================================================
# PHASE 7: MODEL COMPARISON (4 CLASSIFIERS)
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        solver="liblinear"
    ),

    "ANN": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=2500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
    n_estimators=400,        
    max_depth=12,            
    min_samples_leaf=2,      
    min_samples_split=4,
    max_features="sqrt",
    class_weight="balanced", 
    random_state=42,
    n_jobs=-1
),

    "Gaussian NB": GaussianNB(var_smoothing=1e-6)
}

results = []
trained_models = {}

print("\n🚀 Training & Evaluating Models...\n")

for name, model in models.items():
    model.fit(X_train_final, y_train)
    trained_models[name] = model

    y_pred = model.predict(X_test_final)
    report = classification_report(y_test, y_pred, output_dict=True)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": report["macro avg"]["precision"],
        "Recall": report["macro avg"]["recall"],
        "F1-Score": report["macro avg"]["f1-score"],
    })

    print(f"✔ Completed: {name}")

results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)
print(results_df.to_string(index=False, float_format="%.4f"))
print("=" * 70)

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print(f"\n🏆 BEST MODEL SELECTED: {best_model_name}")

# ============================================================
# PHASE 8: BEST MODEL EVALUATION 
# ============================================================

y_pred_best = best_model.predict(X_test_final)

print("\n===== CLASSIFICATION REPORT (BEST MODEL) =====")
print(
    classification_report(
        y_test,
        y_pred_best,
        target_names=label_encoder.classes_
    )
)

cm_best = confusion_matrix(y_test, y_pred_best)

print("\n===== CONFUSION MATRIX (BEST MODEL) =====")
print(cm_best)

# ============================================================
# PHASE 9: FINAL ANN TRAINING FOR BACKEND
# ============================================================

ann_model = trained_models["ANN"]

final_acc = ann_model.score(X_test_final, y_test)
print(f"\n🎯 Final ANN Accuracy: {final_acc:.4f}")

# ============================================================
# PHASE 10: SAVE BACKEND ASSETS 
# ============================================================

train_stats = {
    col: {
        "min": df[col].min(),
        "max": df[col].max(),
        "mean": df[col].mean()
    }
    for col in num_features
}

joblib.dump(
    {
        "model": ann_model,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "location_map": location_freq.to_dict(),
        "feature_names": preprocessor.get_feature_names_out(),
        "train_stats": train_stats
    },
    "reva_ann_model_assets.joblib"
)

print("\n✅ Backend training completed successfully!")
