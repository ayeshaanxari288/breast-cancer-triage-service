import os
import sys
import json
import yaml
import argparse
import joblib
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, accuracy_score, roc_auc_score, make_scorer

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/raw/breast_cancer.csv")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--model-version", type=str, default="v1.0")
    return parser.parse_args()

def main():
    args = parse_args()
    params = load_params()
    seed = params['train']['seed']
    
    # 1. Dataset Load Karein
    df = pd.read_csv(args.data_path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Target 0 = Malignant, Target 1 = Benign (Breast Cancer Dataset format)
    # Malignant recall ke liye pos_label=0 rakha gaya hai
    malignant_recall_scorer = make_scorer(recall_score, pos_label=0)

    # 2. Stratified 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['train']['test_size'], random_state=seed, stratify=y
    )

    # 3. Model Pipelines Define Karein (Two Distinct Families)
    pipelines = {
        "LogisticRegression": Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                C=params['logistic_regression']['C'],
                max_iter=params['logistic_regression']['max_iter'],
                random_state=seed
            ))
        ]),
        "RandomForest": Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                n_estimators=params['random_forest']['n_estimators'],
                max_depth=params['random_forest']['max_depth'],
                random_state=seed
            ))
        ])
    }

    # 4. 5-Fold Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=params['train']['cv_folds'], shuffle=True, random_state=seed)
    scoring = {
        'malignant_recall': malignant_recall_scorer,
        'accuracy': 'accuracy',
        'roc_auc': 'roc_auc'
    }

    cv_results_summary = {}
    best_model_name = None
    best_cv_recall = -1.0

    print("--- Cross-Validation Results ---")
    for name, pipe in pipelines.items():
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        mean_recall = np.mean(scores['test_malignant_recall'])
        mean_acc = np.mean(scores['test_accuracy'])
        mean_auc = np.mean(scores['test_roc_auc'])

        cv_results_summary[name] = {
            'cv_malignant_recall': float(mean_recall),
            'cv_accuracy': float(mean_acc),
            'cv_roc_auc': float(mean_auc)
        }
        print(f"[{name}] Mean CV Recall: {mean_recall:.4f} | Accuracy: {mean_acc:.4f} | ROC AUC: {mean_auc:.4f}")

        if mean_recall > best_cv_recall:
            best_cv_recall = mean_recall
            best_model_name = name

    print(f"\nSelected Best Model (highest CV malignant recall): {best_model_name}")

    # 5. Best Model ko Train Set Par Fit Karein aur Test Set Par Evaluate Karein
    best_pipeline = pipelines[best_model_name]
    best_pipeline.fit(X_train, y_train)

    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    test_recall = recall_score(y_test, y_pred, pos_label=0)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_proba)

    print(f"--- Test Set Performance ({best_model_name}) ---")
    print(f"Malignant Recall: {test_recall:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"ROC AUC: {test_auc:.4f}")

    # 6. Artifacts Directory Banayein
    os.makedirs(args.output_dir, exist_ok=True)
    model_file_path = os.path.join(args.output_dir, "model.joblib")
    card_file_path = os.path.join(args.output_dir, "model_card.json")

    # Serialized Pipeline Save Karein
    joblib.dump(best_pipeline, model_file_path)

    # Model Card JSON Save Karein
    model_card = {
        "model_version": args.model_version,
        "model_family": best_model_name,
        "ordered_feature_names": list(X.columns),
        "cv_metrics": cv_results_summary,
        "test_metrics": {
            "malignant_recall": float(test_recall),
            "accuracy": float(test_accuracy),
            "roc_auc": float(test_auc)
        },
        "seed": seed,
        "scikit_learn_version": sklearn.__version__
    }

    with open(card_file_path, "w") as f:
        json.dump(model_card, f, indent=4)

    print(f"\nSaved model to '{model_file_path}' and model card to '{card_file_path}'")

    # 7. Check Threshold condition (Exit non-zero if test malignant recall < 0.95)
    if test_recall < 0.95:
        print(f"\nERROR: Test malignant recall ({test_recall:.4f}) is below threshold 0.95!")
        sys.exit(1)

if __name__ == "__main__":
    main()