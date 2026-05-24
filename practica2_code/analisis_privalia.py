import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, silhouette_score
import subprocess
import sys
import json
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# CONFIGURACIO INICIAL
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / 'dataset'
FIGURES_DIR = DATASET_DIR / 'resultats_analisi' / 'figures'
CLEAN_DATA_PATH = DATASET_DIR / 'privalia_clean_max_discount.csv'
METRICS_DIR = DATASET_DIR / 'resultats_analisi' / 'metrics'

for d in [FIGURES_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# -------------------------------------------------------------------
# FUNCIO PER CONVERTIR TIPUS NUMPY A NATIUS (PER JSON)
# -------------------------------------------------------------------
def convert_to_native(obj):
    """Converteix recursivament qualsevol objecte NumPy a tipus natiu de Python."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_native(i) for i in obj]
    return obj

# -------------------------------------------------------------------
# CARREGA DE DADES (AMB FALLBACK A NETEJA)
# -------------------------------------------------------------------
if not CLEAN_DATA_PATH.exists():
    print(f"File {CLEAN_DATA_PATH} not found. Running cleaning script...")
    try:
        subprocess.run([sys.executable, str(SCRIPT_DIR / "clean-dataset.py")], check=True)
    except Exception as e:
        print(f"Error running cleaning: {e}")
        sys.exit(1)

print(f"Loading cleaned data from: {CLEAN_DATA_PATH}")
df = pd.read_csv(CLEAN_DATA_PATH)

# Assegurar tipus numerics i eliminar nuls crítics
df['discount_percentage'] = pd.to_numeric(df['discount_percentage'], errors='coerce')
df['original_price'] = pd.to_numeric(df['original_price'], errors='coerce')
df = df.dropna(subset=['discount_percentage', 'original_price', 'brand_name', 'subcategory'])

print(f"Dataset shape: {df.shape}")
print(f"Mean discount: {df['discount_percentage'].mean():.2f}%")
print(f"Mean original price: {df['original_price'].mean():.2f} EUR\n")

# -------------------------------------------------------------------
# FUNCIONS AUXILIARS PER GUARDAR METRIQUES (AMB CONVERSIO)
# -------------------------------------------------------------------
def save_metrics_to_json(metrics_dict, filename):
    """Guarda un diccionari de mètriques a JSON, convertint tipus NumPy."""
    path = METRICS_DIR / filename
    native_dict = convert_to_native(metrics_dict)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(native_dict, f, indent=4, ensure_ascii=False)
    print(f"Metrics saved to {path}")

def save_classification_report_to_csv(report_dict, filename):
    df_report = pd.DataFrame(report_dict).transpose()
    path = METRICS_DIR / filename
    df_report.to_csv(path)
    print(f"Classification report saved to {path}")

# -------------------------------------------------------------------
# 1. CONTRAST D'HIPOTESI
# -------------------------------------------------------------------
def hypothesis_test(df):
    print("\n" + "="*60)
    print("HYPOTHESIS TEST: Discount difference between top 2 brands")
    print("="*60)
    
    top2_brands = df['brand_name'].value_counts().nlargest(2).index.tolist()
    brand1, brand2 = top2_brands[0], top2_brands[1]
    data1 = df[df['brand_name'] == brand1]['discount_percentage']
    data2 = df[df['brand_name'] == brand2]['discount_percentage']
    
    print(f"Comparing '{brand1}' (n={len(data1)}) vs '{brand2}' (n={len(data2)})")
    
    # Normality test (Shapiro-Wilk on sample max 5000)
    sample1 = data1.sample(min(5000, len(data1)))
    sample2 = data2.sample(min(5000, len(data2)))
    _, p_norm1 = stats.shapiro(sample1)
    _, p_norm2 = stats.shapiro(sample2)
    normal = (p_norm1 > 0.05) and (p_norm2 > 0.05)
    
    # Homoscedasticity (Levene)
    _, p_levene = stats.levene(data1, data2)
    homoc = p_levene > 0.05
    
    if normal and homoc:
        stat, p_val = stats.ttest_ind(data1, data2)
        test_name = "T-test (independent)"
    else:
        stat, p_val = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        test_name = "Mann-Whitney U test"
    
    # Save metrics (convertim explicitament el booleà)
    hypothesis_metrics = {
        "test_name": test_name,
        "brand1": brand1, "brand2": brand2,
        "n1": len(data1), "n2": len(data2),
        "mean1": float(data1.mean()), "mean2": float(data2.mean()),
        "std1": float(data1.std()), "std2": float(data2.std()),
        "normality_p1": float(p_norm1), "normality_p2": float(p_norm2),
        "levene_p": float(p_levene),
        "statistic": float(stat),
        "p_value": float(p_val),
        "significant_difference": bool(p_val < 0.05)   # Convertim a bool natiu
    }
    save_metrics_to_json(hypothesis_metrics, "hypothesis_test.json")
    
    print(f"{test_name}: statistic = {stat:.4f}, p-value = {p_val:.5e}")
    if p_val < 0.05:
        print("Conclusion: There is a statistically significant difference in discounts between the two brands.")
    else:
        print("Conclusion: No significant difference detected.")
    
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(14,5))
    subset = df[df['brand_name'].isin(top2_brands)]
    sns.violinplot(x='brand_name', y='discount_percentage', data=subset, ax=ax[0], palette='Set2')
    ax[0].set_title('Discount distribution by brand (violin plot)')
    sns.boxplot(x='brand_name', y='discount_percentage', data=subset, ax=ax[1], palette='Set3')
    ax[1].set_title('Boxplot comparison')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'hypothesis_plots.png', dpi=150)
    plt.close()
    print("Plot saved to figures_avancades/hypothesis_plots.png\n")

# -------------------------------------------------------------------
# 2. MODEL NO SUPERVISAT: K-MEANS CLUSTERING
# -------------------------------------------------------------------
def unsupervised_analysis(df):
    print("\n" + "="*60)
    print("UNSUPERVISED MODEL: K-MEANS CLUSTERING")
    print("="*60)
    
    features = df[['original_price', 'discount_percentage']].dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    
    # Elbow method
    inertias = []
    K_range = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(scaled)
        inertias.append(km.inertia_)
    
    plt.figure()
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.savefig(FIGURES_DIR / 'elbow_method.png', dpi=150)
    plt.close()
    
    # Use k=3 (interpretable)
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled)
    features['Cluster'] = clusters
    
    # Silhouette score
    sil_score = silhouette_score(scaled, clusters)
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Save metrics
    cluster_metrics = {
        "method": "K-Means",
        "n_clusters": k,
        "silhouette_score": float(sil_score),
        "centroids": [
            {"cluster": i, "original_price": float(c[0]), "discount_percentage": float(c[1])}
            for i, c in enumerate(centroids)
        ],
        "cluster_sizes": {int(c): int((clusters == c).sum()) for c in range(k)},
        "inertia": float(kmeans.inertia_)
    }
    save_metrics_to_json(cluster_metrics, "kmeans_metrics.json")
    
    print(f"Silhouette coefficient: {sil_score:.4f}")
    print("Cluster centroids (in original scale):")
    for i, c in enumerate(centroids):
        print(f"  Cluster {i}: price ~ {c[0]:.2f} EUR, discount ~ {c[1]:.1f}%")
    
    # Scatter plot
    plt.figure(figsize=(10,7))
    sns.scatterplot(data=features, x='original_price', y='discount_percentage', hue='Cluster', palette='viridis', alpha=0.6, s=50)
    plt.scatter(centroids[:,0], centroids[:,1], marker='X', s=200, color='red', edgecolor='black', label='Centroids')
    plt.title(f'K-Means Clustering (k={k}) - Silhouette = {sil_score:.3f}')
    plt.xlabel('Original price (EUR)')
    plt.ylabel('Discount (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'kmeans_clusters.png', dpi=150)
    plt.close()
    print("Plot saved to figures_avancades/kmeans_clusters.png\n")

# -------------------------------------------------------------------
# 3. MODEL SUPERVISAT: RANDOM FOREST (CLASSIFICATION)
# -------------------------------------------------------------------
def supervised_analysis(df):
    print("\n" + "="*60)
    print("SUPERVISED MODEL: RANDOM FOREST CLASSIFIER")
    print("Target: high discount (>50%)")
    print("="*60)
    
    # Create binary target
    df['high_discount'] = (df['discount_percentage'] > 50).astype(int)
    target_rate = df['high_discount'].mean()
    print(f"Proportion of high discount products: {target_rate*100:.2f}%")
    
    # Prepare features
    features = ['original_price', 'brand_name', 'subcategory']
    model_df = df[features + ['high_discount']].dropna()
    
    # Reduce cardinality: top 20 brands, rest as 'Altres'
    top_brands = model_df['brand_name'].value_counts().head(20).index
    model_df['brand_name'] = model_df['brand_name'].apply(lambda x: x if x in top_brands else 'Altres')
    
    X = pd.get_dummies(model_df[['original_price', 'brand_name', 'subcategory']], drop_first=True)
    y = model_df['high_discount']
    
    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Cross-validation
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')
    
    # Predictions and probabilities
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    
    # Metrics
    report_dict = classification_report(y_test, y_pred, target_names=['<=50%', '>50%'], output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred)
    conf_mat_percent = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis] * 100
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Save all metrics to JSON
    supervised_metrics = {
        "model": "RandomForestClassifier",
        "parameters": {"n_estimators": 150, "max_depth": 12},
        "target_rate": float(target_rate),
        "train_size": len(X_train), "test_size": len(X_test),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "test_accuracy": float(report_dict['accuracy']),
        "roc_auc": float(roc_auc),
        "classification_report": report_dict,
        "confusion_matrix": conf_mat.tolist(),
        "confusion_matrix_percent": conf_mat_percent.tolist()
    }
    save_metrics_to_json(supervised_metrics, "random_forest_metrics.json")
    
    # Also save classification report as CSV
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(METRICS_DIR / "classification_report.csv")
    print("Classification report saved to metrics/classification_report.csv")
    
    # Print summary
    print(f"\nCross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print(f"Test set accuracy: {report_dict['accuracy']:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")
    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred, target_names=['<=50%', '>50%']))
    
    # Confusion matrix plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['<=50%', '>50%'], yticklabels=['<=50%', '>50%'], ax=ax1)
    ax1.set_title('Confusion Matrix (counts)')
    sns.heatmap(conf_mat_percent, annot=True, fmt='.1f', cmap='Reds',
                xticklabels=['<=50%', '>50%'], yticklabels=['<=50%', '>50%'], ax=ax2)
    ax2.set_title('Confusion Matrix (%)')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'rf_confusion_matrices.png', dpi=150)
    plt.close()
    
    # ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(FIGURES_DIR / 'rf_roc_curve.png', dpi=150)
    plt.close()
    
    # Feature importance (top 20)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[-20:]
    plt.figure(figsize=(10, 8))
    plt.barh(range(20), importances[indices], color='teal')
    plt.yticks(range(20), [X.columns[i] for i in indices])
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'rf_feature_importance.png', dpi=150)
    plt.close()
    
    print("Plots saved to figures_avancades/\n")

# -------------------------------------------------------------------
# 4. GENERAR INFORME TEXT RESUM
# -------------------------------------------------------------------
def generate_summary_report():
    """Llegeix els fitxers de metrics i genera un informe final en text."""
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("FINAL ANALYSIS REPORT - PRACTICE 2")
    report_lines.append("="*70)
    report_lines.append("")
    
    # Hypothesis test
    hypo_file = METRICS_DIR / "hypothesis_test.json"
    if hypo_file.exists():
        with open(hypo_file, 'r') as f:
            hypo = json.load(f)
        report_lines.append("HYPOTHESIS TEST")
        report_lines.append(f"  Test: {hypo['test_name']}")
        report_lines.append(f"  Brands: {hypo['brand1']} vs {hypo['brand2']}")
        report_lines.append(f"  Mean discounts: {hypo['mean1']:.2f}% vs {hypo['mean2']:.2f}%")
        report_lines.append(f"  P-value: {hypo['p_value']:.5e}")
        report_lines.append(f"  Significant difference: {hypo['significant_difference']}")
        report_lines.append("")
    
    # K-means
    km_file = METRICS_DIR / "kmeans_metrics.json"
    if km_file.exists():
        with open(km_file, 'r') as f:
            km = json.load(f)
        report_lines.append("UNSUPERVISED MODEL: K-MEANS")
        report_lines.append(f"  Number of clusters: {km['n_clusters']}")
        report_lines.append(f"  Silhouette score: {km['silhouette_score']:.4f}")
        report_lines.append("  Cluster centroids (price, discount%):")
        for c in km['centroids']:
            report_lines.append(f"    Cluster {c['cluster']}: {c['original_price']:.2f} EUR, {c['discount_percentage']:.1f}%")
        report_lines.append(f"  Cluster sizes: {km['cluster_sizes']}")
        report_lines.append("")
    
    # Random Forest
    rf_file = METRICS_DIR / "random_forest_metrics.json"
    if rf_file.exists():
        with open(rf_file, 'r') as f:
            rf = json.load(f)
        report_lines.append("SUPERVISED MODEL: RANDOM FOREST")
        report_lines.append(f"  Target: high discount (>50%) - prevalence {rf['target_rate']*100:.2f}%")
        report_lines.append(f"  Cross-validation accuracy (5 folds): {rf['cv_accuracy_mean']:.3f} +/- {rf['cv_accuracy_std']:.3f}")
        report_lines.append(f"  Test accuracy: {rf['test_accuracy']:.3f}")
        report_lines.append(f"  ROC AUC: {rf['roc_auc']:.3f}")
        report_lines.append("  Classification report (test set):")
        for label in ['<=50%', '>50%']:
            prec = rf['classification_report'][label]['precision']
            rec = rf['classification_report'][label]['recall']
            f1 = rf['classification_report'][label]['f1-score']
            report_lines.append(f"    {label}: precision={prec:.3f}, recall={rec:.3f}, f1={f1:.3f}")
        report_lines.append("")
    
    report_lines.append("="*70)
    report_lines.append("All figures and detailed metrics are available in:")
    report_lines.append(f"  - Figures: {FIGURES_DIR}")
    report_lines.append(f"  - Metrics (JSON/CSV): {METRICS_DIR}")
    report_lines.append("="*70)
    
    summary_path = METRICS_DIR / "summary_report.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"\nSummary report saved to {summary_path}")

# -------------------------------------------------------------------
# MENU PRINCIPAL
# -------------------------------------------------------------------
def main():
    if len(sys.argv) > 1:
        option = sys.argv[1].strip()
        print(f"Running in automated mode with option: {option}")
    else:
        print("\n" + "="*60)
        print("DATA ANALYSIS TOOLS - PRACTICE 2")
        print("="*60)
        print("Select an option:")
        print("1 - Supervised model only (Random Forest)")
        print("2 - Unsupervised model only (K-Means clustering)")
        print("3 - Both models + hypothesis test")
        print("4 - Exit")
        
        option = input("Option (1/2/3/4): ").strip()
    
    if option == '1':
        supervised_analysis(df)
    elif option == '2':
        unsupervised_analysis(df)
    elif option == '3':
        hypothesis_test(df)
        unsupervised_analysis(df)
        supervised_analysis(df)
    else:
        print("Exiting.")
        sys.exit(0)
    
    generate_summary_report()
    print("\nAnalysis completed successfully.")

if __name__ == "__main__":
    main()