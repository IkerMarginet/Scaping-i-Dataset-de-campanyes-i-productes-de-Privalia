import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import subprocess

# 1. Executar primer el codi de neteja
print("Executant l'script de neteja inicial (clean-dataset.py)...")
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
try:
    subprocess.run(["python", str(SCRIPT_DIR / "clean-dataset.py")], check=True)
except Exception as e:
    print(f"Error executant clean-dataset.py: {e}")

# 2. Carregar les dades netes
DATASET_DIR = PROJECT_ROOT / 'dataset'
clean_data_path = DATASET_DIR / 'privalia_clean_max_discount.csv'
print(f"\nCarregant dades netes des de: {clean_data_path}")
df = pd.read_csv(clean_data_path)

# Directori per figures avançades
figures_dir = DATASET_DIR / 'figures_avancades'
figures_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------
# 4. CONTRAST D'HIPÒTESIS
# Pregunta: Existeix una diferència significativa en el descompte entre les dues marques amb més productes?
# ---------------------------------------------------------
print("\n--- CONTRAST D'HIPÒTESIS ---")
top2_brands = df['brand_name'].value_counts().nlargest(2).index.tolist()
brand1 = top2_brands[0]
brand2 = top2_brands[1]

data_brand1 = df[df['brand_name'] == brand1]['discount_percentage'].dropna()
data_brand2 = df[df['brand_name'] == brand2]['discount_percentage'].dropna()

print(f"Comparant descomptes entre '{brand1}' (n={len(data_brand1)}) i '{brand2}' (n={len(data_brand2)}).")

# Comprovació de normalitat (Shapiro-Wilk)
# Si n > 5000, shapiro no és del tot precís, però ens dóna una idea. Usem mostra si cal.
stat1, p1 = stats.shapiro(data_brand1.sample(min(5000, len(data_brand1))))
stat2, p2 = stats.shapiro(data_brand2.sample(min(5000, len(data_brand2))))

print(f"Test de Normalitat (Shapiro-Wilk) per {brand1}: p-value = {p1:.5f}")
print(f"Test de Normalitat (Shapiro-Wilk) per {brand2}: p-value = {p2:.5f}")

is_normal = (p1 > 0.05) and (p2 > 0.05)

# Comprovació d'Homocedasticitat (Levene)
stat_l, p_l = stats.levene(data_brand1, data_brand2)
print(f"Test d'Homocedasticitat (Levene): p-value = {p_l:.5f}")
is_homocedastic = p_l > 0.05

# Test: Si normal i homocedastic -> T-Student, sinó Mann-Whitney U
if is_normal and is_homocedastic:
    print("Assumptes complerts: Utilitzant T-Student per mostres independents.")
    stat_t, p_t = stats.ttest_ind(data_brand1, data_brand2)
    print(f"T-test p-value: {p_t:.5e}")
else:
    print("Assumptes NO complerts: Utilitzant U de Mann-Whitney (no paramètric).")
    stat_u, p_u = stats.mannwhitneyu(data_brand1, data_brand2, alternative='two-sided')
    print(f"Mann-Whitney U p-value: {p_u:.5e}")

# Gràfic de violí comparatiu
plt.figure(figsize=(8, 6))
sns.violinplot(x='brand_name', y='discount_percentage', data=df[df['brand_name'].isin(top2_brands)])
plt.title(f'Distribució de Descomptes: {brand1} vs {brand2}')
plt.savefig(figures_dir / 'hypothesis_violin.png')
plt.close()

# ---------------------------------------------------------
# 5. MODEL NO SUPERVISAT: CLUSTERING (K-MEANS)
# Agrupar productes segons preu original i descompte
# ---------------------------------------------------------
print("\n--- MODEL NO SUPERVISAT: K-MEANS ---")
cluster_data = df[['original_price', 'discount_percentage']].dropna()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

# Aplicar K-Means (amb 3 clusters com a exemple per buscar perfils com: premium, bàsic, alt descompte)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_data['Cluster'] = kmeans.fit_predict(scaled_data)

# Interpretar els clusters
print("Centroids (inversament transformats):")
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
for i, c in enumerate(centroids):
    print(f"Cluster {i}: Preu Original ~ {c[0]:.2f}€, Descompte ~ {c[1]:.1f}%")

plt.figure(figsize=(10, 6))
sns.scatterplot(data=cluster_data, x='original_price', y='discount_percentage', hue='Cluster', palette='viridis', alpha=0.6)
plt.title('K-Means Clustering: Perfils de Productes')
plt.xlabel('Preu Original (€)')
plt.ylabel('Percentatge de Descompte (%)')
plt.savefig(figures_dir / 'kmeans_clusters.png')
plt.close()

df.loc[cluster_data.index, 'Cluster'] = cluster_data['Cluster']

# ---------------------------------------------------------
# 6. MODEL SUPERVISAT: CLASSIFICACIÓ (RANDOM FOREST)
# Predir si un producte tindrà "Alt Descompte" (>50%)
# ---------------------------------------------------------
print("\n--- MODEL SUPERVISAT: CLASSIFICACIÓ ---")
# Crear variable objectiu: Alt Descompte (1) si > 50%, (0) en cas contrari
df['high_discount'] = (df['discount_percentage'] > 50).astype(int)

# Seleccionar característiques
features = ['original_price', 'brand_name', 'subcategory']
df_model = df[features + ['high_discount']].dropna()

# One-hot encoding per categòriques (agafo només top 20 marques per reduir dimensionalitat)
top20_brands = df_model['brand_name'].value_counts().head(20).index
df_model['brand_name'] = df_model['brand_name'].apply(lambda x: x if x in top20_brands else 'Altres')

X = pd.get_dummies(df_model[['original_price', 'brand_name', 'subcategory']], drop_first=True)
y = df_model['high_discount']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar model
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_train, y_train)

# Predicció i avaluació
y_pred = rf.predict(X_test)
print("Avaluació Random Forest:")
print(classification_report(y_test, y_pred))

# Matriu de confusió
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['<50%', '>50%'], yticklabels=['<50%', '>50%'])
plt.title('Matriu de Confusió - Alt Descompte')
plt.ylabel('Real')
plt.xlabel('Predicció')
plt.savefig(figures_dir / 'rf_confusion_matrix.png')
plt.close()

# Importància de les variables (top 10)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:10]

plt.figure(figsize=(10, 6))
plt.title("Top 10 Variables més importants")
plt.bar(range(10), importances[indices], align="center")
plt.xticks(range(10), [X.columns[i] for i in indices], rotation=45, ha='right')
plt.tight_layout()
plt.savefig(figures_dir / 'rf_feature_importance.png')
plt.close()

print("\nAnàlisi avançat completat! Figures desades a 'dataset/figures_avancades'.")
