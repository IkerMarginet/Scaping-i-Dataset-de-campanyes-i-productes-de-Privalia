# 04_final_cleaning_keep_max_discount.py
# Neteja completa conservant el descompte màxim per producte (no es modifiquen outliers)

import pandas as pd
import re
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------------------------
# Configuració de rutes
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / 'dataset'

campaigns_file = DATASET_DIR / 'privalia_campaigns.csv'
products_file = DATASET_DIR / 'privalia_products.csv'
output_file = DATASET_DIR / 'privalia_clean_max_discount.csv'
html_report = DATASET_DIR / 'informe_resultats_max_discount.html'
figures_dir = DATASET_DIR / 'figures'
figures_dir.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# Funcions de neteja (només netegen format, NO canvien valors)
# -------------------------------------------------------------------
def clean_price(value):
    if pd.isna(value) or str(value).strip() == '':
        return None
    cleaned = str(value).replace('€', '').replace(',', '.').strip()
    cleaned = re.sub(r'[^0-9.]', '', cleaned)
    try:
        return float(cleaned)
    except:
        return None

def clean_pct(value):
    # Neteja de percentatge: elimina '-' i '%', retorna l'enter
    if pd.isna(value) or str(value).strip() == '':
        return None
    cleaned = str(value).replace('-', '').replace('%', '').strip()
    try:
        return int(cleaned)
    except:
        return None

def extract_brand_name(raw):
    if pd.isna(raw):
        return 'Desconegut'
    s = str(raw).strip()
    parts = re.split(r'[-–—]|\.\s+', s)
    first_part = parts[0].strip()
    words = first_part.split()
    if len(words) > 3 and len(words[0]) <= 15:
        return words[0]
    if len(words) > 1 and ' '.join(words[:len(words)//2]) == ' '.join(words[len(words)//2:]):
        return ' '.join(words[:len(words)//2])
    return first_part

def clean_subcategory(value):
    if pd.isna(value):
        return 'desconegut'
    blacklist = ['todos los productos disponibles', 'mira todas las categorias', '2', '3', '4']
    cleaned = str(value)
    for bad in blacklist:
        cleaned = cleaned.replace(bad, '')
    cleaned = re.sub(r',\s*,', ',', cleaned)
    cleaned = re.sub(r',\s*$', '', cleaned)
    cleaned = re.sub(r'^\s*,', '', cleaned)
    if not cleaned or cleaned.strip() == '':
        return 'altres'
    return cleaned.strip().split(',')[0]

def clean_color(value):
    if pd.isna(value) or str(value).strip() == '':
        return 'desconegut'
    first_color = str(value).split(' | ')[0].strip()
    return first_color if first_color else 'desconegut'

def count_sizes(status):
    if pd.isna(status) or status == 'unknown':
        return 0
    return status.count(':')

# -------------------------------------------------------------------
# 1. Càrrega i integració
# -------------------------------------------------------------------
print("Carregant dades originals...")
campaigns = pd.read_csv(campaigns_file)
products = pd.read_csv(products_file)
print(f"Campanyes: {campaigns.shape}, Productes: {products.shape}")

# Seleccionar columnes de campanyes
campaign_cols = ['campaign_id', 'brand_name', 'subcategories', 'end_date_text',
                 'total_products_count', 'unique_products_count']
campaign_cols = [c for c in campaign_cols if c in campaigns.columns]
df = products.merge(campaigns[campaign_cols], on='campaign_id', how='left')
print(f"Després merge: {df.shape}")

# -------------------------------------------------------------------
# 2. Conservar el descompte MÀXIM per product_id (no l'últim timestamp)
# -------------------------------------------------------------------
# Convertir discount_percentage a numèric net per poder comparar
df['discount_pct_num'] = df['discount_percentage'].apply(clean_pct).fillna(0)

# Per a cada product_id, mantenir la fila amb el discount_pct_num més alt
# En cas d'empat, conservem la primera (o la que té timestamp més recent, però el màxim és prioritari)
idx_max = df.groupby('product_id')['discount_pct_num'].idxmax()
df = df.loc[idx_max].reset_index(drop=True)
print(f"Després de conservar el descompte màxim per producte: {df.shape}")

# Eliminem la columna auxiliar
df.drop(columns=['discount_pct_num'], inplace=True)

# -------------------------------------------------------------------
# 3. Neteja de columnes (sense eliminar files per preus nuls, només neteja)
# -------------------------------------------------------------------
print("Aplicant neteja...")
df['original_price'] = df['original_price'].apply(clean_price)
df['discount_price'] = df['discount_price'].apply(clean_price)
df['discount_percentage'] = df['discount_percentage'].apply(clean_pct)
df['brand_name'] = df['brand_name'].apply(extract_brand_name)
df['subcategory'] = df['subcategory'].apply(clean_subcategory)
df['color'] = df['color'].apply(clean_color)
df['sizes_status'] = df['sizes_status'].fillna('unknown')

# Els discount_percentage que quedin com a None els posem a 0 (productes sense descompte)
df['discount_percentage'] = df['discount_percentage'].fillna(0)

# Eliminar només files on original_price sigui nul (no podem analitzar preu)
initial_len = len(df)
df = df.dropna(subset=['original_price'])
print(f"Files eliminades per preu original nul: {initial_len - len(df)}")

# Afegir columna auxiliar d'esgotament
df['has_out'] = df['sizes_status'].apply(lambda x: 'OUT' in str(x))
df['num_sizes'] = df['sizes_status'].apply(count_sizes)

# Convertir categòriques
categorical_cols = ['brand_name', 'subcategory', 'color', 'sizes_status']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# -------------------------------------------------------------------
# 4. Detecció d'outliers (només informativa)
# -------------------------------------------------------------------
def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = series[(series < lower) | (series > upper)]
    return outliers, lower, upper

outliers_price, low_price, high_price = detect_outliers_iqr(df['original_price'])
outliers_disc, low_disc, high_disc = detect_outliers_iqr(df['discount_percentage'])

print(f"\nOutliers en original_price: {len(outliers_price)} valors fora de [{low_price:.2f}, {high_price:.2f}]")
print(f"Outliers en discount_percentage: {len(outliers_disc)} valors fora de [{low_disc:.2f}, {high_disc:.2f}]")
print(f"Descompte màxim actual: {df['discount_percentage'].max()}%")

# -------------------------------------------------------------------
# 5. Estadístiques i gràfiques (igual que abans)
# -------------------------------------------------------------------
brand_discount = df.groupby('brand_name', observed=False)['discount_percentage'].agg(['mean', 'count']).round(1)
brand_discount = brand_discount.sort_values('mean', ascending=False)

max_discount = brand_discount['mean'].max()
top_brands = brand_discount[brand_discount['mean'] == max_discount].index.tolist()
if len(top_brands) == 1:
    top_brand_msg = f"{top_brands[0]} ({max_discount:.1f}%)"
else:
    top_brand_msg = f"Variscampanyes: {', '.join(top_brands)} (veure Taula completa de marques)"

bottom_brand = brand_discount.iloc[-1].name

out_prop = df.groupby(['brand_name', 'subcategory'], observed=False)['has_out'].mean().reset_index()
out_prop.columns = ['brand_name', 'subcategory', 'proportion_out']
global_out = df.groupby('subcategory', observed=False)['has_out'].mean().sort_values(ascending=False)
brand_counts = df['brand_name'].value_counts()
top_brand_products = brand_counts.index[0]

variety = df.groupby(['brand_name', 'subcategory'], observed=False)['num_sizes'].mean().reset_index()
variety.columns = ['brand_name', 'subcategory', 'avg_num_sizes']
variety = variety.sort_values('avg_num_sizes', ascending=False)
brand_variety = df.groupby('brand_name', observed=False)['num_sizes'].mean().sort_values(ascending=False)
top_brand_variety = brand_variety.index[0] if not brand_variety.empty else None

# Gràfica de pastís (top 15)
plt.figure(figsize=(12, 10))
top15 = brand_discount.head(15)
sizes = top15['mean']
labels = [f"{idx}\n({val:.1f}%)" for idx, val in zip(top15.index, top15['mean'])]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, labeldistance=1.05)
plt.title('Top 15 marques amb major percentatge de descompte mitjà (conservant màxims originals)')
plt.axis('equal')
plt.tight_layout()
plt.savefig(figures_dir / 'discount_top15_pie_max.png', dpi=150)
plt.close()

# Boxplot preus
plt.figure(figsize=(10, 6))
plt.boxplot(df['original_price'], vert=False)
plt.title('Distribució del preu original (boxplot)')
plt.xlabel('Preu original (€)')
plt.tight_layout()
plt.savefig(figures_dir / 'outliers_original_price_max.png')
plt.close()

# Esgotament subcategories
plt.figure(figsize=(10, 6))
top_out = global_out.head(10)
plt.barh(top_out.index, top_out.values, color='salmon')
plt.xlabel('Proporció de productes amb talles esgotades')
plt.title('Top 10 subcategories que més s\'esgoten')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(figures_dir / 'out_by_subcategory_max.png')
plt.close()

# Varietat talles
plt.figure(figsize=(10, 6))
top_variety = brand_variety.head(10)
plt.barh(top_variety.index, top_variety.values, color='mediumseagreen')
plt.xlabel('Nombre mitjà de talles per producte')
plt.title('Varietat de talles per marca (top 10)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(figures_dir / 'size_variety_by_brand_max.png')
plt.close()

# -------------------------------------------------------------------
# 6. Desar dataset net final
# -------------------------------------------------------------------
df.to_csv(output_file, index=False)
print(f"\nDataset net final (amb descomptes màxims) desat a: {output_file}")

# -------------------------------------------------------------------
# 7. Generar informe HTML
# -------------------------------------------------------------------
print("Generant informe HTML...")
out_prop_filtered = out_prop[out_prop['proportion_out'] > 0] if not out_prop.empty else pd.DataFrame()

html_content = f"""
<!DOCTYPE html>
<html lang="ca">
<head>
    <meta charset="UTF-8">
    <title>Informe Final (Màxims Descomptes) – Pràctica 2</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .container {{ max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #2c3e50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .figure {{ text-align: center; margin: 20px 0; }}
        .figure img {{ max-width: 100%; border-radius: 5px; box-shadow: 0 0 5px rgba(0,0,0,0.2); }}
        .footer {{ margin-top: 30px; font-size: 0.8em; color: #7f8c8d; text-align: center; }}
        .note {{ background-color: #d4edda; border-left: 5px solid #28a745; padding: 10px; margin: 15px 0; }}
    </style>
</head>
<body>
<div class="container">
    <h1>📊 Informe Final (conservant descomptes màxims)</h1>
    <div class="note">
        <strong>✓ Nota:</strong> S'ha conservat el descompte més alt per a cada producte (en cas de múltiples extraccions).<br>
        Descompte màxim al dataset: <strong>{df['discount_percentage'].max()}%</strong>.
    </div>
    <p><strong>Dataset net final:</strong> {len(df)} productes únics (amb el seu millor descompte).</p>

    <h2>📈 Descomptes per marca</h2>
    <p>➤ Marca amb <strong>descompte més alt</strong>: {top_brand_msg}</p>
    <p>➤ Marca amb <strong>descompte més baix</strong>: {bottom_brand} ({brand_discount.loc[bottom_brand, 'mean']:.1f}%)</p>
    <div class="figure">
        <img src="figures/discount_top15_pie_max.png" alt="Top 15 marques per descompte">
    </div>
    <h3>Taula completa de marques</h3>
    {brand_discount.reset_index().to_html(index=False, classes='dataframe')}

    <h2>📉 Subcategories que s'esgoten abans</h2>
    <h3>Per marca (subcategoria amb més proporció d'esgotament):</h3>
    <ul>
"""
if not out_prop_filtered.empty:
    for brand in out_prop_filtered['brand_name'].unique():
        brand_data = out_prop_filtered[out_prop_filtered['brand_name'] == brand]
        best = brand_data.loc[brand_data['proportion_out'].idxmax()]
        html_content += f"<li><strong>{brand}</strong>: {best['subcategory']} (esgotament: {best['proportion_out']:.1%})</li>"
else:
    html_content += "<li>No s'han detectat productes amb talles esgotades.</li>"

html_content += f"""
    </ul>
    <h3>Globalment, la subcategoria que s'esgota abans és:</h3>
    <p><strong>{global_out.index[0] if not global_out.empty else 'N/A'}</strong> (proporció d'esgotament: {global_out.iloc[0] if not global_out.empty else 0:.1%})</p>
    <div class="figure">
        <img src="figures/out_by_subcategory_max.png" alt="Esgotament per subcategoria">
    </div>

    <h2>🏷️ Marques amb més productes</h2>
    <p>➤ <strong>{top_brand_products}</strong> és la marca amb més productes ({brand_counts.iloc[0]} productes).</p>
    {brand_counts.reset_index().head(10).to_html(index=False, header=['Marca', 'Nombre de productes'], classes='dataframe')}

    <h2>📏 Varietat de talles</h2>
    <p>➤ Marca amb més varietat de talles per producte: <strong>{top_brand_variety if top_brand_variety else 'N/A'}</strong> 
    {f'({brand_variety.iloc[0]:.1f} talles per producte)' if top_brand_variety else ''}</p>
    <div class="figure">
        <img src="figures/size_variety_by_brand_max.png" alt="Varietat de talles per marca">
    </div>
    <h3>Top 10 combinacions marca + subcategoria amb més varietat:</h3>
    {variety.head(10).to_html(index=False, classes='dataframe') if not variety.empty else '<p>No hi ha dades.</p>'}

    <div class="footer">
        Informe generat conservant el descompte màxim per producte – Pràctica 2.
    </div>
</div>
</body>
</html>
"""

with open(html_report, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n✅ Informe final generat: {html_report}")
print(f"📁 Gràfiques disponibles a: {figures_dir}")