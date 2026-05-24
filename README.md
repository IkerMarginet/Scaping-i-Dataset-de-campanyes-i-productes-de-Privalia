# Projecte Privalia: Web-Scraping, Neteja i Anàlisi Avançada de Dades 🛍️📊

Aquest projecte és un cas pràctic complet que abasta tot el cicle de vida de la dada: des de l'extracció automatitzada (*web-scraping*) de dades del lloc web de [Privalia](https://es.privalia.com/) fins a la seva posterior integració, neteja, modelització avançada i contrast d'hipòtesis estadístiques.

Ha estat desenvolupat en el marc del **Màster en Data Science** de la **Universitat Oberta de Catalunya (UOC)** per a l'assignatura *Tipologia i Cicle de vida de les dades* (Pràctica 1 i Pràctica 2).

---

## 👥 Membres de l'equip
*   **Iker Marginet Ballester**
*   **Albert Pérez Costa**

---

## 🗂️ Estructura del Repositori i Descripció del Projecte

A continuació, es detalla l'estructura organitzativa i endreçada del repositori de cara al lliurament final:

```text
privalia-scraping/
├── dataset/                           # Directori central de dades i gràfics
│   ├── figures/                       # Gràfics de la fase d'anàlisi exploratòria (EDA) inicial
│   │   ├── discount_top15_pie_max.png # Distribució de descomptes de les top marques (gràfic pastís)
│   │   ├── out_by_subcategory_max.png # Rati d'esgotament de talles per subcategoria (gràfic barres)
│   │   └── size_variety_by_brand_max.png # Varietat de talles mitjana per producte per marca
│   ├── resultats_analisi/             # Sortides de la fase d'anàlisi avançada i models
│   │   ├── figures/                   # Gràfics de models (Elbow method, Clústers, ROC, etc.)
│   │   └── metrics/                   # Fitxers JSON i CSV amb mètriques formals obtingudes
│   ├── informe_resultats_max_discount.html # Informe exploratori HTML auto-generat pel script de neteja
│   ├── privalia_campaigns.csv         # Dataset original de campanyes extretes (Pràctica 1)
│   ├── privalia_products.csv          # Dataset original de productes extrets (Pràctica 1)
│   └── privalia_clean_max_discount.csv # Dataset final netejat, unificat i deduplicat (Pràctica 2)
├── docs/                              # Depuració i captures auxiliars del web-scraper
├── practica2_code/                    # Codi font de la Pràctica 2 (Neteja i Anàlisi de Dades)
│   ├── clean-dataset.py               # Script de neteja, gestió de nuls, outliers i enginyeria de variables
│   └── analisis_privalia.py           # Script d'anàlisi avançada (contrastos, K-Means i Random Forest)
├── source/                            # Codi font de la Pràctica 1 (Orquestració del Web-Scraper)
│   ├── main.py                        # Executor principal interactiu del crawler i parseig
│   ├── config.py                      # Configuracions, credencials i limits de descàrrega
│   ├── crawler.py                     # Interacció robòtica per Selenium (scroll dinàmic)
│   ├── parser.py                      # Extracció estàtica d'etiquetes mitjançant BeautifulSoup
│   └── storage.py                     # Gestió de l'emmagatzematge i escriptura en CSV
├── requirements.txt                   # Llistat de biblioteques base i dependències de Python
└── README.md                          # Fitxer informatiu complet del projecte (aquest fitxer)
```

---

## ⚙️ Requisits i Instal·lació

El projecte s'ha programat íntegrament en **Python 3.8+**. Per a fer-lo funcionar:

1. **Clona aquest repositori** i accedeix a la carpeta principal:
   ```bash
   cd privalia-scraping
   ```
2. **Crea i activa un entorn virtual** (recomanat):
   ```bash
   # Creació
   python -m venv venv
   # Activació (Windows)
   venv\Scripts\activate
   # Activació (Mac/Linux)
   source venv/bin/activate
   ```
3. **Instal·la totes les dependències necessàries**:
   ```bash
   pip install -r requirements.txt
   ```

*Nota: Per a la Pràctica 1 (web-scraping), es requereix tenir instal·lat un navegador compatible amb Selenium (com Google Chrome o Edge).*

---

## 🖥️ Execució i Ús de les Eines

### 1. Neteja de dades (Pràctica 2)
L'script de neteja realitza la unió de campanyes i productes, neteja els nuls, analitza outliers, converteix formats bruts, genera variables d'enginyeria de característiques (`num_sizes`, `has_out`) i crea els gràfics EDA descriptius primaris.

Per a executar-lo:
```bash
python practica2_code/clean-dataset.py
```
Aquest script:
*   Llegirà `dataset/privalia_campaigns.csv` i `dataset/privalia_products.csv`.
*   Generarà el fitxer unificat net a `dataset/privalia_clean_max_discount.csv`.
*   Generarà les figures exploratòries a `dataset/figures/`.
*   Generarà l'informe gràfic interactiu a `dataset/informe_resultats_max_discount.html`.

### 2. Anàlisi Avançada i Models (Pràctica 2)
L'script d'anàlisi de dades realitza el contrast d'hipòtesi no paramètric (Mann-Whitney U), el clustering per perfils de preu (K-Means) i la classificació predictiva de gangues (Random Forest).

#### Mode Interactiu (Menú de consola)
```bash
python practica2_code/analisis_privalia.py
```
Se't demanarà triar quins models vols entrenar i avaluar (només supervisat, només no supervisat, o l'anàlisi complet).

#### Mode Automatitzat (Útil per a execucions des de terminals o plataformes d'integració)
Pots passar directament per paràmetre l'opció que vulguis per tal d'evitar el menú de consola:
```bash
# Executa l'anàlisi complet directament (opció 3: models + contrast) i desa els resultats
python practica2_code/analisis_privalia.py 3
```

Aquest script:
*   Llegirà el dataset net `dataset/privalia_clean_max_discount.csv` (i cridarà automàticament a `clean-dataset.py` si no existís).
*   Emmagatzemarà totes les visualitzacions avançades (Boxplot+Violin, Corba del Colze, Clústers, Corba ROC, Matrius de confusió percentuals i Feature Importances) a `dataset/resultats_analisi/figures/`.
*   Desarà totes les mètriques estructurades de rendiment en fitxers JSON/CSV i un resum textual d'execució a `dataset/resultats_analisi/metrics/`.

### 3. Web-Scraper (Pràctica 1)
Per arrencar el procés d'extracció dinàmica de dades de Privalia:
```bash
python source/main.py
```
*(Es faran preguntes sobre si es vol activar el mode depuració o si es vol reprendre l'extracció per a evitar duplicar marques ja descarregades)*.

---

## 📊 Dataset i DOI de Zenodo

El dataset generat durant el projecte es troba preservat i catalogat a la plataforma de dades obertes d'investigació **Zenodo** per garantir-ne la replicabilitat completa de la recerca:

*   **DOI del Dataset**: `[https://doi.org/10.5281/zenodo.19441893]`

---

## ⚖️ Consideracions Ètiques i d'Ús

Les dades recollides s'utilitzen exclusivament amb finalitats acadèmiques i didàctiques. El *scraper* dissenyat a la Pràctica 1 respecta polítiques ètiques de descàrrega mitjançant l'aplicació de retards constants i pauses temporitzades que eviten el tràfic abusiu sobre la plataforma de Privalia, emulant el comportament d'un usuari estàndard.

---
**Tipologia i Cicle de vida de les dades | Màster en Data Science | UOC**
