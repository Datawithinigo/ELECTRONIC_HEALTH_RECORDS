	
# Title: Early Clinical Characteristics of ICU Patients and Their Association with Hospital Outcomes 

Group Members: Gabriel Torres, Iñigo Arriazu, Andrea Pérez  

## Brief Description:  

This project aims to explore the relationship between early demographic characteristics, pre-existing health conditions, vital signs, and initial laboratory results of critically ill ICU patients and their hospital outcomes. Using routinely collected ICU data from MIMIC-IV, we will identify patterns in early patient presentation and assess their association with hospital mortality and early ICU outcomes. 

## 1) Idea clave del proyecto (mensaje ordenado)

**Objetivo:** trabajar con *datos de UCI de muy alta frecuencia* (signos vitales) y demostrar cómo:

1. **Extraer/filtrar** los signos vitales de las **primeras 24 horas** (lo importante del estudio).
2. **Tratar alta granularidad** (p. ej. medidas cada 10s) → resampling/aggregations sin perder sentido clínico.
3. **Calcular scores** de severidad:

   * **APACHE II**: usar los **peores valores en las primeras 24h** (es como se define en el score). ([PubMed][1])
   * **SOFA**: idealmente calculado “por día”, pero en tu caso puedes reportar **SOFA en primeras 24h** y, si quieres, **max/mean** de ventanas (últimas 24h / últimos 3 días) como variante analítica. ([PubMed][2])
4. **Modelo estadístico**: aplicar vuestro script **PCA + SVM** a un dataframe final (features) y reportar performance.
5. **Conectar con un paper** (uno de scores + uno de ML en UCI).

---

## 2) Traducción a “comentarios” que puedes meter en el script

### a) Ventana de primeras 24h (lo central)

* “Para este estudio usamos **solo las primeras 24 horas** desde `icu_intime`.”
* “Los signos vitales pueden venir cada X tiempo (incluso 10s), así que **normalizamos a una rejilla temporal** (p. ej. 1 min / 5 min) y guardamos agregados.”
* “En base de datos, guardamos el dato en formato **long/tidy**: (stay_id, charttime, variable, value). Esto escala bien con alta frecuencia.”

### b) Scores

* “**APACHE II** se calcula con **el peor valor** de cada variable durante las primeras 24h.” ([PubMed][1])
* “**SOFA** evalúa 6 sistemas (respiratorio, coagulación, hígado, cardiovascular, SNC, renal). Podemos calcularlo para primeras 24h o por días.” ([PubMed][2])
* “Si no tenemos todos los componentes, calculamos una versión **parcial** (y lo declaramos explícitamente).”

### c) Modelo

* “Con las features (incluyendo scores o sus componentes), aplicamos un pipeline: imputación → escalado → PCA → SVM → métricas.”

---

## 3) “Cómo hacerlo en código” (plantilla lista para adaptar)

### 3.1. Extraer primeras 24h + resample (pandas)

```python
import pandas as pd

def first_24h(events: pd.DataFrame, icu_intime: pd.Timestamp) -> pd.DataFrame:
    """
    events: columnas esperadas: ['stay_id','charttime','variable','value']
    """
    t0 = pd.to_datetime(icu_intime)
    t1 = t0 + pd.Timedelta(hours=24)
    e = events.copy()
    e["charttime"] = pd.to_datetime(e["charttime"])
    return e[(e["charttime"] >= t0) & (e["charttime"] < t1)]

def resample_vitals(events_24h: pd.DataFrame, rule="5min", agg="median") -> pd.DataFrame:
    """
    Devuelve una tabla wide: index=charttime, columnas=variables, valores agregados.
    """
    e = events_24h.copy()
    e = e.dropna(subset=["value"])
    # long -> wide
    w = (e.pivot_table(index="charttime", columns="variable", values="value", aggfunc=agg)
           .sort_index())
    # resample temporal
    w = w.resample(rule).median()  # o mean/min/max según variable
    return w
```

### 3.2. Features “tipo score” (peor valor en 24h)

APACHE II usa “worst value in 24h” para varias variables. Si aún no metes el score completo, al menos puedes crear el “**worst vector**”.

```python
def worst_values_24h(wide_ts: pd.DataFrame) -> pd.Series:
    """
    wide_ts: index tiempo, columnas variables
    Devuelve el peor valor por variable (según definición clínica).
    Ojo: 'peor' no siempre es max: depende de la variable.
    """
    worst = {}

    # Ejemplos típicos (ajusta a tus nombres reales):
    # HR peor = max
    if "HR" in wide_ts: worst["HR_worst"] = wide_ts["HR"].max(skipna=True)
    # MAP peor = min
    if "MAP" in wide_ts: worst["MAP_worst"] = wide_ts["MAP"].min(skipna=True)
    # Temp peor = max desviación o extremos (simplificado aquí)
    if "Temp" in wide_ts:
        worst["Temp_max"] = wide_ts["Temp"].max(skipna=True)
        worst["Temp_min"] = wide_ts["Temp"].min(skipna=True)

    return pd.Series(worst)
```

### 3.3. SOFA (implementación por componentes)

Aquí lo correcto es **usar el peor subscore** por sistema en la ventana (p. ej. primeras 24h). La definición original es Vincent et al. 1996. ([PubMed][2])
Si estás en MIMIC (o similar), puedes apoyarte en el SQL de referencia y adaptarlo. ([GitHub][3])

Ejemplo **muy simplificado** (solo para enseñarte estructura):

```python
def sofa_resp(pao2_fio2):
    # Ejemplo basado en tabla estándar (simplificado)
    if pd.isna(pao2_fio2): 
        return None
    if pao2_fio2 >= 400: return 0
    if pao2_fio2 < 100: return 4
    if pao2_fio2 < 200: return 3
    if pao2_fio2 < 300: return 2
    return 1

def compute_sofa_24h(features: dict) -> dict:
    """
    features: diccionario con inputs (worst o min/max) en 24h
    """
    out = {}
    out["SOFA_resp"] = sofa_resp(features.get("PaO2FiO2_worst"))
    # Repetir para: platelets, bilirubin, creatinine/urine, MAP/vasopressors, GCS
    # ...
    # total (solo si todos presentes)
    parts = [v for k,v in out.items() if k.startswith("SOFA_") and v is not None]
    out["SOFA_total_partial"] = sum(parts) if parts else None
    return out
```

---

## 4) Paper(s) para “relacionar” rápido (y defendible)

* **SOFA (paper original)**: Vincent et al., 1996 (definición del score). ([PubMed][2])
* **APACHE II (paper original)**: Knaus et al., 1985 (definición; usa variables fisiológicas + edad + crónicos; rango 0–71). ([PubMed][1])
* **Por qué “peor valor en 24h” importa**: comparación admission vs worst-24h (Ho 2005). ([PMC][4])
* **ML en UCI con SVM**: ejemplo clásico (Computing in Cardiology Challenge 2012) con SVM para mortalidad. ([CinC][5])

---

Si quieres, pégame aquí **las últimas ~80–150 líneas** del “script nuevo” (donde sale el diccionario de `items`) y te lo dejo **convertido a**:

* comentarios limpios (ES o EN),
* mapping claro de variables → score,
* y un bloque final que construya el dataframe para el PCA+SVM.

[1]: https://pubmed.ncbi.nlm.nih.gov/3928249/?utm_source=chatgpt.com "APACHE II: a severity of disease classification system"
[2]: https://pubmed.ncbi.nlm.nih.gov/8844239/?utm_source=chatgpt.com "The SOFA (Sepsis-related Organ Failure Assessment ..."
[3]: https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/severityscores/sofa.sql?utm_source=chatgpt.com "sofa.sql - MIT-LCP/mimic-code"
[4]: https://pmc.ncbi.nlm.nih.gov/articles/PMC1550848/?utm_source=chatgpt.com "A comparison of admission and worst 24-hour Acute ..."
[5]: https://www.cinc.org/archives/2012/pdf/0257.pdf?utm_source=chatgpt.com "Predicting Mortality of ICU Patients using a Cascaded SVM ..."




Sí: ese “pivot” (diagnóstico agrupado por sistema + necesidad de soporte respiratorio → mortalidad) es **mucho más defendible y acotado** que intentar clavar SOFA/APACHE completos si os faltan variables. Y además encaja perfecto con tu punto fuerte: **intervenciones respiratorias + motivos de ingreso bien caracterizados**.

Aquí tienes una forma **muy clara** de clasificarlo y convertirlo en análisis + modelo.

---

## 1) Cómo clasificar los diagnósticos (3 opciones, de más rápida a más “pro”)

### Opción A (rápida y sólida): capítulos ICD-10 por prefijo

Agrupas por **primera letra/rango** del ICD-10 y lo mapeas a sistema. Esto es defendible porque ICD ya está organizado por capítulos/sistemas. ([doc.ukdataservice.ac.uk][1])

Ejemplo directo (lo que tú ya estás haciendo):

* **I00–I99** → Cardiovascular
* **N00–N99** → Renal/Urinario
* **G00–G99** → Neurológico
* **F00–F99** → Psiquiátrico
* **J00–J99** → Respiratorio
* **K00–K95** → Digestivo
* etc. ([IHACPA][2])

✅ Ventaja: 0 dependencia externa, implementas en 10 min.
⚠️ Límite: menos granular.

### Opción B (mejor para research): CCSR (HCUP)

Usa **Clinical Classifications Software Refined (CCSR)** que agrupa ICD-10-CM en categorías clínicas y **22 “body systems”**. Es literalmente una herramienta hecha para esto. ([hcup-us.ahrq.gov][3])

✅ Ventaja: súper defendible en un paper-style.
⚠️ Límite: requiere bajar el mapping.

### Opción C (tu propia taxonomía)

Válida si lo explicas bien: “Agrupamos diagnósticos en X dominios clínicos para reducir dimensionalidad”.
✅ Ventaja: flexible.
⚠️ Límite: más fácil que te pregunten “¿por qué así?”

**Recomendación práctica:** haced A para entregar seguro; si os da tiempo, añadís B como “sensibility analysis”.

---

## 2) Cómo definir “soporte respiratorio” (exposición) sin liarla

Define una variable ordinal (máxima intervención en primeras 24h):

0. **Sin soporte**
1. **Oxígeno** (nasal/mascarilla)
2. **NIV** (CPAP/BiPAP)
3. **IMV** (intubación / ventilación mecánica invasiva)

Eso os permite:

* tablas claras,
* modelo interpretable,
* y comparación por grupos.

Además hay literatura suficiente para justificar que “tipo de soporte” y “ventilación” se asocian a resultados (aunque hay confusión por severidad). Por ejemplo: ventilación mecánica asociada a mayor mortalidad en cohortes de UCI cardiaca, y estudios comparando NIV vs oxígeno en subpoblaciones. ([PMC][4])

---

## 3) Qué pregunta exacta respondéis (y cómo se ve en resultados)

### Pregunta principal (perfecta para entregar)

**“¿Cómo cambia la mortalidad según (a) el tipo de diagnóstico y (b) la necesidad de soporte respiratorio en primeras 24h?”**

### Hipótesis defendible

El efecto del soporte respiratorio **no es igual** en todos los dominios diagnósticos → interacción.

---

## 4) Análisis estadístico mínimo pero “de calidad”

### 4.1 Descriptivo (1 slide/figura)

Tabla por grupo diagnóstico:

* N pacientes
* % con oxígeno / NIV / IMV
* mortalidad (ICU u hospitalaria)

### 4.2 Modelo principal (logístico con interacción)

**Outcome:** mortalidad (0/1).
**Modelo:**
[
\text{mortalidad} \sim \text{soporte} + \text{grupo_dx} + (\text{soporte} \times \text{grupo_dx}) + \text{covariables}
]

**Covariables mínimas** (para que no os digan “confounding”):

* edad, sexo
* “baseline severity proxy” (por ejemplo: lactato, MAP min, SpO2 min, creatinina max… lo que tengáis)
* comorbilidad simple si existe

Con eso ya puedes reportar:

* OR del soporte respiratorio dentro de cada grupo dx
* y/o diferencias de riesgo predichas

---

## 5) Modelo “ML” (sin pelearse con PCA+SVM)

Si os exigen PCA+SVM, se puede, pero para *presentación* yo haría:

* **Logistic regression regularizada** (interpretable) como baseline
* **SVM** como comparativa

Si insistís con PCA+SVM:

* One-hot a `grupo_dx` y `soporte`
* escalado
* PCA opcional (pero ojo: PCA con muchas dummies a veces no aporta)

---

## 6) Código (plantilla ultra rápida)

### 6.1 Map ICD-10 → grupo (Opción A)

```python
def icd10_to_group(icd):
    if icd is None: 
        return "Unknown"
    icd = icd.strip().upper()

    # usa primera letra y rango si tienes numérico
    letter = icd[0]

    if letter == "I": return "Cardiovascular"
    if letter == "N": return "Renal"
    if letter == "G": return "Neurologic"
    if letter == "F": return "Psychiatric"
    if letter == "J": return "Respiratory"
    if letter == "K": return "Gastrointestinal"
    # ... añade lo que necesitéis
    return "Other"
```

### 6.2 Soporte respiratorio (máximo en 24h)

```python
support_rank = {"none":0, "o2":1, "niv":2, "imv":3}

def max_support_first24h(events):
    # events: lista/df con soporte detectado en 24h
    # devuelve el máximo
    if len(events) == 0:
        return 0
    return max(support_rank[x] for x in events)
```

---

## 7) Papers para justificar (sin volverte loco)

* **Clasificación diagnóstica**: ICD-10 estructura por capítulos/sistemas (os justifica vuestro grouping). ([doc.ukdataservice.ac.uk][1])
* **Grouping estándar**: CCSR (HCUP) agrupa ICD-10-CM en categorías clínicas y 22 sistemas. ([hcup-us.ahrq.gov][3])
* **Respiratory support y mortalidad**: ejemplos en cohortes/ensayos (ventilación asociada a outcomes; NIV vs oxígeno según contexto). ([PMC][4])
* Si queréis algo súper “24h” para ventilación: asociación de parámetros de ventilación tempranos con mortalidad (primeras 24h). ([journal.chestnet.org][5])

---

### Mi recomendación final (práctica)

1. Quedaos con **ICD-10 capítulo → grupo** + **soporte respiratorio en 24h** + **mortalidad**.
2. Haced **descriptivo + logística con interacción**.
3. Si sobra tiempo, añadid el modelo ML (SVM) como “bonus”.

Si me pegas **un ejemplo real** de:

* cómo tienes los diagnósticos (código ICD o texto),
* y cómo detectas las intervenciones respiratorias (campos/eventos),
  te devuelvo **el mapping exacto** y un bloque de código que ya genere el dataframe final (`stay_id`, `grupo_dx`, `support_24h`, `mortality`, covariables).

[1]: https://doc.ukdataservice.ac.uk/doc/8770/mrdoc/pdf/icd-10_international_statistical_classification_of_diseases_and_related_health_problems-v2-eng.pdf?utm_source=chatgpt.com "ICD-10 International statistical classification of diseases ..."
[2]: https://www.ihacpa.gov.au/sites/default/files/2022-08/icd-10-am_chronicle_-_eleventh_edition.pdf?utm_source=chatgpt.com "ICD-10-AM Disease Code List"
[3]: https://hcup-us.ahrq.gov/toolssoftware/ccsr/dxccsr.jsp?utm_source=chatgpt.com "Clinical Classifications Software Refined (CCSR) for ICD-10 ..."
[4]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12332906/?utm_source=chatgpt.com "A propensity score-matched cohort study"
[5]: https://journal.chestnet.org/article/S0012-3692%2825%2900403-9/fulltext?utm_source=chatgpt.com "The Association Between Mechanical Power Within the First ..."



GABRI : PRINCIPALMENTE QUEDARNOS CON demographics y pre-existing conditions