	
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
