import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="AgroSense",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

  /* ── Root Tokens ── */
  :root {
    --soil:   #6B4F3A;
    --leaf:   #a9fcb6;
    --sky:    #5B8DB8;
    --cream:  #7b9678;
    --sand:   #E8DDD0;
    --text:   #2C2416;
    --muted:  #7A6A58;
    --border: #D6CCC0;
    --card:   #34703d;
    --radius: 12px;
  }

  /* ── Base ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
  }

  .stApp {
    background: var(--cream);
  }

  /* ── Hero Header ── */
  .hero {
    background: linear-gradient(135deg, #3A7D44 0%, #2C5F34 60%, #1E4024 100%);
    border-radius: 18px;
    padding: 2.4rem 2.8rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: "🌾";
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.18;
  }
  .hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #FFFFFF;
    margin: 0 0 0.3rem;
    letter-spacing: -0.5px;
  }
  .hero p {
    color: #C8E6C9;
    font-size: 1rem;
    margin: 0;
    font-weight: 300;
  }

  /* ── Mode Toggle ── */
  div[data-testid="stRadio"] > div {
    display: flex;
    gap: 0.5rem;
    flex-direction: row !important;
  }

  /* ── Section Cards ── */
  .section-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem 1rem;
    margin-bottom: 1.2rem;
  }
  .section-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--leaf);
    border-bottom: 1.5px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
  }

  /* ── Texture Bar ── */
  .texture-bar-wrap {
    background: var(--sand);
    border-radius: 8px;
    height: 20px;
    width: 100%;
    overflow: hidden;
    margin: 0.8rem 0;
    display: flex;
  }
  .texture-seg { height: 100%; transition: width 0.4s ease; display: flex; align-items: center; justify-content: center; font-size: 0.65rem; font-weight: 600; color: #fff; }
  .seg-sand  { background: #C9A96E; }
  .seg-clay  { background: #8B5E3C; }
  .seg-silt  { background: #A0B87A; }

  .texture-sum-ok  { color: #3A7D44; font-weight: 600; font-size: 0.82rem; margin-top: 0.3rem; }
  .texture-sum-err { color: #C0392B; font-weight: 600; font-size: 0.82rem; margin-top: 0.3rem; }

  /* ── Predict Button ── */
  .stButton > button {
    background: linear-gradient(135deg, #3A7D44, #2C5F34) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2.5rem !important;
    border-radius: 50px !important;
    border: none !important;
    width: 100%;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.88; }

  /* ── Result Box ── */
  .result-box {
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-top: 1.2rem;
    border-left: 5px solid;
  }
  .result-strongly-acidic   { background:#FDF3F2; border-color:#E74C3C; }
  .result-moderately-acidic { background:#FEF9EC; border-color:#F39C12; }
  .result-neutral           { background:#F0FBF2; border-color:#27AE60; }
  .result-moderately-alkaline{ background:#EFF6FB; border-color:#2980B9; }
  .result-strongly-alkaline { background:#F3EEF9; border-color:#8E44AD; }

  .result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    margin: 0 0 0.25rem;
  }
  .result-confidence { font-size: 0.9rem; color: var(--muted); }

  /* ── Confidence Bar ── */
  .conf-bar-bg { background: #E8E0D8; border-radius: 50px; height: 10px; width: 100%; margin: 0.6rem 0 1.2rem; }
  .conf-bar-fg { height: 10px; border-radius: 50px; transition: width 0.6s ease; }

  /* ── Interpretation Box ── */
  .interp-box {
    background: #F6FAF7;
    border: 1px solid #C8E6C9;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.9rem;
    line-height: 1.65;
    color: var(--text);
  }
  .interp-box strong { color: var(--leaf); }

  /* ── Error Banner ── */
  .error-banner {
    background: #FDF3F2;
    border: 1px solid #E74C3C;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    color: #C0392B;
    font-size: 0.88rem;
    margin-top: 0.8rem;
  }

  /* ── Batch Table ── */
  .batch-result-header {
    font-weight: 600;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
    margin-top: 1.5rem;
  }

  /* ── Tooltips ── */
  abbr[title] {
    cursor: help;
    text-decoration-style: dotted;
    text-decoration-color: var(--muted);
  }

  /* ── Metric display ── */
  .pct-display {
    background: var(--sand);
    border-radius: 8px;
    padding: 0.4rem 0.8rem;
    font-size: 0.82rem;
    font-family: monospace;
    color: var(--text);
    display: inline-block;
    margin-top: 0.2rem;
  }

  /* ── Uploader ── */
  [data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    background: var(--cream) !important;
  }

  /* small helper label */
  .field-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--muted);
    margin-bottom: 0.1rem;
  }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load('Weight Model/Ensemble (Multi-Layer Perceptrons, XGBoost, Random Forest).pkl')

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

PH_LABELS = {
    0: 'Strongly Acidic',
    1: 'Moderately Acidic',
    2: 'Neutral',
    3: 'Moderately Alkaline',
    4: 'Strongly Alkaline',
}

PH_COLORS = {
    'Strongly Acidic':    '#E74C3C',
    'Moderately Acidic':  '#F39C12',
    'Neutral':            '#27AE60',
    'Moderately Alkaline':'#2980B9',
    'Strongly Alkaline':  '#8E44AD',
}

PH_CSS_CLASS = {
    'Strongly Acidic':    'result-strongly-acidic',
    'Moderately Acidic':  'result-moderately-acidic',
    'Neutral':            'result-neutral',
    'Moderately Alkaline':'result-moderately-alkaline',
    'Strongly Alkaline':  'result-strongly-alkaline',
}

INTERPRETATIONS = {
    'Strongly Acidic': (
        "<strong>Soil is strongly acidic (pH < 5.0).</strong> "
        "Most nutrients — especially phosphorus, calcium, and magnesium — become less available at this level. "
        "Heavy liming with agricultural lime (CaCO₃) or dolomite is recommended to raise pH. "
        "Sensitive crops may experience aluminium and manganese toxicity. "
        "Test for micronutrient deficiencies and consider acid-tolerant cover crops while liming."
    ),
    'Moderately Acidic': (
        "<strong>Soil is moderately acidic (pH 5.0–6.4).</strong> "
        "Many crops grow well in this range, but phosphorus availability begins to decline below pH 6.0. "
        "Light liming applications can improve nutrient uptake. "
        "Nitrogen-fixing legumes and most vegetables thrive here; monitor for calcium and magnesium deficiencies."
    ),
    'Neutral': (
        "<strong>Soil is neutral (pH 6.5–7.3).</strong> "
        "This is the optimal range for most crops — nutrient availability is at its peak and microbial activity is high. "
        "Maintain organic matter and balanced fertilisation to sustain productivity. "
        "No immediate corrective action is needed; periodic monitoring is sufficient."
    ),
    'Moderately Alkaline': (
        "<strong>Soil is moderately alkaline (pH 7.4–8.5).</strong> "
        "Iron, manganese, zinc, and boron may become deficient. Phosphorus fixation also increases. "
        "Applying sulphur, organic matter, or ammonium-based fertilisers can gradually lower pH. "
        "Chlorosis in young leaves may indicate iron or manganese deficiency requiring foliar correction."
    ),
    'Strongly Alkaline': (
        "<strong>Soil is strongly alkaline (pH > 8.5).</strong> "
        "Severe micronutrient lockout is likely. Sodium accumulation (indicated by high SAR/ESP) "
        "may also damage soil structure. Gypsum (CaSO₄) application can displace sodium and improve tilth. "
        "Elemental sulphur or acid-forming amendments should be considered. Specialist soil remediation may be required."
    ),
}

TOOLTIPS = {
    "EC":   "EC (dS m⁻¹) — Measures soil electrical conductivity, an indicator of salinity.",
    "OM":   "OM (mass %, Walkley-Black) — Organic matter percentage; key for soil fertility and water retention.",
    "BD":   "BD (g cm⁻³) — Bulk density: dry soil mass per unit volume.",
    "N":    "N (mass %, Kjeldahl) — Soil nitrogen content; essential for plant growth and leaf development.",
    "P":    "P (mg kg⁻¹, Olsen/Bray) — Soil phosphorus; critical for root development and flowering.",
    "K":    "K (Cmol(+) kg⁻¹) — Soil potassium; regulates water, enzymes, and disease resistance.",
    "Ca":   "Ca (Cmol(+) kg⁻¹) — Soil calcium; stabilises pH, soil structure, and root growth.",
    "Mg":   "Mg (Cmol(+) kg⁻¹) — Soil magnesium; central atom in chlorophyll, vital for photosynthesis.",
    "Na":   "Na (Cmol(+) kg⁻¹) — Soil sodium; high levels cause sodicity and structural damage.",
    "CEC":  "CEC (Cmol(+) kg⁻¹, NH₄OAc) — Cation Exchange Capacity: soil's ability to retain and exchange positive ions.",
    "SAR":  "SAR — Sodium Adsorption Ratio: index of sodium proportion relative to Ca and Mg.",
    "ESP":  "ESP (%) — Exchangeable Sodium Percentage: fraction of CEC occupied by Na⁺.",
    "SAND": "SAND (mass %, Bouyoucos) — Coarse particle fraction; affects drainage and aeration.",
    "SILT": "SILT (mass %, Bouyoucos) — Medium particle fraction; improves water retention and texture.",
    "CLAY": "CLAY (mass %, Bouyoucos) — Fine particle fraction; affects shrink-swell and nutrient holding.",
}

REQUIRED_COLS = ["P", "SAND", "CLAY", "N", "K", "Ca", "Mg", "Na", "CEC", "SAR", "ESP", "% Ca", "% Mg", "% K"]
BATCH_INPUT_COLS = ["P", "SAND", "CLAY", "N", "K", "Ca", "Mg", "Na", "CEC", "SAR", "ESP"]


def tip(key):
    """Return a Streamlit help string from the TOOLTIPS dict."""
    return TOOLTIPS.get(key, "")


def compute_percentages(Ca, Mg, K):
    total = Ca + Mg + K
    if total > 0:
        return (Ca / total) * 100, (Mg / total) * 100, (K / total) * 100
    return 0.0, 0.0, 0.0


def build_input_df(P, SAND, CLAY, N, K, Ca, Mg, Na, CEC, SAR, ESP):
    perc_Ca, perc_Mg, perc_K = compute_percentages(Ca, Mg, K)
    return pd.DataFrame([[P, SAND, CLAY, N, K, Ca, Mg, Na, CEC, SAR, ESP, perc_Ca, perc_Mg, perc_K]],
                        columns=REQUIRED_COLS)


def predict_single(df):
    aligned = df[model.feature_names_in_]
    raw = model.predict(aligned)
    rounded = round(float(raw[0]))
    label = PH_LABELS.get(rounded, "Unknown")

    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(aligned)[0]
            confidence = float(proba[rounded]) * 100
        except Exception:
            pass
    return label, confidence, rounded


def confidence_bar_html(conf, color):
    if conf is None:
        return ""
    return f"""
    <div class="conf-bar-bg">
      <div class="conf-bar-fg" style="width:{conf:.1f}%; background:{color};"></div>
    </div>
    """


def render_result(label, confidence):
    color = PH_COLORS.get(label, "#888")
    css   = PH_CSS_CLASS.get(label, "")
    conf_str = f"{confidence:.1f}%" if confidence is not None else "N/A"
    bar = confidence_bar_html(confidence, color)
    st.markdown(f"""
    <div class="result-box {css}">
      <div class="result-label" style="color:{color};">{label}</div>
      <div class="result-confidence">Confidence: <strong>{conf_str}</strong></div>
      {bar}
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="interp-box">{INTERPRETATIONS.get(label, "")}</div>', unsafe_allow_html=True)


st.markdown("""
<div class="hero">
  <h1>🌱 AgroSense</h1>
  <p>Soil pH prediction powered by ensemble machine learning — enter soil parameters to classify pH category.</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Model could not be loaded. Please check the model path.\n\n`{model_error}`")
    st.stop()

mode = st.radio("Prediction mode", ["Single Prediction", "Batch Prediction"], horizontal=True, label_visibility="collapsed")

st.markdown("---")

if mode == "Single Prediction":

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🪨 Soil Texture</div>', unsafe_allow_html=True)
        st.caption("Sand + Clay + Silt must sum to exactly 100 %.")

        t_col1, t_col2, t_col3 = st.columns(3)
        with t_col1:
            SAND = st.number_input("Sand (%)", min_value=0.0, max_value=100.0,
                                   value=40.0, step=0.5, help=tip("SAND"))
        with t_col2:
            CLAY = st.number_input("Clay (%)", min_value=0.0, max_value=100.0,
                                   value=30.0, step=0.5, help=tip("CLAY"))
        with t_col3:
            SILT = st.number_input("Silt (%)", min_value=0.0, max_value=100.0,
                                   value=30.0, step=0.5, help=tip("SILT"))

        texture_sum = SAND + CLAY + SILT
        sand_w = SAND / 100 if texture_sum > 0 else 0
        clay_w = CLAY / 100 if texture_sum > 0 else 0
        silt_w = SILT / 100 if texture_sum > 0 else 0

        st.markdown(f"""
        <div class="texture-bar-wrap">
          <div class="texture-seg seg-sand"  style="width:{sand_w*100:.1f}%">{"Sand"  if sand_w > 0.12 else ""}</div>
          <div class="texture-seg seg-clay"  style="width:{clay_w*100:.1f}%">{"Clay"  if clay_w > 0.12 else ""}</div>
          <div class="texture-seg seg-silt"  style="width:{silt_w*100:.1f}%">{"Silt"  if silt_w > 0.12 else ""}</div>
        </div>
        """, unsafe_allow_html=True)

        if abs(texture_sum - 100.0) < 0.01:
            st.markdown('<div class="texture-sum-ok">✓ Total: 100.0 %</div>', unsafe_allow_html=True)
            texture_valid = True
        else:
            st.markdown(f'<div class="texture-sum-err">✗ Total: {texture_sum:.1f} % — must equal 100 %</div>', unsafe_allow_html=True)
            texture_valid = False

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🌿 Soil Nutrients</div>', unsafe_allow_html=True)

        n_col1, n_col2 = st.columns(2)
        with n_col1:
            N = st.number_input("N — Nitrogen (mass %)", min_value=0.00, max_value=1.00,
                                value=0.20, step=0.01, format="%.3f", help=tip("N"))
        with n_col2:
            P = st.number_input("P — Phosphorus (mg kg⁻¹)", min_value=0.0, max_value=300.0,
                                value=15.0, step=0.5, format="%.2f", help=tip("P"))

        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚗️ Base Chemicals (Cmol(+) kg⁻¹)</div>', unsafe_allow_html=True)

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            Ca = st.number_input("Ca — Calcium", min_value=0.00, max_value=60.00,
                                 value=12.00, step=0.1, format="%.2f", help=tip("Ca"))
            K  = st.number_input("K — Potassium", min_value=0.00, max_value=5.00,
                                 value=0.50, step=0.01, format="%.3f", help=tip("K"))
        with b_col2:
            Mg = st.number_input("Mg — Magnesium", min_value=0.00, max_value=20.00,
                                 value=3.00, step=0.1, format="%.2f", help=tip("Mg"))
            Na = st.number_input("Na — Sodium", min_value=0.00, max_value=20.00,
                                 value=0.50, step=0.01, format="%.3f", help=tip("Na"))

        perc_Ca, perc_Mg, perc_K = compute_percentages(Ca, Mg, K)
        st.markdown(f"""
        <div style="display:flex; gap:0.6rem; margin-top:0.4rem; flex-wrap:wrap;">
          <span class="pct-display">% Ca: {perc_Ca:.1f}</span>
          <span class="pct-display">% Mg: {perc_Mg:.1f}</span>
          <span class="pct-display">% K: {perc_K:.1f}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Soil Metrics</div>', unsafe_allow_html=True)

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            CEC = st.number_input("CEC", min_value=0.0, max_value=100.0,
                                  value=20.0, step=0.5, format="%.2f", help=tip("CEC"))
        with m_col2:
            SAR = st.number_input("SAR", min_value=0.0, max_value=60.0,
                                  value=2.0, step=0.1, format="%.2f", help=tip("SAR"))
        with m_col3:
            ESP = st.number_input("ESP (%)", min_value=0.0, max_value=100.0,
                                  value=5.0, step=0.5, format="%.2f", help=tip("ESP"))

        st.markdown('</div>', unsafe_allow_html=True)

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        predict_clicked = st.button("🔍 Predict Soil pH", use_container_width=True)

    if predict_clicked:
        errors = []
        if not texture_valid:
            errors.append("Sand + Clay + Silt must sum to exactly 100 %.")

        if errors:
            for e in errors:
                st.markdown(f'<div class="error-banner">⚠️ {e}</div>', unsafe_allow_html=True)
        else:
            try:
                df = build_input_df(P, SAND, CLAY, N, K, Ca, Mg, Na, CEC, SAR, ESP)
                label, confidence, _ = predict_single(df)
                render_result(label, confidence)
            except Exception as ex:
                st.markdown(f'<div class="error-banner">⚠️ Prediction failed: {ex}</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="section-card">
      <div class="section-title">📂 Batch Prediction</div>
    """, unsafe_allow_html=True)

    st.write("Upload a CSV file with the following columns:")
    st.code(", ".join(BATCH_INPUT_COLS), language="text")
    st.caption("The model will automatically compute **% Ca**, **% Mg**, and **% K** from Ca, Mg, K values.")

    template_df = pd.DataFrame(columns=BATCH_INPUT_COLS)
    csv_template = template_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download CSV template",
        data=csv_template,
        file_name="agrosense_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        batch_errors = []

        try:
            df_batch = pd.read_csv(uploaded_file)
        except Exception as ex:
            st.markdown(f'<div class="error-banner">⚠️ Could not read CSV: {ex}</div>', unsafe_allow_html=True)
            st.stop()

        missing_cols = [c for c in BATCH_INPUT_COLS if c not in df_batch.columns]
        extra_cols   = [c for c in df_batch.columns  if c not in BATCH_INPUT_COLS]

        if missing_cols:
            batch_errors.append(f"Missing columns: **{', '.join(missing_cols)}**")

        if batch_errors:
            for e in batch_errors:
                st.markdown(f'<div class="error-banner">⚠️ {e}</div>', unsafe_allow_html=True)
            st.stop()

        non_numeric = []
        for col in BATCH_INPUT_COLS:
            if not pd.api.types.is_numeric_dtype(df_batch[col]):
                non_numeric.append(col)

        if non_numeric:
            st.markdown(f'<div class="error-banner">⚠️ Non-numeric values found in: {", ".join(non_numeric)}. '
                        f'Please ensure all columns contain numbers.</div>', unsafe_allow_html=True)
            st.stop()

        rows_with_nan = df_batch[BATCH_INPUT_COLS].isnull().any(axis=1)
        if rows_with_nan.any():
            bad_rows = list(rows_with_nan[rows_with_nan].index + 2) 
            st.markdown(f'<div class="error-banner">⚠️ Missing values detected in rows: {bad_rows}. '
                        f'Please fill or remove incomplete rows before uploading.</div>', unsafe_allow_html=True)
            st.stop()

        if all(c in df_batch.columns for c in ["SAND", "CLAY"]):
            pass

        try:
            df_batch["% Ca"], df_batch["% Mg"], df_batch["% K"] = zip(
                *df_batch.apply(lambda r: compute_percentages(r["Ca"], r["Mg"], r["K"]), axis=1)
            )

            df_model = df_batch[model.feature_names_in_]
            raw_preds = model.predict(df_model)
            rounded_preds = [round(float(v)) for v in raw_preds]
            labels = [PH_LABELS.get(r, "Unknown") for r in rounded_preds]

            if hasattr(model, "predict_proba"):
                try:
                    probas = model.predict_proba(df_model)
                    confidences = [f"{probas[i][r]*100:.1f}%" for i, r in enumerate(rounded_preds)]
                except Exception:
                    confidences = ["N/A"] * len(labels)
            else:
                confidences = ["N/A"] * len(labels)

            df_batch["Predicted pH Class"] = labels
            df_batch["Confidence"]         = confidences

        except Exception as ex:
            st.markdown(f'<div class="error-banner">⚠️ Prediction error: {ex}</div>', unsafe_allow_html=True)
            st.stop()

        st.success(f"✅ {len(df_batch)} rows processed successfully.")

        summary = df_batch["Predicted pH Class"].value_counts().reset_index()
        summary.columns = ["pH Class", "Count"]
        col_sum, col_tbl = st.columns([1, 2], gap="large")

        with col_sum:
            st.markdown('<div class="batch-result-header">Class Distribution</div>', unsafe_allow_html=True)
            for _, row in summary.iterrows():
                color = PH_COLORS.get(row["pH Class"], "#888")
                pct   = row["Count"] / len(df_batch) * 100
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.5rem;">
                  <div style="width:12px;height:12px;border-radius:50%;background:{color};flex-shrink:0;"></div>
                  <div style="flex:1;font-size:0.85rem;">{row['pH Class']}</div>
                  <div style="font-weight:600;font-size:0.85rem;">{row['Count']} <span style="color:var(--muted);font-weight:400;">({pct:.0f}%)</span></div>
                </div>
                """, unsafe_allow_html=True)

        with col_tbl:
            st.markdown('<div class="batch-result-header">Preview (first 10 rows)</div>', unsafe_allow_html=True)
            preview_cols = BATCH_INPUT_COLS + ["Predicted pH Class", "Confidence"]
            st.dataframe(df_batch[preview_cols].head(10), use_container_width=True, hide_index=True)

        output_cols = BATCH_INPUT_COLS + ["% Ca", "% Mg", "% K", "Predicted pH Class", "Confidence"]
        csv_out = df_batch[output_cols].to_csv(index=False)
        st.download_button(
            label="⬇️ Download full results CSV",
            data=csv_out,
            file_name="agrosense_batch_results.csv",
            mime="text/csv",
        )

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:var(--muted);font-size:0.8rem;padding:0.5rem 0 1rem;'>"
    "AgroSense · Ensemble ML (MLP + XGBoost + Random Forest) · For research and advisory use only"
    "</div>",
    unsafe_allow_html=True
)