import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.interpolate import interp1d
import random
import os

try:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

# ── Helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@st.cache_data(show_spinner=False)
def get_population(seed, n_samples, n_features, n_informative, n_redundant,
                   flip_y, minority_frac, class_sep):
    n_redundant = min(n_redundant, n_features - n_informative)
    weights = [1 - minority_frac, minority_frac]
    X_raw, y_raw = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        flip_y=flip_y,
        weights=weights,
        class_sep=class_sep,
        random_state=seed,
    )
    cols = [f"f{i}" for i in range(X_raw.shape[1])]
    return pd.DataFrame(X_raw, columns=cols), pd.Series(y_raw)


def run_simulation(X_pop, y_pop, seed, test_size, sample_sizes,
                   n_simulations, use_lgbm, use_lr, progress_bar):
    set_seed(seed)
    X_train_pool, X_test, y_train_pool, y_test = train_test_split(
        X_pop, y_pop, test_size=test_size, stratify=y_pop, random_state=seed
    )
    scaler = StandardScaler()
    X_tr_sc = pd.DataFrame(scaler.fit_transform(X_train_pool), columns=X_train_pool.columns)
    X_te_sc = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    results = []
    total_steps = len(sample_sizes)

    for step, n in enumerate(sample_sizes):
        lgbm_auc, lgbm_br = [], []
        log_auc,  log_br  = [], []

        for s in range(n_simulations):
            sim_seed = seed + n + s

            try:
                X_samp, _, y_samp, _ = train_test_split(
                    X_tr_sc, y_train_pool,
                    train_size=n, stratify=y_train_pool, random_state=sim_seed
                )
            except ValueError:
                continue  # skip if n > pool size

            if use_lgbm and LGBM_AVAILABLE:
                try:
                    X_t, X_v, y_t, y_v = train_test_split(
                        X_samp, y_samp, test_size=0.2, stratify=y_samp, random_state=sim_seed
                    )
                    lgbm = LGBMClassifier(
                        n_estimators=500, learning_rate=0.05,
                        min_data_in_leaf=max(5, n // 10), num_leaves=15,
                        deterministic=True, force_col_wise=True,
                        verbosity=-1, random_state=seed
                    )
                    lgbm.fit(X_t, y_t, eval_set=[(X_v, y_v)], eval_metric="auc",
                             callbacks=[early_stopping(15, verbose=False),
                                        log_evaluation(period=0)])
                    probs = lgbm.predict_proba(X_te_sc)[:, 1]
                    lgbm_auc.append(roc_auc_score(y_test, probs))
                    lgbm_br.append(brier_score_loss(y_test, probs))
                except Exception:
                    pass

            if use_lr:
                try:
                    lr = LogisticRegression(max_iter=1000, random_state=seed)
                    lr.fit(X_samp, y_samp)
                    probs = lr.predict_proba(X_te_sc)[:, 1]
                    log_auc.append(roc_auc_score(y_test, probs))
                    log_br.append(brier_score_loss(y_test, probs))
                except Exception:
                    pass

        row = {"n": n}
        if use_lgbm and lgbm_auc:
            row.update(lgbm_auc=np.mean(lgbm_auc), lgbm_auc_s=np.std(lgbm_auc),
                       lgbm_br=np.mean(lgbm_br),   lgbm_br_s=np.std(lgbm_br))
        if use_lr and log_auc:
            row.update(LR_auc=np.mean(log_auc), LR_auc_s=np.std(log_auc),
                       LR_br=np.mean(log_br),   LR_br_s=np.std(log_br))
        results.append(row)
        progress_bar.progress((step + 1) / total_steps)

    return pd.DataFrame(results)


def plot_results(df, threshold, palette):
    models = [c.replace("_auc", "") for c in df.columns if c.endswith("_auc")]
    if not models:
        return None, {}

    colors = {"lgbm": palette[0], "LR": palette[1]}
    model_labels = {"lgbm": "LightGBM", "LR": "Logistic Regression"}

    n_models = len(models)
    fig, axes = plt.subplots(n_models, 2, figsize=(13, 4.5 * n_models),
                             facecolor="#0e1117", squeeze=False)
    plt.rcParams.update({"font.family": "sans-serif"})

    results_out = {}

    for i, model in enumerate(models):
        color = colors.get(model, f"C{i}")
        ax1, ax2 = axes[i, 0], axes[i, 1]
        label = model_labels.get(model, model.upper())

        n_vals = df["n"].values
        auc   = df[f"{model}_auc"].values
        auc_s = df[f"{model}_auc_s"].values
        br    = df[f"{model}_br"].values
        br_s  = df[f"{model}_br_s"].values

        target_auc = auc.max() * threshold
        target_br  = br.min() + (1 - threshold) * (br.max() - br.min())

        f_auc = interp1d(auc, n_vals, bounds_error=False, fill_value="extrapolate")
        f_br  = interp1d(br,  n_vals, bounds_error=False, fill_value="extrapolate")
        n_auc_t = float(f_auc(target_auc))
        n_br_t  = float(f_br(target_br))
        results_out[model] = {
            "label": label,
            "n_for_auc": round(n_auc_t),
            "auc_target": round(target_auc, 3),
            "n_for_brier": round(n_br_t),
            "brier_target": round(target_br, 4),
        }

        for ax, y, y_s, title, ylabel, ylim, n_t, target, marker in [
            (ax1, auc, auc_s, f"ROC-AUC — {label}", "AUC", (0.45, 1.02),
             n_auc_t, target_auc, "o"),
            (ax2, br,  br_s,  f"Brier Score — {label}", "Brier Score ↓",
             (0.0, 0.32), n_br_t, target_br, "s"),
        ]:
            ax.set_facecolor("#161b22")
            ax.plot(n_vals, y, f"{marker}-", color=color, lw=2, label=label)
            ax.fill_between(n_vals, y - 1.96*y_s, y + 1.96*y_s, alpha=0.18, color=color)
            ax.axvline(n_t, color="white", ls="--", lw=1, alpha=0.55)
            ax.annotate(f"n ≈ {int(n_t)}", xy=(n_t, target),
                        xytext=(10, 10), textcoords="offset points",
                        color="white", fontsize=9, fontweight="bold",
                        bbox=dict(facecolor="#21262d", alpha=0.85, edgecolor="none", pad=3))
            ax.set_xscale("log")
            ax.set_xlim(n_vals.min() * 0.8, n_vals.max() * 1.2)
            ax.set_ylim(*ylim)
            ax.set_title(title, color="white", fontsize=12, pad=8)
            ax.set_xlabel("Sample Size (n)", color="#8b949e")
            ax.set_ylabel(ylabel, color="#8b949e")
            ax.tick_params(colors="#8b949e")
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")
            ax.grid(True, which="major", color="#21262d", lw=0.8)
            ax.grid(True, which="minor", color="#161b22", lw=0.4)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.legend(facecolor="#21262d", labelcolor="white", framealpha=0.9)

    plt.tight_layout(pad=2.0)
    return fig, results_out


# ── Write .streamlit/config.toml next to this script at startup ──────────────
# Guarantees primaryColor (drives slider fill colour) is always applied,
# regardless of which directory the app is launched from.
import pathlib as _pl
_cfg = _pl.Path(__file__).parent / ".streamlit"
_cfg.mkdir(exist_ok=True)
(_cfg / "config.toml").write_text(
    "[theme]\n"
    "base                     = \"dark\"\n"
    "backgroundColor          = \"#0d1117\"\n"
    "secondaryBackgroundColor = \"#161b22\"\n"
    "primaryColor             = \"#79c0ff\"\n"
    "textColor                = \"#e6edf3\"\n"
    'font                     = "sans serif"\n'
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sample Size Simulator",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
  /* ── Base app background ── */
  .stApp, [data-testid="stAppViewContainer"] {
      background-color: #0d1117;
      color: #e6edf3;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
      background-color: #010409 !important;
      border-right: 1px solid #21262d;
  }
  [data-testid="stSidebar"] * { color: #e6edf3 !important; }

  /* ── Sidebar section headers ── */
  [data-testid="stSidebar"] h1,
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {
      color: #79c0ff !important;
      font-size: 13px !important;
      font-weight: 700 !important;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-top: 6px !important;
      margin-bottom: 2px !important;
  }

  /* ── Slider & widget labels — bright white, bigger ── */
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stCheckbox label,
  [data-testid="stSidebar"] .stNumberInput label,
  [data-testid="stSidebar"] .stTextInput label,
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
      color: #e6edf3 !important;
      font-size: 13px !important;
      font-weight: 600 !important;
  }

  /* Slider value readout */
  [data-testid="stSidebar"] [data-testid="stTickBarMin"],
  [data-testid="stSidebar"] [data-testid="stTickBarMax"],
  [data-testid="stSidebar"] .stSlider [data-testid="stMarkdownContainer"] p {
      color: #8b949e !important;
      font-size: 11px !important;
  }

  /* ── Slider colours set via primaryColor in config.toml ── */

  /* ── Select boxes ── */
  [data-testid="stSidebar"] [data-baseweb="select"] > div {
      background-color: #161b22 !important;
      border-color: #30363d !important;
      color: #e6edf3 !important;
  }

  /* ── Number input ── */
  [data-testid="stSidebar"] input[type="number"],
  [data-testid="stSidebar"] input[type="text"] {
      background-color: #161b22 !important;
      border-color: #30363d !important;
      color: #e6edf3 !important;
  }

  /* ── Checkbox ── */
  [data-testid="stSidebar"] [data-baseweb="checkbox"] span {
      background-color: #21262d !important;
      border-color: #388bfd !important;
  }

  /* ── Dividers ── */
  [data-testid="stSidebar"] hr { border-color: #21262d !important; }

  /* ── Help tooltip icon ── */
  [data-testid="stSidebar"] [data-testid="tooltipHoverTarget"] svg { fill: #58a6ff !important; }

  /* ── Main content area ── */
  [data-testid="stMainBlockContainer"],
  [data-testid="block-container"] {
      background-color: #0d1117;
  }

  /* Headings */
  h1, h2, h3 { color: #e6edf3 !important; }

  /* st.caption / st.info / st.warning */
  [data-testid="stCaptionContainer"] { color: #8b949e !important; }
  [data-testid="stAlert"] {
      background-color: #161b22 !important;
      border-color: #30363d !important;
      color: #e6edf3 !important;
  }

  /* st.metric */
  [data-testid="stMetricValue"]  { color: #79c0ff !important; font-size: 26px !important; }
  [data-testid="stMetricLabel"]  { color: #8b949e !important; font-size: 12px !important; }
  [data-testid="metric-container"] {
      background-color: #161b22;
      border: 1px solid #21262d;
      border-radius: 8px;
      padding: 12px 16px;
  }

  /* Progress bar */
  [data-testid="stProgressBar"] > div { background-color: #388bfd !important; }
  [data-testid="stProgressBar"] { background-color: #21262d !important; }

  /* Expander */
  [data-testid="stExpander"] {
      background-color: #161b22 !important;
      border: 1px solid #21262d !important;
      border-radius: 8px;
  }
  [data-testid="stExpander"] summary { color: #79c0ff !important; }

  /* DataFrame */
  [data-testid="stDataFrame"] { background-color: #161b22 !important; }

  /* Download button */
  [data-testid="stDownloadButton"] button {
      background-color: #21262d !important;
      border-color: #30363d !important;
      color: #e6edf3 !important;
  }

  /* Run button */
  [data-testid="stSidebar"] button[kind="primary"] {
      background-color: #238636 !important;
      border-color: #2ea043 !important;
      color: #ffffff !important;
      font-weight: 700 !important;
      letter-spacing: 0.04em;
  }
  [data-testid="stSidebar"] button[kind="primary"]:hover {
      background-color: #2ea043 !important;
  }

  /* ── Result metric cards ── */
  .metric-card {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 16px 20px;
      margin-bottom: 10px;
  }
  .metric-card h4  { color: #79c0ff; margin: 0 0 8px 0; font-size: 15px; font-weight: 700; }
  .metric-card .val { color: #e6edf3; font-size: 24px; font-weight: 700; }
  .metric-card .sub { color: #8b949e; font-size: 12px; margin-top: 3px; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #0d1117; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


st.title("📊 Sample Size Effect Simulator")
st.caption("Compare LightGBM vs Logistic Regression performance across sample sizes on a synthetic population.")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🧬 Population Config")

    n_samples   = st.select_slider("Population size",
                    options=[5_000, 10_000, 20_000, 50_000, 100_000], value=20_000)
    n_features  = st.slider("Total features",   5,  50, 20)
    n_info_max  = max(2, n_features - 1)
    n_informative = st.slider("Informative features", 2, n_info_max,
                               min(10, n_info_max))
    n_redundant = st.slider("Redundant features", 0,
                             max(0, n_features - n_informative), 4)
    class_sep   = st.slider("Class separation", 0.3, 3.0, 1.0, 0.1,
                             help="Higher = easier problem; < 1 = overlapping classes")
    minority_frac = st.slider("Minority class fraction", 0.05, 0.50, 0.50, 0.05,
                               help="Fraction of positive (Class 1) samples")
    flip_y      = st.slider("Label noise (flip_y)", 0.0, 0.3, 0.05, 0.01)

    st.divider()
    st.header("⚙️ Simulation Config")

    seed        = st.number_input("Random seed", 0, 9999, 42, step=1)
    test_size   = st.slider("Test set fraction", 0.2, 0.8, 0.7, 0.05)
    n_sims      = st.slider("Simulations per sample size", 5, 100, 20,
                             help="More = smoother curves, but slower")
    threshold   = st.slider("Performance threshold", 0.80, 0.99, 0.95, 0.01,
                             help="Fraction of max AUC / min Brier to annotate 'sufficient n'")

    ss_presets = {
        "Fine (15 sizes)": [50,100,150,200,250,300,400,500,750,1000,1500,2000,3000,4000,5000],
        "Coarse (8 sizes)": [50,100,250,500,1000,2000,3500,5000],
        "Custom": None,
    }
    preset = st.selectbox("Sample sizes preset", list(ss_presets.keys()))
    if preset == "Custom":
        raw = st.text_input("Enter sizes (comma-separated)", "50,100,250,500,1000,2000,5000")
        try:
            sample_sizes = sorted(set(int(x) for x in raw.split(",") if x.strip()))
        except ValueError:
            st.error("Invalid input — using default")
            sample_sizes = [50,100,250,500,1000,2000,5000]
    else:
        sample_sizes = ss_presets[preset]

    st.divider()
    st.header("🤖 Models")
    use_lr   = st.checkbox("Logistic Regression", value=True)
    use_lgbm = st.checkbox("LightGBM", value=LGBM_AVAILABLE,
                            disabled=not LGBM_AVAILABLE,
                            help="LightGBM not installed" if not LGBM_AVAILABLE else "")

    st.divider()
    st.header("🎨 Plot colours")
    col_lgbm = st.color_picker("LightGBM", "#d2a8ff")
    col_lr   = st.color_picker("Logistic Regression", "#58a6ff")

    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────

if not use_lr and not use_lgbm:
    st.warning("Select at least one model in the sidebar.")
    st.stop()

if run_btn:
    with st.spinner("Generating population…"):
        X_pop, y_pop = get_population(
            seed, n_samples, n_features, n_informative, n_redundant,
            flip_y, minority_frac, class_sep
        )

    class_counts = y_pop.value_counts().sort_index()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Population size", f"{len(y_pop):,}")
    c2.metric("Class 0 (majority)", f"{class_counts.get(0,0):,}")
    c3.metric("Class 1 (minority)", f"{class_counts.get(1,0):,}")
    c4.metric("Imbalance ratio", f"1 : {class_counts.get(0,1) / max(class_counts.get(1,1),1):.1f}")

    st.divider()
    prog = st.progress(0, text="Running simulation…")
    df_res = run_simulation(
        X_pop, y_pop, int(seed), test_size, sample_sizes,
        n_sims, use_lgbm, use_lr, prog
    )
    prog.empty()

    if df_res.empty:
        st.error("Simulation returned no results. Try larger sample sizes or a bigger population.")
        st.stop()

    st.subheader("📈 Results")
    fig, metrics = plot_results(df_res, threshold, [col_lgbm, col_lr])
    if fig:
        st.pyplot(fig, use_container_width=True)

    # Summary cards
    if metrics:
        st.subheader("🎯 Sample Size Recommendations")
        cols = st.columns(len(metrics))
        for col, (model_key, m) in zip(cols, metrics.items()):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <h4>{m['label']}</h4>
                  <div class="val">n ≈ {m['n_for_auc']:,}</div>
                  <div class="sub">to reach AUC ≥ {m['auc_target']} ({threshold*100:.0f}% of max)</div>
                  <br>
                  <div class="val">n ≈ {m['n_for_brier']:,}</div>
                  <div class="sub">for Brier ≤ {m['brier_target']} (threshold)</div>
                </div>
                """, unsafe_allow_html=True)

    # Raw data table
    with st.expander("📋 Raw simulation data"):
        st.dataframe(df_res.round(4), use_container_width=True)
        csv = df_res.to_csv(index=False).encode()
        st.download_button("⬇ Download CSV", csv, "simulation_results.csv", "text/csv")

else:
    st.info("Configure the parameters in the sidebar and click **▶ Run Simulation** to begin.")
    with st.expander("ℹ️ How this works"):
        st.markdown("""
**Population generation** uses `sklearn.make_classification` to create a synthetic dataset
with controllable class imbalance, feature informativeness, and class separability.

**Simulation loop**
1. A fixed test set is held out from the population.
2. For each sample size *n*, the experiment draws *k* stratified random samples from the remaining pool.
3. Each model is trained on the sample and evaluated on the held-out test set.
4. Mean ± 95% CI of AUC and Brier Score are plotted against *n*.

**Threshold annotation** interpolates the curve to find the *n* where performance
first reaches the chosen fraction (e.g. 95%) of its maximum.

**Models compared**
- **Logistic Regression** — linear, low variance, works well at small *n*
- **LightGBM** — gradient boosted trees, higher capacity, needs more data to shine
        """)