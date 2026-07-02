import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. PAGE CONFIG & INSTITUTIONAL THEME
# ============================================================
st.set_page_config(
    page_title="Securitisation Risk & Waterfall Engine | ZAAUTO2024-1",
    layout="wide",
    page_icon="🏦",
    initial_sidebar_state="expanded",
)

DARK_BG, PANEL_BG, GRID_C, TEXT_C = "#0F172A", "#1E293B", "#334155", "#CBD5E1"
COLORS = {"senior": "#3B82F6", "mezz": "#F59E0B", "equity": "#10B981", "loss": "#EF4444", "neutral": "#94A3B8"}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1400px; }

.deal-banner {
    background: linear-gradient(120deg, #0F172A 0%, #1E293B 60%, #0F172A 100%);
    border: 1px solid #334155; border-radius: 14px; padding: 22px 28px; margin-bottom: 18px;
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
    box-shadow: 0 6px 20px rgba(0,0,0,.4);
}
.deal-title { font-size: 22px; font-weight: 800; color: #F8FAFC; letter-spacing: .3px; }
.deal-sub   { font-size: 12.5px; color: #94A3B8; margin-top: 4px; font-weight: 500; }
.deal-badge {
    background: rgba(59,130,246,.12); border: 1px solid #3B82F6; color: #93C5FD;
    padding: 6px 14px; border-radius: 999px; font-size: 11.5px; font-weight: 700;
    letter-spacing: .5px; text-transform: uppercase;
}
.kpi-row { display: flex; gap: 14px; margin-bottom: 20px; flex-wrap: wrap; }
.kpi-card {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    border: 1px solid #334155; border-radius: 12px; padding: 18px 22px;
    flex: 1; min-width: 180px; box-shadow: 0 4px 12px rgba(0,0,0,.35);
    position: relative; overflow: hidden; transition: border-color .15s ease;
}
.kpi-card:hover { border-color: #3B82F6; }
.kpi-card::before { content: ""; position: absolute; top:0; left:0; width:3px; height:100%; background: var(--accent,#3B82F6); }
.kpi-label { font-size: 10.5px; color: #94A3B8; text-transform: uppercase; letter-spacing:.9px; margin-bottom:7px; font-weight:600; }
.kpi-value { font-size: 25px; font-weight: 800; color: #F1F5F9; line-height: 1.1; }
.kpi-sub   { font-size: 11.5px; color: #64748B; margin-top: 5px; font-weight: 500; }

.section-header {
    background: #0F172A; border-left: 4px solid #3B82F6; padding: 9px 16px; border-radius: 6px;
    color: #F1F5F9; font-size: 14.5px; font-weight: 700; margin: 20px 0 12px 0; letter-spacing: .2px;
}
.risk-badge { display:inline-block; padding:4px 12px; border-radius:999px; font-size:11px; font-weight:700; letter-spacing:.4px; text-transform:uppercase; }
.risk-ok   { background: rgba(16,185,129,.15); color:#6EE7B7; border:1px solid #10B981; }
.risk-warn { background: rgba(245,158,11,.15); color:#FCD34D; border:1px solid #F59E0B; }
.risk-bad  { background: rgba(239,68,68,.15);  color:#FCA5A5; border:1px solid #EF4444; }

.exec-box {
    background: #111C33; border: 1px solid #334155; border-left: 4px solid #10B981;
    border-radius: 10px; padding: 16px 20px; margin-bottom: 20px; font-size: 13.5px; color: #E2E8F0; line-height:1.6;
}
section[data-testid="stSidebar"] { background: #0B1220; border-right: 1px solid #1E293B; }
section[data-testid="stSidebar"] .stMarkdown h3 { color: #E2E8F0; font-size: 13.5px; letter-spacing: .4px; }
button[data-baseweb="tab"] { font-weight: 600; font-size: 13.5px; }
[data-testid="stDataFrame"] { border: 1px solid #334155; border-radius: 8px; overflow: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


def kpi_card(label, value, sub, accent="#3B82F6"):
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

def risk_badge(ok: bool):
    return '<span class="risk-badge risk-ok">Fully Covered</span>' if ok else '<span class="risk-badge risk-bad">Deficit</span>'

PLOTLY_LAYOUT = dict(
    paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
    font=dict(color=TEXT_C, family="Inter, sans-serif", size=12),
    margin=dict(l=50, r=20, t=30, b=45),
    legend=dict(bgcolor=PANEL_BG, bordercolor=GRID_C, borderwidth=1, orientation="h", y=1.12, x=0),
    xaxis=dict(gridcolor=GRID_C, zerolinecolor=GRID_C, linecolor=GRID_C),
    yaxis=dict(gridcolor=GRID_C, zerolinecolor=GRID_C, linecolor=GRID_C),
    hoverlabel=dict(bgcolor=PANEL_BG, font_color=TEXT_C, bordercolor=GRID_C),
)

def inr_axis(fig, axis="yaxis"):
    fig.update_layout(**{axis: dict(tickprefix="₹", tickformat=",.2s")})

# ============================================================
# 2. MATH UTILITIES & NUMERICAL SOLVERS
# ============================================================
def clean_to_float(val) -> float:
    while isinstance(val, (list, np.ndarray, pd.Series)):
        val = val[0] if (hasattr(val, '__len__') and len(val) > 0) else 0.0
    chars = [c for c in str(val).strip() if c.isdigit() or c == '.']
    out, seen_dot = [], False
    for c in chars:
        if c == '.':
            if not seen_dot: out.append(c); seen_dot = True
        else: out.append(c)
    s = "".join(out)
    return float(s) if s and s != '.' else 0.0

def pct(val: float, decimals: int = 2) -> str:
    return f"{val * 100:.{decimals}f}%"

def pct_irr(val: float, decimals: int = 2, cap: float = 9.99) -> str:
    """
    Display-safe IRR formatter. Raw IRR math can be technically correct yet
    practically meaningless when a tranche's stated invested capital is far
    smaller than the cash it actually receives (a sizing mismatch, not a bug).
    Caps the displayed figure instead of printing something like
    '10,988,564,737%' and flags that the input sizing should be reviewed.
    """
    if val > cap:
        return f">{cap*100:,.0f}% ⚠"
    if val < -0.99:
        return "<-99%"
    return f"{val * 100:.{decimals}f}%"

def compute_irr(initial: float, inflows: list) -> float:
    """Robust monthly IRR via Newton-Raphson with bisection fallback, then annualised."""
    cfs = [-float(initial)] + [float(x) for x in inflows]
    if initial <= 0 or sum(inflows) <= 0:
        return 0.0

    def npv(r):
        return sum(cf / ((1 + r) ** t) for t, cf in enumerate(cfs))

    def dnpv(r):
        return sum(-t * cf / ((1 + r) ** (t + 1)) for t, cf in enumerate(cfs) if t > 0)

    r, converged = 0.02, False
    for _ in range(100):
        f, fp = npv(r), dnpv(r)
        if abs(fp) < 1e-10:
            break
        r_new = r - f / fp
        if r_new <= -0.999:
            r_new = -0.5
        if abs(r_new - r) < 1e-10:
            r, converged = r_new, True
            break
        r = r_new

    if not converged or not np.isfinite(r) or r < -0.99 or r > 5:
        lo, hi = -0.99, 5.0
        f_lo, f_hi = npv(lo), npv(hi)
        if f_lo * f_hi > 0:
            return 0.0
        mid = r
        for _ in range(200):
            mid = (lo + hi) / 2
            f_mid = npv(mid)
            if abs(f_mid) < 1e-6:
                break
            if f_lo * f_mid < 0:
                hi = mid
            else:
                lo, f_lo = mid, f_mid
        r = mid

    return ((1 + r) ** 12) - 1

def compute_wal(months: np.ndarray, payments: np.ndarray) -> float:
    denom = payments.sum()
    return float((payments * months).sum() / denom) if denom > 0 else float('inf')

def run_waterfall(df_dyn: pd.DataFrame, senior_tgt: float, mezz_tgt: float,
                   senior_cpn: float, mezz_cpn: float,
                   svc_fee: float, col_mult: float, loss_mult: float) -> pd.DataFrame:
    """
    Sequential-pay, coupon-bearing amortising waterfall:
      1) Senior interest (coupon accrual on outstanding balance, unpaid amounts carry forward)
      2) Senior principal (amortises the outstanding balance)
      3) Mezzanine interest
      4) Mezzanine principal
      5) Equity/Unrated — TRUE RESIDUAL: receives 100% of whatever cash remains, uncapped
         (matches real ABS structuring; equity is the first-loss/residual claimant, not a
         fixed-target tranche — capping it caused cash to silently disappear in earlier versions)
    """
    s_bal, m_bal = senior_tgt, mezz_tgt
    s_short = m_short = 0.0
    records = []
    for i, row in df_dyn.iterrows():
        gross = clean_to_float(row["CollectionsTotal"]) * col_mult
        loss  = clean_to_float(row["NetLoss_ThisMonth"]) * loss_mult
        net   = max(0.0, gross - loss)
        cash  = max(0.0, net - svc_fee)

        s_int_due  = s_bal * (senior_cpn / 12) + s_short
        s_int_paid = min(cash, s_int_due); cash -= s_int_paid; s_short = s_int_due - s_int_paid
        s_prin     = min(cash, s_bal);     cash -= s_prin;     s_bal  -= s_prin

        m_int_due  = m_bal * (mezz_cpn / 12) + m_short
        m_int_paid = min(cash, m_int_due); cash -= m_int_paid; m_short = m_int_due - m_int_paid
        m_prin     = min(cash, m_bal);     cash -= m_prin;     m_bal  -= m_prin

        eq_paid = cash  # residual — no cap

        records.append(dict(
            Month=i + 1, Date=row["ReportingDate"], GrossInflow=gross, CreditLoss=loss, NetCash=net,
            SeniorInterest=s_int_paid, SeniorPrincipal=s_prin, SeniorTotal=s_int_paid + s_prin,
            SeniorOutstanding=s_bal, SeniorIntShortfall=s_short,
            MezzInterest=m_int_paid, MezzPrincipal=m_prin, MezzTotal=m_int_paid + m_prin,
            MezzOutstanding=m_bal, MezzIntShortfall=m_short,
            EqPaid=eq_paid,
        ))
    return pd.DataFrame(records)

# ============================================================
# 3. DATA INGESTION — visible validation instead of silent fallback
# ============================================================
@st.cache_data(show_spinner=False)
def load_all_data():
    load_warnings = []

    try:
        df_dyn = pd.read_csv("dynamic_loss_monthly.csv")
        df_dyn.columns = df_dyn.columns.str.strip()
        df_dyn["ReportingDate"] = pd.to_datetime(df_dyn["ReportingDate"], errors="coerce")
        if df_dyn["ReportingDate"].isna().any():
            load_warnings.append("dynamic_loss_monthly.csv: some ReportingDate values could not be parsed and were dropped.")
            df_dyn = df_dyn.dropna(subset=["ReportingDate"])
    except Exception as e:
        load_warnings.append(f"dynamic_loss_monthly.csv not found/unreadable ({type(e).__name__}) — using synthetic fallback data.")
        df_dyn = pd.DataFrame({
            "ReportingDate": pd.date_range("2024-01-01", periods=12, freq="ME"),
            "CollectionsTotal": [19500000.0] * 12,
            "NetLoss_ThisMonth": [200000.0] * 12
        })

    try:
        df_loan = pd.read_csv("auto_loan_securitisation_data.csv")
        df_loan.columns = df_loan.columns.str.strip()
        if df_loan.empty or "Region" not in df_loan.columns:
            raise ValueError("required columns missing")
    except Exception as e:
        load_warnings.append(f"auto_loan_securitisation_data.csv not found/invalid ({type(e).__name__}) — using synthetic fallback pool.")
        df_loan = pd.DataFrame(columns=["Region", "IFRS9_Stage", "CurrentBalance", "OriginalLoanAmount",
                                        "InterestRate", "RemainingTerm", "DelinquencyDays", "DelinquencyStatus",
                                        "CIBIL_Score_Current", "LTV_Current", "ECL_Provision", "PD_Estimate",
                                        "LGD_Estimate", "VehicleType", "EmploymentType", "InsuranceType", "LoanID"])

    try:
        df_dpd = pd.read_csv("dpd_snapshot_history.csv")
        df_dpd.columns = df_dpd.columns.str.strip()
        if df_dpd.empty:
            raise ValueError("empty file")
    except Exception as e:
        load_warnings.append(f"dpd_snapshot_history.csv not found/empty ({type(e).__name__}) — using synthetic fallback roll-rate data.")
        df_dpd = pd.DataFrame(columns=["SnapshotDate", "DPD_Bucket", "DPD_Bucket_Prior", "CurrentBalance", "DPD_Days"])

    try:
        df_vint = pd.read_csv("static_pool_vintage_data.csv")
        df_vint.columns = df_vint.columns.str.strip()
        if df_vint.empty:
            raise ValueError("empty file")
    except Exception as e:
        load_warnings.append(f"static_pool_vintage_data.csv not found/empty ({type(e).__name__}) — using synthetic fallback vintages.")
        df_vint = pd.DataFrame(columns=["VintageID", "MonthsOnBook", "CumulativeNetLossRate", "PoolFactor", "OriginalPoolBalance"])

    dup_count = int(df_dyn.duplicated(subset=["ReportingDate"]).sum())
    if dup_count > 0:
        load_warnings.append(f"dynamic_loss_monthly.csv had {dup_count} duplicate ReportingDate row(s) — merged (summed) automatically.")
        df_dyn = df_dyn.groupby("ReportingDate", as_index=False).agg({"CollectionsTotal": "sum", "NetLoss_ThisMonth": "sum"})
    df_dyn = df_dyn.sort_values("ReportingDate").reset_index(drop=True)

    N_ROWS = max(500, len(df_loan))
    if df_loan.empty or "Region" not in df_loan.columns:
        df_loan = pd.DataFrame({
            "Region": np.random.choice(["North", "South", "East", "West", "Central"], size=N_ROWS),
            "IFRS9_Stage": np.random.choice([1, 2, 3], size=N_ROWS, p=[0.85, 0.10, 0.05]),
            "CurrentBalance": np.random.uniform(100000, 1500000, size=N_ROWS),
            "OriginalLoanAmount": np.random.uniform(150000, 2000000, size=N_ROWS),
            "InterestRate": np.random.uniform(9.5, 16.5, size=N_ROWS),
            "RemainingTerm": np.random.randint(12, 60, size=N_ROWS),
            "DelinquencyDays": np.random.choice([0, 15, 45, 75, 120], size=N_ROWS, p=[0.80, 0.10, 0.05, 0.03, 0.02]),
            "DelinquencyStatus": np.random.choice(["Current", "1-29 DPD", "30-59 DPD", "60-89 DPD", "90-119 DPD", "120+ DPD", "Default", "Repossessed"], size=N_ROWS),
            "CIBIL_Score_Current": np.random.randint(600, 850, size=N_ROWS),
            "LTV_Current": np.random.uniform(0.45, 0.95, size=N_ROWS),
            "VehicleType": np.random.choice(["Sedan", "SUV", "Hatchback", "Commercial"], size=N_ROWS),
            "EmploymentType": np.random.choice(["Salaried", "Self-Employed", "Business"], size=N_ROWS),
            "InsuranceType": np.random.choice(["Comprehensive", "Third-Party"], size=N_ROWS),
            "LoanID": range(1000, 1000 + N_ROWS)
        })
    df_loan["ECL_Provision"] = df_loan["CurrentBalance"] * np.where(df_loan["IFRS9_Stage"] == 3, 0.35, np.where(df_loan["IFRS9_Stage"] == 2, 0.12, 0.02))
    df_loan["PD_Estimate"] = np.where(df_loan["IFRS9_Stage"] == 3, 1.00, np.where(df_loan["IFRS9_Stage"] == 2, 0.15, 0.02))
    df_loan["LGD_Estimate"] = 0.45

    if df_dpd.empty:
        df_dpd = pd.DataFrame({
            "SnapshotDate": pd.date_range("2024-01-01", periods=1000, freq="h").tolist()[:N_ROWS],
            "DPD_Bucket": np.random.choice(["Current", "1-29 DPD", "30-59 DPD", "60-89 DPD", "90-119 DPD", "120+ DPD", "Default", "Repossessed"], size=N_ROWS),
            "DPD_Bucket_Prior": np.random.choice(["Current", "1-29 DPD", "30-59 DPD", "60-89 DPD"], size=N_ROWS),
            "CurrentBalance": np.random.uniform(100000, 800000, size=N_ROWS),
            "DPD_Days": np.random.randint(0, 150, size=N_ROWS)
        })

    if df_vint.empty:
        records_v = []
        for v_id in ["2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4", "2024-Q1"]:
            for mob in range(1, 25):
                records_v.append({
                    "VintageID": v_id, "MonthsOnBook": mob,
                    "CumulativeNetLossRate": (0.0015 * mob),
                    "PoolFactor": max(0.05, 1.0 - (0.035 * mob)), "OriginalPoolBalance": 100000000.0
                })
        df_vint = pd.DataFrame(records_v)

    return df_dyn, df_loan, df_dpd, df_vint, load_warnings

df_dyn, df_loan, df_dpd, df_vint, load_warnings = load_all_data()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🏦 ZAAUTO2024-1")
    st.caption("Auto Loan ABS · Structuring Console")

    if load_warnings:
        with st.expander(f"⚠️ Data Validation ({len(load_warnings)})", expanded=False):
            for w in load_warnings:
                st.warning(w, icon="⚠️")
    else:
        st.success("All 4 source files loaded cleanly", icon="✅")

    st.markdown("---")
    with st.expander("🔧 Structural Sizing", expanded=True):
        senior_tgt = st.number_input("Senior Bond Target (AAA) ₹", value=18_000_000.0, step=1_000_000.0, format="%.0f")
        senior_cpn = st.number_input("Senior Coupon Rate (% p.a.)", value=8.75, step=0.25, format="%.2f") / 100
        mezz_tgt   = st.number_input("Mezzanine Target (BBB) ₹", value=3_000_000.0,  step=500_000.0,   format="%.0f")
        mezz_cpn   = st.number_input("Mezzanine Coupon Rate (% p.a.)", value=11.50, step=0.25, format="%.2f") / 100
        eq_tgt     = st.number_input("Equity Invested Capital ₹ (for IRR calc)", value=1_000_000.0,  step=100_000.0,   format="%.0f",
                                      help="Equity is the residual tranche — it receives ALL leftover cash each month, uncapped. This field is only used as the initial investment for calculating Equity IRR.")
        svc_fee    = st.number_input("Monthly Servicer Fee ₹", value=100_000.0,    step=10_000.0,    format="%.0f")

    with st.expander("📉 Macro Stress Dials", expanded=True):
        col_mult  = st.slider("Collections Multiplier",  0.50, 1.50, 1.00, 0.05)
        loss_mult = st.slider("Credit Loss Multiplier",  0.50, 4.00, 1.00, 0.10)

    with st.expander("🔍 Pool Filters", expanded=True):
        region_filter = st.multiselect("Region", sorted(df_loan["Region"].unique()), default=sorted(df_loan["Region"].unique()))
        stage_filter  = st.multiselect("IFRS 9 Stage", [1, 2, 3], default=[1, 2, 3])

    st.markdown("---")
    st.caption("Built for Intern Assessment Review")

df_f = df_loan[df_loan["Region"].isin(region_filter) & df_loan["IFRS9_Stage"].isin(stage_filter)].copy()
if df_f.empty: df_f = df_loan.copy()

df_wf = run_waterfall(df_dyn, senior_tgt, mezz_tgt, senior_cpn, mezz_cpn, svc_fee, col_mult, loss_mult)

total_eq_received = df_wf["EqPaid"].sum()
if eq_tgt > 0 and total_eq_received > eq_tgt * 20:
    st.sidebar.info(
        f"💡 Equity Invested Capital (₹{eq_tgt:,.0f}) is small relative to the residual cash "
        f"Equity actually receives (₹{total_eq_received:,.0f}) — this is why Equity IRR shows as "
        f"an extreme/capped number. In a real deal, size Equity capital closer to the pool's "
        f"expected excess spread for a meaningful IRR.", icon="💡")

total_cash = df_wf["NetCash"].sum()
s_irr = compute_irr(senior_tgt, df_wf["SeniorTotal"].values)
m_irr = compute_irr(mezz_tgt,   df_wf["MezzTotal"].values)
e_irr = compute_irr(eq_tgt,     df_wf["EqPaid"].values)
s_wal = compute_wal(df_wf["Month"].values, df_wf["SeniorPrincipal"].values)
m_wal = compute_wal(df_wf["Month"].values, df_wf["MezzPrincipal"].values)
s_outstanding_final = df_wf["SeniorOutstanding"].iloc[-1]
m_outstanding_final = df_wf["MezzOutstanding"].iloc[-1]

pool_bal = df_f["CurrentBalance"].sum()
pool_orig = df_f["OriginalLoanAmount"].sum()
pool_factor = pool_bal / pool_orig if pool_orig else 0.0
wac = (df_f["InterestRate"] * df_f["CurrentBalance"]).sum() / pool_bal if pool_bal else 12.5
wam = (df_f["RemainingTerm"] * df_f["CurrentBalance"]).sum() / pool_bal if pool_bal else 36.0
ecl_total = df_f["ECL_Provision"].sum()
npa_rate = df_f[df_f["DelinquencyDays"] >= 90]["CurrentBalance"].sum() / pool_bal if pool_bal else 0.0
delinq30_rate = df_f[df_f["DelinquencyDays"] >= 30]["CurrentBalance"].sum() / pool_bal if pool_bal else 0.0
avg_cibil = df_f["CIBIL_Score_Current"].mean() if pd.notnull(df_f["CIBIL_Score_Current"].mean()) else 720.0
avg_ltv = df_f["LTV_Current"].mean() if pd.notnull(df_f["LTV_Current"].mean()) else 0.75

# ============================================================
# TOP BANNER
# ============================================================
st.markdown(f"""
<div class="deal-banner">
  <div>
    <div class="deal-title">🏦 Securitisation Risk &amp; Waterfall Analytics</div>
    <div class="deal-sub">Deal ID: ZAAUTO2024-1 &nbsp;·&nbsp; Auto Loan ABS &nbsp;·&nbsp; India RMBS/ABS Desk</div>
  </div>
  <div class="deal-badge">Pool Factor {pool_factor:.2%}</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🌊 Waterfall Engine", "📊 Pool Analytics", "🔴 DPD & Roll Rate", "📈 Vintage Loss Curves", "⚠️ IFRS 9 ECL", "🛡️ Stress Testing"])

with tabs[0]:
    exec_lines = [
        f"Under the current sizing (Senior ₹{senior_tgt/1e7:.2f}Cr @ {senior_cpn*100:.2f}% / Mezz ₹{mezz_tgt/1e7:.2f}Cr @ {mezz_cpn*100:.2f}% / Equity residual) "
        f"and stress dials ({col_mult:.2f}× collections, {loss_mult:.2f}× losses), the pool generated ₹{total_cash/1e7:.2f}Cr "
        f"net cash across {len(df_wf)} periods.",
        f"Senior AAA is {'fully repaid' if s_outstanding_final <= 0 else f'short by ₹{s_outstanding_final:,.0f}'} "
        f"by period end (IRR {pct_irr(s_irr)}, WAL {s_wal:.1f}m). "
        f"Mezzanine BBB is {'fully repaid' if m_outstanding_final <= 0 else f'short by ₹{m_outstanding_final:,.0f}'} "
        f"(IRR {pct_irr(m_irr)}, WAL {m_wal:.1f}m). Equity residual IRR: {pct_irr(e_irr)}."
    ]
    st.markdown(f'<div class="exec-box"><b>📌 Executive Summary</b><br>{" ".join(exec_lines)}</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("Senior AAA — IRR", pct_irr(s_irr), f"WAL: {s_wal:.1f}m  " + risk_badge(s_outstanding_final <= 0), COLORS["senior"])
    with c2: kpi_card("Mezzanine BBB — IRR", pct_irr(m_irr), f"WAL: {m_wal:.1f}m  " + risk_badge(m_outstanding_final <= 0), COLORS["mezz"])
    with c3: kpi_card("Equity Unrated — IRR", pct_irr(e_irr), "Residual claimant (uncapped)", COLORS["equity"])
    with c4: kpi_card("Cumulative Pool Cash (Net)", f"₹{total_cash/1e7:.2f}Cr", f"{len(df_wf)} collection periods", COLORS["neutral"])
    with c5: kpi_card("Stress Dials", f"{col_mult:.2f}× Coll.", f"Loss Mult: {loss_mult:.2f}×", COLORS["neutral"])

    col_chart, col_table = st.columns([1.4, 1])
    with col_chart:
        st.markdown('<div class="section-header">📈 Multi-Month Tranche Repayment (Interest + Principal)</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_wf["Month"], y=df_wf["SeniorTotal"], mode="lines+markers", name="Senior AAA",
                                  line=dict(color=COLORS["senior"], width=2.5), marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=df_wf["Month"], y=df_wf["MezzTotal"], mode="lines+markers", name="Mezzanine BBB",
                                  line=dict(color=COLORS["mezz"], width=2.5), marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=df_wf["Month"], y=df_wf["EqPaid"], mode="lines+markers", name="Equity Unrated",
                                  line=dict(color=COLORS["equity"], width=2.5), marker=dict(size=6)))
        fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Collection Period (Month Index)", yaxis_title="Cash Distributed", height=380)
        inr_axis(fig)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">🌊 Latest Month Capital Waterfall</div>', unsafe_allow_html=True)
        last = df_wf.iloc[-1]
        fee_val = min(svc_fee, max(0.0, last["GrossInflow"] - last["CreditLoss"]))
        wf_fig = go.Figure(go.Waterfall(
            x=["Gross Inflow", "− Credit Loss", "− Servicer Fee", "Senior AAA", "Mezzanine BBB", "Equity Residual"],
            measure=["absolute", "relative", "relative", "relative", "relative", "relative"],
            y=[last["GrossInflow"], -last["CreditLoss"], -fee_val, -last["SeniorTotal"], -last["MezzTotal"], -last["EqPaid"]],
            connector=dict(line=dict(color=GRID_C)),
            decreasing=dict(marker=dict(color=COLORS["loss"])),
            increasing=dict(marker=dict(color=COLORS["senior"])),
            totals=dict(marker=dict(color=COLORS["neutral"])),
            text=[f"₹{v/1e6:.1f}M" for v in [last["GrossInflow"], -last["CreditLoss"], -fee_val, -last["SeniorTotal"], -last["MezzTotal"], -last["EqPaid"]]],
            textposition="outside",
        ))
        wf_fig.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False)
        inr_axis(wf_fig)
        st.plotly_chart(wf_fig, use_container_width=True)

    with col_table:
        st.markdown('<div class="section-header">📋 Monthly Cash-Flow Ledger</div>', unsafe_allow_html=True)
        disp_raw = df_wf[["Month", "Date", "NetCash", "SeniorTotal", "MezzTotal", "MezzOutstanding", "EqPaid"]].copy()
        disp = disp_raw.copy()
        for col in ["NetCash", "SeniorTotal", "MezzTotal", "MezzOutstanding", "EqPaid"]: disp[col] = disp[col].map(lambda v: f"₹{v:,.0f}")
        disp.columns = ["#", "Date", "Net Cash", "Senior", "Mezz", "Mezz Outstanding", "Equity"]
        st.dataframe(disp, height=430, use_container_width=True, hide_index=True)
        st.download_button("⬇ Download Ledger (CSV)", disp_raw.to_csv(index=False).encode(), "cashflow_ledger_ZAAUTO2024-1.csv", "text/csv", use_container_width=True)

with tabs[1]:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("Active Pool Balance", f"₹{pool_bal/1e7:.2f}Cr", f"Pool Factor: {pool_factor:.4f}", COLORS["senior"])
    with c2: kpi_card("Weighted Avg Coupon", f"{wac:.2f}%", f"WAM: {wam:.1f} months", COLORS["mezz"])
    with c3: kpi_card("30+ DPD Rate", pct(delinq30_rate), f"NPA (90+ DPD): {pct(npa_rate)}", COLORS["loss"])
    with c4: kpi_card("Avg CIBIL Score", f"{avg_cibil:.0f}", f"Avg LTV: {avg_ltv:.2%}", COLORS["equity"])
    with c5: kpi_card("Total ECL Provision", f"₹{ecl_total/1e7:.2f}Cr", f"{pct(ecl_total/pool_bal if pool_bal else 0)} of pool", COLORS["neutral"])

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.markdown('<div class="section-header">DPD Bucket Distribution</div>', unsafe_allow_html=True)
        bkt_bal = df_f.groupby("DelinquencyStatus")["CurrentBalance"].sum().sort_values()
        fig = go.Figure(go.Bar(x=bkt_bal.values, y=bkt_bal.index, orientation="h", marker_color=COLORS["senior"]))
        fig.update_layout(**PLOTLY_LAYOUT, height=340)
        inr_axis(fig, "xaxis")
        st.plotly_chart(fig, use_container_width=True)
    with r1c2:
        st.markdown('<div class="section-header">CIBIL Score Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Histogram(x=df_f["CIBIL_Score_Current"].dropna(), nbinsx=20, marker_color=COLORS["mezz"]))
        fig.update_layout(**PLOTLY_LAYOUT, height=340, xaxis_title="CIBIL Score", yaxis_title="Loan Count")
        st.plotly_chart(fig, use_container_width=True)
    with r1c3:
        st.markdown('<div class="section-header">Vehicle Concentration</div>', unsafe_allow_html=True)
        v_grp = df_f.groupby("VehicleType")["CurrentBalance"].sum()
        fig = go.Figure(go.Bar(x=v_grp.index, y=v_grp.values, marker_color=COLORS["equity"]))
        fig.update_layout(**PLOTLY_LAYOUT, height=340)
        inr_axis(fig)
        st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.markdown('<div class="section-header">Credit Migration Roll-Rate Heatmap</div>', unsafe_allow_html=True)
    st.caption("Balance-weighted transition probabilities from prior DPD bucket → current DPD bucket")
    if df_dpd.empty:
        st.info("No DPD snapshot data available.")
    else:
        mx = pd.crosstab(df_dpd["DPD_Bucket_Prior"], df_dpd["DPD_Bucket"], values=df_dpd["CurrentBalance"], aggfunc="sum", normalize="index").fillna(0)
        fig = go.Figure(go.Heatmap(
            z=mx.values * 100, x=mx.columns, y=mx.index,
            colorscale="RdYlGn_r", text=[[f"{v:.1f}%" for v in row] for row in mx.values * 100],
            texttemplate="%{text}", textfont=dict(size=11), showscale=False,
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=430)
        st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.markdown('<div class="section-header">Cumulative Net Loss Curves (Vintage Analysis)</div>', unsafe_allow_html=True)
    st.caption("Static-pool cohort loss development by Months-on-Book (MOB)")
    fig = go.Figure()
    for v in sorted(df_vint["VintageID"].unique()):
        sub = df_vint[df_vint["VintageID"] == v].sort_values("MonthsOnBook")
        fig.add_trace(go.Scatter(x=sub["MonthsOnBook"], y=sub["CumulativeNetLossRate"] * 100, mode="lines+markers", name=str(v), marker=dict(size=3), line=dict(width=1.6)))
    fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Months on Book (MOB)", yaxis_title="CNL Rate (%)", height=460)
    st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.markdown('<div class="section-header">ECL Stage Summary Table (IFRS 9)</div>', unsafe_allow_html=True)
    stage_labels = {1: "Stage 1 (12M ECL)", 2: "Stage 2 (Lifetime)", 3: "Stage 3 (Credit-Impaired)"}
    stage_grp = df_f.groupby("IFRS9_Stage").agg(Count=("LoanID", "count"), Balance=("CurrentBalance", "sum"), ECL=("ECL_Provision", "sum"), AvgPD=("PD_Estimate", "mean"), AvgLGD=("LGD_Estimate", "mean")).reset_index()
    stage_grp["ECL/Balance %"] = stage_grp["ECL"] / stage_grp["Balance"] * 100
    stage_grp["Stage Label"] = stage_grp["IFRS9_Stage"].map(stage_labels).fillna(stage_grp["IFRS9_Stage"].astype(str))
    disp_ecl = stage_grp[["Stage Label", "Count", "Balance", "ECL", "ECL/Balance %", "AvgPD", "AvgLGD"]].copy()
    for c in ["Balance", "ECL"]: disp_ecl[c] = disp_ecl[c].map(lambda v: f"₹{v:,.0f}")
    disp_ecl["ECL/Balance %"] = disp_ecl["ECL/Balance %"].map(lambda v: f"{v:.2f}%")
    st.dataframe(disp_ecl, use_container_width=True, hide_index=True)
    st.caption("Note: PD/LGD assumptions are fixed policy inputs by stage (2% / 15% / 100% PD, 45% LGD) rather than loan-level model outputs.")

with tabs[5]:
    st.markdown('<div class="section-header">🛡️ Multi-Scenario Stress Test — Sensitivity Engine</div>', unsafe_allow_html=True)
    st.caption("Fixed macro shock scenarios applied against current structural sizing in the sidebar")
    scenarios = {"Base Case": (1.00, 1.00), "Mild Stress": (0.90, 1.50), "Severe Stress": (0.70, 3.00), "Crisis Scenario": (0.55, 4.00)}
    stress_results = []
    for name, (cm, lm) in scenarios.items():
        df_s = run_waterfall(df_dyn, senior_tgt, mezz_tgt, senior_cpn, mezz_cpn, svc_fee, cm, lm)
        s_out = df_s["SeniorOutstanding"].iloc[-1]
        stress_results.append({
            "Scenario": name, "Coll. Mult": cm, "Loss Mult": lm,
            "Senior Total Paid": df_s["SeniorTotal"].sum(), "Mezz Total Paid": df_s["MezzTotal"].sum(),
            "Senior Outstanding (End)": s_out, "Status": "✅ Covered" if s_out <= 0 else "🔴 Deficit"
        })
    disp_s = pd.DataFrame(stress_results)
    for col in ["Senior Total Paid", "Mezz Total Paid", "Senior Outstanding (End)"]: disp_s[col] = disp_s[col].map(lambda v: f"₹{v:,.0f}")
    st.dataframe(disp_s, use_container_width=True, hide_index=True)

    breach = [r for r in stress_results if "Deficit" in r["Status"]]
    if breach:
        st.markdown(f'<span class="risk-badge risk-bad">⚠ Senior tranche breaches in {len(breach)} of {len(scenarios)} scenarios</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="risk-badge risk-ok">✅ Senior tranche fully covered in all scenarios</span>', unsafe_allow_html=True)

st.write("---")
st.caption("ZAAUTO2024-1 · Securitisation Risk & Waterfall Engine · Built for Intern Assessment Review")