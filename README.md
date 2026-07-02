# Securitisation Risk & Waterfall Analytics Terminal (Deal ID: ZAAUTO2024-1)

A production-grade, interactive quantitative analytics console designed for structured finance desks to evaluate an Auto Loan Asset-Backed Securities (ABS) pool consisting of 500 underlying retail loan contracts (Total Pool Size: ₹54.3 Cr).

---

## 🛠️ Core Quantitative Features

*   **Sequential Priority Waterfall Engine:** Implements a strict sequential paydown logic routing cash from asset collections through servicing fees down to Senior (AAA), Mezzanine (BBB), and Equity tranches.
*   **Localized IRR Solver (Newton-Raphson):** Bypasses unstable monthly polynomial solvers with a bounded financial convergence algorithm to calculate mathematically sound, annualized Internal Rate of Return (IRR) and Weighted Average Life (WAL) for capital tranches.
*   **Macro Sensitivity Stress Testing:** Evaluates structural cash flow resilience against multiple macroeconomic shock vectors (Base Case up to Systemic Liquidity Crisis) with built-in capital impairment trackers.
*   **Collateral Stratification Analytics:** Dynamic, interactive Plotly visualization matrices analyzing asset data tape distribution including CIBIL credit score migration, Loan-to-Value (LTV) bands, regional exposure, and vehicle segments.
*   **IFRS 9 Impairment Ledger:** Automatically maps loans into Stage 1, Stage 2, and Stage 3 frameworks utilizing forward-looking Probability of Default (PD) and Loss Given Default (LGD) credit provisioning rules.
*   **Credit Migration Transition Density:** Generates balance-weighted transition probability matrices via dynamic heatmaps tracking prior vs. current delinquency day buckets (DPD).

---

## 💻 Tech Stack & Architecture

*   **Language:** Python 3.10
*   **Interface Layer:** Streamlit Community Cloud (Custom Dark-Navy Institutional Styling Engine)
*   **Data Science Engine:** Pandas (vectorized cash-flow mapping), NumPy
*   **Visualization Layer:** Plotly Express & Plotly Graph Objects (High-fidelity interactive charts)
