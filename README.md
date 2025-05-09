# Diagnostic-Agent

A simplified **Bayesian Decision Support System** for health diagnostics.

## 📁 Repository Structure

```
Diagnostic-Agent/
├─ data/
│  └─ normalized_health_data.csv
├─ src/
│  ├─ diagnostic_agent.py
│  ├─ analyze_marginals.py
│  ├─ visualization.py
│  └─ models/bayesian_model.py
├─ notebook/
│  └─ visualizations.ipynb
├─ tests/
│  ├─ test_diagnostic_agent.py
│  └─ test_full_feature_agent.py
├─ venv/                 # your virtual environment
├─ main.py
├─ requirements.txt
├─ pytest.ini
└─ README.md
```

## 🛠️ Setup

1. **Create a fresh virtual environment**

   ```bash
   python -m venv venv
   ```

2. **Activate the venv**

   * **Windows cmd.exe**

     ```bat
     venv\Scripts\activate.bat
     ```
   * **PowerShell (temporary bypass)**

     ```powershell
     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
     .\venv\Scripts\Activate.ps1
     ```
   * **macOS/Linux**

     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 🚀 Running the Core Agent

From the project root:

```bash
python main.py
```

This will:

1. Load and normalize the data.
2. Build both joint-distribution and Naïve-Bayes models.
3. Compute marginals, perform sample inferences, and print results.
4. Produce and display all visualizations (symptom marginals, full marginal panel, heatmap, ROC, calibration, PR curves, confusion matrix).

## 📊 Jupyter Notebook Visualizations

If you want interactive plots in Jupyter:

1. **Ensure your venv is activated** (see above).
2. **Launch from the project root**:

   ```bash
   cd notebook
   python -m notebook
   ```
3. Open **visualizations.ipynb** in your browser and run all cells.

## 🧪 Testing

Run all unit tests with:

```bash
python -m pytest
```

## 🔧 Adding New Plots

All plotting functions live in `src/visualization.py`. You can:

* **`plot_symptom_marginals(joint)`**
* **`plot_full_marginal_distributions(joint, alpha=0)`**
* **`plot_heatmap_top1(joint)`**
* **Calibration, PR and confusion helpers**

They’re all invoked from `main.py` already, but you can import and call them directly:

```python
from src.visualization import plot_full_marginal_distributions
plot_full_marginal_distributions(joint_prob, df, k=10)
```

## 📂 Saving Output

By default, the full-marginals panel is saved to:

```
output/marginals_top10.png
```

and you can override the directory or filename in the call.

---

Feel free to tweak any of the parameters (e.g. number of top conditions `k`, DPI, figure sizes) to suit your reports and slides.
