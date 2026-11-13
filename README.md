# DS 5020 — Walmart Sales: Bayes Theorem & Naive Bayes

Complete step-by-step Jupyter solution for DS 5020 Mini-Project 02, **Will This Be a Strong Sales Week?**

## Files

- `DS5020_Walmart_Bayes_Complete.ipynb` — fully executed Jupyter notebook with warm-up W1–W7 and P1–P3, B1–B7, N1–N7, J1.
- `Walmart.csv` — original 6,435-row Walmart data.
- `DS5020_Walmart_Answer_Sheet.csv` — completed assignment answer sheet.
- `note.txt` — five-sentence stakeholder note.
- `DS5020_Walmart_Step_by_Step_Cells.md` — copy/paste-friendly code and explanations.
- The original assignment PDF is included in the downloadable project ZIP but not uploaded here; the notebook follows its task structure.

## Run

1. Download the repository or clone it locally.
2. Open a terminal in this folder and run `pip install -r requirements.txt` (ideally in your Python environment).
3. Open `DS5020_Walmart_Bayes_Complete.ipynb` in Jupyter or VS Code.
4. Choose **Restart & Run All**. The notebook reads `Walmart.csv` from the same folder.

## Important methodology

- Part P/B: `S` means `Weekly_Sales >= 1.10 × store's average over the complete data`.
- Part N: 2010–2011 are training weeks and 2012 is the test set. Store averages for **both** train and test labels are computed from training weeks only.
- Categorical Naive Bayes uses Laplace smoothing (`alpha=1`) in manual profile scores and in scikit-learn.
- The standard 0.5-threshold classifier has **0% recall** on this test set; this is a genuine evaluated outcome, not an error. The optional threshold exercise explores alternatives.

## Reproduced checks

The notebook was executed successfully with all 20 code cells. The dataset checks are 6,435 rows, 45 stores, 143 weeks and 450 holiday store-weeks.

For class submission, follow your instructor's requirements about whether to submit all supporting files, and review the notebook/explanations as your own work before submitting.
