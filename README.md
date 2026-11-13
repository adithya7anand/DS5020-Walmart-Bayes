# DS 5020 Mini-Project 02 — Walmart Sales Analysis

**Will This Be a Strong Sales Week?**  
*Bayes' Theorem, Conditional Probability, and Naive Bayes Classification*

This project studies Walmart's historical weekly sales to answer a practical retail-planning question: **when should a store expect a stronger-than-usual sales week?** It combines probability calculations, Bayes' theorem, and a Naive Bayes classifier to compare simple staffing rules with a data-driven prediction model.

The analysis is organized as a step-by-step Jupyter notebook, progressing from event definitions and hand calculations to a time-based machine-learning evaluation.

## Project objectives

- Define a **strong sales week** relative to each store's usual sales, rather than using one fixed sales threshold for every store.
- Measure the probability of strong sales and their association with holidays, the November–December season, temperature, fuel prices, unemployment, and store size.
- Use conditional probability, Bayes' theorem, and the law of total probability to explain relationships among events.
- Compare two staffing heuristics: **staff only on holiday weeks** and **staff during November–December**.
- Train a Naive Bayes classifier using historical weeks and evaluate it against simple baselines on unseen weeks.
- Interpret model precision, recall, and the limitations of the available data.

## Dataset

**Source:** [Walmart Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/walmart-dataset)

The supplied `Walmart.csv` contains weekly records for **45 stores** across **143 weeks**, from February 2010 through October 2012. Each row represents one **store-week**.

| Column | Description |
| --- | --- |
| `Store` | Store identifier (1–45) |
| `Date` | Week-ending date, parsed with `dayfirst=True` |
| `Weekly_Sales` | Total weekly sales in US dollars |
| `Holiday_Flag` | Whether the week is designated as a holiday week |
| `Temperature` | Average temperature in °F |
| `Fuel_Price` | Regional fuel price per gallon |
| `CPI` | Consumer Price Index |
| `Unemployment` | Regional unemployment rate |

Stores operate at different sales volumes, so a store's weekly sales are compared against **that store's own mean**.

## Event definitions

Seven binary events are used consistently throughout the probability analysis:

| Symbol | Event | Condition |
| --- | --- | --- |
| **S** | Strong sales week | `Weekly_Sales >= 1.10 * store_average` |
| **H** | Holiday week | `Holiday_Flag == 1` |
| **Q** | Holiday season | Month is November or December |
| **C** | Cold week | `Temperature < 40` °F |
| **F** | Expensive fuel | `Fuel_Price >= 3.50` |
| **J** | High unemployment | `Unemployment >= 8` |
| **G** | Large store | `store_average >= 1_000_000` USD |

**Important distinction:** Parts **P** and **B** calculate store averages using the complete dataset for descriptive analysis. Part **N** calculates the store averages from **2010–2011 training records only** and applies those same averages when labeling 2012 test records, preventing information from the test period from leaking into training.

## Analysis walkthrough

### 1. Probability warm-up (W1–W7)

A small, 20-observation contingency table is used to practice:

- Marginal and conditional probabilities
- Bayes' theorem and the law of total probability
- Independence checks
- Precision and recall of a holiday-based rule
- Hand-calculated Naive Bayes scores and normalized probabilities

### 2. Event and joint probabilities (P1–P3)

Load the CSV, parse dates, define the seven events, and compute:

- `P(S)`, `P(H)`, `P(Q)`, `P(C)`, `P(F)`, `P(J)`, and `P(G)`
- Joint probabilities such as `P(S ∩ H)` and `P(S ∩ Q)`
- Complement-based checks and direct-count comparisons

### 3. Bayes' theorem and staffing rules (B1–B7)

Examine the difference between `P(S | H)` and `P(H | S)`, break sales down by specific holiday, and rebuild the overall strong-week probability using the law of total probability.

The analysis also compares staffing precision and recall for holiday-only and November–December rules, explores alternative strong-week base rates, and checks whether candidate features are independent of the target.

### 4. Naive Bayes classification (N1–N7)

- **Train:** weeks from 2010 and 2011
- **Test:** weeks from 2012
- **Inputs:** the six binary features `H, Q, C, F, J, G`
- **Target:** strong sales week `S`
- **Estimator:** scikit-learn's `CategoricalNB(alpha=1)`, with Laplace smoothing

The notebook builds class-prior and feature-likelihood tables, calculates two profile predictions manually, compares those posterior probabilities with scikit-learn, tests the conditional-independence assumption, and evaluates multiple classification thresholds.

### 5. Evaluation and outputs

The 2012 evaluation reports confusion matrices, accuracy, precision, and recall for:

1. **Never strong:** always predict a normal sales week.
2. **Holiday rule:** predict strong sales for holiday weeks.
3. **Naive Bayes:** predict using the six features and the chosen threshold.

The notebook also generates a filled answer sheet and a five-sentence note intended for a retail operations stakeholder.

## Selected findings

| Finding | Result |
| --- | ---: |
| Strong sales weeks across the complete dataset | **12.65%** |
| Chance of strong sales in a holiday week | **27.11%** |
| Chance of strong sales in a non-holiday week | **11.56%** |
| Chance of strong sales in November–December | **41.11%** |
| Strongest holiday by strong-week rate | **Thanksgiving (82.22%)** |
| Holiday-only rule recall across all store-weeks | **14.99%** |

On the **2012 test set**, the holiday-only rule achieves **9.02% recall**. The Naive Bayes classifier at the default **0.5 threshold** predicts no strong weeks, giving **0% recall**; the notebook explores lower thresholds to show how the precision–recall balance changes.

These findings illustrate why a rule can have a useful conditional probability yet still miss most strong weeks, and why **accuracy alone is insufficient** when strong weeks are relatively uncommon.

## Repository contents

```text
DS5020-Walmart-Bayes/
├── DS5020_Walmart_Bayes_Complete.ipynb
├── DS5020_Walmart_Step_by_Step_Cells.md
├── DS5020_Walmart_Answer_Sheet.csv
├── DS5020_Walmart_Detailed_Answer_Sheet.md
├── Walmart.csv
├── note.txt
├── requirements.txt
├── README.md
└── .gitignore
```

- **Notebook:** Complete explanation and executable analysis in assignment order.
- **Step-by-step Markdown:** Code snippets and explanations for following the workflow.
- **Dataset:** Original Walmart weekly-sales CSV.
- **Answer sheet:** Consolidated probabilities and performance results.
- **Detailed answer sheet:** Every task worked through with its formula, substituted counts, verification, and interpretation (`DS5020_Walmart_Detailed_Answer_Sheet.md`).
- **Stakeholder note:** Five-sentence summary of operational findings.
- **Requirements:** Python packages needed to reproduce the workflow.

## Getting started

**Requirements:** Python 3 and Jupyter Notebook or JupyterLab. The notebook also works in VS Code with a Python/Jupyter environment.

Clone the repository:

```bash
git clone https://github.com/adithya7anand/DS5020-Walmart-Bayes.git
cd DS5020-Walmart-Bayes
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook DS5020_Walmart_Bayes_Complete.ipynb
```

In Jupyter, select **Kernel → Restart & Run All** to execute the cells in order. The notebook reads `Walmart.csv` directly from the repository folder and exports `DS5020_Walmart_Answer_Sheet.csv` and `note.txt`.

## Interpretation and limitations

- **Time-based evaluation:** Splitting by year is more realistic than randomly mixing weeks from the same historical period.
- **Incomplete 2012 holiday coverage:** The test records stop in October 2012, before Thanksgiving and Christmas, so performance during those late-year peaks is not directly evaluated.
- **Naive Bayes independence assumption:** The model treats features as conditionally independent given the class, although holiday weeks and November–December overlap.
- **Class imbalance and threshold choice:** Most weeks are not strong; a classifier can achieve high accuracy while identifying few or no strong weeks at a conservative threshold.
- **Descriptive associations:** These probabilities capture patterns in this dataset and do not by themselves establish that a holiday, weather condition, or economic factor causes higher sales.

## Tools and concepts

**Python · pandas · NumPy · scikit-learn · Jupyter Notebook**  
Conditional probability · Bayes' theorem · Total probability · Naive Bayes · Confusion matrix · Precision · Recall
