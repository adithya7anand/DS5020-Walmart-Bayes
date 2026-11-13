# DS 5020 — Walmart Sales: Computational Walkthrough

The code sections correspond to the analysis in `DS5020_Walmart_Bayes_Complete.ipynb`.

# DS 5020 – Mini-Project 02: Will This Be a Strong Sales Week?
**Bayes’ theorem and Naive Bayes analysis on Walmart sales data**

The analysis uses `Walmart.csv` stored alongside the notebook. Parts P/B use each store’s full-dataset average; Part N uses **training-only store averages from 2010–2011** to avoid leakage into the 2012 test period.

**Definitions (R1):** a store-week is one row in the dataset (R2). The seven events are:

|Event|Meaning|
|---|---|
|S|Weekly_Sales ≥ 1.10 × that store's average weekly sales|
|H|Holiday_Flag = 1|
|Q|Date month is November or December|
|C|Temperature < 40°F|
|F|Fuel_Price ≥ $3.50|
|J|Unemployment ≥ 8%|
|G|Store average weekly sales ≥ $1,000,000|

**Verification (R5):** Probabilities are checked by complements, counts, sums to one, or independent formulas. **Presentation (R4):** Store identifiers only; no store addresses or personal identifiers.

## Warm-up W1–W7 (by hand, using the 20-row table on p. 3)
The table has 20 store-weeks: 6 strong and 14 not strong. There are 4 holiday weeks, 5 November–December weeks, 3 strong holiday weeks, and 4 strong November–December weeks.

**W1. Marginal probabilities:**

- \(P(S)=6/20=0.30\).
- \(P(H)=4/20=0.20\).
- \(P(Q)=5/20=0.25\).

**W2. Conditionals:** \(P(S\mid H)=3/4=0.75\), and \(P(S\mid H^c)=3/16=0.1875\). A strong week is **4 times as likely** on a holiday week: \(0.75/0.1875=4\).

**W3. Bayes:** \(P(H\mid S)=\frac{P(S\mid H)P(H)}{P(S)}=\frac{0.75(0.20)}{0.30}=0.50\). Check: 3 of the 6 strong weeks are holidays, \(3/6=0.50\).

**W4. Total probability:** \(P(S)=P(S\mid H)P(H)+P(S\mid H^c)P(H^c)=0.75(0.20)+0.1875(0.80)=0.30\), matching W1.

**W5. Independence:** \(P(S\cap H)=3/20=0.15\), whereas \(P(S)P(H)=0.30(0.20)=0.06\). They are **not independent**.

**W6. Holiday staffing rule:** precision \(=P(S\mid H)=0.75\); recall \(=P(H\mid S)=0.50\); fraction of strong weeks missed \(=1-0.50=0.50\) (**50%**).

**W7. Naive Bayes by hand (no smoothing for the small table):**

For a week with **H=1, Q=1**:

- Strong score = \(P(S)P(H\mid S)P(Q\mid S)=(6/20)(3/6)(4/6)=0.100000\).
- Not-strong score = \(P(S^c)P(H\mid S^c)P(Q\mid S^c)=(14/20)(1/14)(1/14)=0.00357143\).
- Normalized \(P(S\mid H,Q)\approx0.100000/(0.100000+0.00357143)=\mathbf{0.9655}\). True observed fraction in that table column: \(2/2=1.0000\).

For a week with **H=0, Q=0**:

- Strong score = \((6/20)(3/6)(2/6)=0.050000\).
- Not-strong score = \((14/20)(13/14)(13/14)=0.60357143\).
- Normalized \(P(S\mid H^c,Q^c)\approx0.050000/(0.050000+0.60357143)=\mathbf{0.0765}\). True observed fraction: \(1/13\approx0.0769\).

**Interpretation:** Naive Bayes approximately matches the nonholiday/nonseason case; for holiday+season it predicts ~96.55% versus 100% in the small sample. These are not necessarily identical because the independence assumption is approximate.

## Cell 1 — Imports and P1: load/check the data

Import the libraries, read the file, parse the date as day–month–year, and check the dataset against the assignment.

```python
import numpy as np
import pandas as pd
from pathlib import Path
from IPython.display import display
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

CSV_PATH = Path("Walmart.csv")

df = pd.read_csv(CSV_PATH)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

print("Rows:", len(df))
print("Stores:", df["Store"].nunique())
print("Unique weeks:", df["Date"].nunique())
print("Holiday store-weeks:", df["Holiday_Flag"].sum())
print("Missing values:", df.isna().sum().sum())
display(df.head())

assert len(df) == 6435
assert df["Store"].nunique() == 45
assert df["Date"].nunique() == 143
assert df["Holiday_Flag"].sum() == 450
assert df.isna().sum().sum() == 0
```

## Cell 2 — Define all seven events (R1)

Create a reusable function. For P/B use every store’s overall mean; later, N1 calls this function with training-only means.

```python
def add_events(data, store_avg):
    data = data.copy()
    avg = data["Store"].map(store_avg)
    assert avg.notna().all(), "All stores must have a training average."
    data["S"] = data["Weekly_Sales"] >= 1.10 * avg
    data["H"] = data["Holiday_Flag"] == 1
    data["Q"] = data["Date"].dt.month.isin([11, 12])
    data["C"] = data["Temperature"] < 40
    data["F"] = data["Fuel_Price"] >= 3.50
    data["J"] = data["Unemployment"] >= 8
    data["G"] = avg >= 1_000_000
    return data

full_store_avg = df.groupby("Store")["Weekly_Sales"].mean()
ev = add_events(df, full_store_avg)
EVENTS = ["S", "H", "Q", "C", "F", "J", "G"]
FEATURES = ["H", "Q", "C", "F", "J", "G"]
display(ev[["Store", "Date", "Weekly_Sales"] + EVENTS].head())
```

## Cell 3 — P2: event probabilities with complement checks

For each event, calculate count/6435 and independently confirm P(E) + P(not E) = 1.

```python
p2_rows = []
for event in EVENTS:
    count = int(ev[event].sum())
    p_count = count / len(ev)
    p_complement = (~ev[event]).sum() / len(ev)
    p2_rows.append({
        "Event": event, "Count": count, "P(Event)": p_count,
        "P(Not event)": p_complement, "Sum (check=1)": p_count + p_complement
    })
p2 = pd.DataFrame(p2_rows)
display(p2.round(6))
assert np.allclose(p2["Sum (check=1)"], 1.0)
```

## Cell 4 — P3: joint probabilities

For each pair, count rows satisfying both events; check against the mean of the Boolean intersection.

```python
pairs_p3 = [("S", "H"), ("S", "Q"), ("H", "Q"), ("S", "C")]
p3_rows = []
for a, b in pairs_p3:
    together = ev[a] & ev[b]
    p_count = together.sum() / len(ev)
    p_mean = together.mean()
    assert np.isclose(p_count, p_mean)
    p3_rows.append({"Joint event": f"P({a} ∩ {b})", "Count": int(together.sum()),
                    "Probability": p_count, "Second check": p_mean})
p3 = pd.DataFrame(p3_rows)
display(p3.round(6))
```

## Cell 5 — B1: strong sales conditional on holiday/nonholiday

Compute P(S|H) and P(S|Hᶜ) using counts, check with group means, and compare.

```python
strong_h = int((ev["S"] & ev["H"]).sum())
n_h = int(ev["H"].sum())
strong_nonh = int((ev["S"] & ~ev["H"]).sum())
n_nonh = int((~ev["H"]).sum())
p_s_given_h = strong_h / n_h
p_s_given_nonh = strong_nonh / n_nonh
assert np.isclose(p_s_given_h, ev.loc[ev["H"], "S"].mean())
assert np.isclose(p_s_given_nonh, ev.loc[~ev["H"], "S"].mean())
print(f"P(S|H) = {strong_h}/{n_h} = {p_s_given_h:.6f} ({100*p_s_given_h:.2f}%)")
print(f"P(S|Hᶜ) = {strong_nonh}/{n_nonh} = {p_s_given_nonh:.6f} ({100*p_s_given_nonh:.2f}%)")
print(f"A strong week is {p_s_given_h/p_s_given_nonh:.2f} times as likely on a holiday week.")
```

## Cell 6 — B2: Bayes’ theorem and counting check

Use P(H|S) = P(S|H)P(H)/P(S), then compute the same probability directly among strong rows.

```python
p_s = ev["S"].mean()
p_h = ev["H"].mean()
p_h_given_s_bayes = p_s_given_h * p_h / p_s
p_h_given_s_count = (ev["H"] & ev["S"]).sum() / ev["S"].sum()
print(f"P(H|S) by Bayes = ({p_s_given_h:.6f} × {p_h:.6f})/{p_s:.6f} = {p_h_given_s_bayes:.6f}")
print(f"P(H|S) by counting = {int((ev['H'] & ev['S']).sum())}/{int(ev['S'].sum())} = {p_h_given_s_count:.6f}")
assert np.isclose(p_h_given_s_bayes, p_h_given_s_count)
```

## Cell 7 — B3: total probability across five holiday groups

Assign the specific holiday dates listed in the PDF; calculate P(group), P(S|group), contribution, and P(group|S).

```python
holiday_dates = {
    "Super Bowl": ["2010-02-12", "2011-02-11", "2012-02-10"],
    "Labor Day": ["2010-09-10", "2011-09-09", "2012-09-07"],
    "Thanksgiving": ["2010-11-26", "2011-11-25"],
    "Christmas": ["2010-12-31", "2011-12-30"],
}
ev["Holiday_Group"] = "No holiday"
for group, dates in holiday_dates.items():
    ev.loc[ev["Date"].isin(pd.to_datetime(dates)), "Holiday_Group"] = group
assert (ev.loc[ev["H"], "Holiday_Group"] != "No holiday").all()
assert (ev.loc[~ev["H"], "Holiday_Group"] == "No holiday").all()

b3_rows = []
for group in ["Super Bowl", "Labor Day", "Thanksgiving", "Christmas", "No holiday"]:
    sub = ev.loc[ev["Holiday_Group"] == group]
    pg = len(sub) / len(ev)
    psg = sub["S"].mean()
    b3_rows.append({
        "Group": group, "Store-weeks": len(sub), "Strong weeks": int(sub["S"].sum()),
        "P(group)": pg, "P(S|group)": psg,
        "P(S and group)": pg * psg,
        "P(group|S)": int(sub["S"].sum()) / int(ev["S"].sum())
    })
b3 = pd.DataFrame(b3_rows)
display(b3.round(6))
p_s_rebuilt = b3["P(S and group)"].sum()
print(f"P(S) rebuilt = {p_s_rebuilt:.6f}; directly = {p_s:.6f}")
print("Highest P(S|group):", b3.loc[b3["P(S|group)"].idxmax(), "Group"])
print("Largest share P(group|S):", b3.loc[b3["P(group|S)"].idxmax(), "Group"])
assert np.isclose(p_s_rebuilt, p_s)
assert np.isclose(b3["P(group)"].sum(), 1)
assert np.isclose(b3["P(group|S)"].sum(), 1)
```

## Cell 8 — B4: November–December conditionals

The denominators differ: P(S|Q) conditions on November–December; P(Q|S) conditions on strong weeks.

```python
n_s_and_q = int((ev["S"] & ev["Q"]).sum())
p_s_given_q = n_s_and_q / int(ev["Q"].sum())
p_q_given_s = n_s_and_q / int(ev["S"].sum())
assert np.isclose(p_s_given_q, ev.loc[ev["Q"], "S"].mean())
assert np.isclose(p_q_given_s, ev.loc[ev["S"], "Q"].mean())
print(f"P(S|Q) = {n_s_and_q}/{int(ev['Q'].sum())} = {p_s_given_q:.6f}")
print(f"P(Q|S) = {n_s_and_q}/{int(ev['S'].sum())} = {p_q_given_s:.6f}")
print("They differ because they use different reference groups (different denominators).")
```

## Cell 9 — B5: holiday vs November–December staffing rules

Compute each rule’s precision with the reverse Bayes formula; check precision and recall by counting.

```python
def staffing_rule(event):
    p_e_given_s = ev.loc[ev["S"], event].mean()
    p_e_given_not_s = ev.loc[~ev["S"], event].mean()
    precision_bayes = (p_e_given_s * p_s) / (
        p_e_given_s * p_s + p_e_given_not_s * (1 - p_s)
    )
    precision_count = ev.loc[ev[event], "S"].mean()
    recall_count = ev.loc[ev["S"], event].mean()
    assert np.isclose(precision_bayes, precision_count)
    return {
        "Rule": "Holiday only" if event == "H" else "November–December",
        "Precision": precision_bayes,
        "Recall": recall_count,
        "Strong weeks missed": 1 - recall_count,
        "Flagged store-weeks": int(ev[event].sum())
    }
b5 = pd.DataFrame([staffing_rule("H"), staffing_rule("Q")])
display(b5.round(6))
print("A larger recall means the rule misses fewer truly strong weeks.")
```

## Cell 10 — B6: hypothetical base rates

Hold the holiday likelihoods P(H|S) and P(H|Sᶜ) fixed, but change the prior P(S).

```python
p_h_given_not_s = ev.loc[~ev["S"], "H"].mean()
def holiday_precision_given_prior(new_prior):
    return (p_h_given_s_count * new_prior) / (
        p_h_given_s_count * new_prior + p_h_given_not_s * (1 - new_prior)
    )
b6 = pd.DataFrame([
    {"Scenario": "Twice as common", "Assumed P(S)": 2 * p_s,
     "Holiday precision": holiday_precision_given_prior(2 * p_s)},
    {"Scenario": "Half as common", "Assumed P(S)": p_s / 2,
     "Holiday precision": holiday_precision_given_prior(p_s / 2)},
])
display(b6.round(6))
assert 2 * p_s < 1
```

## Cell 11 — B7: independence checks

Compare P(S ∩ E) against P(S)P(E); a nonzero difference means dependence in this dataset.

```python
b7_rows = []
for feature in FEATURES:
    joint = (ev["S"] & ev[feature]).mean()
    if_independent = ev["S"].mean() * ev[feature].mean()
    b7_rows.append({
        "Pair": f"S and {feature}", "P(S∩E)": joint,
        "P(S)P(E)": if_independent, "Difference": joint - if_independent,
        "Abs. difference": abs(joint - if_independent)
    })
b7 = pd.DataFrame(b7_rows)
display(b7.round(6))
print("Largest absolute departures from independence:")
display(b7.sort_values("Abs. difference", ascending=False)[["Pair", "Difference"]].round(6))
```

## Cell 12 — N1: time-based train/test split (prevent leakage)

Train on 2010–2011 (4,500 store-weeks), test on 2012 (1,935). Recompute store averages using only train weeks.

```python
train_raw = df.loc[df["Date"].dt.year.isin([2010, 2011])].copy()
test_raw = df.loc[df["Date"].dt.year == 2012].copy()
train_store_avg = train_raw.groupby("Store")["Weekly_Sales"].mean()
train = add_events(train_raw, train_store_avg)
test = add_events(test_raw, train_store_avg)
print(f"Training: {len(train)} rows; P(S) = {train['S'].mean():.6f} ({100*train['S'].mean():.2f}%)")
print(f"Test:     {len(test)} rows; P(S) = {test['S'].mean():.6f} ({100*test['S'].mean():.2f}%)")
assert len(train) == 4500 and len(test) == 1935
assert train["Date"].max() < test["Date"].min()
assert train["Store"].nunique() == test["Store"].nunique() == 45
```

## Cell 13 — N2: priors and six binary likelihood tables

First show frequencies estimated by counting. Also show add-one (Laplace) smoothed probabilities (alpha=1) to match CategoricalNB in N5.

```python
n_train = len(train)
n_s = int(train["S"].sum())
n_not_s = n_train - n_s
prior_s = n_s / n_train
prior_not_s = n_not_s / n_train
print(f"P(S)={n_s}/{n_train}={prior_s:.6f}")
print(f"P(Sᶜ)={n_not_s}/{n_train}={prior_not_s:.6f}")
assert np.isclose(prior_s + prior_not_s, 1)

n2_rows = []
for feature in FEATURES:
    for value in (0, 1):
        yes_count = int(((train[feature].astype(int) == value) & train["S"]).sum())
        no_count = int(((train[feature].astype(int) == value) & ~train["S"]).sum())
        n2_rows.append({
            "Feature": feature, "Value": value,
            "Count if S": yes_count, "Count if Sᶜ": no_count,
            "P(value|S) raw": yes_count / n_s,
            "P(value|Sᶜ) raw": no_count / n_not_s,
            "P(value|S) smoothed": (yes_count + 1) / (n_s + 2),
            "P(value|Sᶜ) smoothed": (no_count + 1) / (n_not_s + 2),
        })
n2 = pd.DataFrame(n2_rows)
display(n2.round(6))
for feature in FEATURES:
    sub = n2.loc[n2["Feature"] == feature]
    for column in ["P(value|S) raw", "P(value|Sᶜ) raw",
                   "P(value|S) smoothed", "P(value|Sᶜ) smoothed"]:
        assert np.isclose(sub[column].sum(), 1)
```

## Cell 14 — N3: manual Naive Bayes scores and normalization

Profile (a): holiday, Nov–Dec, cold, cheap fuel, low unemployment, large store. Profile (b) is the opposite. Use the smoothed conditional probabilities from N2.

```python
profile_a = {"H": 1, "Q": 1, "C": 1, "F": 0, "J": 0, "G": 1}
profile_b = {"H": 0, "Q": 0, "C": 0, "F": 1, "J": 1, "G": 0}

def manual_nb_posterior(profile, label):
    score_s = prior_s
    score_not_s = prior_not_s
    steps = []
    for feature, value in profile.items():
        p_x_given_s = (int(((train[feature].astype(int) == value) & train["S"]).sum()) + 1) / (n_s + 2)
        p_x_given_not_s = (int(((train[feature].astype(int) == value) & ~train["S"]).sum()) + 1) / (n_not_s + 2)
        score_s *= p_x_given_s
        score_not_s *= p_x_given_not_s
        steps.append({"Feature": feature, "Value": value,
                      "P(value|S) smoothed": p_x_given_s,
                      "P(value|Sᶜ) smoothed": p_x_given_not_s})
    posterior = score_s / (score_s + score_not_s)
    other_posterior = score_not_s / (score_s + score_not_s)
    print(f"\nProfile {label}: {profile}")
    display(pd.DataFrame(steps).round(6))
    print(f"Score(S)  = P(S) × ∏P(feature|S)   = {score_s:.12f}")
    print(f"Score(Sᶜ) = P(Sᶜ) × ∏P(feature|Sᶜ) = {score_not_s:.12f}")
    print(f"P(S|x) = {score_s:.12f}/({score_s:.12f}+{score_not_s:.12f}) = {posterior:.9f}")
    assert np.isclose(posterior + other_posterior, 1)
    return posterior

posterior_a = manual_nb_posterior(profile_a, "a")
posterior_b = manual_nb_posterior(profile_b, "b")
```

## Cell 15 — N4: check conditional independence of H and Q

Test whether P(H ∩ Q|S) equals P(H|S) × P(Q|S) in the training data.

```python
strong_train = train.loc[train["S"]]
p_hq_given_s = (strong_train["H"] & strong_train["Q"]).mean()
p_h_s_times_p_q_s = strong_train["H"].mean() * strong_train["Q"].mean()
print(f"P(H∩Q|S) = {p_hq_given_s:.6f}")
print(f"P(H|S)P(Q|S) = {p_h_s_times_p_q_s:.6f}")
print(f"Difference = {p_hq_given_s - p_h_s_times_p_q_s:.6f}")
print("H and Q are not independent: Thanksgiving and Christmas holidays fall in Nov–Dec.")
```

## Cell 16 — N5: fit CategoricalNB and verify N3

Use exactly the same six binary features and alpha=1 Laplace smoothing as in N3.

```python
X_train = train[FEATURES].astype(int)
y_train = train["S"].astype(int)
X_test = test[FEATURES].astype(int)
y_test = test["S"].astype(int)

nb = CategoricalNB(alpha=1)
nb.fit(X_train, y_train)
profiles_df = pd.DataFrame([profile_a, profile_b], index=["Profile a", "Profile b"])[FEATURES]
nb_posteriors = nb.predict_proba(profiles_df)[:, list(nb.classes_).index(1)]
comparison_n5 = pd.DataFrame({"Profile": profiles_df.index,
                              "Manual P(S|x)": [posterior_a, posterior_b],
                              "sklearn P(S|x)": nb_posteriors})
display(comparison_n5.round(9))
assert np.allclose([posterior_a, posterior_b], nb_posteriors, atol=1e-12)
print("The manual calculations match scikit-learn.")
```

## Cell 17 — N6: confusion matrices and baseline comparisons

Evaluate on 2012 only: never strong, holiday-only rule, and Naive Bayes with the default 0.5 decision threshold.

```python
def evaluation_row(name, actual, predicted):
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    precision_count = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_count = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    assert np.isclose(precision_count, precision_score(actual, predicted, zero_division=0))
    assert np.isclose(recall_count, recall_score(actual, predicted, zero_division=0))
    return {"Method": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "Accuracy": accuracy_score(actual, predicted),
            "Precision": precision_count, "Recall": recall_count}

pred_never = np.zeros(len(test), dtype=int)
pred_holiday = test["H"].astype(int).to_numpy()
pred_nb = nb.predict(X_test)
n6 = pd.DataFrame([
    evaluation_row("Never strong", y_test, pred_never),
    evaluation_row("Holiday only", y_test, pred_holiday),
    evaluation_row("Naive Bayes (0.5)", y_test, pred_nb),
])
display(n6.round(6))
for name, predicted in [("Never strong", pred_never), ("Holiday only", pred_holiday),
                        ("Naive Bayes", pred_nb)]:
    print(f"\n{name} confusion matrix [[TN,FP],[FN,TP]]:")
    print(confusion_matrix(y_test, predicted, labels=[0, 1]))
print("\n2012 has no Thanksgiving or Christmas records (data ends in October).")
print("This limits evaluation of holiday-season effects learned from 2010–2011.")
```

## Cell 18 — N7: threshold stretch task

Predict strong when P(S|x) is strictly greater than t; compare four thresholds.

```python
test_proba = nb.predict_proba(X_test)[:, list(nb.classes_).index(1)]
n7_rows = []
for t in [0.1, 0.2, 0.3, 0.5]:
    predicted = (test_proba > t).astype(int)
    row = evaluation_row(f"t={t}", y_test, predicted)
    n7_rows.append({"Threshold": t, "Predicted strong": int(predicted.sum()),
                    "Accuracy": row["Accuracy"], "Precision": row["Precision"],
                    "Recall": row["Recall"]})
n7 = pd.DataFrame(n7_rows)
display(n7.round(6))
print("Lower thresholds usually flag more weeks; precision and recall need not both improve.")
```

## Cell 19 — J1: five-sentence note to Maria (note.txt)

Build the note using measured values. Print and save it as note.txt, as required by the project.

```python
thanksgiving_p = float(b3.loc[b3["Group"] == "Thanksgiving", "P(S|group)"].iloc[0])
holiday_precision = float(b5.loc[b5["Rule"] == "Holiday only", "Precision"].iloc[0])
holiday_recall = float(b5.loc[b5["Rule"] == "Holiday only", "Recall"].iloc[0])
nb_recall = float(n6.loc[n6["Method"] == "Naive Bayes (0.5)", "Recall"].iloc[0])
holiday_test_recall = float(n6.loc[n6["Method"] == "Holiday only", "Recall"].iloc[0])

sentences = [
    f"Across all 6,435 store-weeks, {100*p_s:.2f}% were strong sales weeks.",
    f"Thanksgiving was the strongest holiday ({100*thanksgiving_p:.2f}% strong), while {100*p_s_given_q:.2f}% of November–December store-weeks were strong.",
    f"The holiday-only staffing rule was right {100*holiday_precision:.2f}% of the time it flagged a week, but missed {100*(1-holiday_recall):.2f}% of all strong weeks.",
    f"On the 2012 test set, Naive Bayes at the standard 0.5 cutoff did not beat the holiday rule at finding strong weeks: its recall was {100*nb_recall:.2f}% versus {100*holiday_test_recall:.2f}% for the holiday rule.",
    "One risk is that the 2012 test data end in October, so the model cannot be tested on that year's Thanksgiving or Christmas periods."
]
note_text = " ".join(sentences)
print(note_text)
Path("note.txt").write_text(note_text + "\n", encoding="utf-8")
assert len(sentences) == 5
```

## Cell 20 — Section 7: filled-in answer sheet

Collect the exact requested values into one table and export a CSV. B7 and N7 detailed tables appear above.

```python
highest_holiday = b3.loc[b3.loc[b3["Group"] != "No holiday", "P(S|group)"].idxmax(), "Group"]
strong_share_group = b3.loc[b3["P(group|S)"].idxmax(), "Group"]

answer_rows = [
    ("P1", f"{len(df)} rows; {df.Store.nunique()} stores; {df.Date.nunique()} weeks; {int(df.Holiday_Flag.sum())} holiday store-weeks"),
    ("P2", "; ".join(f"P({e})={ev[e].mean():.6f}" for e in EVENTS)),
    ("P3", "; ".join(f"P({a}∩{b})={(ev[a]&ev[b]).mean():.6f}" for a, b in pairs_p3)),
    ("B1", f"P(S|H)={p_s_given_h:.6f}; P(S|Hᶜ)={p_s_given_nonh:.6f}"),
    ("B2", f"P(H|S) by Bayes={p_h_given_s_bayes:.6f}; counting={p_h_given_s_count:.6f}"),
    ("B3", f"Highest P(S|group): {highest_holiday}; largest P(group|S): {strong_share_group}"),
    ("B4", f"P(S|Q)={p_s_given_q:.6f}; P(Q|S)={p_q_given_s:.6f}"),
    ("B5", "; ".join(f"{r['Rule']}: precision={r['Precision']:.6f}, recall={r['Recall']:.6f}" for _, r in b5.iterrows())),
    ("B6", "; ".join(f"{r['Scenario']}: {r['Holiday precision']:.6f}" for _, r in b6.iterrows())),
    ("N3", f"Profile a={posterior_a:.9f}; Profile b={posterior_b:.9f}"),
    ("N6", "; ".join(f"{r['Method']}: accuracy={r['Accuracy']:.6f}, precision={r['Precision']:.6f}, recall={r['Recall']:.6f}" for _, r in n6.iterrows())),
]
answer_sheet = pd.DataFrame(answer_rows, columns=["Task", "Your value"])
display(answer_sheet)
answer_sheet.to_csv("DS5020_Walmart_Answer_Sheet.csv", index=False)
print("Saved: DS5020_Walmart_Answer_Sheet.csv and note.txt")
```

