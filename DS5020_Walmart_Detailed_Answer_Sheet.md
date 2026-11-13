# DS 5020 — Mini-Project 02: Detailed Answer Sheet
**Will This Be a Strong Sales Week? — Conditional Probability, Bayes’ Theorem, and Naive Bayes**

This solution follows the question order **W1–W7, P1–P3, B1–B7, N1–N7, and J1**. Each result gives the formula, substituted values, arithmetic, and a short check or interpretation. All probabilities use store-weeks as their unit of observation.

## Definitions and calculation conventions (R1–R5)

| Symbol | Event | Exact test for one store-week |
|:--|:--|:--|
| **S** | Strong week | `Weekly_Sales >= 1.10 × that store’s average weekly sales` |
| **H** | Holiday week | `Holiday_Flag == 1` |
| **Q** | November–December | Week-ending month is 11 or 12 |
| **C** | Cold week | `Temperature < 40°F` |
| **F** | Expensive fuel | `Fuel_Price >= $3.50` |
| **J** | High unemployment | `Unemployment >= 8\%` |
| **G** | Large store | Store mean weekly sales `>= $1,000,000` |

- **Unit and sample space:** one record is one store-week; the full dataset contains $N=6,435$ records.
- **Descriptive analysis (P and B):** average sales for each store are calculated from **all weeks**.
- **Predictive analysis (N):** 2010–2011 are training weeks; 2012 is the test year. Each store’s reference average is computed **only from training weeks** and used for both training/test labels. These two definitions of `S` are applied in their respective parts and must not be mixed.
- **Counting:** $P(E)=n(E)/N$, $P(E^c)=1-P(E)$, $P(A\cap B)=n(A\cap B)/N$, and $P(A\mid B)=n(A\cap B)/n(B)$.
- **Precision:** $TP/(TP+FP)$; **recall:** $TP/(TP+FN)$; **accuracy:** $(TP+TN)/(TP+TN+FP+FN)$.
- Rounded decimals are displayed, but checks and calculations use the unrounded fractions.

***
## Part W — Warm-up by hand (20 synthetic store-weeks)
The four columns below are the mutually exclusive combinations of holiday week $H$ and November–December $Q$.

| | $H\cap Q$ | $H\cap Q^c$ | $H^c\cap Q$ | $H^c\cap Q^c$ | Total |
|:--|--:|--:|--:|--:|--:|
| Strong ($S$) | 2 | 1 | 2 | 1 | **6** |
| Not strong ($S^c$) | 0 | 1 | 1 | 12 | **14** |
| **Total** | **2** | **2** | **3** | **13** | **20** |

### W1. Find $P(S)$, $P(H)$, and $P(Q)$.
**Formula:** $P(E)=n(E)/20$.

1. Strong weeks: $n(S)=2+1+2+1=6$, so $P(S)=6/20=\boxed{0.30}$.
2. Holiday weeks: $n(H)=2+2=4$, so $P(H)=4/20=\boxed{0.20}$.
3. November–December weeks: $n(Q)=2+3=5$, so $P(Q)=5/20=\boxed{0.25}$.
4. **Complement checks:** $P(S^c)=14/20=0.70$, $P(H^c)=16/20=0.80$, $P(Q^c)=15/20=0.75$. Each event plus its complement equals 1.

### W2. Find $P(S\mid H)$ and $P(S\mid H^c)$; compare.
**Formula:** $P(S\mid H)=n(S\cap H)/n(H)$.

1. Strong and holiday: $n(S\cap H)=2+1=3$ of 4 holiday weeks, so $P(S\mid H)=3/4=\boxed{0.75}$.
2. Strong and nonholiday: $n(S\cap H^c)=2+1=3$ of 16 nonholiday weeks, so $P(S\mid H^c)=3/16=\boxed{0.1875}$.
3. Ratio $=0.75/0.1875=\boxed{4}$; difference $=0.75-0.1875=0.5625$ (56.25 percentage points).
4. **Interpretation:** A strong week is four times as likely among the holiday observations in this small table.

### W3. Calculate $P(H\mid S)$ using Bayes; verify by counting.
**Formula:** $P(H\mid S)=\frac{P(S\mid H)P(H)}{P(S)}$.

1. Substitute: $P(H\mid S)=\frac{(3/4)(4/20)}{6/20}$.
2. Simplify: $(3/20)/(6/20)=\boxed{3/6=0.50}$.
3. **Count check:** exactly 3 of the 6 strong observations are holiday weeks, so $3/6=0.50$.

### W4. Recover $P(S)$ using the law of total probability.
**Formula:** $P(S)=P(S\mid H)P(H)+P(S\mid H^c)P(H^c)$.

1. Substitute: $P(S)=(3/4)(4/20)+(3/16)(16/20)$.
2. Simplify: $P(S)=3/20+3/20=\boxed{6/20=0.30}$.
3. **Check:** This matches W1.

### W5. Are $S$ and $H$ independent?
**Independence condition:** $P(S\cap H)=P(S)P(H)$.

1. Actual joint probability: $P(S\cap H)=3/20=0.15$.
2. Product of marginals: $P(S)P(H)=(6/20)(4/20)=0.30(0.20)=0.06$.
3. Because $0.15\neq0.06$, **the two events are not independent** in this table; observing a holiday changes the probability of a strong week.

### W6. Precision, recall, and missed strong weeks under the holiday staffing rule.
Under this rule, a holiday is a **predicted strong week**. Thus $TP=3$, $FP=1$, $FN=3$, and $TN=13$.

1. **Precision:** $TP/(TP+FP)=3/(3+1)=\boxed{0.75=75\%}=P(S\mid H)$.
2. **Recall:** $TP/(TP+FN)=3/(3+3)=\boxed{0.50=50\%}=P(H\mid S)$.
3. **Missed fraction:** $FN/(TP+FN)=3/6=1-0.50=\boxed{0.50=50\%}$.
4. The rule is correct for three quarters of flagged weeks but misses half of the truly strong weeks.

### W7. Two Naive Bayes predictions calculated by hand (without smoothing).
**Naive Bayes formula:** for features $H=h,Q=q$, compute two unnormalized scores

$$u_S=P(S)P(H=h\mid S)P(Q=q\mid S),\qquad u_{S^c}=P(S^c)P(H=h\mid S^c)P(Q=q\mid S^c),$$
$$P_{\mathrm{NB}}(S\mid h,q)=\frac{u_S}{u_S+u_{S^c}}.$$

From the strong group ($n=6$): $P(H\mid S)=3/6$, $P(H^c\mid S)=3/6$, $P(Q\mid S)=4/6$, $P(Q^c\mid S)=2/6$.
From the not-strong group ($n=14$): $P(H\mid S^c)=1/14$, $P(H^c\mid S^c)=13/14$, $P(Q\mid S^c)=1/14$, $P(Q^c\mid S^c)=13/14$.

**Profile 1 — holiday + November–December ($H=1,Q=1$)**

1. Strong score: $u_S=(6/20)(3/6)(4/6)=0.10000000$.
2. Not-strong score: $u_{S^c}=(14/20)(1/14)(1/14)=0.00357143$.
3. Normalize: $P_{\mathrm{NB}}(S\mid H,Q)=0.1/(0.1+0.00357143)=\boxed{0.965517=96.55\%}$.
4. **Direct observed fraction:** in the $H\cap Q$ column, 2/2 observations are strong, i.e. **100\%**. The Naive Bayes estimate is lower because the product of marginal likelihoods is a model approximation.

**Profile 2 — neither holiday nor November–December ($H=0,Q=0$)**

1. Strong score: $u_S=(6/20)(3/6)(2/6)=0.05000000$.
2. Not-strong score: $u_{S^c}=(14/20)(13/14)(13/14)=0.60357143$.
3. Normalize: $P_{\mathrm{NB}}(S\mid H^c,Q^c)=0.05/(0.05+0.60357143)=\boxed{0.076503=7.65\%}$.
4. **Direct observed fraction:** in $H^c\cap Q^c$, 1/13 observations are strong, i.e. $\boxed{0.076923=7.69\%}$.

***
## Part P — Probabilities on the complete Walmart dataset

### P1. Load the data, parse dates, and verify dataset size.
**Method:** read `Walmart.csv` with `pandas.read_csv`; convert `Date` via `pandas.to_datetime(..., dayfirst=True)`; calculate row count, distinct stores, distinct weeks, and total holiday indicators.

1. $N=\operatorname{len}(df)=\boxed{6,435}$ store-weeks.
2. Unique store identifiers = $\boxed{45}$.
3. Unique dates = $\boxed{143}$ weeks.
4. Holiday store-weeks = $\sum\mathrm{Holiday\_Flag}=\boxed{450}$.
5. **Cross-check:** $45\times143=6,435$ total store-weeks. Holiday rate $450/6,435=0.069930=6.99\%$.

### P2. Probability of every event and complementary-event verification.
**Formula:** $P(E)=n(E)/6,435$ and $P(E^c)=(6,435-n(E))/6,435$; their sum must be 1.

| Event | Favorable $n(E)$ | Substitution | $P(E)$ | $n(E^c)$ | $P(E^c)$ | Sum |
|:--|--:|:--|--:|--:|--:|--:|
| **S** | 814 | $814/6,435$ | **0.126496** | 5,621 | 0.873504 | 1.000000 |
| **H** | 450 | $450/6,435$ | **0.069930** | 5,985 | 0.930070 | 1.000000 |
| **Q** | 810 | $810/6,435$ | **0.125874** | 5,625 | 0.874126 | 1.000000 |
| **C** | 989 | $989/6,435$ | **0.153691** | 5,446 | 0.846309 | 1.000000 |
| **F** | 3,029 | $3,029/6,435$ | **0.470707** | 3,406 | 0.529293 | 1.000000 |
| **J** | 3,022 | $3,022/6,435$ | **0.469619** | 3,413 | 0.530381 | 1.000000 |
| **G** | 3,146 | $3,146/6,435$ | **0.488889** | 3,289 | 0.511111 | 1.000000 |

**Worked example for S:** a store-week is strong if its sales are at least 1.10 times that store’s full-period mean. There are 814 such records, so $P(S)=814/6,435=0.126496$ and $P(S^c)=5,621/6,435=0.873504$. The sum is 1.000000.

### P3. Four joint probabilities.
**Formula:** $P(A\cap B)=n(A\cap B)/6,435=P(A)P(B\mid A)$.

- **$P(S\cap H)$:** 122 rows satisfy both flags; $P(S\cap H)=122/6,435=\boxed{0.018959}$. **Second check:** $P(S)P(H\mid S)=(814/6,435)(122/814)=122/6,435$.
- **$P(S\cap Q)$:** 333 rows satisfy both flags; $P(S\cap Q)=333/6,435=\boxed{0.051748}$. **Second check:** $P(S)P(Q\mid S)=(814/6,435)(333/814)=333/6,435$.
- **$P(H\cap Q)$:** 180 rows satisfy both flags; $P(H\cap Q)=180/6,435=\boxed{0.027972}$. **Second check:** $P(H)P(Q\mid H)=(450/6,435)(180/450)=180/6,435$.
- **$P(S\cap C)$:** 185 rows satisfy both flags; $P(S\cap C)=185/6,435=\boxed{0.028749}$. **Second check:** $P(S)P(C\mid S)=(814/6,435)(185/814)=185/6,435$.

***
## Part B — Conditional probability, Bayes’ theorem, and staffing decisions

### B1. Compare the strong-week probability with and without a holiday.
**Formula:** $P(S\mid H)=n(S\cap H)/n(H)$ and $P(S\mid H^c)=n(S\cap H^c)/n(H^c)$.

1. $P(S\mid H)=122/450=\boxed{0.271111=27.11\%}$.
2. $P(S\mid H^c)=692/5,985=\boxed{0.115622=11.56\%}$.
3. Difference **15.55 percentage points**; ratio $(122/450)/(692/5,985)=\boxed{2.345}$ times.
4. **Interpretation:** holidays are associated with an elevated chance of strong sales, but most holiday weeks still do not meet the strong-week threshold.

### B2. Reverse the conditional probability using Bayes’ theorem.
**Formula:** $P(H\mid S)=\frac{P(S\mid H)P(H)}{P(S)}$.

1. Substitute the unrounded counts: $P(H\mid S)=\frac{(122/450)(450/6,435)}{814/6,435}$.
2. Simplify: $122/814=\boxed{0.149877=14.99\%}$.
3. **Independent counting check:** 122 of the 814 strong store-weeks are holidays, so direct counting also gives $122/814$.
4. This is the **fraction of all strong weeks falling on holidays**, not the chance of being strong given a holiday.

### B3. Split weeks into five holiday types; apply total probability.
**Group definition:** four named holidays use their listed week-ending dates. Every other week is **No holiday**.

**Formulas:** $P(g)=n(g)/N$, $P(S\mid g)=n(S\cap g)/n(g)$, $P(S\cap g)=n(S\cap g)/N$, $P(g\mid S)=n(S\cap g)/n(S)$.

| Group | $n(g)$ | $n(S\cap g)$ | $P(g)$ | $P(S\mid g)$ | Contribution $P(S\cap g)$ | $P(g\mid S)$ |
|:--|--:|--:|--:|--:|--:|--:|
| Super Bowl | 135 | 26 | 0.020979 | **0.192593** | 0.004040 | 0.031941 |
| Labor Day | 135 | 16 | 0.020979 | **0.118519** | 0.002486 | 0.019656 |
| Thanksgiving | 90 | 74 | 0.013986 | **0.822222** | 0.011500 | 0.090909 |
| Christmas | 90 | 6 | 0.013986 | **0.066667** | 0.000932 | 0.007371 |
| No holiday | 5,985 | 692 | 0.930070 | **0.115622** | 0.107537 | 0.850123 |

**Worked example (Thanksgiving):** $P(g)=90/6,435=0.013986$; $P(S\mid g)=74/90=0.822222$; their product is $74/6,435=0.011500$; $P(g\mid S)=74/814=0.090909$.

**Total-probability reconstruction:**

$$P(S)=\sum_g P(S\mid g)P(g) =\frac{26+16+74+6+692}{6,435}=\frac{814}{6,435}=\boxed{0.126496}.$$

**Checks:** $135+135+90+90+5,985=6,435$; group contributions sum to $P(S)$ and group shares among strong weeks sum to 1.

**Answer:** Thanksgiving has the **highest conditional strong-week rate** ($74/90=82.22\%$); **No holiday** gives the **largest share of all strong weeks** ($692/814=85.01\%$) because nonholiday weeks are much more common.

### B4. Explain $P(S\mid Q)$ versus $P(Q\mid S)$.
**Formulas:** $P(S\mid Q)=n(S\cap Q)/n(Q)$; $P(Q\mid S)=n(S\cap Q)/n(S)$.

1. $P(S\mid Q)=333/810=\boxed{0.411111=41.11\%}$.
2. $P(Q\mid S)=333/814=\boxed{0.409091=40.91\%}$.
3. **Why different?** The first examines the fraction of November–December weeks that are strong (denominator 810); the second examines the fraction of strong weeks taking place in November–December (denominator 814).
4. **Bayes check:** $P(Q\mid S)=P(S\mid Q)P(Q)/P(S)=(333/810)(810/6,435)/(814/6,435)=333/814$.

### B5. Compare staffing rules using reverse Bayes, precision, and recall.
**Bayes formula for any staffing indicator $E$:**

$$\mathrm{Precision}(E)=P(S\mid E)=\frac{P(E\mid S)P(S)}{P(E\mid S)P(S)+P(E\mid S^c)P(S^c)}.$$
$$\mathrm{Recall}(E)=P(E\mid S);\qquad \mathrm{Missed}(E)=1-\mathrm{Recall}(E).$$

**Holiday-only rule ($E=H$):**

1. $P(H\mid S)=122/814=0.149877$; $P(H\mid S^c)=328/5,621=0.058353$.
2. $P(S)=814/6,435$; $P(S^c)=5,621/6,435$.
3. Bayes numerator: $(122/814)(814/6,435)=122/6,435$.
4. Bayes denominator: $122/6,435+328/6,435=450/6,435$.
5. Precision $=(122/6,435)/(450/6,435)=122/450=\boxed{0.271111=27.11\%}$.
6. Recall $=122/814=\boxed{0.149877=14.99\%}$; missed $=692/814=\boxed{85.01\%}$.

**November–December rule ($E=Q$):**

1. $P(Q\mid S)=333/814=0.409091$; $P(Q\mid S^c)=477/5,621=0.084860$.
2. Bayes numerator: $(333/814)(814/6,435)=333/6,435$.
3. Bayes denominator: $333/6,435+477/6,435=810/6,435$.
4. Precision $=333/810=\boxed{0.411111=41.11\%}$.
5. Recall $=333/814=\boxed{0.409091=40.91\%}$; missed $=481/814=\boxed{59.09\%}$.

| Rule | Flagged weeks | True strong flagged | Precision | Recall | Strong weeks missed |
|:--|--:|--:|--:|--:|--:|
| Holiday only | 450 | 122 | 27.11\% | 14.99\% | 85.01\% |
| November–December | 810 | 333 | 41.11\% | 40.91\% | 59.09\% |

**Conclusion:** the November–December rule has both higher precision and higher recall in this dataset; however, it flags more weeks for extra staffing.

### B6. What if strong weeks were twice as common or half as common?
Keep the conditionals $P(H\mid S)$ and $P(H\mid S^c)$ fixed; change only the hypothetical prior $p=P(S)$.

$$\mathrm{Precision}_{new}(p)=\frac{P(H\mid S)p}{P(H\mid S)p+P(H\mid S^c)(1-p)}.$$

$P(H\mid S)=122/814=0.149877$, and $P(H\mid S^c)=328/5,621=0.058353$.

**Twice as common:**
1. New base rate $p=2(814/6,435)=0.252991$.
2. Numerator $(122/814)(0.252991)=0.037917638$.
3. Denominator $0.037917638+(328/5,621)(1-0.252991)=0.081507534$.
4. Precision $=0.037917638/0.081507534=\boxed{0.465204=46.52\%}$.

**Half as common:**
1. New base rate $p=(814/6,435)/2=0.063248$.
2. Numerator $(122/814)(0.063248)=0.009479409$.
3. Denominator $0.009479409+(328/5,621)(1-0.063248)=0.064141338$.
4. Precision $=0.009479409/0.064141338=\boxed{0.147789=14.78\%}$.

**Interpretation:** the exact same holiday indicator becomes more precise when the underlying prevalence of strong weeks increases. These are hypothetical recalibrations, not new observations.

### B7. Test independence of strong weeks and six candidate features.
**Independence test:** $P(S\cap E)\stackrel{?}{=}P(S)P(E)$. If these differ, observing $E$ changes the chance of $S$.

| Feature | $n(S\cap E)$ | Observed $P(S\cap E)$ | Expected if independent $P(S)P(E)$ | Direction |
|:--|--:|--:|--:|:--|
| H | 122 | 0.018959 | 0.008846 | Positive |
| Q | 333 | 0.051748 | 0.015923 | Positive |
| C | 185 | 0.028749 | 0.019441 | Positive |
| F | 291 | 0.045221 | 0.059542 | Negative |
| J | 412 | 0.064025 | 0.059405 | Positive |
| G | 367 | 0.057032 | 0.061842 | Negative |

**Worked example ($H$):** $P(S\cap H)=122/6,435=0.018959$ whereas $P(S)P(H)=(814/6,435)(450/6,435)=0.008846$. These values are not equal, so the events are dependent.

**Conclusion:** none of the six observed pairs is exactly independent. Holidays ($H$) and especially November–December ($Q$) have substantial positive associations with strong weeks. Cold weather ($C$) is also positive; expensive fuel ($F$) is negative; unemployment ($J$) and store-size indicator ($G$) show smaller differences. Association does not prove causation.

***
## Part N — Naive Bayes classifier (time-based train/test split)

### N1. Split by year, avoid leakage, and report class prevalence.
**Procedure:** define training rows with `Date.year <= 2011`, test rows with `Date.year == 2012`; calculate store means from the training set; redefine $S$ and $G$ for both sets using those means.

1. Training observations: $n_{train}=4,500$ store-weeks; training strong: 574; $P_{train}(S)=574/4500=\boxed{0.127556=12.76\%}$.
2. Test observations: $n_{test}=1,935$ store-weeks; test strong: 255; $P_{test}(S)=255/1935=\boxed{0.131783=13.18\%}$.
3. Non-strong training rows $=4,500-574=3,926$; non-strong test rows $=1,935-255=1,680$.
4. **Leakage check:** the threshold for Store $i$ in both years is $1.10\times\operatorname{mean}(\mathrm{WeeklySales}_{i,\mathrm{2010-2011}})$; 2012 sales are not used to construct it.

### N2. Class priors and feature likelihoods learned by counting.
**Priors:** $P_{train}(S)=574/4,500=0.127556$, $P_{train}(S^c)=3,926/4,500=0.872444$. Check: they sum to 1.

For a binary feature $E$, its **unsmoothed** likelihood is $P(E=1\mid S)=n(E=1,S)/574$ and similarly for $S^c$. The feature-false likelihood is its complement.

| Feature | $n(1,S)$ | $P(1\mid S)$ | $P(0\mid S)$ | $n(1,S^c)$ | $P(1\mid S^c)$ | $P(0\mid S^c)$ |
|:--|--:|--:|--:|--:|--:|--:|
| H | 99 | 0.172474 | 0.827526 | 261 | 0.066480 | 0.933520 |
| Q | 335 | 0.583624 | 0.416376 | 475 | 0.120988 | 0.879012 |
| C | 165 | 0.287456 | 0.712544 | 599 | 0.152573 | 0.847427 |
| F | 108 | 0.188153 | 0.811847 | 1358 | 0.345899 | 0.654101 |
| J | 317 | 0.552265 | 0.447735 | 2085 | 0.531075 | 0.468925 |
| G | 266 | 0.463415 | 0.536585 | 1934 | 0.492613 | 0.507387 |

**Worked example (holiday):** 99 of the 574 training strong weeks are holidays, so $P(H=1\mid S)=99/574=0.172474$. Of the 3,926 training not-strong weeks, 261 are holidays, so $P(H=1\mid S^c)=261/3,926=0.066480$. For $H=0$, subtract each from 1.

**Laplace smoothing for N3 and N5:** `CategoricalNB(alpha=1)` adds one pseudo-count to each of two categories of a binary feature. For class count $n_c$, the smoothed likelihood is

$$\widetilde P(E=v\mid Y=c)=\frac{n(E=v,Y=c)+1}{n_c+2}.$$

| Feature | $\widetilde P(1\mid S)$ | $\widetilde P(0\mid S)$ | $\widetilde P(1\mid S^c)$ | $\widetilde P(0\mid S^c)$ |
|:--|--:|--:|--:|--:|
| H | 0.173611 | 0.826389 | 0.066701 | 0.933299 |
| Q | 0.583333 | 0.416667 | 0.121181 | 0.878819 |
| C | 0.288194 | 0.711806 | 0.152749 | 0.847251 |
| F | 0.189236 | 0.810764 | 0.345978 | 0.654022 |
| J | 0.552083 | 0.447917 | 0.531059 | 0.468941 |
| G | 0.463542 | 0.536458 | 0.492617 | 0.507383 |

**Smoothing check:** for each feature and each class, the probabilities of value 1 and value 0 add to 1. The class priors are **not** smoothed in this calculation; they retain their observed 574/4,500 and 3,926/4,500 values.

### N3. Two six-feature Naive Bayes predictions by hand.
**Model formula:** for profile $x=(h,q,c,f,j,g)$ and class $y$, compute

$$u_y(x)=P(y)\prod_{E\in\{H,Q,C,F,J,G\}} \widetilde P(E=x_E\mid y),\qquad P(S\mid x)=\frac{u_S(x)}{u_S(x)+u_{S^c}(x)}.$$

**Profile a:** holiday, Nov–Dec, cold, cheap fuel, low unemployment, large store. Feature values $(1,1,1,0,0,1)$.

Strong score:

$$\begin{aligned}u_{S} &= (574/4,500) (100/576) \cdot (336/576) \cdot (166/576) \\
&\quad \cdot (467/576) \cdot (258/576) \cdot (267/576) \\
&= 0.000626699995\end{aligned}$$

Not-strong score:

$$\begin{aligned}u_{S^c} &= (3926/4,500) (262/3928) \cdot (476/3928) \cdot (600/3928) \\
&\quad \cdot (2569/3928) \cdot (1842/3928) \cdot (1935/3928) \\
&= 0.000162743293\end{aligned}$$

**Normalize:** $P(S\mid x_{a})=0.000626699995/(0.000626699995+0.000162743293)=\boxed{0.793850558=79.39\%}$.

**Profile b:** nonholiday, outside Nov–Dec, not cold, expensive fuel, high unemployment, small store. Feature values $(0,0,0,1,1,0)$.

Strong score:

$$\begin{aligned}u_{S} &= (574/4,500) (476/576) \cdot (240/576) \cdot (410/576) \\
&\quad \cdot (109/576) \cdot (318/576) \cdot (309/576) \\
&= 0.001752179628\end{aligned}$$

Not-strong score:

$$\begin{aligned}u_{S^c} &= (3926/4,500) (3666/3928) \cdot (3452/3928) \cdot (3328/3928) \\
&\quad \cdot (1359/3928) \cdot (2086/3928) \cdot (1993/3928) \\
&= 0.056519268402\end{aligned}$$

**Normalize:** $P(S\mid x_{b})=0.001752179628/(0.001752179628+0.056519268402)=\boxed{0.030069265=3.01\%}$.

**Check:** every posterior is between 0 and 1, and $P(S\mid x)+P(S^c\mid x)=1$. The following scikit-learn calculation verifies these manual results.

### N4. Test the Naive Bayes conditional-independence assumption.
**Independence given $S$ would require:** $P(H\cap Q\mid S)=P(H\mid S)P(Q\mid S)$.

1. Actual conditional joint probability: $P(H\cap Q\mid S)=80/574=\boxed{0.139373}$.
2. Product: $P(H\mid S)P(Q\mid S)=(99/574)(335/574)=\boxed{0.100660}$.
3. Difference $=0.038713$; they are not equal, so $H$ and $Q$ **are not conditionally independent** in the training data.
4. **Explanation:** Thanksgiving and Christmas holiday weeks occur in November–December, so knowing that a week is in the holiday season changes the holiday-week likelihood. Naive Bayes nevertheless treats these features as conditionally independent.

### N5. Check the hand-calculated predictions with scikit-learn.
Train `CategoricalNB(alpha=1)` on the **six binary input columns** and the training strong-week label. `predict_proba` returns probabilities in class order `[not strong, strong]`.

| Profile | Hand $P(S\mid x)$ from N3 | `CategoricalNB` $P(S\mid x)$ | Difference |
|:--|--:|--:|--:|
| a | 0.793850558 | 0.793850558 | 6.66e-16 |
| b | 0.030069265 | 0.030069265 | 1.70e-16 |

**Conclusion:** the smoothed hand calculations and scikit-learn predictions agree to floating-point precision.

### N6. Compare Naive Bayes with the two test-year baselines.
**Confusion matrix orientation:** rows are actual `[not strong, strong]`; columns are predicted `[not strong, strong]`, so each matrix is $\begin{bmatrix}TN&FP\\FN&TP\end{bmatrix}$.

**Metric formulas:** $\mathrm{Accuracy}=(TN+TP)/N$; $\mathrm{Precision}=TP/(TP+FP)$; $\mathrm{Recall}=TP/(TP+FN)$.

| Rule | $TN$ | $FP$ | $FN$ | $TP$ | Accuracy | Precision | Recall |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Never strong | 1680 | 0 | 255 | 0 | 86.82\% | 0.00\% | 0.00\% |
| Holiday only | 1613 | 67 | 232 | 23 | 84.55\% | 25.56\% | 9.02\% |
| Naive Bayes (t = 0.5) | 1680 | 0 | 255 | 0 | 86.82\% | 0.00\% | 0.00\% |

**Worked metric calculations — holiday rule:**

1. Accuracy $=(1,613+23)/1,935=1,636/1,935=\boxed{0.845478=84.55\%}$.
2. Precision $=23/(23+67)=23/90=\boxed{0.255556=25.56\%}$.
3. Recall $=23/(23+232)=23/255=\boxed{0.090196=9.02\%}$.
4. **Never strong:** $(TN,FP,FN,TP)=(1,680,0,255,0)$, so accuracy $=1,680/1,935=86.82\%$ and recall $=0/255=0\%$.
5. **Naive Bayes at the 0.5 threshold:** the same confusion matrix as never strong. Its accuracy is **86.82\%**, but recall is **0\%**: it does not identify any of the 255 strong weeks.
6. **Precision convention:** when a method predicts *zero* positive weeks ($TP+FP=0$), precision is mathematically undefined. The table reports **0** following `precision_score(..., zero_division=0)`; this is a software reporting convention, not an observed success rate.

**Answer:** the standard-threshold Naive Bayes classifier has higher overall accuracy than the holiday rule (because most weeks are non-strong) but is **worse for finding strong weeks**: it has 0\% recall versus 9.02\%. The 2012 test period stops in October, so it omits Thanksgiving and Christmas; that limits evaluation of the strongest seasonal peak.

### N7. Stretch: change the probability threshold $t$.
**Decision rule:** predict a strong week when $P(S\mid x)>t$, for $t\in\{0.1,0.2,0.3,0.5\}$. All probabilities come from the same trained model.

| Threshold $t$ | $TN$ | $FP$ | $FN$ | $TP$ | Predicted strong | Precision | Recall |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.1 | 1574 | 106 | 236 | 19 | 125 | 15.20\% | 7.45\% |
| 0.2 | 1673 | 7 | 252 | 3 | 10 | 30.00\% | 1.18\% |
| 0.3 | 1678 | 2 | 255 | 0 | 2 | 0.00\% | 0.00\% |
| 0.5 | 1680 | 0 | 255 | 0 | 0 | 0.00\% | 0.00\% |

**Worked example at $t=0.1$:** there are 19 true positives and 106 false positives, so precision $=19/(19+106)=19/125=0.152=15.20\%$; recall $=19/(19+236)=19/255=0.074510=7.45\%$.

**Interpretation:** lowering the threshold to 0.1 makes the model flag 125 weeks and detect 19 strong weeks. At 0.5 it flags none. The threshold illustrates the precision–recall trade-off in these particular data; a low threshold does **not** guarantee performance better than the holiday baseline.

***
## Part J — Five-sentence stakeholder note (J1)

> Across all 6,435 store-weeks, 12.65\% were strong sales weeks. Thanksgiving was the strongest holiday (82.22\% strong), while 41.11\% of November–December store-weeks were strong. The holiday-only staffing rule was right 27.11\% of the time it flagged a week, but missed 85.01\% of all strong weeks. On the 2012 test set, Naive Bayes at the standard 0.5 cutoff did not beat the holiday rule at finding strong weeks: its recall was 0.00\% versus 9.02\% for the holiday rule. One risk is that the 2012 test data end in October, so the model cannot be tested on that year's Thanksgiving or Christmas periods.

**Numerical basis of the note:** 814/6,435 strong weeks; Thanksgiving 74/90; November–December 333/810; holiday-rule precision 122/450 and missed fraction 692/814; test holiday recall 23/255, while Naive Bayes at 0.5 detected 0/255; the 2012 series ends in October.

***
## Consolidated Section 7 answers

| Task | Final value |
|:--|:--|
| P1 | 6,435 rows; 45 stores; 143 weeks; 450 holiday store-weeks |
| P2 | $P(S)=0.126496$, $P(H)=0.069930$, $P(Q)=0.125874$, $P(C)=0.153691$, $P(F)=0.470707$, $P(J)=0.469619$, $P(G)=0.488889$ |
| P3 | $P(S\cap H)=0.018959$; $P(S\cap Q)=0.051748$; $P(H\cap Q)=0.027972$; $P(S\cap C)=0.028749$ |
| B1 | $P(S\mid H)=0.271111$; $P(S\mid H^c)=0.115622$ |
| B2 | $P(H\mid S)=0.149877$ by Bayes and by counting |
| B3 | Highest conditional strong rate: Thanksgiving; largest $P(g\mid S)$: no holiday |
| B4 | $P(S\mid Q)=0.411111$; $P(Q\mid S)=0.409091$ |
| B5 | Holiday: precision 0.271111, recall 0.149877; Nov–Dec: precision 0.411111, recall 0.409091 |
| B6 | Strong weeks doubled: 0.465204; halved: 0.147789 |
| N3 | $P(S\mid x_a)=0.793850558$; $P(S\mid x_b)=0.030069265$ |
| N6 | Never strong: accuracy 0.868217, precision 0, recall 0; holiday rule: accuracy 0.845478, precision 0.255556, recall 0.090196; Naive Bayes (0.5): accuracy 0.868217, precision 0, recall 0 |

**Verification notes:** All counts are computed from `Walmart.csv`; P and B use complete-period averages, while N uses training-only averages. Probabilities are checked using complementary counts, alternate formulas, or normalizing sums where applicable.