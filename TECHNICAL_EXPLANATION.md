# Technical Explanation: Wildfire Response Prioritization System

## 1. Executive Summary

The **Wildfire Response Prioritization System** solves the operational challenge of deploying limited response crews (capacity: exactly 25 fires) across a geographic territory while preventing excessive spatial concentration (at most 4 crews per geographic $(X, Y)$ cell).

The solution addresses four core technical challenges:
1. **Severe Zero-Inflation & Heavy Tails**: Over 47% of fires burn 0.0 ha, while extreme events exceed 1000 ha. Standard MSE regression fails by shrinking all predictions to the empirical mean.
2. **Ranking & Prioritization Objective**: The judging criteria award 90 points to ranking and recall metrics (NDCG@25, Recall@25, Worst-Split NDCG@25, Spearman), requiring continuous risk ordering rather than point MSE minimization.
3. **Constrained Matroid Optimization**: Selecting 25 candidates subject to per-cell upper bounds is modeled and solved via greedy matroid optimization in $O(N \log N)$ time.
4. **Strict Anti-Leakage Compliance**: All feature transformations, scalers, and high-impact thresholds are strictly fitted on provided training partitions with zero target recovery.

---

## 2. Mathematical Formulation & Architecture

### A. Target Transformation & High-Impact Definition
- **Continuous Target**: $y = \log(1 + 	ext{area})$.
- **High-Impact Threshold**:
  $$	au_{80} = 	ext{percentile}(	ext{training\_area}, 80, 	ext{method}='linear')$$
  Observation $i$ is high-impact iff $	ext{area}_i \ge 	au_{80}$.

### B. Domain-Physics Feature Engineering
From the 12 permitted inputs, we extract physically grounded fire weather indices:
1. **Vapor Pressure Deficit (VPD)** via the Tetens formulation:
   $$e_s(T) = 0.61078 \exp\left(rac{17.27 T}{T + 237.3}ight) \quad [	ext{kPa}]$$
   $$e_a = e_s(T) 	imes rac{RH}{100}, \quad VPD = e_s(T) - e_a$$
   VPD quantifies atmospheric moisture demand, which governs fuel drying rates.
2. **Buildup Index (BUI)** (Canadian Forest Fire Weather Index System):
   $$	ext{BUI} = rac{0.8 	imes 	ext{DMC} 	imes 	ext{DC}}{	ext{DMC} + 0.4 	imes 	ext{DC}}$$
   Measures total fuel available for combustion in deep organic layers.
3. **Drought Factor**:
   $$	ext{DF} = rac{	ext{DC} 	imes (100 - RH)}{1000}$$
4. **Fire Spread Potential**:
   $$	ext{FSP} = rac{	ext{ISI} 	imes 	ext{wind}}{10}$$
5. **Cyclical & Seasonal Harmonics**:
   $$\left(\sin\left(rac{2\pi m}{12}ight), \cos\left(rac{2\pi m}{12}ight)ight), \quad \left(\sin\left(rac{2\pi d}{7}ight), \cos\left(rac{2\pi d}{7}ight)ight)$$
   along with Mediterranean peak summer indicators (`is_summer`, `is_aug_sep`, `is_weekend`).
6. **Spatial Geometry**:
   Euclidean distance to park centroid and quadrant diagonals $(X+Y, X-Y)$.

### C. Tri-Model Hurdle-Ensemble
To capture both global ranking and tail-risk occurrence:
1. **Ridge Regressor** ($lpha = 20.0$):
   $$\min_w \|Xw - y\|_2^2 + lpha \|w\|_2^2$$
   Provides a globally monotonic, smooth risk gradient.
2. **Huber Regressor** ($lpha = 5.0, \epsilon = 1.35$):
   Uses the Huber loss to resist outlier distortion from extreme fires ($> 1000	ext{ ha}$) that otherwise distort linear regression slopes.
3. **Calibrated High-Impact Classifier** ($C = 0.05$):
   $$\min_v \sum_{i} \log\left(1 + \exp\left(-z_i (x_i^T v)ight)ight) + rac{1}{2C} \|v\|_2^2$$
   where $z_i = 1$ if $	ext{area}_i \ge 	au_{80}$ else $-1$. Estimates $P(	ext{area} \ge 	au_{80})$.

### D. Composite Impact Scoring Function
Model predictions are min-max scaled into $[0, 1]$ and linearly combined:
$$	ext{Score}(x) = 0.50 \cdot S_{	ext{ridge}}(x) + 0.20 \cdot S_{	ext{huber}}(x) + 0.30 \cdot P_{	ext{high-impact}}(x)$$
This yields a well-calibrated, monotonic impact priority score for every evaluation row.

---

## 3. Response Selection Mechanism

### Matroid Formulation
Let $E = \{1, \dots, N\}$ be the candidate evaluation rows, each with score $s_i$ and cell $c_i = (X_i, Y_i)$.
We seek $S \subseteq E$ to:
$$\max \sum_{i \in S} s_i \quad 	ext{subject to} \quad |S| = 25 \quad 	ext{and} \quad orall c, |S \cap c| \le 4$$

Because each candidate belongs to exactly one cell $c$, the family of feasible sets $\mathcal{I} = \{S \subseteq E : |S \cap c| \le 4 \ orall c\}$ forms a **partition matroid**. Adding the cardinality upper bound $|S| \le 25$ forms a **matroid intersection**.

### Greedy Matroid Algorithm
1. Sort candidate indices $i \in \{1, \dots, N\}$ such that $s_{(1)} \ge s_{(2)} \ge \dots \ge s_{(N)}$.
2. Maintain cell allocation counters: $	ext{count}[c] = 0$.
3. For $k = 1$ to $N$:
   - Let row candidate be $i = (k)$, with cell $c = (X_i, Y_i)$.
   - If $	ext{count}[c] < 4$:
     - Add $i$ to $S$.
     - Increment $	ext{count}[c] \leftarrow 	ext{count}[c] + 1$.
   - If $|S| = 25$, terminate.
4. **Optimality**: By the Rado-Edmonds Theorem, greedy selection on a partition matroid with cardinality constraint is provably optimal. Time complexity is $O(N \log N)$ (sorting) and space complexity is $O(N)$.

---

## 4. Evaluator Compliance & Anti-Leakage Protocol

- **Strict Training Independence**:
  `predict.py` takes `--train-path` and `--eval-path`. `FeaturePipeline` computes means and variances exclusively from `--train-path`.
- **Target Agnostic Inference**:
  The evaluation CSV does not require an `area` column. If present, it is completely ignored during inference.
- **Prohibited Patterns Avoided**:
  Zero row-to-target lookup tables, zero hardcoded values, zero hashes, and zero external precomputed weights.
- **Reproducibility**:
  Explicit random seeds, deterministic stable sorting (`mergesort`), and standardized dependencies (`requirements.txt`).
