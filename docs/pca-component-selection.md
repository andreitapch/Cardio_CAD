# Choosing the number of principal components for segmental strain traces

Methods notes and measured results for the dimensionality-reduction step, written
so the relevant parts can be lifted into a manuscript. Every number here was
measured on the project cohort; the scripts are described well enough to rerun.

Cohort for all measurements below: **480 studies x 18 myocardial segments = 8,640
traces**, GE and Philips pooled, baseline (BL) studies only, loaded from
`st_segmentalStrainInterp_T` and `phi_segmentalStrainInterp_T`. Each interpolated
trace has **128 samples**. Apical-cap segments are excluded upstream, and only
studies contributing all 18 segments are retained.

---

## 1. Temporal normalisation

Strain traces cannot be compared sample-by-sample unless the same index means the
same point of the cardiac cycle in every patient. Raw traces differ in both
duration (heart rate) and in the fraction of the cycle spent in systole, and the
latter is not fixed by simple resampling to a common length.

Two anchors are available for the speckle-tracking data: the start of the trace
(end-diastole, the strain reference) and end-systole (`esFrame`). Together with
the end of the trace these give three anchors, supporting **two-segment
piecewise-linear resampling**:

- systole, `[0, ES]`, resampled to a fixed number of samples;
- diastole, `[ES, end]`, resampled to the remaining samples.

Both targets are cohort-level constants (mean ES frame and mean curve length), so
every stored trace has the same length and end-systole falls at the same index in
every patient. Index *i* therefore denotes the same normalised cardiac phase
across the cohort, which is the condition column-wise methods such as PCA
require.

A single global stretch to a common length would equalise duration but not
phase: a patient with long systole would still have end-systole at a different
index. Anchoring at ES is what makes the columns comparable.

**Cost of the normalisation.** Absolute timing is discarded — cycle length,
`esFrame` and systolic fraction are no longer recoverable from the normalised
trace. If timing is clinically relevant (post-systolic shortening and altered
systolic duration are plausible remodeling markers), it should be carried
alongside the shape as explicit scalar features rather than recovered from the
components. The raw values remain available pre-interpolation.

### 1.1 Endpoint clamping and vendor harmonisation

The resampling pipeline sets the first and last sample of every trace to zero.
The first is definitional: strain is referenced to end-diastole, so it is zero at
`t = 0`. The last encodes an assumption — that the cycle closes — and was
verified rather than assumed.

Distribution of the **raw, pre-interpolation** endpoint value (9,097 traces):

| | GE (n = 5,580) | Philips (n = 3,517) |
|---|---|---|
| First sample, median \|v\| | 0.000 | 0.000 |
| Last sample, median \|v\| | **0.000** | **0.592%** |
| Last sample, p95 \|v\| | 0.000 | 2.358% |
| \|last\| > 0.5% | 0.1% of traces | 56.5% of traces |
| \|last\| > 1.0% | 0.1% | 29.2% |

The two vendors behave differently: GE's software already closes the curve, so
clamping is a no-op there, whereas Philips leaves over half its traces ending
away from zero.

Clamping is retained, for two reasons. First, the Philips residual is
**symmetric about zero** (median -0.013, p25 -0.610, p75 +0.569, mean -0.048),
which is the signature of tracking drift at the cycle boundary rather than of a
physiological effect — incomplete relaxation or ongoing post-systolic shortening
would bias consistently in one direction. Second, in a pooled GE + Philips
cohort, leaving it in place creates a deterministic vendor signature in a known
column, which a classifier can exploit as a scanner proxy. Clamping harmonises
the two sources.

The clamped samples become constant columns and are dropped automatically by the
classifier's uninformative-column filter.

---

## 2. Two representations, and why the choice matters

PCA can be applied either to the 18 segment traces concatenated into one vector
per study (2,304 features) or per segment (128 features, 18 separate bases). The
spectra differ sharply.

| | k=1 | k=3 | k=5 | components for 95% |
|---|---|---|---|---|
| **Per segment** (mean of 18) | 64.9% | **89.8%** | 96.0% | **4.9** |
| **Concatenated** (18 segments) | 15.6% | 28.3% | 37.7% | 59 |

The low cumulative variance of the concatenated representation is **not** a
data-quality problem, and was initially misread as one. Each individual segment
is close to 5-dimensional and remarkably consistent (95% reached at 4–5
components in all 18; three-component variance ranges only 87.5–91.2%). The flat
concatenated spectrum reflects genuine **inter-segment independence**: roughly
18 x 5 = 90 dimensions of real variation, which global PCA compresses to 59 for
95%. That 59 < 90 shows some shared structure across segments, but not much.
Three global components cannot represent 18 regions of largely independent
variation, and should not be expected to.

This is diagnostic rather than pathological — regional heterogeneity is the
clinical rationale for segmental strain in the first place — but it does argue
for the per-segment representation, whose components are attributable to a named
myocardial region and therefore interpretable (and explainable via SHAP at the
segment level).

---

## 3. Selecting the number of components

### 3.1 Reconstruction error does not answer the question

Held-out reconstruction R² was computed over **20 random 85/15 splits**, fitting
PCA on the training split and scoring reconstruction of the held-out studies
(R² = 1 - MSE / Var, variance taken about the training mean):

| Segment | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=10 | k=20 | k=40 |
|---|---|---|---|---|---|---|---|---|---|
| apLat | 0.728 | 0.855 | 0.915 | 0.948 | 0.965 | 0.975 | 0.994 | 1.000 | 1.000 |
| basInf | 0.574 | 0.783 | 0.865 | 0.920 | 0.946 | 0.962 | 0.989 | 0.999 | 1.000 |
| midSept | 0.592 | 0.811 | 0.881 | 0.936 | 0.956 | 0.970 | 0.993 | 1.000 | 1.000 |

The curve rises monotonically to essentially perfect reconstruction and **never
plateaus**. With 480 studies against 128 features no overfitting gap opens, so a
"maximise held-out R²" rule always answers *more components*, and a
one-standard-error rule simply returns whatever upper limit is imposed on the
search. Reconstruction fidelity is therefore not a usable selection criterion for
this data, and in-sample cumulative variance is weaker still — it can only
increase with k, and is measured on the data the basis was fitted to.

### 3.2 Parallel analysis

Horn's parallel analysis provides an explicit null model. Each feature column is
permuted independently, which destroys between-column correlation while
preserving each column's marginal distribution; the eigenvalues of the permuted
data then describe the dimensionality attributable to noise alone. Components
whose real eigenvalue exceeds the 95th percentile of the null eigenvalue
distribution are retained. Here: **30 permutations per segment**.

### 3.3 Result

| Criterion | Median across 18 segments | Range |
|---|---|---|
| **Parallel analysis (Horn's)** | **4** | 4–5 |
| 95% cumulative variance | 5 | 4–5 |
| Held-out R² plateau | none | — |

Two independent criteria agree on **4–5 components per segment**. Marginal
held-out gains corroborate: k=3→4 adds +0.03 to +0.06 R², k=4→5 adds +0.02,
k=5→6 adds +0.01.

**Recommended and defensible: 4 components per segment, by parallel analysis,
corroborated by the 95%-variance rule.**

---

## 4. "Optimal" depends on the objective

Component count for *reconstruction* and for *classification* are different
questions with different answers, and conflating them is a common error.

- **Reconstruction / representation.** Parallel analysis, as above: 4 per
  segment.
- **Classification.** PCA orders components by variance, not by association with
  the outcome. The discriminative direction is not guaranteed to be an early
  component, so the selection criterion must be downstream predictive
  performance — here pooled PR-AUC under nested cross-validation, read against
  the prevalence baseline, with confidence intervals. Candidate widths are
  k = 3, 4, 5 per segment, i.e. **54, 72 or 90 features** across 18 segments.
  With roughly 252 labelled exams and a minority outcome class, the smallest k
  that is statistically tied with the best is the right choice.

Reporting both, and stating which criterion governed which decision, is stronger
than presenting a single number as though it served both purposes.

---

## 5. Reproducing the measurements

All figures were produced against the project database via the viewer's own data
loading (`DimensionalityRedViewer.load_study_nr_dropdown`), so the filtering
matches what the application uses.

- Segment spectra: PCA per `curve_segment` on the 480 x 128 matrix.
- Held-out R²: 20 splits, 15% validation, PCA fitted on the training split only,
  components truncated to each k from a single fit.
- Parallel analysis: 30 column-wise permutations, 95th-percentile threshold.
- Endpoint distribution: raw `st_segmentalStrain_T` / `phi_segmentalStrain_T`
  before interpolation.

### Caveats

- Single cohort, single centre; no external validation set.
- 480 studies here are trace-level; the labelled classification cohort is
  smaller (~252 exams) after joining outcomes, so component counts justified on
  reconstruction grounds are not automatically appropriate as classifier inputs.
- Parallel analysis assumes the permutation null is meaningful for smooth,
  strongly autocorrelated time series. Adjacent samples of a strain curve are
  highly correlated, and permuting within a column removes only between-column
  structure. The close agreement with the 95% rule is reassuring, but the
  estimate should be read as approximate rather than exact.
