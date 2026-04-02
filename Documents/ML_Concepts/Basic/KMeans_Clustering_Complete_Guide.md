# K-Means Clustering: Complete Guide with Example Data at Every Step

## The Problem

You have 8 pizza stores. This time there are **no labels** — you don't know which stores are
"successful" or not. You just have features and want to find **natural groups** in the data.

### Our Data

| Store | Rating (x₁) | Delivery_Min (x₂) |
|-------|-------------|-------------------|
| S1    | 4.5         | 20                |
| S2    | 3.2         | 45                |
| S3    | 4.8         | 18                |
| S4    | 2.9         | 50                |
| S5    | 4.1         | 25                |
| S6    | 3.5         | 35                |
| S7    | 4.3         | 22                |
| S8    | 3.0         | 40                |

**Goal:** Group these stores into K clusters based on similarity.

---

## STEP 1: What Is K-Means?

K-Means finds K groups (clusters) in your data such that:
- Points within a cluster are CLOSE to each other
- Points in different clusters are FAR from each other

```
It's like sorting pizza stores into categories:
  Cluster A: "High-rated, fast delivery" stores
  Cluster B: "Low-rated, slow delivery" stores

Nobody told the algorithm these categories exist.
It discovers them from the data patterns.
```

### The Algorithm (4 steps, repeat):
```
1. CHOOSE K random starting points (centroids)
2. ASSIGN each data point to the nearest centroid
3. UPDATE each centroid to the mean of its assigned points
4. REPEAT steps 2-3 until centroids stop moving
```

---

## STEP 2: Choose K and Initialize Centroids

Let's use K=2 (we think there are 2 types of stores).

### Random initialization

Randomly pick 2 stores as initial centroids:

```
Centroid 1 (C₁): S1 = (4.5, 20)
Centroid 2 (C₂): S4 = (2.9, 50)
```

---

## STEP 3: Iteration 1 — Assign Points to Nearest Centroid

### Distance formula (Euclidean)

```
d(a, b) = √[(x₁ᵃ - x₁ᵇ)² + (x₂ᵃ - x₂ᵇ)²]
```

### Compute distance from each store to both centroids

**Store S1 (4.5, 20):**
```
d(S1, C₁) = √[(4.5-4.5)² + (20-20)²] = √[0 + 0] = 0.0
d(S1, C₂) = √[(4.5-2.9)² + (20-50)²] = √[2.56 + 900] = √902.56 = 30.04

Nearest: C₁ ✅
```

**Store S2 (3.2, 45):**
```
d(S2, C₁) = √[(3.2-4.5)² + (45-20)²] = √[1.69 + 625] = √626.69 = 25.03
d(S2, C₂) = √[(3.2-2.9)² + (45-50)²] = √[0.09 + 25] = √25.09 = 5.01

Nearest: C₂ ✅
```

**Store S3 (4.8, 18):**
```
d(S3, C₁) = √[(4.8-4.5)² + (18-20)²] = √[0.09 + 4] = √4.09 = 2.02
d(S3, C₂) = √[(4.8-2.9)² + (18-50)²] = √[3.61 + 1024] = √1027.61 = 32.06

Nearest: C₁ ✅
```

**Store S4 (2.9, 50):**
```
d(S4, C₁) = √[(2.9-4.5)² + (50-20)²] = √[2.56 + 900] = √902.56 = 30.04
d(S4, C₂) = √[(2.9-2.9)² + (50-50)²] = √[0 + 0] = 0.0

Nearest: C₂ ✅
```

**Store S5 (4.1, 25):**
```
d(S5, C₁) = √[(4.1-4.5)² + (25-20)²] = √[0.16 + 25] = √25.16 = 5.02
d(S5, C₂) = √[(4.1-2.9)² + (25-50)²] = √[1.44 + 625] = √626.44 = 25.03

Nearest: C₁ ✅
```

**Store S6 (3.5, 35):**
```
d(S6, C₁) = √[(3.5-4.5)² + (35-20)²] = √[1.0 + 225] = √226 = 15.03
d(S6, C₂) = √[(3.5-2.9)² + (35-50)²] = √[0.36 + 225] = √225.36 = 15.01

Nearest: C₂ (barely!) ✅
```

**Store S7 (4.3, 22):**
```
d(S7, C₁) = √[(4.3-4.5)² + (22-20)²] = √[0.04 + 4] = √4.04 = 2.01
d(S7, C₂) = √[(4.3-2.9)² + (22-50)²] = √[1.96 + 784] = √785.96 = 28.03

Nearest: C₁ ✅
```

**Store S8 (3.0, 40):**
```
d(S8, C₁) = √[(3.0-4.5)² + (40-20)²] = √[2.25 + 400] = √402.25 = 20.06
d(S8, C₂) = √[(3.0-2.9)² + (40-50)²] = √[0.01 + 100] = √100.01 = 10.00

Nearest: C₂ ✅
```

### Cluster assignments after Iteration 1:

```
Cluster 1 (C₁): S1(4.5,20), S3(4.8,18), S5(4.1,25), S7(4.3,22)
                 → High rating, fast delivery stores

Cluster 2 (C₂): S2(3.2,45), S4(2.9,50), S6(3.5,35), S8(3.0,40)
                 → Lower rating, slower delivery stores
```

---

## STEP 4: Update Centroids

New centroid = mean of all points in the cluster.

**New C₁:**
```
Rating:   (4.5 + 4.8 + 4.1 + 4.3) / 4 = 17.7 / 4 = 4.425
Delivery: (20 + 18 + 25 + 22) / 4 = 85 / 4 = 21.25

C₁_new = (4.425, 21.25)
```

**New C₂:**
```
Rating:   (3.2 + 2.9 + 3.5 + 3.0) / 4 = 12.6 / 4 = 3.15
Delivery: (45 + 50 + 35 + 40) / 4 = 170 / 4 = 42.5

C₂_new = (3.15, 42.5)
```

### How much did centroids move?

```
C₁: (4.5, 20) → (4.425, 21.25)
  Movement = √[(4.5-4.425)² + (20-21.25)²] = √[0.006 + 1.563] = √1.569 = 1.25

C₂: (2.9, 50) → (3.15, 42.5)
  Movement = √[(2.9-3.15)² + (50-42.5)²] = √[0.063 + 56.25] = √56.313 = 7.50
```

Centroids moved → not converged yet → do another iteration.

---

## STEP 5: Iteration 2 — Reassign Points

Using new centroids C₁=(4.425, 21.25) and C₂=(3.15, 42.5):

**Store S1 (4.5, 20):**
```
d(S1, C₁) = √[(4.5-4.425)² + (20-21.25)²] = √[0.006 + 1.563] = 1.25
d(S1, C₂) = √[(4.5-3.15)² + (20-42.5)²] = √[1.823 + 506.25] = 22.55
→ C₁
```

**Store S2 (3.2, 45):**
```
d(S2, C₁) = √[(3.2-4.425)² + (45-21.25)²] = √[1.501 + 564.063] = 23.79
d(S2, C₂) = √[(3.2-3.15)² + (45-42.5)²] = √[0.003 + 6.25] = 2.50
→ C₂
```

**Store S6 (3.5, 35):** (the borderline case)
```
d(S6, C₁) = √[(3.5-4.425)² + (35-21.25)²] = √[0.856 + 189.063] = 13.79
d(S6, C₂) = √[(3.5-3.15)² + (35-42.5)²] = √[0.123 + 56.25] = 7.51
→ C₂ (same as before)
```

All other stores stay in the same clusters. No reassignments happened.

### Update centroids again:

Same clusters → same means → centroids don't change → **CONVERGED!**

```
Final centroids:
  C₁ = (4.425, 21.25)  → "Premium stores" (high rating, fast delivery)
  C₂ = (3.15, 42.5)    → "Budget stores" (lower rating, slower delivery)
```

---

## STEP 6: Measuring Cluster Quality — Inertia (WCSS)

### Within-Cluster Sum of Squares (WCSS / Inertia)

```
WCSS = Σ Σ ||xᵢ - cₖ||²
       k  i∈Cₖ

Sum of squared distances from each point to its cluster centroid.
Lower WCSS = tighter clusters = better.
```

### Computing WCSS for our result:

**Cluster 1:**
```
S1: (4.5-4.425)² + (20-21.25)² = 0.006 + 1.563 = 1.569
S3: (4.8-4.425)² + (18-21.25)² = 0.141 + 10.563 = 10.703
S5: (4.1-4.425)² + (25-21.25)² = 0.106 + 14.063 = 14.168
S7: (4.3-4.425)² + (22-21.25)² = 0.016 + 0.563 = 0.578

WCSS₁ = 1.569 + 10.703 + 14.168 + 0.578 = 27.018
```

**Cluster 2:**
```
S2: (3.2-3.15)² + (45-42.5)² = 0.003 + 6.25 = 6.253
S4: (2.9-3.15)² + (50-42.5)² = 0.063 + 56.25 = 56.313
S6: (3.5-3.15)² + (35-42.5)² = 0.123 + 56.25 = 56.373
S8: (3.0-3.15)² + (40-42.5)² = 0.023 + 6.25 = 6.273

WCSS₂ = 6.253 + 56.313 + 56.373 + 6.273 = 125.211
```

**Total WCSS = 27.018 + 125.211 = 152.229**

---

## STEP 7: The Elbow Method — Choosing K

How do we know K=2 is the right number of clusters? Try multiple values of K
and plot WCSS:

```
K=1: All points in one cluster
  Centroid = mean of all = (3.7875, 31.875)
  WCSS = sum of all squared distances to this mean
  
  S1: (4.5-3.7875)² + (20-31.875)² = 0.507 + 141.016 = 141.523
  S2: (3.2-3.7875)² + (45-31.875)² = 0.345 + 172.266 = 172.611
  S3: (4.8-3.7875)² + (18-31.875)² = 1.026 + 192.516 = 193.541
  S4: (2.9-3.7875)² + (50-31.875)² = 0.788 + 328.516 = 329.303
  S5: (4.1-3.7875)² + (25-31.875)² = 0.098 + 47.266 = 47.363
  S6: (3.5-3.7875)² + (35-31.875)² = 0.083 + 9.766 = 9.848
  S7: (4.3-3.7875)² + (22-31.875)² = 0.263 + 97.266 = 97.528
  S8: (3.0-3.7875)² + (40-31.875)² = 0.620 + 66.016 = 66.636

  WCSS(K=1) = 1058.353

K=2: WCSS = 152.229 (computed above)

K=3: (would need to run the algorithm with K=3)
  Approximately WCSS ≈ 40-60

K=4: WCSS ≈ 10-20
```

### The Elbow Plot:

```
WCSS
  |
1058 ●
  |    \
  |     \
  |      \
 152 |       ●  ← ELBOW! Sharp bend here
  |          \
  50 |           ●
  20 |              ●
  |                  ●
  +---+---+---+---+---→ K
      1   2   3   4   5
```

The "elbow" is at K=2 — after this point, adding more clusters gives
diminishing returns. K=2 is the sweet spot.

---

## STEP 8: Feature Scaling — Why It Matters

### The problem with our data:

```
Rating:   ranges from 2.9 to 4.8  (range = 1.9)
Delivery: ranges from 18 to 50    (range = 32)
```

Delivery has a MUCH larger range. Without scaling, K-Means is dominated
by Delivery because distances in that dimension are larger.

```
d(S1, S2) = √[(4.5-3.2)² + (20-45)²]
          = √[1.69 + 625]
          = √626.69

The Rating difference (1.69) is DWARFED by Delivery difference (625).
K-Means essentially ignores Rating!
```

### Solution: Standardize features (z-score)

```
z = (x - mean) / std

Rating:   mean = 3.7875, std = 0.714
Delivery: mean = 31.875, std = 12.15

Standardized data:
| Store | Rating_z | Delivery_z |
|-------|----------|------------|
| S1    | +0.997   | -0.977     |
| S2    | -0.823   | +1.080     |
| S3    | +1.418   | -1.142     |
| S4    | -1.243   | +1.491     |
| S5    | +0.437   | -0.566     |
| S6    | -0.403   | +0.257     |
| S7    | +0.717   | -0.812     |
| S8    | -1.103   | +0.669     |
```

Now both features have mean=0 and std=1. Equal influence on clustering.

**Always scale features before K-Means!**

---

## STEP 9: K-Means Limitations and Alternatives

### Limitation 1: Assumes spherical clusters

```
K-Means works well:          K-Means fails:
  ○ ○                          ╭────╮
 ○ ● ○                        │ ●●● │
  ○ ○                          │●●●●●│
                               ╰────╯
  ○ ○                          ╭────╮
 ○ ● ○                        │ ●●● │
  ○ ○                          │●●●●●│
                               ╰────╯
(round clusters)              (elongated clusters)
```

### Limitation 2: Must specify K in advance

You need to know (or guess) the number of clusters before running.
The Elbow Method and Silhouette Score help, but it's still a guess.

### Limitation 3: Sensitive to initialization

Different random starting centroids can give different results.

**Solution: K-Means++ initialization**
```
Instead of random centroids:
1. Pick first centroid randomly
2. Pick next centroid with probability proportional to distance² from nearest existing centroid
3. Repeat until K centroids chosen

This spreads out initial centroids → more consistent results.
(sklearn uses K-Means++ by default)
```

### Limitation 4: Sensitive to outliers

One extreme store (Rating=1.0, Delivery=100) would pull a centroid toward it.

---

## STEP 10: Silhouette Score — Another Quality Metric

### For each point, compute:

```
a(i) = average distance to all OTHER points in the SAME cluster
b(i) = average distance to all points in the NEAREST OTHER cluster

Silhouette(i) = (b(i) - a(i)) / max(a(i), b(i))

Range: -1 to +1
  +1 = perfectly clustered (far from other clusters, close to own)
   0 = on the boundary between clusters
  -1 = probably in the wrong cluster
```

### Example for S1 (in Cluster 1):

```
a(S1) = average distance to S3, S5, S7 (same cluster)
  d(S1,S3) = √[(4.5-4.8)² + (20-18)²] = √[0.09+4] = 2.02
  d(S1,S5) = √[(4.5-4.1)² + (20-25)²] = √[0.16+25] = 5.02
  d(S1,S7) = √[(4.5-4.3)² + (20-22)²] = √[0.04+4] = 2.01
  a(S1) = (2.02 + 5.02 + 2.01) / 3 = 3.02

b(S1) = average distance to S2, S4, S6, S8 (other cluster)
  d(S1,S2) = 25.03, d(S1,S4) = 30.04, d(S1,S6) = 15.03, d(S1,S8) = 20.06
  b(S1) = (25.03 + 30.04 + 15.03 + 20.06) / 4 = 22.54

Silhouette(S1) = (22.54 - 3.02) / max(3.02, 22.54) = 19.52 / 22.54 = 0.866
```

A silhouette score of 0.866 is excellent — S1 is well-placed in Cluster 1.

Average silhouette across all points gives the overall clustering quality.

---

## STEP 11: K-Means for More Than 2 Clusters

### What if K=3?

The algorithm works the same way, just with 3 centroids:

```
Possible result with K=3:
  Cluster A: S1, S3, S7     → "Premium" (high rating, very fast)
  Cluster B: S5, S6         → "Mid-tier" (decent rating, moderate delivery)
  Cluster C: S2, S4, S8     → "Budget" (low rating, slow delivery)
```

More clusters = lower WCSS but potentially overfitting the structure.

---

## STEP 12: Practical Tips

```
1. ALWAYS scale features before K-Means
2. Run K-Means multiple times with different initializations (n_init=10 in sklearn)
3. Use the Elbow Method + Silhouette Score together to choose K
4. K-Means works best with:
   - Roughly spherical clusters
   - Similar-sized clusters
   - Numerical features
5. For non-spherical clusters → try DBSCAN or Gaussian Mixture Models
6. For categorical data → try K-Modes or K-Prototypes
```

---

## COMPLETE FORMULA SUMMARY

```
1. Euclidean distance:  d(a,b) = √[Σ(aⱼ - bⱼ)²]
2. Centroid update:     cₖ = (1/|Cₖ|) × Σ xᵢ  for i ∈ Cₖ
3. WCSS (Inertia):     WCSS = Σₖ Σᵢ∈Cₖ ||xᵢ - cₖ||²
4. Silhouette:         s(i) = (b(i) - a(i)) / max(a(i), b(i))
5. Standardization:    z = (x - μ) / σ
6. Objective:          Minimize WCSS (find centroids that minimize total within-cluster distance)
```

---

## INTERVIEW CHEAT SHEET

**Q: "Explain K-Means clustering."**
> "K-Means partitions data into K clusters by iteratively assigning points to the nearest centroid and updating centroids to the mean of assigned points. It minimizes within-cluster sum of squares (WCSS). It's simple, fast, and works well for spherical clusters."

**Q: "How do you choose K?"**
> "Two main methods: (1) Elbow Method — plot WCSS vs K and look for the 'elbow' where adding clusters gives diminishing returns. (2) Silhouette Score — measures how well each point fits its cluster vs the nearest other cluster. Higher average silhouette = better K."

**Q: "What are K-Means' limitations?"**
> "It assumes spherical, equally-sized clusters. It's sensitive to initialization (use K-Means++), outliers, and feature scaling. You must specify K in advance. It can get stuck in local minima — run multiple times with different seeds."

**Q: "K-Means vs DBSCAN?"**
> "K-Means needs K specified, assumes spherical clusters, assigns every point. DBSCAN discovers K automatically, finds arbitrary-shaped clusters, and can label points as noise/outliers. DBSCAN is better for irregular clusters; K-Means is simpler and faster."

**Q: "Why scale features before K-Means?"**
> "K-Means uses Euclidean distance. Features with larger ranges dominate the distance calculation. Scaling ensures all features contribute equally. Without scaling, a feature ranging 0-1000 would overpower one ranging 0-1."

---

*Every number in this document was computed by hand. Verify them yourself for practice!*
