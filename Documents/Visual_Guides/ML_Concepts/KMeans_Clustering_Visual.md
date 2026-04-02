# K-Means Clustering: Visual Guide with Mermaid Diagrams

> Visual companion to `Documents/KMeans_Clustering_Complete_Guide.md`.
> Every diagram has explanatory text — what it shows, why it matters, and how to read it.

---

## 1. What Is K-Means?

K-Means is an unsupervised algorithm — there are no labels. It discovers natural groups in data by iteratively assigning points to the nearest centroid and updating centroids to the mean of their assigned points. The diagram shows the core loop.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    START["📊 Unlabeled data<br/>8 pizza stores<br/>Only features, NO labels"] --> CHOOSE["1. Choose K=2<br/>Pick 2 random centroids"]
    CHOOSE --> ASSIGN["2. ASSIGN each store<br/>to nearest centroid"]
    ASSIGN --> UPDATE["3. UPDATE centroids<br/>= mean of assigned points"]
    UPDATE --> CHECK{"Centroids<br/>moved?"}
    CHECK -->|"Yes"| ASSIGN
    CHECK -->|"No"| DONE["✅ Converged!<br/>Final clusters found"]

    style START fill:#252840,stroke:#f5b731,color:#c8cfe0
    style CHOOSE fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style ASSIGN fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
    style UPDATE fill:#1a1d2e,stroke:#22d3a7,color:#e2e8f0
    style DONE fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

The loop between ASSIGN and UPDATE is the heart of K-Means. It typically converges in 5-20 iterations. The key difference from supervised learning: we never use labels — the algorithm finds structure purely from feature patterns.

---

## 2. Our Data — No Labels This Time

Unlike the supervised learning guides, we only have features. No "Successful" column. The scatter plot shows the stores in feature space — the two natural clusters are visible to us, but the algorithm has to discover them.

```
  Delivery
  (min)
     55 ┤
        │  ╔══════════════════╗
     50 ┤  ║  S4 (2.9, 50)    ║
     45 ┤  ║  S2 (3.2, 45)    ║    Cluster ?
     40 ┤  ║  S8 (3.0, 40)    ║    (low rating, slow)
     35 ┤  ║  S6 (3.5, 35)    ║
        │  ╚══════════════════╝
     30 ┤
        │                  ╔══════════════════╗
     25 ┤                  ║  S5 (4.1, 25)    ║
     22 ┤                  ║  S7 (4.3, 22)    ║    Cluster ?
     20 ┤                  ║  S1 (4.5, 20)    ║    (high rating, fast)
     18 ┤                  ║  S3 (4.8, 18)    ║
        │                  ╚══════════════════╝
        └──┬─────┬─────┬─────┬─────┬─────┬──→ Rating
          2.5   3.0   3.5   4.0   4.5   5.0

  We can SEE two groups. K-Means must FIND them using only distances.
```

---

## 3. Initialization — Pick Random Centroids

We randomly pick S1 (4.5, 20) and S4 (2.9, 50) as initial centroids. Why these two? They're just random choices — in practice, K-Means picks K random data points as starting centroids. Different random picks can lead to different final clusters, which is why we run K-Means multiple times (sklearn uses n_init=10 by default) and keep the best result. K-Means++ is a smarter initialization that spreads centroids apart, reducing the chance of a bad start. The diagram shows the starting positions. These are just starting guesses — the algorithm will move them to the true cluster centers.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    subgraph INIT["Step 1: Random Initialization (K=2)"]
        C1["⭐ Centroid 1<br/>= S1 position<br/>(4.5, 20)"]
        C2["⭐ Centroid 2<br/>= S4 position<br/>(2.9, 50)"]
    end

    INIT --> NOTE["These are just starting guesses.<br/>Different starting points can give different results.<br/>KMeans plus plus picks smarter starting points."]

    style INIT fill:#0e1117,stroke:#f5b731,color:#e2e8f0
    style C1 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style C2 fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style NOTE fill:#252840,stroke:#f5b731,color:#c8cfe0
```

Green centroid = will attract the "good" stores. Red centroid = will attract the "struggling" stores. But the algorithm doesn't know this yet — it just picked two random points.

---

## 4. Iteration 1 — Assign to Nearest Centroid

For each store, we compute the Euclidean distance to both centroids and assign it to the closer one. The diagram shows the assignment results — all high-rating stores go to C1, all low-rating stores go to C2.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    subgraph CLUSTER1["Cluster 1 (nearest to C1)"]
        S1C["S1 (4.5, 20)<br/>d to C1: 0.0<br/>d to C2: 30.04"]
        S3C["S3 (4.8, 18)<br/>d to C1: 2.02<br/>d to C2: 32.06"]
        S5C["S5 (4.1, 25)<br/>d to C1: 5.02<br/>d to C2: 25.03"]
        S7C["S7 (4.3, 22)<br/>d to C1: 2.01<br/>d to C2: 28.03"]
    end

    subgraph CLUSTER2["Cluster 2 (nearest to C2)"]
        S2C["S2 (3.2, 45)<br/>d to C1: 25.03<br/>d to C2: 5.01"]
        S4C["S4 (2.9, 50)<br/>d to C1: 30.04<br/>d to C2: 0.0"]
        S6C["S6 (3.5, 35)<br/>d to C1: 15.03<br/>d to C2: 15.01 ← barely!"]
        S8C["S8 (3.0, 40)<br/>d to C1: 20.06<br/>d to C2: 10.00"]
    end

    style CLUSTER1 fill:#0e1117,stroke:#22d3a7,color:#e2e8f0
    style CLUSTER2 fill:#0e1117,stroke:#f45d6d,color:#e2e8f0
```

Notice S6 — it's almost equidistant from both centroids (15.03 vs 15.01). It barely goes to Cluster 2. Borderline points like this can flip between clusters depending on initialization. The distance values show exactly why each store was assigned where it was.

---

## 5. Update Centroids

New centroid = mean of all points in the cluster. The diagram shows the calculation and how much each centroid moved from its starting position.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    subgraph BEFORE["Before (Initial)"]
        BC1["C1 = (4.5, 20)<br/><i>= S1's position</i>"]
        BC2["C2 = (2.9, 50)<br/><i>= S4's position</i>"]
    end

    BEFORE -->|"Compute mean<br/>of each cluster"| AFTER

    subgraph AFTER["After (Updated)"]
        AC1["C1 = (<b>4.425, 21.25</b>)<br/>mean of S1,S3,S5,S7<br/><i>Moved 1.25 units</i>"]
        AC2["C2 = (<b>3.15, 42.5</b>)<br/>mean of S2,S4,S6,S8<br/><i>Moved 7.50 units</i>"]
    end

    AFTER --> CHECK["Centroids moved → NOT converged<br/>→ Do another iteration"]

    style BEFORE fill:#0e1117,stroke:#5eaeff,color:#e2e8f0
    style AFTER fill:#0e1117,stroke:#22d3a7,color:#e2e8f0
    style CHECK fill:#252840,stroke:#f5b731,color:#c8cfe0
```

C2 moved much more (7.50 units) because it started at S4's extreme position and got pulled toward the center of its cluster. C1 barely moved (1.25 units) because S1 was already close to the cluster center. When centroids stop moving, the algorithm has converged.

---

## 6. Iteration 2 — Convergence

We reassign all stores using the updated centroids. Nobody changes clusters → centroids don't move → converged! The diagram shows the final stable state.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    REASSIGN["Reassign all stores<br/>with new centroids"] --> RESULT["No stores changed clusters!"]
    RESULT --> CONVERGED["Centroids unchanged → <b>CONVERGED</b>"]

    subgraph FINAL["Final Clusters"]
        FC1["✅ Cluster 1: 'Premium Stores'<br/>S1, S3, S5, S7<br/>Centroid: (4.425, 21.25)<br/><i>High rating, fast delivery</i>"]
        FC2["❌ Cluster 2: 'Budget Stores'<br/>S2, S4, S6, S8<br/>Centroid: (3.15, 42.5)<br/><i>Lower rating, slower delivery</i>"]
    end

    CONVERGED --> FINAL

    style REASSIGN fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style CONVERGED fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style FC1 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style FC2 fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
```

The algorithm discovered the same two groups we could see visually — without any labels. The cluster names ("Premium" and "Budget") are our interpretation; the algorithm only knows them as Cluster 1 and Cluster 2.

---

## 7. The Elbow Method — Choosing K

**Why we need the Elbow Method:** K-Means requires you to specify K (number of clusters) upfront, but you usually don't know the right K. WCSS (Within-Cluster Sum of Squares) measures how tight the clusters are — lower WCSS means points are closer to their centroids, which sounds like "lower = better." But WCSS *always* decreases as K increases — at the extreme, K=N (every point is its own cluster) gives WCSS=0, which is useless. The Elbow Method looks for the K where adding more clusters stops giving meaningful improvement — the "elbow" in the WCSS-vs-K curve. It's a tradeoff between cluster tightness and model simplicity: you want the fewest clusters that still capture the real structure in your data.

How do we know K=2 is right? Run K-Means for K=1,2,3,4,5 and plot the WCSS (within-cluster sum of squares). Look for the "elbow" — the point where adding more clusters gives diminishing returns.

```
  WCSS
  1058 ●
       │╲
       │ ╲
       │  ╲
       │   ╲
   152 │    ● ← ELBOW (K=2)
       │     ╲
    50 │      ●
    20 │       ●
    10 │        ●
       └──┬──┬──┬──┬──┬──→ K
          1  2  3  4  5

  K=1: Everything in one cluster (WCSS=1058, terrible)
  K=2: Two clusters (WCSS=152, big improvement!)
  K=3: Three clusters (WCSS≈50, diminishing returns)
  K=4+: Marginal improvement, risk overfitting
```

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph LR
    K1["K=1<br/>WCSS=1058<br/><i>One big blob</i>"] -->|"Huge drop"| K2["K=2<br/>WCSS=152<br/><i>Two clear groups</i><br/>🏆 ELBOW"]
    K2 -->|"Smaller drop"| K3["K=3<br/>WCSS≈50<br/><i>Diminishing returns</i>"]
    K3 -->|"Tiny drop"| K4["K=4 or more<br/><i>Overfitting structure</i>"]

    style K1 fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style K2 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style K3 fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style K4 fill:#252840,stroke:#f5b731,color:#c8cfe0
```

The biggest drop happens from K=1 to K=2 — that's where the real structure is. After K=2, improvements are marginal. The elbow tells us: 2 clusters capture the main pattern; more would be splitting hairs.

---

## 8. Feature Scaling — Why It's Critical

K-Means uses Euclidean distance, which is dominated by features with larger ranges. Without scaling, Delivery (range 18-50) overpowers Rating (range 2.9-4.8). The diagram shows the problem and the fix.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    subgraph PROBLEM["❌ Without Scaling"]
        P1["Rating difference: (4.5-3.2)² = <b>1.69</b>"]
        P2["Delivery difference: (20-45)² = <b>625</b>"]
        P3["Total distance dominated by Delivery!<br/>Rating is essentially ignored"]
    end

    subgraph SOLUTION["✅ With Standardization"]
        S1["z = (x - mean) / std"]
        S2["Rating_z: range ≈ -1.2 to 1.4"]
        S3["Delivery_z: range ≈ -1.1 to 1.5"]
        S4["Both features contribute equally!"]
    end

    PROBLEM -->|"Apply z-score<br/>standardization"| SOLUTION

    style PROBLEM fill:#0e1117,stroke:#f45d6d,color:#e2e8f0
    style SOLUTION fill:#0e1117,stroke:#22d3a7,color:#e2e8f0
```

The numbers tell the story: 625 vs 1.69 — Delivery is 370× more influential than Rating. After standardization, both features have similar ranges and contribute equally to distance calculations. Always scale before K-Means.

---

## 9. Silhouette Score — Measuring Cluster Quality

**Why we need the Silhouette Score:** The Elbow Method only looks at within-cluster distance (how tight each cluster is). But a good clustering should also have well-separated clusters — points should be much closer to their own cluster than to any other cluster. The Silhouette Score captures both dimensions: it asks "is this point closer to its own cluster or to the nearest other cluster?" A point with silhouette near +1 is well-clustered (tight within its cluster and far from others). Near 0 means it sits on the boundary between two clusters. Near -1 means it's probably assigned to the wrong cluster entirely. Unlike WCSS (which gives one global number), the Silhouette Score gives a per-point quality measure — you can identify exactly which points are well-placed and which are problematic.

For each point, the silhouette score measures how well it fits its own cluster vs the nearest other cluster. The diagram explains the formula and what the values mean.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    FORMULA["Silhouette(i) = (b - a) / max(a, b)"]

    subgraph COMPONENTS["For each point i:"]
        A["a(i) = avg distance to<br/>points in SAME cluster<br/><i>How tight is my cluster?</i>"]
        B["b(i) = avg distance to<br/>points in NEAREST OTHER cluster<br/><i>How far is the next cluster?</i>"]
    end

    FORMULA --> COMPONENTS

    subgraph INTERPRET["Interpretation"]
        GOOD["Score near 1: Perfectly clustered<br/><i>Far from others, close to own</i>"]
        MID["0: On the boundary<br/><i>Between two clusters</i>"]
        BAD["Score near −1: Wrong cluster<br/><i>Closer to another cluster</i>"]
    end

    COMPONENTS --> INTERPRET

    style FORMULA fill:#252840,stroke:#f5b731,color:#c8cfe0
    style GOOD fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style MID fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style BAD fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
```

For our S1: a=3.02 (close to cluster mates), b=22.54 (far from other cluster), silhouette=0.866 (excellent). Average silhouette across all points gives the overall clustering quality — use it alongside the Elbow Method to choose K.

---

## 10. K-Means Limitations

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    subgraph LIMITS["⚠️ K Means Limitations"]
        L1["Must specify K in advance<br/><i>Use Elbow and Silhouette to guess</i>"]
        L2["Assumes spherical clusters<br/><i>Fails on elongated/irregular shapes</i>"]
        L3["Sensitive to initialization<br/><i>Use KMeans plus plus (sklearn default)</i>"]
        L4["Sensitive to outliers<br/><i>One extreme point pulls centroid</i>"]
        L5["Only numerical features<br/><i>Use K-Modes for categorical</i>"]
    end

    subgraph ALTERNATIVES["🔄 Alternatives"]
        A1["DBSCAN: finds K automatically<br/>handles arbitrary shapes"]
        A2["Gaussian Mixture: soft assignments<br/>elliptical clusters"]
        A3["Hierarchical: no K needed<br/>produces dendrogram"]
    end

    LIMITS --> ALTERNATIVES

    style LIMITS fill:#0e1117,stroke:#f45d6d,color:#e2e8f0
    style ALTERNATIVES fill:#0e1117,stroke:#22d3a7,color:#e2e8f0
```

Red = limitations to be aware of. Green = alternatives that address specific limitations. K-Means is the go-to first choice for clustering, but knowing when to switch to DBSCAN or GMM is important.

---

## 11. Complete Algorithm Flowchart

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
flowchart TD
    START(["Input: Data X, K clusters"]) --> SCALE["Scale features<br/>(standardize)"]
    SCALE --> INIT["Initialize K centroids<br/>(KMeans plus plus or random)"]
    INIT --> ASSIGN["Assign each point<br/>to nearest centroid<br/>(Euclidean distance)"]
    ASSIGN --> UPDATE["Update centroids<br/>= mean of assigned points"]
    UPDATE --> CHECK{"Centroids<br/>changed?"}
    CHECK -->|"Yes"| ASSIGN
    CHECK -->|"No"| EVAL["Evaluate:<br/>WCSS, Silhouette Score"]
    EVAL --> OUTPUT(["Output: K clusters<br/>and centroid positions"])

    style START fill:#252840,stroke:#f5b731,color:#c8cfe0
    style SCALE fill:#1a1d2e,stroke:#f45d6d,color:#e2e8f0
    style INIT fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style ASSIGN fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
    style UPDATE fill:#1a1d2e,stroke:#22d3a7,color:#e2e8f0
    style OUTPUT fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

The red "Scale" step at the top is critical and often forgotten. The purple-green loop (assign → update) is the core algorithm. The evaluation step at the end tells you if K was a good choice.

---

## 12. Interview Decision Tree 🎯

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    Q1{"Explain K-Means?"} -->|Answer| A1["Partition data into K clusters<br/>Iteratively: assign to nearest centroid<br/>then update centroids to cluster means<br/>Minimizes within-cluster sum of squares"]
    Q1 -->|Next Q| Q2{"How to choose K?"}
    Q2 -->|Answer| A2["Elbow Method: plot WCSS vs K<br/>Silhouette Score: measures cluster quality<br/>Use both together"]
    Q2 -->|Next Q| Q3{"Limitations?"}
    Q3 -->|Answer| A3["Must specify K, assumes spherical clusters<br/>Sensitive to init (use KMeans plus plus)<br/>Sensitive to outliers and scaling"]
    Q3 -->|Next Q| Q4{"K-Means vs DBSCAN?"}
    Q4 -->|Answer| A4["K-Means: need K, spherical, assigns all points<br/>DBSCAN: finds K, arbitrary shapes, labels outliers<br/>DBSCAN better for irregular clusters"]

    style Q1 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style Q2 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style Q3 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style Q4 fill:#252840,stroke:#f5b731,color:#c8cfe0
    style A1 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style A2 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style A3 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style A4 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

---

> 💡 **How to view:** GitHub (native), VS Code (Mermaid extension), Obsidian (built-in), or [mermaid.live](https://mermaid.live)
