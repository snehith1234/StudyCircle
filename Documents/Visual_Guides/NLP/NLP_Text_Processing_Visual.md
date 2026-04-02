# NLP Text Processing: Visual Guide with Mermaid Diagrams

> Visual companion to `Documents/NLP/Basic/Phase4_NLP_Text_Processing_Complete_Guide.md`.
> Every diagram has explanatory text — what it shows, why it matters, and how to read it.

---

## 1. The Core Problem — Text to Numbers

ML models only understand numbers. Text is words. This diagram shows the fundamental NLP pipeline: raw text goes through preprocessing, then vectorization converts it to numbers, and finally any ML model can make predictions. Everything in NLP is about making this pipeline better.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph LR
    RAW["📝 Raw Text<br/>'Great pizza, fast delivery!'"] --> CLEAN["🧹 Preprocess<br/>lowercase, tokenize,<br/>remove stop words"]
    CLEAN --> VEC["🔢 Vectorize<br/>BoW, TF-IDF,<br/>or Embeddings"]
    VEC --> MODEL["🤖 ML Model<br/>LogReg, XGBoost,<br/>Neural Network"]
    MODEL --> PRED["✅ Prediction<br/>Positive or Negative"]

    style RAW fill:#252840,stroke:#f5b731,color:#c8cfe0
    style CLEAN fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style VEC fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
    style MODEL fill:#1a1d2e,stroke:#22d3a7,color:#e2e8f0
    style PRED fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Read left to right: yellow = raw input, blue = cleaning, purple = the key step (text to numbers), green = model and output. The purple "Vectorize" step is where all the NLP magic happens.

---

## 2. Text Preprocessing Pipeline

Raw text is messy — mixed case, punctuation, filler words. Each preprocessing step removes noise while keeping meaning. The diagram shows the pipeline applied to one review, with the output of each step feeding into the next.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    INPUT["'Great Pizza, Fast Delivery!'"] --> LOWER["1. Lowercase<br/>'great pizza, fast delivery!'"]
    LOWER --> PUNCT["2. Remove Punctuation<br/>'great pizza fast delivery'"]
    PUNCT --> TOKEN["3. Tokenize<br/>great, pizza, fast, delivery"]
    TOKEN --> STOP["4. Remove Stop Words<br/>great, pizza, fast, delivery<br/><i>no stop words here</i>"]
    STOP --> STEM["5. Stem or Lemmatize<br/>great, pizza, fast, deliveri<br/><i>stemming chops endings</i>"]

    style INPUT fill:#252840,stroke:#f5b731,color:#c8cfe0
    style LOWER fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style PUNCT fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style TOKEN fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
    style STOP fill:#1a1d2e,stroke:#22d3a7,color:#e2e8f0
    style STEM fill:#1a1d2e,stroke:#22d3a7,color:#e2e8f0
```

Each step is simple on its own. Together they transform messy human text into clean tokens ready for vectorization. Tokenization (purple) is the most important step — it defines what a "word" is.

---

## 3. Bag of Words — The Simplest Vectorization

### Why we need it

ML models need fixed-length numerical vectors. Bag of Words creates a vocabulary of all unique words, then represents each document as a count vector. It's the simplest way to go from words to numbers.

### How it works

Build a vocabulary from all reviews, then count each word per review. The diagram shows how two reviews become numerical vectors using the same vocabulary.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    VOCAB["📖 Vocabulary: 24 unique words<br/>amazing, awful, bad, best, cold,<br/>crust, delivery, fast, great, ..."]

    VOCAB --> R1["R1: 'great pizza fast delivery'<br/>→ great=1, pizza=1, fast=1, delivery=1<br/>→ rest = 0"]
    VOCAB --> R2["R2: 'terrible cold pizza slow'<br/>→ terrible=1, cold=1, pizza=1, slow=1<br/>→ rest = 0"]

    R1 --> FEED["Now feed to LogReg, XGBoost, etc.<br/>Each review = a 24 number vector"]
    R2 --> FEED

    style VOCAB fill:#252840,stroke:#f5b731,color:#c8cfe0
    style R1 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style R2 fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style FEED fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
```

Green = positive review, Red = negative review. Both become 24-number vectors. The limitation: "pizza" (appears everywhere) gets the same weight as "great" (appears once). TF-IDF fixes this.

---

## 4. TF-IDF — Smart Word Weighting

### Why we need it

In Bag of Words, "pizza" (in 7/8 reviews) gets the same count as "amazing" (in 1/8 reviews). But "amazing" is far more informative for classification. TF-IDF automatically downweights common words and upweights rare distinctive ones.

### How the formula works

TF-IDF multiplies two things: how frequent a word is in THIS document (TF) and how rare it is across ALL documents (IDF). The specific numbers: TF('great' in R1) = 1/4 = 0.25 (1 occurrence out of 4 words). IDF('great') = log(8/1) = 2.08 (appears in only 1 of 8 reviews — very rare). TF-IDF = 0.25 × 2.08 = 0.52. For 'pizza': TF = 1/4 = 0.25, IDF = log(8/7) = 0.13 (appears in 7 of 8 reviews — very common). TF-IDF = 0.25 × 0.13 = 0.03. The 16× difference (0.52 vs 0.03) shows TF-IDF's power: it automatically identified 'great' as the distinctive word. A word that's frequent locally but rare globally gets the highest score — it's the most distinctive feature of that document.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    TF["TF = count in this doc / total words<br/><i>How important is this word HERE?</i>"]
    IDF["IDF = log(total docs / docs with word)<br/><i>How rare is this word EVERYWHERE?</i>"]

    TF --> TFIDF["TF-IDF = TF × IDF"]
    IDF --> TFIDF

    TFIDF --> EX1["'pizza': TF=0.25, IDF=0.13<br/>TF-IDF = <b>0.03</b> (common = low)"]
    TFIDF --> EX2["'great': TF=0.25, IDF=2.08<br/>TF-IDF = <b>0.52</b> (rare = high)"]

    EX1 --> INSIGHT["'great' gets 16× more weight than 'pizza'<br/>TF-IDF automatically found the distinctive word"]
    EX2 --> INSIGHT

    style TF fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style IDF fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
    style TFIDF fill:#252840,stroke:#f5b731,color:#c8cfe0
    style EX1 fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style EX2 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style INSIGHT fill:#252840,stroke:#f5b731,color:#c8cfe0
```

Blue = TF (local importance), Purple = IDF (global rarity). Red = common word (low score), Green = rare word (high score). The yellow insight box shows the practical result.

---

## 5. Word Embeddings — Words as Meaningful Vectors

### Why we need them

BoW and TF-IDF treat every word as independent — "great" and "amazing" are completely unrelated features. But we know they mean similar things. Word embeddings represent words as dense vectors where similar words are close together in vector space.

### How Word2Vec learns

The core principle: "You know a word by the company it keeps." Words appearing in similar contexts get similar vectors. The diagram shows how this works.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    CONTEXT["Training: words in similar contexts<br/>get similar vectors"]

    CONTEXT --> W1["'great' appears near:<br/>pizza, delivery, crust"]
    CONTEXT --> W2["'amazing' appears near:<br/>pizza, delivery, crust"]

    W1 --> SIMILAR["Similar contexts → Similar vectors!<br/>'great' = 0.8, 0.2, 0.9<br/>'amazing' = 0.7, 0.3, 0.8<br/>distance = 0.17 (very close)"]
    W2 --> SIMILAR

    SIMILAR --> VS["vs 'terrible' = opposite context<br/>'terrible' = neg0.7, 0.2, neg0.8<br/>distance from 'great' = 2.40 (far!)"]

    style CONTEXT fill:#252840,stroke:#f5b731,color:#c8cfe0
    style W1 fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style W2 fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style SIMILAR fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style VS fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
```

Green = similar words cluster together. Red = opposite words are far apart. The network learns this structure automatically from reading millions of sentences.

---

## 6. The Evolution — BoW to Transformers

Each generation of NLP solved a limitation of the previous one. This diagram shows the progression and what each step added.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    BOW["1990s: Bag of Words<br/>Words as independent counts<br/><i>Limitation: all words equal</i>"]
    BOW -->|"Add importance<br/>weighting"| TFIDF["2000s: TF-IDF<br/>Words weighted by rarity<br/><i>Limitation: no word meaning</i>"]
    TFIDF -->|"Add semantic<br/>meaning"| W2V["2013: Word2Vec / GloVe<br/>Words as meaningful vectors<br/><i>Limitation: no context</i>"]
    W2V -->|"Add context<br/>understanding"| TRANS["2017: Transformers<br/>Words understood in context<br/><i>'bank' near 'river' vs 'money'</i>"]
    TRANS -->|"Scale up"| LLM["2020s: LLMs<br/>GPT, BERT, Claude<br/><i>Generate human-like text</i>"]

    style BOW fill:#252840,stroke:#f5b731,color:#c8cfe0
    style TFIDF fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style W2V fill:#1a1d2e,stroke:#7c6aff,color:#e2e8f0
    style TRANS fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style LLM fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
```

Each arrow label tells you what problem was solved. Follow top to bottom: each generation is strictly better than the previous one, but also more complex and data-hungry.

---

## 7. Cosine Similarity — Measuring Text Similarity

### Why we need it

Once text is a vector, we need to measure "how similar are two texts?" Euclidean distance fails because it's affected by document length — a long review and a short review about the same topic would seem far apart. Cosine similarity measures the angle between vectors, ignoring length.

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    FORMULA["Cosine Similarity = A·B / (‖A‖ × ‖B‖)<br/>Measures angle, ignores magnitude"]

    FORMULA --> SCORE1["Score near 1.0<br/>Same direction = same topic<br/><i>'great pizza' vs 'amazing pizza'</i>"]
    FORMULA --> SCORE0["Score near 0.0<br/>Perpendicular = unrelated<br/><i>'great pizza' vs 'stock market'</i>"]
    FORMULA --> SCOREN["Score near neg1.0<br/>Opposite direction = opposite meaning<br/><i>'great pizza' vs 'terrible pizza'</i>"]

    SCORE1 --> RAG["Used in RAG: find most similar<br/>documents to a query<br/>= semantic search"]

    style FORMULA fill:#252840,stroke:#f5b731,color:#c8cfe0
    style SCORE1 fill:#1a2a1f,stroke:#22d3a7,color:#c8d8c0
    style SCORE0 fill:#1a1d2e,stroke:#5eaeff,color:#e2e8f0
    style SCOREN fill:#2a1a1f,stroke:#f45d6d,color:#d8a8b8
    style RAG fill:#252840,stroke:#f5b731,color:#c8cfe0
```

Green = similar, Blue = unrelated, Red = opposite. This is the foundation of semantic search in RAG systems (Phase 5).

---

## 8. Interview Decision Tree 🎯

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'darkMode': true, 'background': '#0e1117', 'primaryColor': '#1a1d2e', 'primaryTextColor': '#e2e8f0', 'primaryBorderColor': '#2d3148', 'lineColor': '#8892b0', 'secondaryColor': '#252840', 'tertiaryColor': '#1a1d2e', 'fontSize': '14px', 'edgeLabelBackground': '#0e1117'}, 'flowchart': {'nodeSpacing': 30, 'rankSpacing': 40, 'padding': 15, 'htmlLabels': true}}}%%
graph TD
    Q1{"How to convert<br/>text to numbers?"} -->|Answer| A1["BoW: count words<br/>TF-IDF: weight by rarity<br/>Embeddings: dense meaningful vectors"]
    Q1 -->|Next Q| Q2{"TF-IDF vs BoW?"}
    Q2 -->|Answer| A2["TF-IDF downweights common words<br/>IDF = log(N / docs with word)<br/>Rare distinctive words get high scores"]
    Q2 -->|Next Q| Q3{"Explain Word2Vec?"}
    Q3 -->|Answer| A3["Learns vectors from context prediction<br/>Similar contexts = similar vectors<br/>king minus man plus woman = queen"]
    Q3 -->|Next Q| Q4{"Why cosine similarity<br/>for text?"}
    Q4 -->|Answer| A4["Measures angle not magnitude<br/>Long and short docs about same topic<br/>score high despite different lengths"]

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
