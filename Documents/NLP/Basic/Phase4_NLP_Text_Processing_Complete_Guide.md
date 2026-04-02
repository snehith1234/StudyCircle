# NLP Text Processing: Complete Guide with Example Data at Every Step

## The Problem

You have 8 pizza store **customer reviews**. You want to predict: **Is this review positive (1) or negative (0)?**

### Our Data

| Review ID | Text | Sentiment |
|-----------|------|-----------|
| R1 | "Great pizza, fast delivery!" | 1 (Positive) |
| R2 | "Terrible cold pizza, very slow" | 0 (Negative) |
| R3 | "Best pizza in town, loved it" | 1 (Positive) |
| R4 | "Awful service, never ordering again" | 0 (Negative) |
| R5 | "Good pizza but delivery was slow" | 1 (Positive) |
| R6 | "Worst pizza I ever had, disgusting" | 0 (Negative) |
| R7 | "Amazing crust, quick delivery, perfect" | 1 (Positive) |
| R8 | "Bad pizza, cold and late delivery" | 0 (Negative) |

**The challenge:** ML models need NUMBERS, not words. How do we convert text → numbers?

---

## STEP 1: Why Can't We Just Feed Text to a Model?

```
Logistic Regression expects: [4.5, 20] → numbers
XGBoost expects:             [4.5, 20] → numbers
Any ML model expects:        [x₁, x₂, ...] → numbers

But we have: "Great pizza, fast delivery!" → words

We need a pipeline: Text → Clean Text → Numerical Features → Model
```

This is the fundamental NLP problem: **text representation** — converting human language into mathematical vectors that preserve meaning.

---

## STEP 2: Text Preprocessing Pipeline

Raw text is messy. Before converting to numbers, we clean it.

### Step 2a: Lowercasing

```
"Great Pizza, Fast Delivery!" → "great pizza, fast delivery!"

Why: "Pizza" and "pizza" mean the same thing.
     Without lowercasing, the model treats them as different words.
```

### Step 2b: Removing Punctuation

```
"great pizza, fast delivery!" → "great pizza fast delivery"

Why: Punctuation doesn't carry sentiment information for basic models.
     (Advanced models like BERT do use punctuation.)
```

### Step 2c: Tokenization — Splitting into Words

```
"great pizza fast delivery" → ["great", "pizza", "fast", "delivery"]

This is TOKENIZATION — breaking text into individual tokens (words).
Each token becomes a potential feature.
```

### Step 2d: Removing Stop Words

Stop words are common words that don't carry meaning: "the", "is", "a", "in", "and", etc.

```
R5: ["good", "pizza", "but", "delivery", "was", "slow"]
  → ["good", "pizza", "delivery", "slow"]

Removed: "but", "was" — they don't help distinguish positive from negative.
```

### Step 2e: Stemming vs Lemmatization

Both reduce words to their root form, but differently:

```
STEMMING (crude, fast — chops off endings):
  "delivery" → "deliveri"    (not a real word!)
  "ordering" → "order"
  "loved"    → "love"
  "disgusting" → "disgust"

LEMMATIZATION (smart, slower — uses dictionary):
  "delivery" → "delivery"    (keeps real words)
  "ordering" → "order"
  "loved"    → "love"
  "disgusting" → "disgusting"

Stemming is faster but produces non-words.
Lemmatization is slower but produces real words.
For most tasks, either works fine.
```

### Our Cleaned Reviews

| ID | Original | After Preprocessing |
|----|----------|-------------------|
| R1 | "Great pizza, fast delivery!" | ["great", "pizza", "fast", "delivery"] |
| R2 | "Terrible cold pizza, very slow" | ["terrible", "cold", "pizza", "slow"] |
| R3 | "Best pizza in town, loved it" | ["best", "pizza", "town", "loved"] |
| R4 | "Awful service, never ordering again" | ["awful", "service", "never", "ordering"] |
| R5 | "Good pizza but delivery was slow" | ["good", "pizza", "delivery", "slow"] |
| R6 | "Worst pizza I ever had, disgusting" | ["worst", "pizza", "ever", "disgusting"] |
| R7 | "Amazing crust, quick delivery, perfect" | ["amazing", "crust", "quick", "delivery", "perfect"] |
| R8 | "Bad pizza, cold and late delivery" | ["bad", "pizza", "cold", "late", "delivery"] |

---

## STEP 3: Bag of Words (BoW) — The Simplest Representation

### The Idea

Create a vocabulary of ALL unique words across all reviews.
For each review, count how many times each word appears.

### Building the Vocabulary

```
All unique words (sorted):
amazing, awful, bad, best, cold, crust, delivery, disgusting,
ever, fast, good, great, late, loved, never, ordering, perfect,
pizza, quick, service, slow, terrible, town, worst

Total vocabulary size: 24 words
```

### The BoW Matrix

Each review becomes a vector of 24 numbers (one per word):

```
         amazing awful bad best cold crust delivery disgusting ever fast good great late loved never ordering perfect pizza quick service slow terrible town worst
R1 (+):    0      0    0   0    0     0      1         0       0    1    0     1    0    0     0      0        0      1     0      0     0      0     0     0
R2 (-):    0      0    0   0    1     0      0         0       0    0    0     0    0    0     0      0        0      1     0      0     1      1     0     0
R3 (+):    0      0    0   1    0     0      0         0       0    0    0     0    0    1     0      0        0      1     0      0     0      0     1     0
R4 (-):    0      1    0   0    0     0      0         0       0    0    0     0    0    0     1      1        0      0     0      1     0      0     0     0
R5 (+):    0      0    0   0    0     0      1         0       0    0    1     0    0    0     0      0        0      1     0      0     1      0     0     0
R6 (-):    0      0    0   0    0     0      0         1       1    0    0     0    0    0     0      0        0      1     0      0     0      0     0     1
R7 (+):    1      0    0   0    0     1      1         0       0    0    0     0    0    0     0      0        1      0     1      0     0      0     0     0
R8 (-):    0      0    1   0    1     0      1         0       0    0    0     0    1    0     0      0        0      1     0      0     0      0     0     0
```

### Now We Can Use Any ML Model!

```
R1 vector: [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
Label: 1 (Positive)

Feed this to Logistic Regression, XGBoost, Random Forest — they all work now!
```

### BoW Limitations

```
Problem 1: "pizza" appears in almost every review → not useful for classification
Problem 2: Word order is lost → "not good" and "good not" are identical
Problem 3: Rare meaningful words get same weight as common meaningless ones

Solution: TF-IDF (next section)
```

---

## STEP 4: TF-IDF — Smarter Word Weighting

### Why We Need It

In Bag of Words, "pizza" (appears in 7/8 reviews) gets the same weight as "amazing" (appears in 1/8 reviews). But "amazing" is far more informative — if a review contains "amazing", it's almost certainly positive. TF-IDF fixes this by downweighting common words and upweighting rare, distinctive ones.

### The Formula

```
TF-IDF(word, document) = TF(word, document) × IDF(word)

TF  = Term Frequency    = count of word in this document / total words in document
IDF = Inverse Document Frequency = log(total documents / documents containing this word)
```

### Why This Formula?

- **TF** captures: "How important is this word in THIS review?" A word mentioned 3 times in a review matters more than one mentioned once.
- **IDF** captures: "How rare is this word ACROSS ALL reviews?" A word in every review (like "pizza") gets low IDF. A word in only one review (like "amazing") gets high IDF.
- **Multiplied together:** A word that's frequent in one review but rare overall gets the highest score — it's the most distinctive feature of that review.

### Computing TF for R1: "great pizza fast delivery"

```
Total words in R1: 4

TF("great", R1)    = 1/4 = 0.25
TF("pizza", R1)    = 1/4 = 0.25
TF("fast", R1)     = 1/4 = 0.25
TF("delivery", R1) = 1/4 = 0.25
```

### Computing IDF for Key Words

```
Total documents N = 8

"pizza":    appears in 7 reviews → IDF = log(8/7) = log(1.14) = 0.13  (very common → low IDF)
"delivery": appears in 4 reviews → IDF = log(8/4) = log(2.0)  = 0.69  (moderate)
"great":    appears in 1 review  → IDF = log(8/1) = log(8.0)  = 2.08  (rare → high IDF!)
"fast":     appears in 1 review  → IDF = log(8/1) = log(8.0)  = 2.08  (rare → high IDF!)
"terrible": appears in 1 review  → IDF = log(8/1) = log(8.0)  = 2.08  (rare → high IDF!)
"cold":     appears in 2 reviews → IDF = log(8/2) = log(4.0)  = 1.39  (somewhat rare)
"slow":     appears in 2 reviews → IDF = log(8/2) = log(4.0)  = 1.39  (somewhat rare)
```

### Computing TF-IDF for R1

```
TF-IDF("great", R1)    = 0.25 × 2.08 = 0.520  ← HIGH (rare + present)
TF-IDF("pizza", R1)    = 0.25 × 0.13 = 0.033  ← LOW  (too common)
TF-IDF("fast", R1)     = 0.25 × 2.08 = 0.520  ← HIGH (rare + present)
TF-IDF("delivery", R1) = 0.25 × 0.69 = 0.173  ← MEDIUM
```

**Key insight:** "great" and "fast" get 16× more weight than "pizza"! TF-IDF automatically figured out that "great" and "fast" are the distinctive words in R1, while "pizza" is just noise (everyone mentions pizza).

### TF-IDF vs BoW Comparison

```
BoW for R1:     pizza=1, great=1, fast=1, delivery=1  (all equal!)
TF-IDF for R1:  pizza=0.03, great=0.52, fast=0.52, delivery=0.17  (weighted by importance!)

The model now knows: "great" and "fast" matter most in this review.
```

---

## STEP 5: Word Embeddings — Words as Vectors with Meaning

### The Limitation of BoW and TF-IDF

```
BoW/TF-IDF treat every word as independent:
  "great" and "amazing" → completely different features (no relationship)
  "cold" and "hot"      → completely different features (no relationship)

But we KNOW "great" ≈ "amazing" (similar meaning)
And "cold" is opposite of "hot"

Can we represent words so that similar words are CLOSE together?
```

### The Word2Vec Idea

Represent each word as a dense vector (e.g., 3 numbers instead of 24).
Words with similar meanings get similar vectors.

```
"great"   → [0.8, 0.2, 0.9]
"amazing" → [0.7, 0.3, 0.8]   ← close to "great"!
"terrible"→ [-0.7, 0.2, -0.8]  ← opposite direction from "great"!
"pizza"   → [0.1, 0.9, 0.1]   ← different dimension entirely

Distance("great", "amazing")  = 0.17  (very close — similar meaning)
Distance("great", "terrible") = 2.40  (very far — opposite meaning)
Distance("great", "pizza")    = 1.20  (moderate — unrelated)
```

### How Word2Vec Learns These Vectors

The core idea: **"You shall know a word by the company it keeps."**

```
Training data (context windows):
  "great pizza fast delivery" → "great" appears near "pizza", "fast"
  "amazing crust quick delivery" → "amazing" appears near "crust", "quick"

Since "great" and "amazing" appear in similar contexts
(near "pizza", "delivery", "crust"), they get similar vectors.

The model learns: words in similar contexts → similar vectors
```

### Two Word2Vec Architectures

```
CBOW (Continuous Bag of Words):
  Input: context words ["great", "fast", "delivery"]
  Predict: center word "pizza"
  "Given the surrounding words, what's the middle word?"

Skip-gram:
  Input: center word "pizza"
  Predict: context words ["great", "fast", "delivery"]
  "Given one word, what words appear nearby?"

Skip-gram works better for rare words.
CBOW is faster for frequent words.
```

### From Word Vectors to Document Vectors

To represent an entire review, average the word vectors:

```
R1: "great pizza fast delivery"
  great    = [0.8, 0.2, 0.9]
  pizza    = [0.1, 0.9, 0.1]
  fast     = [0.6, 0.1, 0.7]
  delivery = [0.3, 0.8, 0.2]

  R1 vector = average = [(0.8+0.1+0.6+0.3)/4, (0.2+0.9+0.1+0.8)/4, (0.9+0.1+0.7+0.2)/4]
            = [0.45, 0.50, 0.475]

This 3-number vector captures the "meaning" of the entire review.
Feed it to any ML model for classification.
```

---

## STEP 6: Cosine Similarity — Measuring Text Similarity

### Why Euclidean Distance Fails for Text

```
Document A: mentions "pizza" 10 times (long review)
Document B: mentions "pizza" 2 times (short review)

Euclidean distance says they're far apart (10 vs 2).
But they're about the SAME TOPIC — just different lengths!
```

### Cosine Similarity — Direction, Not Magnitude

```
Cosine Similarity = (A · B) / (||A|| × ||B||)

It measures the ANGLE between two vectors, ignoring length.
Range: -1 (opposite) to 0 (unrelated) to +1 (identical direction)
```

### Computing for Two Reviews

```
R1 vector: [0.52, 0.03, 0.52, 0.17]  (TF-IDF: great, pizza, fast, delivery)
R3 vector: [0.00, 0.03, 0.00, 0.00, 0.52, 0.52]  (TF-IDF: best, loved, pizza, town)

These share "pizza" but differ on sentiment words.
Cosine similarity would be low — they use different positive words.

R1 vector vs R7 vector: both have "delivery" + positive words
Cosine similarity would be higher — similar sentiment pattern.
```

### Why Cosine Similarity Matters for RAG

```
In RAG systems (Phase 5), you'll use cosine similarity to find
the most relevant documents for a query:

Query: "How fast is delivery?"
Compare query embedding to all document embeddings using cosine similarity.
Return the documents with highest similarity scores.

This is SEMANTIC SEARCH — finding meaning, not just keyword matches.
```

---

## STEP 7: Text Classification Pipeline — Putting It All Together

### The Complete Pipeline

```
Raw Text → Preprocess → Vectorize → Model → Prediction

Step 1: "Great pizza, fast delivery!" 
Step 2: ["great", "pizza", "fast", "delivery"]     (preprocess)
Step 3: [0.52, 0.03, 0.52, 0.17, 0, 0, ...]       (TF-IDF vectorize)
Step 4: Logistic Regression → P(positive) = 0.89    (predict)
Step 5: 0.89 ≥ 0.5 → Positive ✅                    (classify)
```

### Which Vectorization to Use?

```
| Method      | Pros                          | Cons                           | Best For              |
|-------------|-------------------------------|--------------------------------|-----------------------|
| Bag of Words| Simple, interpretable         | No word importance weighting   | Quick baseline        |
| TF-IDF      | Weights important words       | Still no word relationships    | Classical ML + text   |
| Word2Vec    | Captures word meaning         | Needs lots of training data    | When meaning matters  |
| BERT embed. | Best semantic understanding   | Slow, needs GPU                | Production NLP        |
```

---

## STEP 8: Named Entity Recognition (NER)

### What Is NER?

NER identifies and classifies named entities in text — people, places, organizations, dates, etc.

```
"I ordered from Pizza Palace on Main Street last Friday"

NER output:
  "Pizza Palace" → ORGANIZATION
  "Main Street"  → LOCATION
  "last Friday"  → DATE

This is INFORMATION EXTRACTION — pulling structured data from unstructured text.
```

### Why NER Matters for Agentic AI

```
In RAG systems: NER helps chunk documents around entities
In Agents: NER extracts parameters from user requests
  "Book a table at Pizza Palace for 2 people on Friday"
  → restaurant: "Pizza Palace", party_size: 2, date: "Friday"
```

---

## STEP 9: The Evolution — From BoW to Transformers

```
1990s: Bag of Words      → Words as independent counts
2000s: TF-IDF            → Words weighted by importance  
2013:  Word2Vec/GloVe    → Words as meaningful vectors
2017:  Transformers      → Words understood in CONTEXT
2018:  BERT              → Bidirectional context understanding
2020+: GPT, LLMs        → Generate human-like text

Each step solved a limitation of the previous one:
  BoW → "all words equal"     → TF-IDF weights them
  TF-IDF → "no word meaning"  → Word2Vec adds meaning
  Word2Vec → "no context"     → Transformers add context
  ("bank" near "river" vs "bank" near "money" → different vectors!)
```

---

## COMPLETE FORMULA SUMMARY

```
1. TF(word, doc)  = count(word in doc) / total words in doc
2. IDF(word)      = log(N / docs containing word)
3. TF-IDF         = TF × IDF
4. Cosine Sim     = (A · B) / (||A|| × ||B||)
5. Word2Vec       = Neural network trained on context prediction
6. Doc vector     = average of word vectors (simple approach)
```

---

## INTERVIEW CHEAT SHEET

**Q: "How do you convert text to numbers for ML?"**
> "Three main approaches: (1) Bag of Words — count word occurrences, simple but treats all words equally. (2) TF-IDF — weight words by importance, downweights common words like 'the'. (3) Word embeddings (Word2Vec, BERT) — represent words as dense vectors where similar words are close together. For classical ML, TF-IDF is the go-to. For deep learning, embeddings are standard."

**Q: "What is TF-IDF and why use it over Bag of Words?"**
> "TF-IDF = Term Frequency × Inverse Document Frequency. It solves BoW's main problem: common words like 'pizza' get the same weight as rare distinctive words like 'amazing'. IDF downweights words that appear in many documents and upweights rare ones. The result: each word's score reflects how distinctive it is for that particular document."

**Q: "Explain Word2Vec."**
> "Word2Vec learns dense vector representations of words by training on context prediction — 'you know a word by the company it keeps.' Words appearing in similar contexts get similar vectors. Two architectures: CBOW predicts the center word from context, Skip-gram predicts context from the center word. The result: king - man + woman ≈ queen."

**Q: "What is cosine similarity and why use it for text?"**
> "Cosine similarity measures the angle between two vectors, ignoring magnitude. For text, this is important because a long document and a short document about the same topic should be similar, even though their word counts differ. Range is -1 to +1. It's the standard similarity metric for semantic search and RAG retrieval."

---

*Every number in this document was computed by hand. Verify them yourself for practice!*
