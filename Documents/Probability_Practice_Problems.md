# 10 Probability Word Problems with Step-by-Step Solutions

Practice these problems to master probability concepts for data science and interviews.

---

## PART 1: BASIC PROBABILITY (Problems 1-3)

### Formula Reminder
```
P(event) = Number of favorable outcomes / Total number of outcomes
```

---

### Problem 1: Email Campaign

**Question:**
You send 500 marketing emails. 75 people click the link. What's the probability a random recipient clicked?

**Given:**
- Total emails sent = 500
- People who clicked = 75

**Solution:**
```
P(click) = clicked / total
P(click) = 75 / 500
P(click) = 0.15 = 15%
```

**Answer:** There's a 15% probability (or 0.15) that a random recipient clicked the link.

**Interpretation:** If you pick any email recipient at random, there's about a 1 in 7 chance they clicked.

---

### Problem 2: Quality Control

**Question:**
A factory produces 1000 items. 23 are defective. If you pick one randomly, what's P(defective)?

**Given:**
- Total items = 1000
- Defective items = 23

**Solution:**
```
P(defective) = defective / total
P(defective) = 23 / 1000
P(defective) = 0.023 = 2.3%
```

**Answer:** P(defective) = 0.023 or 2.3%

**Interpretation:** About 2-3 out of every 100 items are defective. This is your defect rate.

---

### Problem 3: Customer Segments

**Question:**
40% of customers are "Premium", 60% are "Basic". What's the probability of randomly selecting a Premium customer?

**Given:**
- P(Premium) = 40% = 0.40
- P(Basic) = 60% = 0.60

**Solution:**
```
P(Premium) = 0.40 = 40%
```

**Answer:** P(Premium) = 0.40 or 40%

**Verification:** P(Premium) + P(Basic) = 0.40 + 0.60 = 1.00 ✓ (probabilities must sum to 1)

**Interpretation:** If you randomly pick a customer, there's a 40% chance they're Premium.

---

## PART 2: CONDITIONAL PROBABILITY (Problems 4-6)

### Formula Reminder
```
P(A|B) = P(A and B) / P(B)

Read as: "Probability of A, given that B happened"
```

---

### Problem 4: Churn Analysis

**Question:**
- 15% of customers complained last month
- Of those who complained, 40% churned
- Of those who didn't complain, 5% churned

Find: P(churn | complained) and P(churn | no complaint)

**Given:**
- P(complained) = 0.15
- P(no complaint) = 1 - 0.15 = 0.85
- P(churn | complained) = 0.40
- P(churn | no complaint) = 0.05

**Solution:**
These are already given as conditional probabilities!

```
P(churn | complained) = 0.40 = 40%
P(churn | no complaint) = 0.05 = 5%
```

**Answer:** 
- P(churn | complained) = 40%
- P(churn | no complaint) = 5%

**Interpretation:** Customers who complained are 8× more likely to churn (40% vs 5%). Complaints are a strong warning signal!

**Bonus - What's the overall churn rate?**
```
P(churn) = P(churn | complained) × P(complained) + P(churn | no complaint) × P(no complaint)
P(churn) = (0.40 × 0.15) + (0.05 × 0.85)
P(churn) = 0.06 + 0.0425
P(churn) = 0.1025 = 10.25%
```

---

### Problem 5: A/B Test

**Question:**
- Version A: 1000 visitors, 50 converted
- Version B: 1000 visitors, 65 converted

What's P(convert | saw version A) vs P(convert | saw version B)?

**Given:**
- Version A: 1000 visitors, 50 conversions
- Version B: 1000 visitors, 65 conversions

**Solution:**
```
P(convert | version A) = conversions_A / visitors_A
P(convert | version A) = 50 / 1000
P(convert | version A) = 0.05 = 5%

P(convert | version B) = conversions_B / visitors_B
P(convert | version B) = 65 / 1000
P(convert | version B) = 0.065 = 6.5%
```

**Answer:**
- P(convert | version A) = 5.0%
- P(convert | version B) = 6.5%

**Interpretation:** Version B has a 30% higher conversion rate than A ((6.5-5)/5 = 30%). But is this statistically significant? That requires a hypothesis test!

---

### Problem 6: Delivery Times

**Question:**
- 70% of orders are delivered on time
- Of late orders, 60% result in complaints
- Of on-time orders, 10% result in complaints

What's P(complaint | late)?

**Given:**
- P(on time) = 0.70
- P(late) = 1 - 0.70 = 0.30
- P(complaint | late) = 0.60
- P(complaint | on time) = 0.10

**Solution:**
```
P(complaint | late) = 0.60 = 60%
```

**Answer:** P(complaint | late) = 60%

**Interpretation:** 60% of late deliveries result in complaints, compared to only 10% of on-time deliveries. Late delivery is 6× more likely to cause a complaint.

**Bonus - What's the overall complaint rate?**
```
P(complaint) = P(complaint | late) × P(late) + P(complaint | on time) × P(on time)
P(complaint) = (0.60 × 0.30) + (0.10 × 0.70)
P(complaint) = 0.18 + 0.07
P(complaint) = 0.25 = 25%
```

---

## PART 3: BAYES' THEOREM (Problems 7-9)

### Formula Reminder
```
P(A|B) = P(B|A) × P(A) / P(B)

Or expanded:
P(A|B) = P(B|A) × P(A) / [P(B|A) × P(A) + P(B|not A) × P(not A)]
```

Bayes lets you "flip" a conditional probability.

---

### Problem 7: Spam Filter

**Question:**
- 20% of emails are spam
- Spam filter catches 95% of spam: P(flagged | spam) = 0.95
- Filter incorrectly flags 5% of legitimate emails: P(flagged | not spam) = 0.05

If an email is flagged, what's the probability it's actually spam?

**Given:**
- P(spam) = 0.20
- P(not spam) = 0.80
- P(flagged | spam) = 0.95
- P(flagged | not spam) = 0.05

**Find:** P(spam | flagged)

**Solution:**

Step 1: Calculate P(flagged) using total probability
```
P(flagged) = P(flagged | spam) × P(spam) + P(flagged | not spam) × P(not spam)
P(flagged) = (0.95 × 0.20) + (0.05 × 0.80)
P(flagged) = 0.19 + 0.04
P(flagged) = 0.23
```

Step 2: Apply Bayes' Theorem
```
P(spam | flagged) = P(flagged | spam) × P(spam) / P(flagged)
P(spam | flagged) = (0.95 × 0.20) / 0.23
P(spam | flagged) = 0.19 / 0.23
P(spam | flagged) = 0.826 = 82.6%
```

**Answer:** P(spam | flagged) = 82.6%

**Interpretation:** If an email is flagged, there's an 82.6% chance it's actually spam. That means 17.4% of flagged emails are false positives (legitimate emails incorrectly flagged).

---

### Problem 8: Medical Test (Classic Bayes Problem!)

**Question:**
- Disease affects 1% of population: P(sick) = 0.01
- Test is 99% accurate for sick people: P(positive | sick) = 0.99
- Test is 95% accurate for healthy people: P(negative | healthy) = 0.95
  - This means P(positive | healthy) = 0.05 (false positive rate)

If you test positive, what's the probability you're actually sick?

**Given:**
- P(sick) = 0.01
- P(healthy) = 0.99
- P(positive | sick) = 0.99 (true positive rate)
- P(positive | healthy) = 0.05 (false positive rate)

**Find:** P(sick | positive)

**Solution:**

Step 1: Calculate P(positive) using total probability
```
P(positive) = P(positive | sick) × P(sick) + P(positive | healthy) × P(healthy)
P(positive) = (0.99 × 0.01) + (0.05 × 0.99)
P(positive) = 0.0099 + 0.0495
P(positive) = 0.0594
```

Step 2: Apply Bayes' Theorem
```
P(sick | positive) = P(positive | sick) × P(sick) / P(positive)
P(sick | positive) = (0.99 × 0.01) / 0.0594
P(sick | positive) = 0.0099 / 0.0594
P(sick | positive) = 0.167 = 16.7%
```

**Answer:** P(sick | positive) = 16.7%

**Interpretation:** Even with a "99% accurate" test, if you test positive, there's only a 16.7% chance you're actually sick! This is because the disease is rare (1%), so most positive results are false positives.

**This is why:**
- Doctors order follow-up tests
- Screening tests need to be interpreted carefully
- Base rate (prevalence) matters enormously

---

### Problem 9: Customer Prediction

**Question:**
- 30% of visitors are "high-intent" buyers
- 80% of high-intent visitors add to cart
- 20% of low-intent visitors add to cart

If someone added to cart, what's P(high-intent)?

**Given:**
- P(high-intent) = 0.30
- P(low-intent) = 0.70
- P(add to cart | high-intent) = 0.80
- P(add to cart | low-intent) = 0.20

**Find:** P(high-intent | added to cart)

**Solution:**

Step 1: Calculate P(add to cart)
```
P(add to cart) = P(add | high) × P(high) + P(add | low) × P(low)
P(add to cart) = (0.80 × 0.30) + (0.20 × 0.70)
P(add to cart) = 0.24 + 0.14
P(add to cart) = 0.38
```

Step 2: Apply Bayes' Theorem
```
P(high-intent | added to cart) = P(add | high) × P(high) / P(add to cart)
P(high-intent | added to cart) = (0.80 × 0.30) / 0.38
P(high-intent | added to cart) = 0.24 / 0.38
P(high-intent | added to cart) = 0.632 = 63.2%
```

**Answer:** P(high-intent | added to cart) = 63.2%

**Interpretation:** If someone adds to cart, there's a 63% chance they're a high-intent buyer (up from the baseline 30%). Adding to cart is a strong signal of purchase intent.

**Business insight:** Target cart-abandoners with follow-up emails — they're likely high-intent!

---

## PART 4: COMBINED APPLICATION (Problem 10)

### Problem 10: Pizza Store Scenario

**Question:**
Your pizza store data shows:
- 25% of orders are delivery, 75% are pickup
- Delivery orders: 15% have complaints
- Pickup orders: 5% have complaints

Find:
a) What's P(complaint)?
b) If a customer complained, what's P(it was delivery)?
c) Are delivery and complaints independent?

**Given:**
- P(delivery) = 0.25
- P(pickup) = 0.75
- P(complaint | delivery) = 0.15
- P(complaint | pickup) = 0.05

---

**Part (a): What's P(complaint)?**

Use total probability:
```
P(complaint) = P(complaint | delivery) × P(delivery) + P(complaint | pickup) × P(pickup)
P(complaint) = (0.15 × 0.25) + (0.05 × 0.75)
P(complaint) = 0.0375 + 0.0375
P(complaint) = 0.075 = 7.5%
```

**Answer (a):** P(complaint) = 7.5%

---

**Part (b): If a customer complained, what's P(it was delivery)?**

Use Bayes' Theorem:
```
P(delivery | complaint) = P(complaint | delivery) × P(delivery) / P(complaint)
P(delivery | complaint) = (0.15 × 0.25) / 0.075
P(delivery | complaint) = 0.0375 / 0.075
P(delivery | complaint) = 0.50 = 50%
```

**Answer (b):** P(delivery | complaint) = 50%

**Interpretation:** Even though only 25% of orders are delivery, they account for 50% of complaints! Delivery is over-represented in complaints.

---

**Part (c): Are delivery and complaints independent?**

Two events are independent if: P(A and B) = P(A) × P(B)

Check:
```
P(delivery and complaint) = P(complaint | delivery) × P(delivery)
P(delivery and complaint) = 0.15 × 0.25 = 0.0375

P(delivery) × P(complaint) = 0.25 × 0.075 = 0.01875
```

Compare:
```
P(delivery and complaint) = 0.0375
P(delivery) × P(complaint) = 0.01875

0.0375 ≠ 0.01875
```

**Answer (c):** No, delivery and complaints are NOT independent.

**Interpretation:** Delivery orders are more likely to have complaints than you'd expect by chance. There's a relationship between delivery and complaints — delivery causes more complaints.

---

## SUMMARY: KEY FORMULAS

| Concept | Formula |
|---------|---------|
| Basic Probability | P(A) = favorable / total |
| Complement | P(not A) = 1 - P(A) |
| Conditional | P(A\|B) = P(A and B) / P(B) |
| Total Probability | P(A) = P(A\|B)×P(B) + P(A\|not B)×P(not B) |
| Bayes' Theorem | P(A\|B) = P(B\|A) × P(A) / P(B) |
| Independence Test | Independent if P(A and B) = P(A) × P(B) |

---

## PRACTICE TIPS

1. **Always write out what's given** — list all probabilities
2. **Identify what's asked** — which probability do you need?
3. **Draw a tree diagram** for complex problems
4. **Check your answer** — probabilities must be between 0 and 1
5. **Interpret in plain English** — what does this mean for business?

---

## INTERVIEW TIPS

When solving probability in interviews:
- Talk through your reasoning out loud
- State your assumptions
- Write the formula before plugging in numbers
- Sanity check: "Does this answer make sense?"
- Connect to business impact: "This means we should..."

---

*Practice these until they feel natural. Probability is the foundation of all data science!*
