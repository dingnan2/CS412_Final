# Case Study Report


## TP Cases

Businesses correctly predicted to remain open.

**Common Important Features:**

- `category_encoded` (in 5/5 cases)
- `category_avg_success_rate` (in 5/5 cases)
- `review_count` (in 5/5 cases)
- `avg_review_stars` (in 5/5 cases)
- `weighted_avg_rating` (in 4/5 cases)

**Sample Case:**

- **Business ID**: G7UzqSWntpAZ5LFv00D2jA
- **Business Name**: North Valley Automotive
- **Location**: Reno, NV
- **Categories**: Automotive, Auto Repair
- **Prediction Year**: 2019
- **Prediction Probability**: 0.970

Top Contributing Features:

1. **category_encoded**: 0.909 (higher)
2. **category_avg_success_rate**: 0.913 (higher)
3. **review_count**: 29.000 (lower)
4. **weighted_avg_rating**: 4.780 (higher)
5. **weighted_sentiment**: 0.890 (higher)

## TN Cases

Businesses correctly predicted to close.

**Common Important Features:**

- `review_count` (in 5/5 cases)
- `avg_reviewer_tenure` (in 4/5 cases)
- `review_frequency` (in 4/5 cases)
- `size_activity_interaction` (in 4/5 cases)
- `days_since_first_review` (in 4/5 cases)

**Sample Case:**

- **Business ID**: FV4n-PzaiBi2YkYbxOULnA
- **Business Name**: MAC - New Orleans
- **Location**: New Orleans, LA
- **Categories**: Beauty & Spas, Makeup Artists, Shopping, Cosmetics & Beauty Supply
- **Prediction Year**: 2019
- **Prediction Probability**: 0.499

Top Contributing Features:

1. **review_count**: 7.000 (lower)
2. **weighted_avg_rating**: 1.907 (lower)
3. **avg_reviewer_tenure**: 12.047 (higher)
4. **weighted_sentiment**: -0.547 (lower)
5. **avg_text_length**: 994.714 (higher)

## FP Cases

Businesses predicted to stay open but actually closed (model errors).

**Common Important Features:**

- `review_count` (in 5/5 cases)
- `size_activity_interaction` (in 4/5 cases)
- `review_frequency` (in 4/5 cases)
- `category_encoded` (in 4/5 cases)
- `avg_reviewer_tenure` (in 4/5 cases)

**Sample Case:**

- **Business ID**: nVI9wI9ujmrutJRMgHKSIg
- **Business Name**: Fiesta Maya Mexican Grill
- **Location**: Springfield, PA
- **Categories**: Mexican, Restaurants
- **Prediction Year**: 2019
- **Prediction Probability**: 0.500

Top Contributing Features:

1. **review_count**: 80.500 (higher)
2. **size_activity_interaction**: 0.459 (higher)
3. **review_frequency**: 0.094 (higher)
4. **engagement_recent_vs_all**: 1.758 (higher)
5. **city_avg_success_rate**: 0.810 (higher)

**Why the model made this error (FP):**

Key distinguishing features (vs correctly predicted cases):

- `std_reviewer_credibility`: higher than expected (+6.35σ)
- `engagement_recent_vs_all`: higher than expected (+4.74σ)
- `weighted_useful_votes`: higher than expected (+2.34σ)

## FN Cases

Businesses predicted to close but actually stayed open (model errors).

**Common Important Features:**

- `review_count` (in 5/5 cases)
- `days_since_first_review` (in 5/5 cases)
- `avg_reviewer_tenure` (in 4/5 cases)
- `category_encoded` (in 3/5 cases)
- `category_avg_success_rate` (in 3/5 cases)

**Sample Case:**

- **Business ID**: SJlaPiAHtwQHxceuQblklQ
- **Business Name**: The Surly Mermaid
- **Location**: Saint Petersburg, FL
- **Categories**: Food, Seafood, American (Traditional), Burgers, Fish & Chips, Restaurants, Food Trucks, Sandwiches
- **Prediction Year**: 2020
- **Prediction Probability**: 0.500

Top Contributing Features:

1. **review_count**: 17.000 (lower)
2. **days_since_first_review**: 1047.000 (lower)
3. **avg_reviewer_tenure**: 10.323 (lower)
4. **weighted_avg_rating**: 4.516 (higher)
5. **weighted_sentiment**: 0.758 (higher)

**Why the model made this error (FN):**

Key distinguishing features (vs correctly predicted cases):

- `rating_recent_vs_early`: lower than expected (-5.71σ)
- `city_encoded`: lower than expected (-1.94σ)
- `total_useful_votes`: higher than expected (+1.64σ)

## Key Insights

### False Positives (Prediction Errors)

Common patterns in businesses predicted to stay open but closed:

- May have had stable historical performance
- Recent decline not captured by current features
- External factors (location changes, competition) not modeled

### False Negatives (Missed Survivors)

Common patterns in businesses predicted to close but stayed open:

- May have had temporary difficulties
- Recovery signals not captured
- Strong intangible factors (loyal customer base, unique offerings)

## Recommendations

Based on case study analysis:

1. **Feature Engineering**: Add features capturing recent trends more accurately
2. **External Data**: Consider incorporating location-based economic indicators
3. **Temporal Dynamics**: Better model recovery patterns after temporary decline
4. **Ensemble Methods**: Combine models with different strengths to reduce errors
