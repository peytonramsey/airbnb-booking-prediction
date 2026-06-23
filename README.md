# Airbnb Booking Prediction and Market Study

This project builds a national scale classifier on US Airbnb listings, then adapts the same framework to a single Paris neighborhood to answer a concrete investor question about gentrification and commercial hosting.

## Project overview

The work has two parts, both framed as binary classification of “high booking” listings (booking conversion rate above 5 percent).

- Part I, US national model  
  Trained a CatBoost classifier on about 374,000 US listings to predict whether a listing achieves a high booking rate, with 75 engineered features across pricing, host behavior, text, and regulatory signals. The final model reached an AUC of about 0.87 on the Kaggle leaderboard.

- Part II, Popincourt, Paris investor study  
  Reused the modeling framework on 93,578 listing month observations from the Popincourt neighborhood of Paris, a gentrifying market with strict rental regulations. The local model achieved a 0.966 holdout AUC and revealed a structural 6x booking rate gap between casual and commercial hosts that price and location do not explain away.

The repository is designed as a portfolio quality example of end to end tabular ML with careful leakage control and business facing interpretation.

## Business questions

The project is centered on questions that matter to an investor deciding how to operate in Popincourt.

- National question  
  Given listing level features, can a model reliably predict which US listings will achieve a high booking rate?

- Popincourt questions  
  1. Is price the primary lever controlling booking success in this neighborhood?  
  2. Does host type (casual, semi professional, commercial) independently explain booking performance, or is it just a proxy for pricing?  
  3. Does sub neighborhood geography create exploitable micro market advantages?

## Data

- Source  
  - Inside Airbnb, US national snapshot for Part I.  
  - Inside Airbnb, Popincourt neighborhood panel (2015 to 2024) for Part II.

- Scale  
  - US national: about 374,000 listings and 56 raw columns, after cleaning.  
  - Popincourt: 93,578 listing month rows and 95 columns, about 86,871 rows after removing zero price entries and top 1 percent price outliers.

- Target  
  Binary label `high_booking` defined as booking conversion rate above 5 percent. 

Each dataset is loaded with Polars, with explicit handling of messy null markers and price fields stored as strings. 

## Methods and modeling

### Feature engineering

Across both parts, around 75 features were engineered into nine conceptual groups.

- Regulatory  
  Categorical regulatory type plus flags intended to capture hosts using minimum night settings or maximum night settings as regulatory workarounds.

- Pricing  
  Log price, total cost per night including cleaning and security deposit, price per person, price per bedroom, monthly and weekly discount ratios, and a price tier bucket (budget, mid, premium, luxury).

- Neighborhood context  
  Neighborhood level medians for price, reviews per month, review score, and capacity, computed out of fold with Bayesian shrinkage to avoid leaking target information into features.

- Host quality and tenure  
  Response rate and tier, missing response rate flags, host acceptance rate with domain informed imputation, host account age proxies based on profile URL patterns and IDs, and experience proxies from “about” text length.

- Amenities  
  Counts of total, luxury, and practical amenities, separating “nice to have” features from core usability features.

- Text (US model only)  
  A per fold TF IDF plus TruncatedSVD pipeline on combined listing text fields to create dense latent text features without leaking validation data.

- Geography and Popincourt specifics  
  Haversine distances to three gentrification anchors (Oberkampf, Bastille, Roquette), a host type classifier (casual, semi professional, commercial) from host listing counts, snapshot year, and host cohort year. 

### Models and training

- Model choice  
  CatBoost was chosen for native categorical handling, ordered target statistics to avoid leakage, and strong performance on mixed tabular data without feature scaling. 

- Hyperparameter tuning  
  Optuna ran about 40 Bayesian trials to tune depth, learning rate, regularization, subsampling, and tree parameters for the US model. The Popincourt model reused the same configuration with a slightly shallower depth to reflect the smaller dataset. 

- Validation strategy  
  - US national model: 5 fold stratified cross validation with TF IDF and neighborhood features recomputed inside the fold structure. Kaggle leaderboard AUC around 0.87, cross validation AUC about 0.85, and a same distribution 80/20 holdout AUC around 0.94.
  - Popincourt model: stratified 80/20 split with an AUC of 0.966, accuracy about 0.91, and F1 near 0.88 at a tuned threshold around 0.54.

- Interpretability  
  SHAP values from CatBoost were used to rank features and explain individual predictions, with a focus on investor actionable signals like reviews per month, price per bedroom, log price, minimum nights, and host type. 

## Key findings

The Popincourt analysis produced several non intuitive findings that contrast with more commercial US markets.

- Price as the main lever  
  Budget listings around 49 per night booked at about 54 percent, mid tier around 79 per night at about 43 percent, and premium around 167 per night at about 22 percent. This 32 percentage point spread is monotonic and consistent across host types, which means aggressive premium pricing sharply reduces booking conversion.

- Host type and operating style  
  Casual hosts with a single listing booked at about 44 percent on average and captured roughly 62 percent of estimated revenue, despite being under 80 percent of the market. Commercial hosts with six or more listings booked at around 12 percent and captured only about 15 percent of revenue even though they charged about 50 percent more per night. In the extreme, casual budget listings reached about 59 percent booking, while commercial premium listings were around 9 to 10 percent, roughly a 6x gap. 

- Geography is not a lever  
  K means clustering on latitude and longitude showed only about 4 percentage points of booking variation across spatial clusters, and cluster membership correlated with the target at about 0.02. High price and high booking listings were scattered rather than concentrated into a simple “best zone”. 

- Regulatory workarounds  
  A 30 night minimum stay, used by some commercial operators to shift into medium term rentals and avoid nightly caps, did not rescue commercial performance. Casual hosts with 30 night minimums had some of the highest booking rates in the dataset, while commercial hosts with the same flag had some of the lowest. 

In short, the project shows that the conventional professional “many premium listings” playbook produces the worst outcomes in this neighborhood.

## What I learned

This project reinforced a few habits that matter in real data science work.

- Domain research has to come before heavy modeling. Understanding how Inside Airbnb structures its snapshots made it possible to extract snapshot dates and cohort years from picture URLs and date fields and to design regulatory and temporal features that were not obvious from the column names alone.

- Leakage prevention is a design problem, not just an implementation detail. Keeping TF IDF fits, neighborhood aggregates, and Bayesian shrinkage inside the cross validation fold structure was critical to getting honest validation metrics, and a few early missteps showed how easily leakier versions can inflate scores.

- It is important to let the data overturn initial assumptions. The starting story emphasized location and premium positioning, but the final models and EDA showed that conservative pricing and resident style operation are what actually win in Popincourt. Being willing to change the narrative around what the data says was key.

