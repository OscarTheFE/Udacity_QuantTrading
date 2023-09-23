# Udacity_QuantTrading
This repo contains the projects I did for Udacity AI for Trading nanodegree.
# Summary of each project
## 1. Trading with Momentum. [Project](https://github.com/OscarTheFE/Udacity_QuantTrading/tree/main/Project1_TradingWithMomentum/home)
- Learn basics of stock markets. Learn how to calculate stock returns and design momentum trading strategy.
- 
## 2. Breakout Strategy. Project
- Learn the importance of outliers and how to detect them. Learn about methods designed to handle outliers.
- Learn about regression, and related statistical tools that pre-process data before regression analysis. Learn commonly-used time series models.
- Learn about stock volatility, and how the GARCH model analysis volatility. See how volatility is used in equity trading.
- Learn about pair trading, and implemented linear regression for hedge ratio and statistical tests used to check cointegration.

## 3. Smart beta and portfolio optimization. Project
- Overview of stocks, indices, and funds. Learn about ETFs.
- Learn how to optimize portfolios to meet certain criteria and constraints.

## 4. Alpha Research and Factor Modeling. Project
- Learn factors and how to convert factor values into portfolio weights in a dollar neutral portfolio with leverage ratio equals to 1 (i.e., standardize factor values).
- Learn fundamentals of factor models and type of factors. Learn how to compute portfolio variance using risk factor models. Learn time series and cross-sectional risk models.
- Learn how to use PCA to build risk factor models.

## 5. Intro to NLP. Project
- NLP pipeline consists of text processing, feature extraction, and modeling.
- Text processing: Learn text acquisition (plane text, tabular data, and online resources), simple data cleaning with python regex and BeautifulSoup, using nltk (natural language toolkit) for tokenization, stemming, and lemmatization.
- Financial Statement: Learn how to apply Regexes to 10Ks, how BeautifulSoup can ease the parse of (perfectly formatted) html and xml downloaded using request library.
- Basic NLP Analysis: Learn quantitatively measure readability of documents using readability indices, how to convert document into vectors using bag of word and TF-IDF weighting, and metrics to compare similarities between documents.

## 6. Sentiment Analysis with Neural Networks. Project
- Neural Network Basics: Learn maximum likelihood, cross entropy, logistic regression, gradient decent, regularization, and practical heuristics for training neural networks.
- Deep Learning with PyTorch.
- Recurrence Neutral Networks:
  - Learn to use RNN to predict simple Time Series and train Character-Level LSTM to generate new text based on the text from the book.
  - Learn Word2Vec algorithm using the Skip-gram Architecture and with Negative Sampling.
  - Sentiment Analysis RNN: Implement a recurrent neural network that can predict if the text of a movie review is positive or negative.

## 7. Combining Signals for Enhanced Alpha. Project
- Decision Tree: Learn how to branching decision tree using entropy and information gain. Implement decision tree using sklearn for Titanic Survival Exploration and visualize the decision tree using graphviz.
- Model Testing and Evaluation: Learn Type 1 and Type 2 errors, Precision vs. Recall, Cross validation for time series, and using learning curve to determine underfitting and overfitting.
- Random Forest: Learn the ensemble random forest method and implement it in sklearn.
- Feature Engineering: Certain alphas perform better or worse depending on market conditions. Feature engineering creates additional inputs to give models more contexts about the current market condition so that the model can adjust its prediction accordingly.
- Overlapping Labels: Mitigate the problem when features are dependent on each other (non-IID).
- Feature Importance: Company would prefer simple interpretable models to black-box complex models. interpretability opens the door for complex models to be readily acceptable. One way to interpret a model is to measure how much each feature contributed to the model prediction called feature importance. Learn how sklearn computes features importance for tree-based method. Learn how to calculate shap for feature importance of a single sample.

## 8. Backtesting. Project
- Basics: Learn best practices of backtesting and see what overfitting can "look like" in practice.
- Learn how to optimization a portfolio with transaction cost. Learn some additional ways to design your optimization with efficiency in mind. This is really helpful when backtesting, because having reasonably shorter runtimes allows you to test and iterate on your alphas more quickly.
