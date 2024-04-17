# Sentiment Analysis on IMDB 50k Dataset using Logistic Regression and Neural Network

This project performs sentiment analysis on the IMDB 50k dataset using logistic regression and a simple neural network. The goal is to classify movie reviews as either positive or negative based on their textual content.

## Dataset
```
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
```
The IMDB 50k dataset consists of 50,000 movie reviews, labeled as either positive or negative. The dataset is divided into two sets: 35000 reviews for training and 15000 reviews for testing.

##Libraries

```python
import pandas  as pd
import numpy as np
import seaborn as sns
import regex as re
import nltk
from nltk.corpus import stopwords
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,f1_score

```

## Preprocessing

Before applying the machine learning models, the text data needs to be preprocessed. The following steps are performed:

1. **Tokenization**: The text is split into individual words or tokens.
2. **Stopword Removal**: Common words like "the," "a," "is," etc., are removed as they do not contribute much to the sentiment analysis.
3. **Stemming/Lemmatization**: Words are reduced to their root form to handle different verb tenses and plural forms.
4. **URL Remover**
5. **Special Character Removal**

## Vectorization

After preprocessing, the text data needs to be converted into a numerical representation that can be understood by the machine learning models. In this project, the `CountVectorizer` from the `scikit-learn` library is used.

The `CountVectorizer` converts the text data into a sparse matrix, where each row represents a document (movie review), and each column represents a unique word in the corpus. The values in the matrix represent the count of each word in the corresponding document.

```python

custom_params = {
    'max_features': 1000,
    'ngram_range': (1, 2),
    'max_df': 0.8,
    'min_df': 5,
    'binary': True
cv=CountVectorizer(**custom_params)
}

```
`max_features`: 1000: This parameter specifies the maximum number of features (words or n-grams) to consider in the vectorized representation of the text data.

`ngram_range`: (1, 2): This parameter determines the range of n-gram values to be included in the vectorized representation

`max_df`: 0.8: This parameter specifies the maximum document frequency for a word to be included as a feature

`min_df`: 5: This parameter specifies the minimum document frequency for a word to be included as a feature

`binary`: True: This parameter determines whether to use binary feature vectors or term frequency-based feature vectors



## Logistic Regression

Logistic regression is a supervised machine learning algorithm that can be used for binary classification tasks, such as sentiment analysis. It models the probability of the target variable (positive or negative sentiment) as a function of the input features (word counts).

In this project, the logistic regression model is trained on the preprocessed and vectorized training data. The trained model is then evaluated on the test data to assess its performance.

## Evaluation

The performance of both the logistic regression and neural network models is evaluated using various metrics, such as accuracy, precision, recall, and F1-score. 
```python
Accuracy:0.8580666666666666
Precision Score:0.8533523168521874
Recall score:0.8687574120437476
F1 Score:0.8609859614756775
```
