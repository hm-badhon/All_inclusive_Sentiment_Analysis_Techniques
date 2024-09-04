
# Sentiment Analysis with Naive Bayes and TF-IDF

## Overview

This project performs sentiment analysis on movie reviews using a Naive Bayes classifier with TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. Sentiment analysis helps determine if a piece of text expresses a positive or negative opinion. This project involves text preprocessing, feature extraction, model training, evaluation, and prediction.

## How It Works

1. **Data Preparation**: We use a dataset of movie reviews provided by the NLTK library, which includes labeled reviews as either positive or negative.
2. **Text Preprocessing**: The text data is cleaned by converting it to lowercase, removing punctuation, and eliminating stopwords. Additionally, words are lemmatized to their base forms.
3. **Feature Extraction**: TF-IDF vectorization is used to convert text data into numerical vectors that represent the importance of words in the reviews.
4. **Model Training**: We train a Naive Bayes classifier with hyperparameter tuning.
5. **Evaluation**: The model is evaluated using cross-validation and tested on a separate test set.
6. **Prediction**: We use the trained model to predict the sentiment of new reviews.

## Steps to Run the Project

### 1. Install Required Libraries

Before running the code, make sure you have the following Python libraries installed:

```bash
pip install nltk pandas scikit-learn
```

### 2. Download the Data

The script automatically downloads the necessary datasets using the `nltk` library. If needed, you can manually download them by running:

```python
import nltk
nltk.download("movie_reviews")
nltk.download("stopwords")
nltk.download("wordnet")
```

### 3. Run the Code

The script performs the following steps:

- **Loads the Data**: Retrieves movie reviews and their sentiments.
- **Preprocesses the Text**: Cleans the reviews by converting to lowercase, removing punctuation and stopwords, and lemmatizing words.
- **Converts Text to Feature Vectors**: Uses TF-IDF to represent the text data numerically.
- **Trains the Model**: Fits a Naive Bayes classifier to the training data.
- **Evaluates the Model**: Measures the model’s performance using cross-validation and a test set.
- **Predicts Sentiment**: Classifies new reviews as positive or negative.

### 4. Understand the Output

After running the code, you will see:

- **Cross-validation Accuracy**: The average accuracy of the model during cross-validation, which is approximately 79.69%.
- **Test Accuracy**: The accuracy of the model on the test set, which is approximately 81.75%.
- **Classification Report**: A detailed report showing precision, recall, and F1-score for each class (positive and negative). Here’s the report:

```
Classification Report:
              precision    recall  f1-score   support

         neg       0.82      0.81      0.82       199
         pos       0.82      0.82      0.82       201

    accuracy                           0.82       400
   macro avg       0.82      0.82      0.82       400
weighted avg       0.82      0.82      0.82       400
```

- **Precision**: The proportion of positive identifications that were actually correct.
- **Recall**: The proportion of actual positives that were correctly identified.
- **F1-score**: The harmonic mean of precision and recall.
- **Support**: The number of actual occurrences of the class in the specified dataset.

### 5. Prediction Results

The `predict_sentiment` function allows you to predict the sentiment of new reviews. For example, running:

```python
print(predict_sentiment("I absolutely loved this movie! It was fantastic."))
print(predict_sentiment("It was a terrible film. I hated it."))
print(predict_sentiment("The movie was okay, nothing special."))
```

Will provide output similar to:

```
positive
negative
negative
```

(Note: The actual output will depend on the model’s predictions and the data used.)

## Explanation of Key Components

### TF-IDF Vectorization

- **What Is It?**: TF-IDF is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents (corpus).
- **How It Works**: It combines two metrics:
  - **Term Frequency (TF)**: How often a term appears in a document.
  - **Inverse Document Frequency (IDF)**: How important a term is by considering its frequency across all documents.
- **Purpose**: TF-IDF helps in highlighting the significance of words while reducing the influence of common words that appear frequently across many documents.

### Naive Bayes Classifier

- **What Is It?**: A probabilistic machine learning model based on Bayes' theorem, assuming that features are independent given the class label.
- **How It Works**: Calculates the probability of each class given the features and predicts the class with the highest probability.

### Text Preprocessing

- **Lowercasing**: Converts all characters in the text to lowercase to maintain consistency.
- **Punctuation Removal**: Eliminates punctuation marks that do not contribute to the meaning.
- **Stopwords Removal**: Removes common words like "the," "is," and "and" that do not add significant meaning.
- **Lemmatization**: Reduces words to their base or root form to ensure consistency in word representation.

## Conclusion

This project demonstrates a comprehensive approach to sentiment analysis using a Naive Bayes classifier with TF-IDF vectorization. It showcases how to preprocess text, convert it to numerical features, train and evaluate a model, and make predictions.
