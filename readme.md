

# All_inclusive_Sentiment_Analysis_Techniques

## Overview

This project explores various sentiment analysis strategies, each using different techniques and models. The strategies covered are:

1. **Sentiment Analysis with Naive Bayes**
2. **Sentiment Analysis with Naive Bayes and TF-IDF**
3. **Sentiment Analysis with Logistic Regression and TF-IDF**
4. **Sentiment Analysis with BERT and PyTorch**

Each approach is evaluated to understand its effectiveness and compare performance metrics.

## Strategies

### 1. Sentiment Analysis with Naive Bayes

**Approach**: Uses a Naive Bayes classifier to predict sentiment based on raw text data.

- **Data Preparation**: Loads and preprocesses movie reviews from NLTK.
- **Feature Extraction**: Converts text data into feature vectors using CountVectorizer.
- **Model Training**: Trains a Multinomial Naive Bayes classifier.
- **Evaluation**: Measures accuracy and provides a classification report.

**Sample Results**:
- **Accuracy**: `0.80`
- **Classification Report**: Precision, recall, and F1-score around `0.80` for both classes.

### 2. Sentiment Analysis with Naive Bayes and TF-IDF

**Approach**: Enhances the Naive Bayes model by using TF-IDF for feature extraction.

- **Data Preparation**: Same as above.
- **Feature Extraction**: Uses TF-IDF vectorization with unigrams and bigrams.
- **Model Training**: Trains a Multinomial Naive Bayes classifier with TF-IDF features.
- **Evaluation**: Measures accuracy and provides a detailed classification report.

**Sample Results**:
- **Cross-validation Accuracy**: `0.80`
- **Test Accuracy**: `0.82`
- **Classification Report**: Precision, recall, and F1-score around `0.82` for both classes.

### 3. Sentiment Analysis with Logistic Regression and TF-IDF

**Approach**: Uses logistic regression with TF-IDF for feature extraction and hyperparameter tuning.

- **Data Preparation**: Same as above.
- **Feature Extraction**: Uses TF-IDF vectorization with unigrams and bigrams.
- **Model Training**: Trains a logistic regression model with hyperparameter tuning using GridSearchCV.
- **Evaluation**: Measures accuracy and provides a classification report.

**Sample Results**:
- **Cross-validation Accuracy**: `0.80`
- **Test Accuracy**: `0.84`
- **Classification Report**: Precision, recall, and F1-score around `0.84` for both classes.

### 4. Sentiment Analysis with BERT and PyTorch

**Approach**: Utilizes BERT for sequence classification with advanced text preprocessing and mixed precision training.

- **Data Preparation**: Tokenizes and encodes movie reviews using BERT’s tokenizer.
- **Feature Extraction**: Encodes text data into input IDs and attention masks.
- **Model Training**: Fine-tunes a BERT model with mixed precision training.
- **Evaluation**: Measures accuracy and provides a classification report.

**Sample Results**:
- **Validation Accuracy**: `0.84`
- **Classification Report**: Precision, recall, and F1-score around `0.84` for both classes.

## Comparison

| **Strategy**                        | **Accuracy** | **Precision (pos)** | **Recall (pos)** | **F1-score (pos)** | **Precision (neg)** | **Recall (neg)** | **F1-score (neg)** |
|-------------------------------------|--------------|---------------------|------------------|--------------------|---------------------|------------------|--------------------|
| Naive Bayes                         | `0.80`       | `0.80`              | `0.80`           | `0.80`             | `0.80`              | `0.80`           | `0.80`             |
| Naive Bayes with TF-IDF             | `0.82`       | `0.82`              | `0.82`           | `0.82`             | `0.82`              | `0.82`           | `0.82`             |
| Logistic Regression with TF-IDF     | `0.84`       | `0.84`              | `0.84`           | `0.84`             | `0.84`              | `0.84`           | `0.84`             |
| BERT with PyTorch                    | `0.84`       | `0.83`              | `0.85`           | `0.84`             | `0.84`              | `0.83`           | `0.84`             |

## Analysis

- **Naive Bayes**: Provides a baseline performance with a good balance of precision and recall. However, its performance can be improved with more sophisticated feature extraction methods.
- **Naive Bayes with TF-IDF**: Shows improved performance over basic Naive Bayes by incorporating TF-IDF, which enhances feature representation.
- **Logistic Regression with TF-IDF**: Further improves accuracy with more robust feature extraction and hyperparameter tuning, demonstrating its effectiveness in capturing nuanced patterns.
- **BERT with PyTorch**: Achieves the highest accuracy and balanced precision and recall, thanks to the advanced capabilities of BERT. It effectively handles the context and subtleties of the text.

## Conclusion

Each strategy demonstrates different strengths in sentiment analysis. BERT with PyTorch provides the best overall performance but requires more computational resources. Logistic Regression with TF-IDF offers a good balance between performance and resource efficiency, while Naive Bayes models are simpler but less effective in capturing complex patterns.

This comprehensive analysis helps in selecting the appropriate strategy based on the specific requirements and constraints of the sentiment analysis task.
