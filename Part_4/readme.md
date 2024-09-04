
# Sentiment Analysis with BERT and PyTorch

## Overview

This project performs sentiment analysis on movie reviews using BERT (Bidirectional Encoder Representations from Transformers) with PyTorch. BERT is a powerful pre-trained transformer model that captures contextual information from text, making it highly effective for various NLP tasks. This project includes data preprocessing, tokenization, model training with mixed precision, evaluation, and prediction.

## How It Works

1. **Data Preparation**: The dataset consists of movie reviews from the NLTK library, labeled as either positive or negative. The data is split into training and testing sets.
2. **Tokenization**: The BERT tokenizer converts text data into token IDs and attention masks required by the BERT model.
3. **Model Training**: A BERT model for sequence classification is fine-tuned using the training data. Mixed precision training is used to optimize training efficiency.
4. **Evaluation**: The model is evaluated on a test set, and performance metrics are provided, including accuracy and a classification report.

## Steps to Run the Project

### 1. Install Required Libraries

Before running the code, ensure you have the necessary Python libraries installed. You can install them using:

```bash
pip install torch transformers nltk pandas scikit-learn tqdm
```

### 2. Download the Data

The script automatically downloads the necessary NLTK data. If needed, you can manually download it by running:

```python
import nltk
nltk.download("movie_reviews")
```

### 3. Run the Code

The script performs the following steps:

- **Load and Prepare Data**: Retrieves movie reviews from NLTK and preprocesses them.
- **Tokenize Data**: Converts text data into input IDs and attention masks using BERT's tokenizer.
- **Train the Model**: Fine-tunes the BERT model on the training data using mixed precision training.
- **Evaluate the Model**: Calculates accuracy and generates a classification report on the test set.
- **Prediction**: The model can be used to predict sentiments for new reviews (not included in this script but can be added for further use).

### 4. Understand the Output

After running the code, you will see:

- **Validation Accuracy**: The accuracy of the model on the test set. For example:

  ```
  Validation Accuracy: 0.84
  ```

- **Classification Report**: A detailed report showing precision, recall, and F1-score for each class (positive and negative). Here's an example report:

  ```
  Classification Report:
                precision    recall  f1-score   support

           neg       0.82      0.85      0.83       199
           pos       0.86      0.82      0.84       201

      accuracy                           0.84       400
     macro avg       0.84      0.84      0.84       400
  weighted avg       0.84      0.84      0.84       400
  ```

  - **Precision**: The proportion of positive identifications that were actually correct.
  - **Recall**: The proportion of actual positives that were correctly identified.
  - **F1-score**: The harmonic mean of precision and recall.
  - **Support**: The number of actual occurrences of each class.

### 5. Explanation of Key Components

#### BERT Tokenization

- **What Is It?**: Tokenization is the process of converting text into a format suitable for the BERT model, including token IDs and attention masks.
- **How It Works**: The BERT tokenizer splits the text into tokens and generates input IDs and attention masks that tell the model which tokens are important.

#### BERT Model Training

- **What Is It?**: Fine-tuning BERT for sequence classification adapts the pre-trained model to classify text into categories.
- **How It Works**: The model is trained on the labeled movie reviews, using mixed precision training to improve performance and reduce memory usage.

#### Mixed Precision Training

- **What Is It?**: A technique that uses both 16-bit and 32-bit floating-point operations to speed up training and reduce memory usage.
- **How It Works**: The GradScaler and autocast from `torch.cuda.amp` handle scaling of gradients and automatic mixed precision, optimizing the training process.

## Conclusion

This project demonstrates how to leverage the BERT model for sentiment analysis using PyTorch, including advanced techniques like mixed precision training. The provided code can be adapted to other text classification tasks and further customized for different datasets.

