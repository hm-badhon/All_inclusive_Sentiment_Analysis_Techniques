# Achieving 90% accuracy in sentiment analysis can be challenging, especially with a simple model like Naive Bayes. 
# However, we can try several advanced techniques and strategies to further improve the model's performance:

### 1. **Advanced Text Preprocessing**
   - **Word Embeddings:** Use pre-trained word embeddings like Word2Vec, GloVe, or FastText instead of traditional TF-IDF vectors. These embeddings capture semantic meaning better.
   - **Advanced Tokenization:** Consider using advanced tokenizers such as spaCy, which can handle multi-word expressions and named entities more effectively.

   ```python
   import spacy
   from sklearn.feature_extraction.text import TfidfVectorizer

   nlp = spacy.load("en_core_web_sm")

   def preprocess_text_spacy(text):
       doc = nlp(text.lower())
       tokens = [
           token.lemma_ for token in doc 
           if not token.is_stop and not token.is_punct and token.lemma_ != '-PRON-'
       ]
       return " ".join(tokens)

   df["cleaned_review"] = df["review"].apply(preprocess_text_spacy)
   ```

### 2. **More Sophisticated Models**
   - **Logistic Regression:** Often outperforms Naive Bayes for text classification.
   - **Support Vector Machines (SVM):** Known for being powerful for classification tasks.
   - **Ensemble Methods:** Use ensemble methods like Random Forest, Gradient Boosting, or XGBoost.

   ```python
   from sklearn.linear_model import LogisticRegression

   model = LogisticRegression(max_iter=200)
   model.fit(X_train, y_train)
   ```

### 3. **Deep Learning Models**
   - **Recurrent Neural Networks (RNNs) or LSTM/GRU:** Handle sequences of words better, capturing the context more effectively.
   - **Transformers (e.g., BERT):** State-of-the-art models for NLP tasks. Pre-trained BERT models can be fine-tuned on your dataset to significantly boost accuracy.

   ```python
   from transformers import BertTokenizer, BertForSequenceClassification
   from transformers import Trainer, TrainingArguments

   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

   def preprocess_for_bert(texts):
       return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

   # Example for using BERT (requires PyTorch)
   model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
   ```

### 4. **Hyperparameter Tuning**
   - Use tools like `GridSearchCV` or `RandomizedSearchCV` to find the best hyperparameters for your models.

   ```python
   from sklearn.model_selection import GridSearchCV

   param_grid = {'C': [0.1, 1, 10, 100]}
   grid = GridSearchCV(LogisticRegression(), param_grid, refit=True, verbose=3)
   grid.fit(X_train, y_train)
   ```

### 5. **Handling Class Imbalance**
   - If there is class imbalance (not the case in the movie_reviews dataset, but generally), techniques like oversampling, undersampling, or using class weights can help.

   ```python
   model = LogisticRegression(class_weight='balanced', max_iter=200)
   model.fit(X_train, y_train)
   ```

### 6. **Feature Selection**
   - **SelectKBest:** Use feature selection techniques to reduce dimensionality and focus on the most informative features.

   ```python
   from sklearn.feature_selection import SelectKBest, chi2

   selector = SelectKBest(chi2, k=1500)
   X_train_selected = selector.fit_transform(X_train, y_train)
   X_test_selected = selector.transform(X_test)
   ```

### 7. **More Data**
   - **Data Augmentation:** Use data augmentation techniques to artificially increase the size of the dataset.
   - **External Datasets:** Combine your dataset with other sentiment analysis datasets to provide more training data.

### 8. **Error Analysis**
   - Manually analyze the misclassifications to understand what the model is missing. You might identify specific types of reviews that need more attention, which could guide further feature engineering or model tuning.

### Implementation Strategy:
We can start by implementing a more sophisticated model like Logistic Regression or SVM with proper hyperparameter tuning and advanced text preprocessing. If these don't yield the desired results, move on to using deep learning models like BERT, which are highly effective for text classification tasks.

