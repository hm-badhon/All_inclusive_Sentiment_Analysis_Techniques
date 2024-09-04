import nltk
import pandas as pd
import string
from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Setup - Install necessary libraries (if not installed)
# !pip install nltk pandas scikit-learn

# Step 2: Data Preparation
nltk.download("movie_reviews")
nltk.download("stopwords")
nltk.download("wordnet")

# Load the dataset
documents = [
    (" ".join(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

# Convert to DataFrame
df = pd.DataFrame(documents, columns=["review", "sentiment"])

# Text Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords and lemmatize
    text = " ".join(
        lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words
    )
    return text

df["cleaned_review"] = df["review"].apply(preprocess_text)

# Step 3: Model Training

# Convert text data to feature vectors using TF-IDF with bigrams
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["cleaned_review"])
y = df["sentiment"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Hyperparameter Tuning with GridSearchCV

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'max_iter': [100, 200, 300]    # Number of iterations
}

grid = GridSearchCV(LogisticRegression(), param_grid, refit=True, cv=5, verbose=2)
grid.fit(X_train, y_train)

# Best parameters found by GridSearchCV
print(f"Best parameters: {grid.best_params_}")

# Train the model using the best parameters
best_model = grid.best_estimator_

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Step 5: Prediction

def predict_sentiment(text):
    # Preprocess the input text
    cleaned_text = preprocess_text(text)
    # Transform the text to feature vector
    text_vector = vectorizer.transform([cleaned_text])
    # Predict the sentiment
    prediction = best_model.predict(text_vector)
    return prediction[0]

# Test the prediction function
print(predict_sentiment("I absolutely loved this movie! It was fantastic."))
print(predict_sentiment("It was a terrible film. I hated it."))
print(predict_sentiment("The movie was okay, nothing special."))
