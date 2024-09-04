import nltk
import pandas as pd
import string
from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
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

# Convert text data to feature vectors using TF-IDF
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["cleaned_review"])
y = df["sentiment"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a Naive Bayes classifier with hyperparameter tuning
model = MultinomialNB(alpha=0.1)
model.fit(X_train, y_train)

# Evaluate the model using cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-validation accuracy: {scores.mean()}")

# Evaluate the model on the test set
y_pred = model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Step 4: Prediction

def predict_sentiment(text):
    # Preprocess the input text
    cleaned_text = preprocess_text(text)
    # Transform the text to feature vector
    text_vector = vectorizer.transform([cleaned_text])
    # Predict the sentiment
    prediction = model.predict(text_vector)
    return prediction[0]

# Test the prediction function
print(predict_sentiment("I absolutely loved this movie! It was fantastic."))
print(predict_sentiment("It was a terrible film. I hated it."))
print(predict_sentiment("The movie was okay, nothing special."))
