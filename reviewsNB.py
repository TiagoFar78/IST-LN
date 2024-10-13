import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import re

nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_plot(text):
    text.lower()

    text = re.sub(r'[^\w\s]', '', text)

    #words = [lemmatizer.lemmatize(word) for word in text.split()]
    words = [stemmer.stem(word) for word in text.split()]

    return ' '.join(words)

# Load and preprocess data
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            title, movie_from, genre, director, plot = line.strip().split('\t')
            combined_features = f"{title} {movie_from} {plot}"
            data.append((preprocess_plot(combined_features), genre))
    return pd.DataFrame(data, columns=["plot", "genre"])

# Load data from file
train_data = load_data('train.txt')

# Train-test split (80% train, 20% test) with stratified sampling by genre
train_df, test_df = train_test_split(train_data, test_size=0.2, stratify=train_data['genre'])

# Initialize CountVectorizer (you can also limit the max number of features)
vectorizer = CountVectorizer(max_features=7000, stop_words='english')

# Fit and transform the train plots, and transform the test plots
X_train = vectorizer.fit_transform(train_df['plot'])
X_test = vectorizer.transform(test_df['plot'])

# Labels (target) mapping genres to numbers
genre_mapping = {genre: i for i, genre in enumerate(train_data['genre'].unique())}
y_train = train_df['genre'].map(genre_mapping)
y_test = test_df['genre'].map(genre_mapping)

# Train the Multinomial Naive Bayes model
nb_model = MultinomialNB()
cv_scores = cross_val_score(nb_model, X_train, y_train, cv=5, scoring='accuracy')

# Print the results of cross-validation
print(f"Cross-Validation Scores (5-fold): {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}")
nb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=genre_mapping.keys(), yticklabels=genre_mapping.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ---- Extract misclassified examples ----

# Create a DataFrame with true and predicted labels
test_df['true_genre'] = y_test.map({v: k for k, v in genre_mapping.items()})
test_df['predicted_genre'] = [list(genre_mapping.keys())[list(genre_mapping.values()).index(pred)] for pred in y_pred]

# Filter out misclassified examples
misclassified = test_df[test_df['true_genre'] != test_df['predicted_genre']]

# Specify genres of interest (comedy, drama, romance)
genres_of_interest = ['comedy', 'drama', 'romance']

# Filter misclassified examples involving only comedy, drama, and romance
misclassified_filtered = misclassified[
    (misclassified['true_genre'].isin(genres_of_interest)) & 
    (misclassified['predicted_genre'].isin(genres_of_interest))
]

# Select relevant columns to save to the file
misclassified_output = misclassified_filtered[['plot', 'true_genre', 'predicted_genre']]

# Write the filtered misclassified examples to a CSV file
misclassified_output.to_csv('filtered_misclassified_examples.csv', index=False)

print("Filtered misclassified examples written to 'filtered_misclassified_examples.csv'.")