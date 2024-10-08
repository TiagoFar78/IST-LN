import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(lemmatized_words)

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            title, movie_from, genre, director, plot = line.strip().split('\t')
            plot = preprocess_text(plot)
            data.append((plot, genre))
    return pd.DataFrame(data, columns=["plot", "genre"])

def prepare_data(data):
    train_df, test_df = train_test_split(
        data, 
        test_size=0.2, 
        stratify=data['genre'], 
        random_state=42
    )
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df['plot'])
    X_test = vectorizer.transform(test_df['plot'])
    
    genre_mapping = {genre: i for i, genre in enumerate(data['genre'].unique())}
    train_labels = train_df['genre'].map(genre_mapping)
    test_labels = test_df['genre'].map(genre_mapping)
    
    return X_train, X_test, train_labels, test_labels, genre_mapping

def train_model(X_train, train_labels):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, train_labels)
    return model

def evaluate_model(model, X_test, test_labels):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    return predictions

def plot_confusion_matrix(test_labels, predictions, genre_mapping):
    conf_matrix = confusion_matrix(test_labels, predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=genre_mapping.keys(),
        yticklabels=genre_mapping.keys()
    )
    plt.xlabel('Predicted Genre')
    plt.ylabel('True Genre')
    plt.title('Genre Classification Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def main():
    data = load_data('train.txt')
    X_train, X_test, train_labels, test_labels, genre_mapping = prepare_data(data)
    model = train_model(X_train, train_labels)
    predictions = evaluate_model(model, X_test, test_labels)
    plot_confusion_matrix(test_labels, predictions, genre_mapping)

if __name__ == "__main__":
    main()