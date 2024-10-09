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
from collections import Counter
import numpy as np
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
    stemmed_words = [stemmer.stem(word) for word in lemmatized_words]

    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in stemmed_words if word not in stop_words])
    return text

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            title, movie_from, genre, director, plot = line.strip().split('\t')
            plot = preprocess_text(plot)
            director = preprocess_text(director) 
            movie_from = preprocess_text(movie_from)
            data.append((plot, director, movie_from, genre)) 
    return pd.DataFrame(data, columns=["plot", "director", "movie_from", "genre"])

def prepare_data(data):
    train_df, test_df = train_test_split(
        data, 
        test_size=0.2, 
        stratify=data['genre']
    )

    train_df['combined'] = train_df['plot'] + ' ' + train_df['director'] + ' ' + train_df['movie_from']
    test_df['combined'] = test_df['plot'] + ' ' + test_df['director'] + ' ' + test_df['movie_from']

    vectorizer = TfidfVectorizer(max_features=10000, sublinear_tf=True)
    X_train = vectorizer.fit_transform(train_df['combined'])  
    X_test = vectorizer.transform(test_df['combined']) 
    
    genre_mapping = {genre: i for i, genre in enumerate(data['genre'].unique())}
    train_labels = train_df['genre'].map(genre_mapping)
    test_labels = test_df['genre'].map(genre_mapping)
    
    return X_train, X_test, train_labels, test_labels, genre_mapping

def train_model(X_train, train_labels):
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, train_labels)
    return model

def get_top_mismatches(conf_matrix, genre_mapping, top_n=10):
    genre_names = list(genre_mapping.keys())
    
    mismatches = []

    for i in range(len(genre_names)):
        for j in range(len(genre_names)):
            if i != j:  
                mismatches.append(((genre_names[i], genre_names[j]), conf_matrix[i, j]))
    mismatches.sort(key=lambda x: x[1], reverse=True)
    return mismatches[:top_n]

def analyze_word_frequencies(data, true_genre, predicted_genre, genre_mapping):
    misclassified_plots = data[(data['genre'] == true_genre) & (data['predicted_genre'] == predicted_genre)]['plot']
    
    all_words = []
    for plot in misclassified_plots:
        all_words.extend(plot.split())

    word_frequencies = Counter(all_words)
    
    return word_frequencies.most_common(10)

def evaluate_model_with_predictions(model, X_test, test_labels, genre_mapping):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    test_labels_genres = [list(genre_mapping.keys())[list(genre_mapping.values()).index(label)] for label in test_labels]
    predicted_labels_genres = [list(genre_mapping.keys())[list(genre_mapping.values()).index(pred)] for pred in predictions]
    
    return predictions, test_labels_genres, predicted_labels_genres

def plot_confusion_matrix(test_labels, predictions, genre_mapping):
    conf_matrix = confusion_matrix(test_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=genre_mapping.keys(), yticklabels=genre_mapping.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return conf_matrix

def get_top_common_words_across_mismatches(top_mismatches, test_df, genre_mapping, top_n=10):
    aggregated_word_frequencies = Counter()
    
    for (true_genre, predicted_genre), count in top_mismatches:
        common_words = analyze_word_frequencies(test_df, true_genre, predicted_genre, genre_mapping)
        
        # Add frequencies to the aggregated counter
        aggregated_word_frequencies.update(dict(common_words))
    
    # Get the top N common words across all mismatches
    top_common_words = aggregated_word_frequencies.most_common(top_n)
    
    return top_common_words

def main():
    data = load_data('train.txt')
    X_train, X_test, train_labels, test_labels, genre_mapping = prepare_data(data)
    model = train_model(X_train, train_labels)
    predictions, true_genres, predicted_genres = evaluate_model_with_predictions(model, X_test, test_labels, genre_mapping)
    
    # Prepare test dataframe with predictions
    test_df = data.loc[data.index.isin(test_labels.index)].copy()
    test_df['predicted_genre'] = predicted_genres
    
    # Plot confusion matrix and get top mismatches
    conf_matrix = plot_confusion_matrix(test_labels, predictions, genre_mapping)
    top_mismatches = get_top_mismatches(conf_matrix, genre_mapping)
    
    print("Top 10 mismatches and their most common words:")
    for (true_genre, predicted_genre), count in top_mismatches:
        print(f"Mismatched Genres: {true_genre} -> {predicted_genre} (Count: {count})")
        common_words = analyze_word_frequencies(test_df, true_genre, predicted_genre, genre_mapping)
        print(f"Most common words in these misclassified plots: {common_words}\n")

    # Find top 10 common words across all mismatches
    top_common_words = get_top_common_words_across_mismatches(top_mismatches, test_df, genre_mapping)
    
    print(f"Top 10 common words across all mismatches: {top_common_words}")

if __name__ == "__main__":
    main()

