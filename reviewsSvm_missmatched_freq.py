import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from collections import Counter
import nltk

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()

    # Lemmatize and stem words
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    stemmed_words = [stemmer.stem(word) for word in lemmatized_words]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in stemmed_words if word not in stop_words])
    
    return text

# Function to load data from file
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            title, movie_from, genre, director, plot = line.strip().split('\t')
            plot = preprocess_text(plot)  # Preprocess plot
            director = preprocess_text(director) 
            movie_from = preprocess_text(movie_from)
            title = preprocess_text(title)
            data.append((plot, director, movie_from, title, genre)) 
    return pd.DataFrame(data, columns=["plot", "director", "movie_from", "title", "genre"])

# Function to prepare data for training
def prepare_data(data):
    # Split into train and test sets
    train_df, test_df = train_test_split(
        data, 
        test_size=0.2, 
        stratify=data['genre'], 
    )

    train_df['combined'] = train_df['plot'] + ' ' + train_df['director'] + ' ' + train_df['movie_from'] + ' ' + train_df['title'] 
    test_df['combined'] = test_df['plot'] + ' ' + test_df['director'] + ' ' + test_df['movie_from'] + ' ' +  test_df['title'] 
    
    # Use TfidfVectorizer to transform text
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), sublinear_tf=True)  # Use bigrams  
    X_train = vectorizer.fit_transform(train_df['combined'])
    X_test = vectorizer.transform(test_df['combined'])
    
    # Map genres to numerical labels
    genre_mapping = {genre: i for i, genre in enumerate(data['genre'].unique())}
    train_labels = train_df['genre'].map(genre_mapping)
    test_labels = test_df['genre'].map(genre_mapping)
    
    return X_train, X_test, train_labels, test_labels, genre_mapping, test_df

# Function to train the SVC model
def train_model(X_train, train_labels):
    model = SVC(kernel='linear')
    model.fit(X_train, train_labels)
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, test_labels):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    return predictions

# Function to plot the confusion matrix
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

# Function to get mismatched plots
def get_mismatched_plots(test_df, test_labels, predictions):
    mismatched_indices = test_labels != predictions
    mismatched_plots = test_df[mismatched_indices]['plot']
    return mismatched_plots

# Function to get most frequent words from mismatched plots
def get_most_frequent_words(mismatched_plots, top_n=20):
    all_words = " ".join(mismatched_plots).split()
    word_counts = Counter(all_words)
    most_common_words = word_counts.most_common(top_n)
    return most_common_words

# Main function to execute the steps
def main():
    data = load_data('train.txt')  # Load the dataset
    X_train, X_test, train_labels, test_labels, genre_mapping, test_df = prepare_data(data)  # Prepare data
    model = train_model(X_train, train_labels)  # Train the SVC model
    predictions = evaluate_model(model, X_test, test_labels)  # Evaluate model
    plot_confusion_matrix(test_labels, predictions, genre_mapping)  # Plot confusion matrix
    
    # Get mismatched plots and find the most frequent words
    mismatched_plots = get_mismatched_plots(test_df, test_labels, predictions)
    most_frequent_words = get_most_frequent_words(mismatched_plots)
    print("Most frequent words in mismatched plots:", most_frequent_words)

if __name__ == "__main__":
    main()
