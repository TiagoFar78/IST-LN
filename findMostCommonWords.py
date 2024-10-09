import re
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Initialize the PorterStemmer and stopwords set
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to preprocess the text (stopword removal, punctuation removal, and stemming)
def preprocess_text(text):
    # Remove non-alphabetic characters, lowercase, and split into words
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    
    # Tokenize and remove stopwords
    words = [word for word in text.split() if word not in stop_words]
    
    # Apply stemming
    stemmed_words = [stemmer.stem(word) for word in words]
    
    return stemmed_words


def findMostCommonWords():
    # Step 1: Read the file as plain text and filter lines
    filtered_lines = []
    with open('filtered_misclassified_examples.csv', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            # Match lines where true and predicted genres are "drama,romance" or "romance,drama"
            if re.search(r'(drama,romance|romance,drama)$', line):
                filtered_lines.append(line)

    # Step 2: Preprocess each line and extract words
    words = []
    for line in filtered_lines:
        # Only extract the plot text part (assuming it's the first part of the line before the genres)
        plot = line.split(',')[0]  # Adjust this if the plot isn't the first part
        # Preprocess the plot and extract words
        preprocessed_words = preprocess_text(plot)
        words.extend(preprocessed_words)

    # Step 3: Count word frequencies
    word_counts = Counter(words)

    # Step 4: Get the 10 most common words
    most_common_words = word_counts.most_common(10)

    # Output the most common words
    print(most_common_words)

findMostCommonWords()