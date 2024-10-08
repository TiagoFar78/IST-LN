import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import re
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import nltk

# Download stopwords if not already available
nltk.download('stopwords')

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocessing function: lowercase, remove punctuation, and optionally stopwords
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove stopwords
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in words]

    #stop_words = set(stopwords.words('english'))
    #text = " ".join([word for word in lemmatized_words if word not in stop_words])
    
    return ' '.join(lemmatized_words)

# Load and preprocess data
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            title, movie_from, genre, director, plot = line.strip().split('\t')
            plot = preprocess_text(plot)
            data.append((plot, genre))
    return pd.DataFrame(data, columns=["plot", "genre"])

train_data = load_data('train.txt')

# Step 1: Perform stratified train-test split (80% train, 20% test)
train_df, test_df = train_test_split(train_data, test_size=0.2, stratify=train_data['genre'], random_state=42)

# Step 2: Reduce the size of the training and test sets separately while maintaining proportionality
train_df = train_df.groupby('genre').sample(frac=0.1, random_state=42).reset_index(drop=True)
test_df = test_df.groupby('genre').sample(frac=0.1, random_state=42).reset_index(drop=True)

# BERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the data
train_encodings = tokenizer(list(train_df['plot']), truncation=True, padding=True)
test_encodings = tokenizer(list(test_df['plot']), truncation=True, padding=True)

# Create dataset class
class MovieGenreDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Map genres to integer labels
genre_mapping = {genre: i for i, genre in enumerate(train_data['genre'].unique())}
train_labels = [genre_mapping[genre] for genre in train_df['genre']]
test_labels = [genre_mapping[genre] for genre in test_df['genre']]

# Create datasets
train_dataset = MovieGenreDataset(train_encodings, train_labels)
test_dataset = MovieGenreDataset(test_encodings, test_labels)

# Load BERT model for classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(genre_mapping))

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",  # Evaluate at the end of each epoch
    logging_dir='./logs',
    weight_decay=0.01,
)

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=lambda p: {'accuracy': accuracy_score(p.label_ids, p.predictions.argmax(-1))}
)

# Train the model
trainer.train()

# Evaluate the model on the test set
eval_results = trainer.evaluate()

# Print accuracy
print(f"Test Accuracy: {eval_results['eval_accuracy']:.4f}")

# Predict on test set
test_predictions = trainer.predict(test_dataset)
predicted_labels = test_predictions.predictions.argmax(-1)

# Confusion Matrix
conf_matrix = confusion_matrix(test_labels, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=genre_mapping.keys(), yticklabels=genre_mapping.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
