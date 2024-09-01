# wget https://raw.githubusercontent.com/subashgandyer/datasets/main/financial_news_headlines_sentiment.csv use this code in terminal to download
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from gensim.models import Doc2Vec
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import nltk
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('punkt')

def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) <= 0:
                continue
            tokens.append(word.lower())
    return tokens

# Load the data
df = pd.read_csv('financial_news_headlines_sentiment.csv', delimiter=',', encoding='latin-1')

# Rename the columns
df = df.rename(columns={'neutral':'sentiment', 'According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .':'Message'})

# Convert string sentiment to numeric
sentiment  = {'positive': 0,'neutral': 1,'negative':2}
df.sentiment = [sentiment[item] for item in df.sentiment]

# Clean the text
df['Message'] = df['Message'].apply(cleanText)

# Plot the sentiment distribution
cnt_pro = df['sentiment'].value_counts()
plt.figure(figsize=(12,4))
sns.barplot(x=cnt_pro.index, y=cnt_pro.values, alpha=0.8, palette='pastel')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('sentiment', fontsize=12)
plt.xticks(rotation=90)
plt.show()

# Separate features and target variable
X = df['Message']
y = df['sentiment']

# Convert the text data to numerical data
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Apply SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_vectorized, y)

# The resampled data is in a sparse matrix format. We can convert it back to a DataFrame as follows:
X_resampled_df = pd.DataFrame(X_resampled.toarray(), columns=vectorizer.get_feature_names_out())
y_resampled_df = pd.DataFrame(y_resampled, columns=['sentiment'])

# Concatenate the resampled features and target variable
df_resampled = pd.concat([X_resampled_df, y_resampled_df], axis=1)

# Convert the resampled data back to original form
X_original = vectorizer.inverse_transform(X_resampled)
df_resampled_original = pd.DataFrame({'Message': [' '.join(text) for text in X_original], 'sentiment': y_resampled})

# Map the numerical sentiment labels back to their original string representations
sentiment_mapping = {0: 'positive', 1: 'neutral', 2: 'negative'}
df_resampled_original['sentiment'] = df_resampled_original['sentiment'].map(sentiment_mapping)

# Fit the tokenizer on the 'Message' column of the resampled DataFrame
tokenizer = Tokenizer(num_words=500000, split=' ', filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df_resampled_original['Message'].values)

# Convert the text to sequences
X = tokenizer.texts_to_sequences(df_resampled_original['Message'].values)
X = pad_sequences(X, maxlen=50)

# Use the 'sentiment' column as the target variable
y = df_resampled['sentiment'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(500000, embed_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax')) # 3 output classes for sentiment
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
batch_size = 32
model.fit(X_train, y_train, epochs=10, batch_size=batch_size, validation_split=0.2)

# Predict sentiment for testing data
y_pred_lstm = model.predict_classes(X_test)

# Print accuracy, classification report and confusion matrix for LSTM
print("\nAccuracy for LSTM:", accuracy_score(y_test, y_pred_lstm))
print("\nClassification Report for LSTM:\n", classification_report(y_test, y_pred_lstm))
print("\nConfusion Matrix for LSTM:\n", confusion_matrix(y_test, y_pred_lstm))

# Prepare training data in doc2vec format
train_tagged = df_resampled_original.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Message']), tags=[r.sentiment]), axis=1)

# Initialize a Doc2Vec model
model_d2v = Doc2Vec(dm=1, # Use distributed memory (PV-DM)
                    vector_size=300, # Size of the document vectors
                    min_count=1, # Ignore words with frequency less than this
                    workers=4, # Number of worker threads to train the model
                    epochs=20) # Number of iterations (epochs) over the corpus

# Build vocabulary from the training data
model_d2v.build_vocab(train_tagged)

# Train the Doc2Vec model
model_d2v.train(train_tagged, total_examples=model_d2v.corpus_count, epochs=model_d2v.epochs)

# Infer vectors for training documents 
X_train_d2v = np.array([model_d2v.infer_vector(doc.words) for doc in train_tagged])

# Train a logistic regression model on the Doc2Vec vectors
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_d2v, df_resampled_original['sentiment']) # Use the inferred vectors for training data

# Prepare testing data in doc2vec format
test_tagged = df_resampled_original.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['Message']), tags=[r.sentiment]), axis=1)

# Infer vectors for testing documents
X_test_d2v = np.array([model_d2v.infer_vector(doc.words) for doc in test_tagged])

# Predict sentiment for testing data
y_pred_d2v = clf.predict(X_test_d2v)

# Print accuracy, classification report and confusion matrix for Doc2Vec
print("\nAccuracy for Doc2Vec:", accuracy_score(y_test, y_pred_d2v))
print("\nClassification Report for Doc2Vec:\n", classification_report(y_test, y_pred_d2v))
print("\nConfusion Matrix for Doc2Vec:\n", confusion_matrix(y_test, y_pred_d2v))
