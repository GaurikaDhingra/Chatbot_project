

#install necessary libraries 
import pandas as pd

#load dataset 
data_test = pd.read_csv(r"D:/Gaurika Dhingra/Gaurika_CS/Chatbot_Project/datasets/test.csv")
data_train = pd.read_csv(r"D:/Gaurika Dhingra/Gaurika_CS/Chatbot_Project/datasets/training.csv")
data_val = pd.read_csv(r"D:/Gaurika Dhingra/Gaurika_CS/Chatbot_Project/datasets/validation.csv")

## data features:-  
### text - string values 
### label - a classification label, with possible values including 
###         sadness (0), joy (1), love (2), anger (3), fear (4).

print(data_test.info())
print(data_train.info())
print(data_val.info())


#check for null values 
print(data_test.isnull().sum())
print(data_train.isnull().sum())
print(data_val.isnull().sum())


## DATA PRE - PROCESSING  
import string 
import nltk 
from nltk.corpus import stopwords 


## convert text into lower case 
data_test['text'] = data_test['text'].str.lower()
data_train['text'] = data_train['text'].str.lower()
data_val['text'] = data_val['text'].str.lower()

###remove punctuation
punc_remove = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', punc_remove))

data_test['text'] = data_test['text'].apply(lambda text: remove_punctuation(text))
data_train['text'] = data_train['text'].apply(lambda text: remove_punctuation(text))
data_val['text'] = data_train['text'].apply(lambda text: remove_punctuation(text))

##stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return ' '.join(word for word in text.split() if word.lower() not in stop_words)

data_test['text'] = data_test['text'].apply(remove_stopwords)
data_train['text'] = data_train['text'].apply(remove_stopwords)
data_val['text'] = data_val['text'].apply(remove_stopwords)



#Tokenization 
from nltk.tokenize import word_tokenize

data_test['text'] = data_test['text'].apply(word_tokenize)
data_train['text'] = data_train['text'].apply(word_tokenize)
data_val['text'] = data_val['text'].apply(word_tokenize)



# Extract features (X) and labels (y) from the preprocessed text
X_train = data_train['text']
y_train = data_train['label']
X_val = data_val['text']
y_val = data_val['label']
X_test = data_test['text']
y_test = data_test['label']


# Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(data_train['text'])
X_val = vectorizer.transform(data_val['text'])
X_test = vectorizer.transform(data_test['text'])


# Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Validation Performance
y_val_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

# Test Performance
y_test_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))


# Save Model and Vectorizer
import joblib
# Save Model and Vectorizer
joblib.dump(model, "emotion_classifier_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Confirm the file is saved
print("Model saved as 'emotion_classifier_model.pkl'")

