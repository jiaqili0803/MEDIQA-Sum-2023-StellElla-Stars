import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import nltk
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import warnings
warnings.filterwarnings("ignore")

# read data for command line
parser = argparse.ArgumentParser(description='Classification task')
parser.add_argument('--train_csv_file', type=str, required=True, help='Path to the training CSV file')
parser.add_argument('--valid_csv_file', type=str, required=True, help='Path to the validation CSV file')
parser.add_argument('--test_csv_file', type=str, required=True, help='Path to the test CSV file')
parser.add_argument('--output_csv_file', type=str, required=True, help='Path to the output CSV file')
args = parser.parse_args()

train_data = pd.read_csv(args.train_csv_file)
valid_data = pd.read_csv(args.valid_csv_file)
test_data = pd.read_csv(args.test_csv_file)

# read data
#train_data = pd.read_csv('TaskA-TrainingSet.csv')
#valid_data = pd.read_csv('TaskA-ValidationSet.csv')
#test_data = pd.read_csv('taskA_testset4participants_headers_inputConversations.csv')

# assign data and labels
X_train = train_data['dialogue']
y_train = train_data['section_header']
X_valid = valid_data['dialogue']
y_valid = valid_data['section_header']
X_test = test_data['dialogue']

# data cleaning/preprocessing
stop_words = set(stopwords.words('english'))
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

def clean_data(text):
    
    # remove "Doctor:" and "Patient:" labels and timestamps
    text = re.sub(r"(Doctor|Patient|Guest_family):|\d{1,2}[:.]\d{1,2}\s?(AM|PM|am|pm)?", "", text)
    
    # lowercase the text
    text = text.lower()

    # remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]+", "", text)

    # tokenize the text
    words = nltk.word_tokenize(text)

    # remove stop words
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    # lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # join words back into a single string
    text = " ".join(words)

    return text

X_train = X_train.apply(clean_data)
X_valid = X_valid.apply(clean_data)
X_test = X_test.apply(clean_data)

## tfidf_vec = TfidfVectorizer(stop_words='english')
tfidf_vec = TfidfVectorizer(stop_words='english',
                            ngram_range=(1, 2),
                            max_df=0.9,
                            # min_df=2,
                            #max_features=300
                           )
X_train = tfidf_vec.fit_transform(X_train)
X_valid = tfidf_vec.transform(X_valid)
X_test = tfidf_vec.transform(X_test)

## oversampling
ros = RandomOverSampler(random_state = 712)
X_train, y_train = ros.fit_resample(X_train, y_train)


# LogisticRegression model
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
# label prediction of validation data
y_pred = model_lr.predict(X_valid)
report = classification_report(y_valid, y_pred)
print(report)

y_test = model_lr.predict(X_test)
test_data.rename({'ID': 'TestID'}, axis = 1, inplace = True)
test_data.drop('dialogue', axis = 1, inplace = True)
test_data.set_index('TestID', inplace = True)
test_data['SystemOutput'] = y_test

#test_data.to_csv('taskA_StellEllaStars_run3_mediqaSum.csv')
test_data.to_csv(args.output_csv_file)

## LogisticRegression GridSearchCV
#param_grid = {
#    'C': [0.1, 1, 10],
#    'penalty': ['l1', 'l2'],
#    'solver': ['liblinear', 'saga'],
#    'max_iter': [100, 200, 500]
#}
#model_lr = LogisticRegression()
#grid_search = GridSearchCV(model_lr, param_grid, cv = 5)
#grid_search.fit(X_train, y_train)
#print("Best hyperparameters: ", grid_search.best_params_)
#print("Best score: ", grid_search.best_score_)
#best_model = grid_search.best_estimator_
#y_pred = best_model.predict(X_valid)
#report = classification_report(y_valid, y_pred)
#print(report)

