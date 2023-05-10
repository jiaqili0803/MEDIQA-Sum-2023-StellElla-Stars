import numpy as np
np.random.seed(712)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Model
from keras.layers import Embedding, Input, Flatten, concatenate, LSTM, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout, SpatialDropout1D, GRU, Dense
from keras.utils.np_utils import to_categorical
from keras import callbacks
%matplotlib inline


# read data
train_data = pd.read_csv('TaskA-TrainingSet.csv')
valid_data = pd.read_csv('TaskA-ValidationSet.csv')
test_data = pd.read_csv('taskA_testset4participants_headers_inputConversations.csv')

train_X = train_data['dialogue']
valid_X = valid_data['dialogue']
test_X = test_data['dialogue']

# transform categories to numbers (and then back)
sec_to_num = {}
num_to_sec = {}
label_num = -1
for section in set(train_data['section_header'].unique()):
  label_num += 1
  sec_to_num[section] = label_num
  num_to_sec[label_num] = section
  
train_data['label'] = train_data['section_header'].map(sec_to_num)
valid_data['label'] = valid_data['section_header'].map(sec_to_num)

train_y = train_data['label']
valid_y = valid_data['label']
train_y = to_categorical(np.asarray(train_y))
valid_y = to_categorical(np.asarray(valid_y))

# data preprocessing
MAX_NB_WORDS = 20000

## raw text data
train_texts = train_X.astype(str)
valid_texts = valid_X.astype(str)
test_texts = test_X.astype(str)

## vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words = MAX_NB_WORDS, char_level = False)
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
valid_sequences = tokenizer.texts_to_sequences(valid_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

word_index = tokenizer.word_index
index_to_word = dict((i, w) for w, i in tokenizer.word_index.items())

MAX_SEQUENCE_LENGTH = 512
## pad sequences with 0s
train_X = pad_sequences(train_sequences, maxlen = MAX_SEQUENCE_LENGTH)
valid_X = pad_sequences(valid_sequences, maxlen = MAX_SEQUENCE_LENGTH)
test_X = pad_sequences(test_sequences, maxlen = MAX_SEQUENCE_LENGTH)

# oversampling
#!pip install -U imbalanced-learn
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.pipeline import make_pipeline

#smote = SMOTE(random_state = 712)
random = RandomOverSampler(random_state = 712)
#my_pipe = make_pipeline(smote, random)
train_X_resampled, train_y_resampled = random.fit_resample(train_X, train_y)

EMBEDDING_DIM = 512
N_CLASSES = 20


# CBOW model
sequence_input = Input(shape = (MAX_SEQUENCE_LENGTH,), dtype = 'int32')
embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM,
                            input_length = MAX_SEQUENCE_LENGTH,
                            trainable = True)
embedded_sequences = embedding_layer(sequence_input)

average = GlobalAveragePooling1D()(embedded_sequences)
predictions = Dense(N_CLASSES, activation = 'softmax')(average)

model = Model(sequence_input, predictions)
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam', metrics = ['acc'])


earlystopping = callbacks.EarlyStopping(monitor = 'val_loss',
                                            patience = 5,
                                            restore_best_weights = True)

filepath = "weights_best_cbow.hdf5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor = 'val_acc', save_best_only = True, mode = 'max', save_weights_only = True)

#callback = [earlystopping, checkpoint]
callback = [checkpoint]

#model.fit(train_X, train_y, validation_split = 0.1, epochs = 150, batch_size = 128, callbacks = callback)
model.fit(train_X_resampled, train_y_resampled, validation_split = 0.1, epochs = 150, batch_size = 128, callbacks = callback)

model.load_weights("weights_best_cbow.hdf5")

# validation
loss, accuracy = model.evaluate(valid_X, valid_y)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

# test data prediction
y_proba = model.predict(test_X)
y_classes = y_proba.argmax(axis = -1)

# output
test_data.rename({'ID': 'TestID'}, axis = 1, inplace = True)
test_data.drop('dialogue', axis = 1, inplace = True)
test_data.set_index('TestID', inplace = True)
test_data['output'] = y_classes
test_data['SystemOutput'] = test_data['output'].map(num_to_sec)
test_data.drop('output', axis = 1, inplace = True)

test_data.to_csv('taskA_StellEllaStars_run2_mediqaSum.csv')
