import sys, os, re, csv, codecs, numpy as np, pandas as pd

import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

from sklearn.linear_model import Ridge

path = ''
comp = ''
EMBEDDING_FILE = 'glove.6B.50d.txt'
TRAIN_DATA_FILE = 'train.csv'
TEST_DATA_FILE = 'test.csv'

BATCH_SIZE = 64
EPOCHS = 50


embed_size = 50        # how big is each word vector
max_features = 20000   # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100           # max number of words in a comment to use

train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


def get_coefs(word,*arr):
    return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()


word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

def get_bi_lstm_model(maxlen, max_features, embed_size, embedding_matrix):
    inp = Input(shape=(maxlen,))

    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

    x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    # x = GRU(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(x)
    x = GlobalMaxPool1D()(x)

    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(32, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def aggregate_predicts(Y1, Y2):
    assert Y1.shape == Y2.shape
    ratio = 0.63
    return Y1 * ratio + Y2 * (1.0 - ratio)

# print("Fitting Ridge model on training examples...")
# ridge_model = Ridge(
#     solver='auto', fit_intercept=True, alpha=0.5,
#     max_iter=100, normalize=False, tol=0.05,
# )
# ridge_model.fit(X_train, Y_train)


model = get_bi_lstm_model(maxlen, max_features, embed_size, embedding_matrix)
model.fit(X_t, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)


y_test = model.predict([X_te], batch_size=1024, verbose=1)
print(y_test[0])

sample_submission = pd.read_csv('sample_submission.csv')
sample_submission[list_classes] = y_test

sample_submission.to_csv('submission.csv', index=False)




