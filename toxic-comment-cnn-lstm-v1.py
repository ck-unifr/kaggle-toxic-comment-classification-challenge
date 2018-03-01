# The script is modified from
# https://www.kaggle.com/devm2024/cnn-lstm-eda-lb-0-067/notebook

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from keras.layers import Bidirectional
from keras.preprocessing.text import text_to_word_sequence
from sklearn.model_selection import train_test_split

import codecs
from keras.layers import LSTM, Dropout, Dense, Input, GlobalMaxPooling1D
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Conv1D, Embedding


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#Some data cleaning
train.comment_text = train.comment_text.apply(lambda x : re.sub(' u ', 'you', x))
train.comment_text = train.comment_text.apply(lambda x : re.sub('\nu ', 'you', x))
train.comment_text = train.comment_text.apply(lambda x : re.sub(' u\n', 'you', x))
train.comment_text = train.comment_text.apply(lambda x : re.sub("i'm", 'i am', x))
train.comment_text = train.comment_text.apply(lambda x : re.sub("fucksex", 'fuck sex', x))

test.comment_text = test.comment_text.apply(lambda x : re.sub(' u ', 'you', x))
test.comment_text = test.comment_text.apply(lambda x : re.sub('\nu ', 'you', x))
test.comment_text = test.comment_text.apply(lambda x : re.sub(' u\n', 'you', x))
test.comment_text = test.comment_text.apply(lambda x : re.sub("i'm", 'i am', x))
test.comment_text = test.comment_text.apply(lambda x : re.sub("fucksex", 'fuck sex', x))

#Let's see the words which constitutes the toxic comments
def getwordcountdf(data, key):
    filtered=data[data[key]==1]
    sequence=[]
    tr_words=set(stopwords.words('english'))
    for x in filtered.comment_text:
        sequence+=text_to_word_sequence(x,
                                        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                        lower=True,
                                        split=" ")
    filtered_words = [word for word in sequence if word not in tr_words]
    df= pd.DataFrame({'words': filtered_words})
    z= df.groupby('words').size().reset_index(name='counts')
    return z.sort_values('counts',ascending=False)

obscene=getwordcountdf(train, 'obscene')
toxic=getwordcountdf(train, 'toxic')
severe_toxic=getwordcountdf(train, 'severe_toxic')
threat=getwordcountdf(train, 'threat')
insult=getwordcountdf(train, 'insult')
identity_hate=getwordcountdf(train, 'identity_hate')

# obscene.head(50).plot.bar(x='words', y='counts', figsize=(20,4))
# severe_toxic.head(50).plot.bar(x='words', y='counts', figsize=(20,4))
# threat.head(50).plot.bar(x='words', y='counts', figsize=(20,4))
# insult.head(50).plot.bar(x='words', y='counts', figsize=(20,4))
# identity_hate.head(50).plot.bar(x='words', y='counts', figsize=(20,4))

# remove stop words
def cleanupDoc(s):
    stopset = set(stopwords.words('english'))
    stopset.add('wikipedia')
    tokens =sequence=text_to_word_sequence(s,
                                        filters="\"!'#$%&()*+,-˚˙./:;‘“<=·>?@[]^_`{|}~\t\n",
                                        lower=True,
                                        split=" ")
    cleanup = " ".join(filter(lambda word: word not in stopset, tokens))
    return cleanup

test.comment_text=test.comment_text.apply(cleanupDoc)
train.comment_text=train.comment_text.apply(cleanupDoc)

# Standard tokenization of data to get the indexes of embedding matrix
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train.comment_text)
sequences = tokenizer.texts_to_sequences(train.comment_text)
test_sequence = tokenizer.texts_to_sequences(test.comment_text)

# Pad the sequence so that we have the same length input, since keras supports same length input only
data = pad_sequences(sequences, maxlen=150)
t_data = pad_sequences(test_sequence, maxlen=150)

### Load the pretrained glove vectors
print('Indexing word vectors.')
embeddings_index = {}
# glove.6B.300d.txt
f = codecs.open('glove.42B.300d.txt', encoding='utf-8')
for line in f:
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

vector_length=300
length=150
num_classes=6

word_index = tokenizer.word_index
print('Preparing embedding matrix.')
# prepare embedding matrix
nb_words = min(200000, len(word_index))
notfound=[]
embedding_matrix = np.zeros((nb_words, vector_length))
for word, i in word_index.items():
    if i >= nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        notfound.append(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

print(embedding_matrix.shape)


def get_model():
    inp = Input(shape=(length,))
    x = Embedding(nb_words, vector_length, weights=[embedding_matrix])(inp)
    x = Conv1D(256, 3, activation='relu')(x)
    x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

classes = ['toxic','severe_toxic', 'obscene','threat','insult', 'identity_hate']
labels= train[classes]
data_train, data_test, y_train, y_test, comm_train, comm_trst = train_test_split(data, np.array(labels),train.comment_text, test_size=0.20, random_state=42)

model = get_model()
model.fit(data_train, y_train, 32, epochs=2, validation_data=(data_test, y_test))

# submission script
preds = model.predict(t_data)

sample_submission = pd.read_csv('sample_submission.csv')
sample_submission[classes] = preds
sample_submission.to_csv('submission_cnn_lstm_v1.csv', index=False)