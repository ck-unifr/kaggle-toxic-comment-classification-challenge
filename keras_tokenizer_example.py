# The script is modified from
# http://www.orbifold.net/default/2017/01/10/embedding-and-tokenizer-in-keras/


from keras.preprocessing.text import Tokenizer

# --------------------
# Tokenizer

# num_words: the maximum number of words to keep, based
#             on word frequency. Only the most common `num_words` words will
#             be kept.

print('--------------------------------')
print('Tokenizer')
print('--------------------------------')

nb_words = 100
tokenizer = Tokenizer(nb_words=nb_words)

tokenizer.fit_on_texts(["The sun is shining in June!","September is grey.","Life is beautiful in August.",
                        "I like it","This and other things?"])
print(tokenizer.word_index)

print(tokenizer.texts_to_sequences(["June is beautiful and I like it!"]))

print(tokenizer.word_counts)

print("Was lower-case applied to %s sentences?: %s"%(tokenizer.document_count, tokenizer.lower))

# If you want to feed sentences to a network you can’t use arrays of variable lengths, corresponding to variable length sentences.
# So, the trick is to use the texts_to_matrix method to convert the sentences directly to equal size arrays:

X = tokenizer.texts_to_matrix(["June is beautiful and I like it!","Like August"])
print(X.shape)


# ---------------------------------------
# Basic network with textual data

# For example, let’s say you want to detect the word ‘shining’ in the sequences above.
# The most basic way would be to use a layer with some nodes like so:

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

# nb_words = 3
# tokenizer = Tokenizer(nb_words=nb_words)
print('--------------------------------')
print('Basic network with textual data')
print('--------------------------------')

tokenizer = Tokenizer()
texts = ["The sun is shining in June!", "September is grey.", "Life is beautiful in August.",
         "I like it","This and other things?"]
tokenizer.fit_on_texts(texts)
print(tokenizer.word_index)
print(tokenizer.texts_to_sequences(["June is beautiful and I like it!"]))
tokenizer.fit_on_texts(texts)

X = tokenizer.texts_to_matrix(texts)
y = [1, 0, 0, 0, 0]

print(X.shape)

vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Dense(2, input_dim=vocab_size))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop')

model.fit(X, y=y, batch_size=32, nb_epoch=10, verbose=0, validation_split=0.2, shuffle=True)

# You can check that this indeed learned the word:
from keras.utils.np_utils import np as np
np.round(model.predict(X))

# You can also do more sophisticated things.
# If the vocabulary is very large the numerical sequences turn into sparse arrays and it’s more efficient to cast
# everything to a lower dimension with the Embedding layer.

# --------------------
# Embedding
# How does embedding work? An example demonstrates best what is going on.
# Assume you have a sparse vector [0,1,0,1,1,0,0] of dimension seven. You can turn it into a non-sparse 2d vector like so:

print('--------------------------------')
print('Embedding')
print('--------------------------------')

X = tokenizer.texts_to_matrix(texts)
y = [1, 0, 0, 0, 0]

model = Sequential()
model.add(Embedding(input_dim=2, output_dim=2, input_length=7))
model.compile('rmsprop', 'mse')
print(model.predict(np.array([[0,1,0,1,1,0,0]])))

# Where do these numbers come from? It’s a simple map from the given range to a 2d space:
# print(model.layers[0].W.get_value())

# If you want to use the embedding it means that the output of the embedding layer will have dimension (5, 19, 10).
# This works well with LSTM or GRU (see below) but if you want a binary classifier
# you need to flatten this to (5, 19*10):

model = Sequential()
model.add(Embedding(nb_words, 10, input_length = X.shape[1]))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop')
model.fit(X, y=y, batch_size=200, nb_epoch=10, verbose=0, validation_split=0.2, shuffle=True)

# It detects ‘shining’ flawlessly:
print(model.predict(X))

# --------------------
# An LSTM layer has historical memory and so the dimension outputted by the embedding works in this case,
# no need to flatten things:

print('--------------------------------')
print('LSTM')
print('--------------------------------')

model = Sequential()
vocab_size = len(tokenizer.word_index) + 1
model.add(Embedding(vocab_size, 10))
model.add(LSTM(5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop')
model.fit(X, y=y, nb_epoch=10, verbose=0, validation_split=0.2, shuffle=True)

# Obviously, it predicts things as well:
print(model.predict(X))

#-------------------------------
# Using word2vec

# The idea is that instead of mapping sequences of integer numbers to sequences of floats happens in a way
# which preserves the semantic affinity. There are various pretrained word2vec datasets on the net,
# we’ll GloVe since it’s small and straightforward but check out the Google repo as well.

embeddings_index = {}
glove_data = 'glove.6B.50d.txt'
f = open(glove_data)
for line in f:
    values = line.split()
    word = values[0]
    value = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = value
f.close()

print('Loaded %s word vectors.' % len(embeddings_index))


embedding_dimension = 10
word_index = tokenizer.word_index

# The embedding_matrix matrix maps words to vectors in the specified embedding dimension (here 100):

embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimension))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector[:embedding_dimension]

print(embedding_matrix.shape)

embedding_layer = Embedding(embedding_matrix.shape[0],
                            embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            input_length=12)

# In order to use this new embedding you need to reshape the training data X to the basic word-to-index sequences:


from keras.preprocessing.sequence import pad_sequences
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=12)

model = Sequential()
model.add(embedding_layer)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.layers[0].trainable=False
model.compile(loss='binary_crossentropy', optimizer='rmsprop')
model.fit(X, y=y, batch_size=20, nb_epoch=700, verbose=0, validation_split=0.2, shuffle=True)

print(model.predict(X))
