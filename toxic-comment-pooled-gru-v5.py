#Forked from https://github.com/PavelOstyakov/toxic/blob/master/fit_predict.py
# python fit_predict.py train.csv test.csv crawl-300d-2M.vec


import argparse
import numpy as np
import os
import pandas as pd

import nltk
import tqdm

from keras.layers import Dense, Embedding, Input
from keras.layers import Bidirectional, Dropout, CuDNNGRU
from keras.models import Model
from keras.optimizers import RMSprop

from sklearn.metrics import log_loss

def _train_model(model, batch_size, train_x, train_y, val_x, val_y):
    best_loss = -1
    best_weights = None
    best_epoch = 0

    current_epoch = 0

    while True:
        model.fit(train_x, train_y, batch_size=batch_size, epochs=1)
        y_pred = model.predict(val_x, batch_size=batch_size)

        total_loss = 0
        for j in range(6):
            loss = log_loss(val_y[:, j], y_pred[:, j])
            total_loss += loss

        total_loss /= 6.

        print("Epoch {0} loss {1} best_loss {2}".format(current_epoch, total_loss, best_loss))

        current_epoch += 1
        if total_loss < best_loss or best_loss == -1:
            best_loss = total_loss
            best_weights = model.get_weights()
            best_epoch = current_epoch
        else:
            if current_epoch - best_epoch == 5:
                break

    model.set_weights(best_weights)
    return model


def train_folds(X, y, fold_count, batch_size, get_model_func):
    fold_size = len(X) // fold_count
    models = []
    for fold_id in range(0, fold_count):
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_size - 1:
            fold_end = len(X)

        train_x = np.concatenate([X[:fold_start], X[fold_end:]])
        train_y = np.concatenate([y[:fold_start], y[fold_end:]])

        val_x = X[fold_start:fold_end]
        val_y = y[fold_start:fold_end]

        model = _train_model(get_model_func(), batch_size, train_x, train_y, val_x, val_y)
        models.append(model)

    return models

def get_model(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True))(embedding_layer)
    x = Dropout(dropout_rate)(x)
    x = Bidirectional(CuDNNGRU(recurrent_units, return_sequences=False))(x)
    x = Dense(dense_size, activation="relu")(x)
    output_layer = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(clipvalue=1, clipnorm=1),
                  metrics=['accuracy'])

    return model

def read_embedding_list(file_path):
    embedding_word_dict = {}
    embedding_list = []
    with open(file_path) as f:
        for row in tqdm.tqdm(f.read().split("\n")[1:-1]):
            data = row.split(" ")
            word = data[0]
            embedding = np.array([float(num) for num in data[1:-1]])
            embedding_list.append(embedding)
            embedding_word_dict[word] = len(embedding_word_dict)

    embedding_list = np.array(embedding_list)
    return embedding_list, embedding_word_dict


def clear_embedding_list(embedding_list, embedding_word_dict, words_dict):
    cleared_embedding_list = []
    cleared_embedding_word_dict = {}

    for word in words_dict:
        if word not in embedding_word_dict:
            continue
        word_id = embedding_word_dict[word]
        row = embedding_list[word_id]
        cleared_embedding_list.append(row)
        cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)

    return cleared_embedding_list, cleared_embedding_word_dict


def convert_tokens_to_ids(tokenized_sentences, words_list, embedding_word_dict, sentences_length):
    words_train = []

    for sentence in tokenized_sentences:
        current_words = []
        for word_index in sentence:
            word = words_list[word_index]
            word_id = embedding_word_dict.get(word, len(embedding_word_dict) - 2)
            current_words.append(word_id)

        if len(current_words) >= sentences_length:
            current_words = current_words[:sentences_length]
        else:
            current_words += [len(embedding_word_dict) - 1] * (sentences_length - len(current_words))
        words_train.append(current_words)
    return words_train

def tokenize_sentences(sentences, words_dict):
    tokenized_sentences = []
    for sentence in tqdm.tqdm(sentences):
        if hasattr(sentence, "decode"):
            sentence = sentence.decode("utf-8")
        tokens = nltk.tokenize.word_tokenize(sentence)
        result = []
        for word in tokens:
            word = word.lower()
            if word not in words_dict:
                words_dict[word] = len(words_dict)
            word_index = words_dict[word]
            result.append(word_index)
        tokenized_sentences.append(result)
    return tokenized_sentences, words_dict


UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

PROBABILITIES_NORMALIZE_COEFFICIENT = 1.4

SENTENCE_LENGTH = 500

DROPOUT = 0.2
RECURRENT_UNITS = 64
DENSE_SIZE = 32

BATCH_SIZE = 256

RESULT_PATH = "toxic_results"

def main():
    # parser = argparse.ArgumentParser(
    #     description="Recurrent neural network for identifying and classifying toxic online comments")
    #
    # parser.add_argument("train_file_path")
    # parser.add_argument("test_file_path")
    # parser.add_argument("embedding_path")
    # parser.add_argument("--result-path", default="toxic_results")
    # parser.add_argument("--batch-size", type=int, default=256)
    # parser.add_argument("--sentences-length", type=int, default=500)
    # parser.add_argument("--recurrent-units", type=int, default=64)
    # parser.add_argument("--dropout-rate", type=float, default=0.3)
    # parser.add_argument("--dense-size", type=int, default=32)
    # parser.add_argument("--fold-count", type=int, default=10)
    #
    # args = parser.parse_args()
    #
    # if args.fold_count <= 1:
    #     raise ValueError("fold-count should be more than 1")

    print("Loading data...")
    # train_data = pd.read_csv(args.train_file_path)
    # test_data = pd.read_csv(args.test_file_path)

    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')


    list_sentences_train = train_data["comment_text"].fillna(NAN_WORD).values
    list_sentences_test = test_data["comment_text"].fillna(NAN_WORD).values
    y_train = train_data[CLASSES].values

    print("Tokenizing sentences in train set...")
    tokenized_sentences_train, words_dict = tokenize_sentences(list_sentences_train, {})

    print("Tokenizing sentences in test set...")
    tokenized_sentences_test, words_dict = tokenize_sentences(list_sentences_test, words_dict)

    words_dict[UNKNOWN_WORD] = len(words_dict)

    print("Loading embeddings...")

    # embedding_list, embedding_word_dict = read_embedding_list(args.embedding_path)
    embedding_list, embedding_word_dict = read_embedding_list('crawl-300d-2M.vec')

    embedding_size = len(embedding_list[0])

    print("Preparing data...")
    embedding_list, embedding_word_dict = clear_embedding_list(embedding_list, embedding_word_dict, words_dict)

    embedding_word_dict[UNKNOWN_WORD] = len(embedding_word_dict)
    embedding_list.append([0.] * embedding_size)
    embedding_word_dict[END_WORD] = len(embedding_word_dict)
    embedding_list.append([-1.] * embedding_size)

    embedding_matrix = np.array(embedding_list)

    id_to_word = dict((id, word) for word, id in words_dict.items())

    # train_list_of_token_ids = convert_tokens_to_ids(
    #     tokenized_sentences_train,
    #     id_to_word,
    #     embedding_word_dict,
    #     args.sentences_length)
    train_list_of_token_ids = convert_tokens_to_ids(
        tokenized_sentences_train,
        id_to_word,
        embedding_word_dict,
        SENTENCE_LENGTH)

    # test_list_of_token_ids = convert_tokens_to_ids(
    #     tokenized_sentences_test,
    #     id_to_word,
    #     embedding_word_dict,
    #     args.sentences_length)
    test_list_of_token_ids = convert_tokens_to_ids(
        tokenized_sentences_test,
        id_to_word,
        embedding_word_dict,
        SENTENCE_LENGTH)

    X_train = np.array(train_list_of_token_ids)
    X_test = np.array(test_list_of_token_ids)

    # get_model_func = lambda: get_model(
    #     embedding_matrix,
    #     args.sentences_length,
    #     args.dropout_rate,
    #     args.recurrent_units,
    #     args.dense_size)
    get_model_func = lambda: get_model(
        embedding_matrix,
        SENTENCE_LENGTH,
        DROPOUT,
        RECURRENT_UNITS,
        DENSE_SIZE)


    print("Starting to train models...")
    # models = train_folds(X_train, y_train, args.fold_count, args.batch_size, get_model_func)
    models = train_folds(X_train, y_train, 10, BATCH_SIZE, get_model_func)


    # if not os.path.exists(args.result_path):
    #     os.mkdir(args.result_path)

    print("Predicting results...")
    test_predicts_list = []
    for fold_id, model in enumerate(models):
        # model_path = os.path.join(args.result_path, "model{0}_weights.npy".format(fold_id))
        model_path = "model{0}_weights.npy".format(fold_id)

        np.save(model_path, model.get_weights())

        # test_predicts_path = os.path.join(args.result_path, "test_predicts{0}.npy".format(fold_id))
        test_predicts_path = "test_predicts{0}.npy".format(fold_id)

        test_predicts = model.predict(X_test, batch_size = BATCH_SIZE)
        test_predicts_list.append(test_predicts)
        np.save(test_predicts_path, test_predicts)

    test_predicts = np.ones(test_predicts_list[0].shape)
    for fold_predict in test_predicts_list:
        test_predicts *= fold_predict

    test_predicts **= (1. / len(test_predicts_list))
    test_predicts **= PROBABILITIES_NORMALIZE_COEFFICIENT

    test_ids = test_data["id"].values
    test_ids = test_ids.reshape((len(test_ids), 1))

    test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
    test_predicts["id"] = test_ids
    test_predicts = test_predicts[["id"] + CLASSES]

    # submit_path = os.path.join(args.result_path, "submit")
    submit_path = "submission-pooled-gru-v5.csv"

    test_predicts.to_csv(submit_path, index=False)

if __name__ == "__main__":
    main()
