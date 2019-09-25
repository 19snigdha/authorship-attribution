import spacy
import numpy as np
import pandas as pd
nlp = spacy.load('en_vectors_web_lg')
import json
import re
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, SpatialDropout1D
from keras.datasets import imdb
from keras.utils import to_categorical
from keras import backend as K
from keras import layers, Model, models
from keras.layers import Dense,Reshape,concatenate
from keras.layers import Flatten, merge
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate


import pandas as pd
# MAX_DATA = 100
BASE_DIR = ''

#Emoji patterns

emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

# GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B.100d.txt')

def get_ir_data():
    train_data = pd.read_csv("train_tweets.txt", sep="\t", header=None)
    train_tweets = train_data[1]
    train_label = train_data[0]
    # train_tweets= train_tweets[:MAX_DATA]
    # train_label = train_label[:MAX_DATA]
    return train_tweets, train_label

def generate_data_vectors (nlp, tweets, random, max_length):
    sents_as_ids = []
    for sent in tweets:
        sent = re.sub(r'@handle', '', sent)
        sent = re.sub(r'()', '', sent)
        sent = re.sub(r'[^\x00-\x7F]+', '',  sent)
        sent = emoji_pattern.sub(r'', sent)
        sent = re.sub(r"http\S+", "", sent)
        # print(sent)
        doc = nlp(sent)
        word_ids = []
        for i, token in enumerate(doc):
            if token.has_vector and token.vector_norm == 0: continue
            if i > max_length:break
            if token.has_vector: word_ids.append(token.rank + random + 1)
            else: word_ids.append(token.rank % random + 1)

        word_id_vec = np.zeros((max_length), dtype="int")
        clipped_len = min(max_length, len(word_ids))
        word_id_vec[:clipped_len] = word_ids[:clipped_len]
        sents_as_ids.append(word_id_vec)
    return [np.array(sents_as_ids)]

train_tweets, train_label = get_ir_data()
print('this one!')
# labels = map_labels(train_data)
data = generate_data_vectors(nlp, train_tweets, 100, 150)
print(data)
print(data[0].shape)
train_label = to_categorical(train_label)
print(train_label.shape)
# print(set(train_label))
# print(len(set(train_label)))


# def get_embeddings(vocab, embedd_dim):
#     length = max(lex.rank for lex in vocab)+2
#     oov = np.random.normal(size=(embedd_dim, vocab.vectors_length))
#     oov = oov / oov.sum(axis=1, keepdims=True)   
#     embedding_matrix = np.zeros((length + embedd_dim, vocab.vectors_length), dtype='float32')
#     embedding_matrix[1:(embedd_dim + 1), ] = oov
#     for word in vocab:
#         if word.has_vector and word.vector_norm > 0:
#             embedding_matrix[embedd_dim + word.rank + 1] = word.vector / word.vector_norm 

#     return embedding_matrix



def get_embeddings(vocab, embedding_dim):
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    # oov = np.random.normal(size=(embedding_dim, vocab.vectors_length))
    # oov = oov / oov.sum(axis=1, keepdims=True)
    # embedding_matrix[1:(embedding_dim + 1), ] = oov
    for i, word in enumerate(vocab):
        if word.has_vector and word.vector_norm > 0:
            # print(word.vector, "kya hai ye")
            embedding_matrix[i] = word.vector / word.vector_norm
    return embedding_matrix 


embeddings = get_embeddings(nlp.vocab, 300)



print(embeddings.shape)




vocab_size = len(nlp.vocab)
def get_conv_lstm_model(embeddings, vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 300,input_length=150, weights=[embeddings], trainable=False))
    model.add(SpatialDropout1D(0.3))
    model.add(Conv1D(filters=512, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D())
    model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(10001, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



# #Build an LSTM model
# model = Sequential()
# vocab_size = len(nlp.vocab)
# # output_size = len(set(train_label))
# model.add(Embedding(vocab_size, 300,input_length=1000, weights=[embeddings]))
# model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.1))
# model.add(Dense(9978, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
# embed1 = Sequential()
# embed1.add(Embedding(vocab_size, 300, weights=[embeddings], input_length= 150, trainable=False))
# embed2 = Sequential()
# embed2.add(Embedding(vocab_size, 300, weights=[embeddings], input_length=150, trainable=True))
# model = Sequential()
# model = concatenate([embed1, embed2])

# model.add(Merge([embed1, embed2], mode='concat', concat_axis=-1))
# model.add(Reshape((2, 150, 300)))

# ---------------------------------------
# model = Sequential()
# model.add(Embedding(vocab_size, 300,input_length=150, weights=[embeddings], trainable=False))
# model.add(Conv1D(64, 5, activation="relu"))
# model.add(MaxPooling1D(150 - 5 + 1))
# model.add(Flatten())
# model.add(Dense(256, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(10001, activation='softmax'))
# model.compile(loss='categorical_crossentropy',
#                   optimizer='rmsprop',
#                   metrics=['acc'])

# Fit the model
model = get_conv_lstm_model(embeddings, vocab_size)
model.fit(data[0], train_label, epochs=10, batch_size=256, verbose=2)


loss, acc = model.evaluate(data[0], train_label, verbose=0)
print('Test Accuracy: %f' % (acc*100))





# scores, acc = model.evaluate(data[0], train_label, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# print("Training acc", acc)
 
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# model.save_weights("model-lstm.h5")
# print("Saved model to disk")

# import pandas as pd
# from keras.utils import to_categorical
train_data = pd.read_csv("train_tweets.txt", sep="\t", header=None)
train_tweets = train_data[1]
train_label = train_data[0]
columns = to_categorical(train_label)

model_json = model.to_json()
with open("modellstmcnn.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model-cnn-lstm.h5")
print("Saved model to disk")



# test_data = pd.read_csv('test_tweets_unlabeled.txt', sep="\t", header=None)# csv for prediction
# claim_data = (train_data["Claim"].fillna(' ')).tolist()
# evidence_data = (train_data["Evidence"].fillna(' ')).tolist()
# with open('irtest-score.json') as json_f:  
#     final_predict = json.load(json_f)

# test_data = str(test_data)

# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)

# model.load_weights("model-lstm.h5")
# print("Loaded model from disk")


# model.compile(loss='categorical_crossentropy',
#                   optimizer='rmsprop',
#                   metrics=['acc'])


# model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])


# data_to_predict = generate_data_vectors(nlp, test_data, 100, 1000)

# print(data)
# print(data[0].shape)


  
# Predict and write to file
# results = model.predict(data_to_predict[0])
# print('result',results)
# results = pd.DataFrame(results, columns)
# results.insert(1, "id", ids)
# results.to_csv("my_submission.csv", index=False)
# with open('irtest-score.json', 'w') as outfile:  
#     json.dump(final_predict, outfile, indent=4)