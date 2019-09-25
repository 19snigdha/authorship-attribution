import spacy
import numpy as np
nlp = spacy.load('en_vectors_web_lg')
import json
import csv

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb
from keras.utils import to_categorical

import pandas as pd
from keras import backend as K
from keras import layers, Model, models
from keras.models import model_from_json
from keras.models import load_model



json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("model-cnn.h5")
print("Loaded model from disk")
 

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
with open("test_tweets_unlabeled.txt", "r") as f:
    test_data = f.readlines()

# print(f.read())
# test_data = pd.read_csv('test_tweets_unlabeled.txt', sep="\t", header=None)# csv for prediction
# test_data = test_data[0].tolist()
print(len(test_data))



# test_data = str(test_data)

def generate_data_vectors (nlp, tweets, random, max_length):
    sents_as_ids = []
    for sent in tweets:
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


data_to_predict = generate_data_vectors(nlp, test_data, 100, 1000)

print(data_to_predict)
print(data_to_predict[0].shape)

train_data = pd.read_csv("train_tweets.txt", sep="\t", header=None)
train_tweets = train_data[1]
train_label = train_data[0]
columns, indices = np.unique(train_label, return_index=True)

# # new instances where we do not know the answer
# # Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
# # Xnew = scalar.transform(Xnew)
# # # make a prediction
# # ynew = model.predict_classes(Xnew)
# # # show the inputs and predicted outputs
# # for i in range(len(Xnew)):
# # 	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))

# # columns = to_categorical(columns)

# print("col",columns.shape[0]) 
# print(len(test_data)) 


# Predict and write to file
results = model.predict(data_to_predict[0])
print("len of results",len(results))

y_classes = []
y_id = []


for i in range(0,len(test_data)):
    y_id.append(i+1)
    # y_id[i] = i+1
    y_classes.append(np.argmax(results[i]))
     
# print(y_id)

# # final_prediction = pd.DataFrame(y_classes)



# # Df1=pd.DataFrame(columns=[‘Email’])

# # x

print("its finaly done!")


for y_i, y_c in zip(y_id, y_classes):
    fileval = [y_i,y_c]
    with open('my_submission.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(fileval)

csvFile.close()   

	
# prediction = pd.DataFrame(results, columns).to_csv('prediction.csv')

# ynew = model.predict_proba(data_to_predict[0])

# show the inputs and predicted outputs
# for i in range(len(results)):
# 	print("X=%s, Predicted=%s" % (data_to_predict[i], results[i]))
#pd.concat([df1, s1],axis=1)

# results = pd.join([results, columns], axis=1)

# i = 0
# for key in  data_to_predict:
# 	print(i)
# 	y_classes = np.argmax(results[i])
# 	data_to_predict[key]["id"] = columns[y_classes]
# 	i += 1

# results = pd.DataFrame(results, columns)
# results.insert(1, "id", ids)
# results.to_csv("my_submission.csv", index=False)
