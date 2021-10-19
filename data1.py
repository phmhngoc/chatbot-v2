import numpy as np
import json
import pickle
from underthesea import word_tokenize
from pyvi import ViTokenizer
import string
# Loading json data
with open('intent1.json') as file:
  data = json.loads(file.read())

labels = []
text = []
for intent in data["intents"]:
  for pattern in intent["patterns"]:
    text.append(pattern)
    labels.append(intent["tag"])

labels=np.array(labels)
text=np.array(text)
	
def clean_document(doc):
    doc = ViTokenizer.tokenize(doc) #Pyvi Vitokenizer library
    doc = doc.lower() #Lower
    tokens = doc.split() #Split in_to words
    table = str.maketrans('', '', string.punctuation.replace("_", "")) #Remove all punctuation
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word]
    return tokens

from sklearn.model_selection import train_test_split
train_txt,test_txt,train_label,test_labels = train_test_split(text,labels,test_size = 0.3)

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
max_num_words = 40000
classes = np.unique(labels)

train_txt_temp = []
for txt in range(len(train_txt)):
  tokens=clean_document(train_txt[txt])
  line = " ".join(tokens)
  train_txt_temp.append(line)
train_txt=np.array(train_txt_temp)
tokenizer = Tokenizer(num_words=max_num_words, filters='!"#$%&()*+,-./:;<=>?@[]^`{|}~ ')
tokenizer.fit_on_texts(train_txt)
word_index = tokenizer.word_index


ls=[]
for c in train_txt:
    ls.append(len(c.split()))
maxLen=int(np.percentile(ls, 98))

train_sequences = tokenizer.texts_to_sequences(train_txt)
train_sequences = pad_sequences(train_sequences, maxlen=maxLen, padding='post')
test_sequences = tokenizer.texts_to_sequences(test_txt)
test_sequences = pad_sequences(test_sequences, maxlen=maxLen, padding='post')

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(classes)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoder.fit(integer_encoded)
train_label_encoded = label_encoder.transform(train_label)
train_label_encoded = train_label_encoded.reshape(len(train_label_encoded), 1)
train_label = onehot_encoder.transform(train_label_encoded)
test_labels_encoded = label_encoder.transform(test_labels)
test_labels_encoded = test_labels_encoded.reshape(len(test_labels_encoded), 1)
test_labels = onehot_encoder.transform(test_labels_encoded)

embeddings_index={}
with open('word2vec_vi_words_100dims.txt', encoding='utf8') as f:
    for line in f:
      values = line.split()
      word = values[0]
      values = np.where(type(values)!=string)
      coefs = np.asarray(values, dtype='float32')
      embeddings_index[word] = coefs

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
num_words = min(max_num_words, len(word_index))+1
embedding_dim=100
embedding_matrix = np.random.normal(emb_mean, emb_std, (num_words, embedding_dim))
for word, i in word_index.items():
    if i >= max_num_words:
        break
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Input, Dropout, CuDNNLSTM, Activation, Bidirectional,Embedding

model = Sequential()
model.add(Embedding(num_words, 100, trainable=False,input_length=train_sequences.shape[1], weights=[embedding_matrix]))
model.add(Bidirectional(CuDNNLSTM(256, return_sequences=True), 'concat'))
model.add(Dropout(0.3))
model.add(CuDNNLSTM(256, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(classes.shape[0], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(train_sequences, train_label, epochs = 30,
          batch_size = 64, shuffle=True,
          validation_data=[test_sequences, test_labels])

model.save('intents.h5')

with open('classes.pkl','wb') as file:
   pickle.dump(classes,file)

with open('tokenizer.pkl','wb') as file:
   pickle.dump(tokenizer,file)

with open('label_encoder.pkl','wb') as file:
   pickle.dump(label_encoder,file)