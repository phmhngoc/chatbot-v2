from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from transformer import FeatureTransformer
# from sklearn.svm import SVC
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


with open('intent1.json') as file:
  data = json.loads(file.read())

# test_percent = 0.2
 
# text = []
# label = []

# for intent in data["intents"]:
#     for pattern in intent["patterns"]:
#         label.append(intent['tag'])
#         text.append(pattern)

# # print(text)       
# X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=test_percent, random_state=42)

# label_encoder = LabelEncoder()
# label_encoder.fit(y_train)
# y_train = label_encoder.transform(y_train)
# y_test = label_encoder.transform(y_test)

# print(X_train)
class NaiveBayes_Model(object):
    def __init__(self):
        self.clf = self._init_pipeline()

    @staticmethod
    def _init_pipeline():
        pipe_line = Pipeline([
            ("transformer", FeatureTransformer()),#sử dụng pyvi tiến hành word segmentation
            ("vect", CountVectorizer(ngram_range=(1,1),
                                             max_df=0.8,
                                             max_features=None)),#bag-of-words
            ("tfidf", TfidfTransformer()),#tf-idf
            ("clf", MultinomialNB())#model naive bayes
        ])
        return pipe_line

class LogisticRegression_Model(object):
    def __init__(self):
        self.clf = self._init_pipeline()

    @staticmethod
    def _init_pipeline():
        pipe_line = Pipeline([
            ("transformer", FeatureTransformer()),
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", LogisticRegression(C=212.10, max_iter=10000, solver='lbfgs', multi_class='multinomial'))
        ])

        return pipe_line

# class SVM_Model(object):
#     def __init__(self):
#         self.clf = self._init_pipeline()

#     @staticmethod
#     def _init_pipeline():
#         pipe_line = Pipeline([
#             ("transformer", FeatureTransformer()),
#             ("vect", CountVectorizer()),
#             ("tfidf", TfidfTransformer()),
#             ("clf", SVC(kernel='sigmoid', C=500, gamma='scale', probability=True, class_weight='balanced'))
#         ])
#         return pipe_line

# model = LogisticRegression_Model()
# clf = model.clf.fit(X_train, y_train)
# model2=NaiveBayes_Model()
# clf2=model2.clf.fit(X_train, y_train)
# pickle.dump(clf, 'logistic.pkl', 'wb')
# pickle.dump(clf2, 'naive.pkl', 'wb')
