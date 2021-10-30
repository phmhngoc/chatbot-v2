import test_data
from text_preprocess import text_preprocess
from model import LogisticRegression_Model, NaiveBayes_Model, SVM_Model
import numpy as np
import pandas as pd
from text_preprocess import text_preprocess

class TextClassificationPredict(object):
    def __init__(self):
        self.test = None
    
    def get_train_data(self):
        train_data = test_data.get_dbtrain()
        df_train = pd.DataFrame(train_data)
        model = LogisticRegression_Model()
        model2 = NaiveBayes_Model()
        model3= SVM_Model()
        data_answer = pd.DataFrame(test_data.get_dbanswers())
        # Print predicted result
        while True:
            question = input()
            question = text_preprocess(question)
            # data_test.append({"Question": question})
            df_test = pd.DataFrame([{"Question": text_preprocess(question)}])
            # print(df_train["Question"])
            clf = model.clf.fit(df_train["Question"], df_train.Intent)
            clf1 = model2.clf.fit(df_train["Question"], df_train.Intent)
            clf2= model3.clf.fit(df_train["Question"], df_train.Intent)
            predicted = clf.predict(df_test["Question"])
            predicted2 = clf1.predict(df_test["Question"])
            predicted3= clf2.predict(df_test["Question"])
            if (np.ndarray.max(clf.predict_proba(df_test["Question"])) < 0.5):
                # return ({"predicted": predicted, "proba": np.ndarray.max(clf.predict_proba(df_test["Question"]))})
                print (predicted)
                print (np.ndarray.max(clf.predict_proba(df_test["Question"])))
                print("Nếu không, vui lòng nhập chi tiết hơn")
                print(predicted2)
                print (np.ndarray.max(clf1.predict_proba(df_test["Question"])))
                print(predicted3)
                print (np.ndarray.max(clf2.predict_proba(df_test["Question"])))
            else:
                # return ({"predicted": predicted, "proba": np.ndarray.max(clf.predict_proba(df_test["Question"]))})
                print (predicted)
                print (np.ndarray.max(clf.predict_proba(df_test["Question"])))
                # print(predicted2)
                # print (np.ndarray.max(clf1.predict_proba(df_test["Question"])))
                print(predicted3)
                print (np.ndarray.max(clf2.predict_proba(df_test["Question"])))

if __name__ == '__main__':
    tcp = TextClassificationPredict()
    tcp.get_train_data()