import data
from model import NaiveBayes_Model
import numpy as np
import pandas as pd

class TextClassificationPredict(object):
    def __init__(self):
        self.test = None
    
    def get_train_data(self, question):
        train_data = data.get_dbtrain()
        df_train = pd.DataFrame(train_data)
        model = NaiveBayes_Model()
        data_answer = pd.DataFrame(data.get_dbanswers())
        # Print predicted result
        # while True:
        # question = input()
        # data_test.append({"Question": question})
        df_test = pd.DataFrame([{"Question": question}])
        clf = model.clf.fit(df_train["Question"], df_train.Intent)
        predicted = clf.predict(df_test["Question"]).tolist()
        if (np.ndarray.max(clf.predict_proba(df_test["Question"])) < 0.5):
            return ({"predicted": predicted, "proba": np.ndarray.max(clf.predict_proba(df_test["Question"]))})
            # print (predicted)
            # print("Nếu không, vui lòng nhập chi tiết hơn")
            # print (np.ndarray.max(clf.predict_proba(df_test["Question"])))
        else:
            return ({"predicted": predicted, "proba": np.ndarray.max(clf.predict_proba(df_test["Question"]))})


# if __name__ == '__main__':
#     tcp = TextClassificationPredict()
    # tcp.get_train_data()