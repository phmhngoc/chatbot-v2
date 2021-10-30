import data
from model import LogisticRegression_Model, SVM_Model
import numpy as np
import pandas as pd
import random
import math


class TextClassificationPredict(object):
    def __init__(self):
        self.test = None
    
    def get_train_data(self, question):
        train_data = data.get_dbtrain()
        df_train = pd.DataFrame(train_data)
        model = LogisticRegression_Model()
        model2 = SVM_Model()
        data_answer = pd.DataFrame(data.get_dbanswers())
        df_test = pd.DataFrame([{"Question": (question)}])
        clf = model.clf.fit(df_train["Question"], df_train.Intent)
        clf1 = model2.clf.fit(df_train["Question"], df_train.Intent)
        predicted = clf.predict(df_test["Question"])
        predicted2 = clf1.predict(df_test["Question"])
        maxPredictProb = np.ndarray.max(clf.predict_proba(df_test["Question"]))
        try:
            if (maxPredictProb < 0.5):
                # return ({"predicted": predicted, "proba": np.ndarray.max(clf.predict_proba(df_test["Question"]))})
                return {"mess": "Vui lòng nhập chi tiết hơn", "predictProb": maxPredictProb}
            else:
                # return ({"predicted": predicted, "proba": np.ndarray.max(clf.predict_proba(df_test["Question"]))})
                print (predicted)
                print (np.ndarray.max(clf.predict_proba(df_test["Question"])))
                print(predicted2)
                s = data_answer.loc[data_answer['Intent'] == " ".join(predicted), 'Answers']
                if(len(s.iat[0])>1):
                    return {"mess": s.iat[0][math.trunc(random.random()*len(s.iat[0]))], "predictProb": maxPredictProb}
                else:
                    return {"mess": s.iat[0], "predictProb": maxPredictProb}
                # print (np.ndarray.max(clf1.predict_proba(df_test["Question"])))
        except ValueError:
            print(ValueError)

            


# if __name__ == '__main__':
#     tcp = TextClassificationPredict()
#     tcp.get_train_data(question)