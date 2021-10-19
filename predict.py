import data
from model import NaiveBayes_Model

import pandas as pd

class TextClassificationPredict(object):
    def __init__(self):
        self.test = None
    def get_train_data(self):
        train_data = data.get_dbtrain()
        df_train = pd.DataFrame(train_data)
        
        data_test = []
        data_test.append({"Question": "nói người khác không nghe"})
        df_test = pd.DataFrame(data_test)

        model = NaiveBayes_Model()

        clf = model.clf.fit(df_train["Question"], df_train.Intent)

        predicted = clf.predict(df_test["Question"])
        # Print predicted result
        print (predicted)
        print (clf.predict_proba(df_test["Question"]))

if __name__ == '__main__':
    tcp = TextClassificationPredict()
    tcp.get_train_data()