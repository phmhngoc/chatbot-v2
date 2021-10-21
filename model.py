from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from transformer import FeatureTransformer


class NaiveBayes_Model(object):
    def __init__(self):
        self.clf = self._init_pipeline()

    @staticmethod
    def _init_pipeline():
        pipe_line = Pipeline([
            ("transformer", FeatureTransformer()),#sử dụng pyvi tiến hành word segmentation
            ("vect", CountVectorizer()),#bag-of-words
            ("tfidf", TfidfTransformer()),#tf-idf
            ("clf", MultinomialNB(alpha=0.1))#model naive bayes
        ])
        return pipe_line
