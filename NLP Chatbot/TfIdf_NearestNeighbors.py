import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def read_data(path):
    corpus = pd.read_csv(path, sep='\t')
    return corpus


corpus = read_data('insurance_qna_dataset.csv').iloc[:, 1:]
corpus = corpus.groupby('Question', as_index=False).agg(lambda x: np.unique(x).tolist())
corpus_q = corpus['Question']

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus_q)

print('Enter new question')
new_q = input()
N = 25
vector_new = vectorizer.transform([new_q])
# find 10 nearest neighbors using ball tree algorythm
nbrs = NearestNeighbors(n_neighbors=N, algorithm='kd_tree').fit(X)
print(corpus_q.iloc[nbrs.kneighbors(vector_new)[1].flatten()])