import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk import word_tokenize
import gensim.downloader
from sklearn.neighbors import NearestNeighbors

nltk.download('punkt')

def read_data(path):
    cr = pd.read_csv(path, sep='\t')
    return cr


porter = PorterStemmer()
wv = gensim.downloader.load('word2vec-google-news-300')


def stem_sent(sent):
    lem_sent = []
    for word in word_tokenize(sent):
        if word.isalnum():
            lem_sent.append(porter.stem(word))
    return " ".join(lem_sent)


def sent_w2v(sent):
    vector_list = []
    for token in word_tokenize(stem_sent(sent)):
        try:
            vector_list.append(wv[token])
        except:
            pass
    return vector_list


def average_vectors(vector_list):
    return np.mean(vector_list, axis=0)


corpus = read_data('insurance_qna_dataset.csv').iloc[:, 1:]
corpus = corpus.groupby('Question', as_index=False).agg(lambda x: np.unique(x).tolist())
corpus_q = corpus['Question']

all_vec = []
for i in range(corpus_q.shape[0]):
    all_vec.append(average_vectors(sent_w2v(corpus_q[i])))
all_vec = np.array(all_vec)
model = NearestNeighbors(n_neighbors=25, algorithm='ball_tree').fit(all_vec)

print("Input question: ")

new_q = input()

new_vec = [average_vectors(sent_w2v(new_q))]

sim_questions = model.kneighbors(new_vec)
print(corpus_q[sim_questions[1].flatten()])
