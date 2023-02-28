import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk import word_tokenize
import gensim.downloader
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')


def read_data(path):
    cr = pd.read_csv(path, sep='\t')
    return cr


corpus = read_data('insurance_qna_dataset.csv').iloc[:, 1:]
corpus = corpus.groupby('Question', as_index=False).agg(lambda x: np.unique(x).tolist())
corpus_q = corpus['Question']

vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b')
X = vectorizer.fit_transform(corpus_q)

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
            vector_list.append(np.zeros(300))
    return vector_list


def average_vectors(vector_list):
    return np.mean(vector_list, axis=0)


idf_dict = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))


def find_idf(sent, idf_dict):
    idf_list = []
    for token in word_tokenize(sent):
        if token.isalnum():
            idf_list.append(idf_dict.get(token.lower()))

    return idf_list


def add_idf(sent):
    new_vec = []
    for i in range(len(sent_w2v(sent))):
        new_vec.append(sent_w2v(sent)[1] * find_idf(sent, idf_dict)[1])
    return new_vec


all_vec = []
for i in range(corpus_q.shape[0]):
    try:
        all_vec.append(average_vectors(add_idf(corpus_q[i])))
    except:
        pass
all_vec = np.array(all_vec)
model = NearestNeighbors(n_neighbors=25, algorithm='ball_tree').fit(all_vec)

print("Input question: ")

new_q = input()

new_vec = [average_vectors(add_idf(new_q))]

sim_questions = model.kneighbors(new_vec)
print(corpus_q[sim_questions[1].flatten()])