import numpy as np
import pandas as pd
import nltk
from sklearn.neighbors import NearestNeighbors
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
import gensim.downloader

nltk.download('punkt')


def read_data(path):
    cr = pd.read_csv(path, sep='\t')
    return cr


lemmatizer = WordNetLemmatizer()
wv = gensim.downloader.load('word2vec-google-news-300')


def lem_sent(sent):
    lem_sentence = []
    for word in word_tokenize(sent):
        if word.isalnum():
            lem_sentence.append(lemmatizer.lemmatize(word))
    return " ".join(lem_sentence)


def sent_w2v(sent):
    vector_list = []
    for token in word_tokenize(lem_sent(sent)):
        try:
            vector_list.append(wv[token])
        except:
            pass
    return vector_list


def sum_vectors(vector_list):
    return np.sum(vector_list, axis=0)


corpus = read_data('insurance_qna_dataset.csv').iloc[:, 1:]
corpus = corpus.groupby('Question', as_index=False).agg(lambda x: np.unique(x).tolist())
corpus_q = corpus['Question']


vectorizer = TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b')
X = vectorizer.fit_transform(corpus_q)


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
        all_vec.append(sum_vectors(add_idf(corpus_q[i])))
    except:
        pass
model = NearestNeighbors(n_neighbors=25, algorithm='kd_tree').fit(all_vec)

print("Input question: ")
new_q = input()

new_vec = [sum_vectors(add_idf(new_q))]

sim_questions = model.kneighbors(new_vec)
print(corpus_q[sim_questions[1].flatten()])
