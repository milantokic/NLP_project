import numpy as np
import pandas as pd
import nltk
from sklearn.neighbors import NearestNeighbors
from nltk.stem import WordNetLemmatizer
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
