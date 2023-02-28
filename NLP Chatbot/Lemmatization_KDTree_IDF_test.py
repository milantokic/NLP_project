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

def transform_new_question(vector):
    return [sum_vectors(add_idf(vector))]


def similar_questions(ques, model):
    vector = transform_new_question(ques)
    sim_questions = model.kneighbors(vector)
    return corpus_q[sim_questions[1].flatten()]


test_questions = {'0': 'What Is The Best Life Insurance To Buy? ',
                  '1': 'What Is The Comprehensive Life Insurance To Buy? ',
                  '2': 'What Is The Comprehensive Life Insurance To Purchase? ',
                  '3': 'What Is The Most Outstanding Guarantee To Purchase? ',

                  '4': 'How Much Does A Whole Life Insurance Policy Typically Cost? ',
                  '5': 'How Much Does A Whole Life Insurance Policy Usually Cost? ',
                  '6': 'What Is The Typically Price Of Whole Life Insurance Policy? ',
                  '7': 'At What Price Could Life Insurance Be Bought? ',

                  '8': 'Who Should Get Critical Illness Insurance? ',
                  '9': 'Who Should Buy Critical Illness Insurance? ',
                  '10': 'Critical Illness Insurance Should Be Got By Who? ',
                  '11': 'Which People Should Buy Censorious Illness Insurance? ',

                  '12': 'Where To Buy Good Life Insurance? ',
                  '13': 'Where To Get Quality Life Insurance? ',
                  '14': 'Where Is The Best Place To Buy Quality Life Insurance? ',
                  '15': 'Which Is The Best Insurance Company to Get Satisfying Life Insurance? ',

                  '16': 'How Can I Find Who My Car Insurance Is With? ',
                  '17': 'How To Find Who My Car Insurance Is With? ',
                  '18': 'How To Find Out Which Is My Vehicle Insurance Company? ',
                  '19': 'At Which Company My Vehicle Is Insured? ',

                  '20': 'What Does Home Insurance Cost? ',
                  '21': 'What Does Property Insurance Cost? ',
                  '22': 'What Is The Price Of Home Insurance? ',
                  '23': 'What Does Insuring My House Cost? ',

                  '24': 'What Is The Purpose Of Life Insurance? ',
                  '25': 'What Is The Motive Of Life Insurance? ',
                  '26': 'What Is The Idea Of Life Indemnity? ',
                  '27': 'What Is The Idea Of Buying Life Indemnity? ',

                  '28': 'Are New Cars Cheaper To Insure Than Older Cars? ',
                  '29': 'Are New Vehicles Cheaper To Insure Than Older? ',
                  '30': 'Is It Cheaper To Insure, Old Or New Cars? ',
                  '31': 'Is It More Expensive To Insure New Vehicles Or Old? ',

                  '32': 'What Is The Best Disability Insurance Company? ',
                  '33': 'What Is The Best Invalidity Insurance Company? ',
                  '34': 'What Is The Recommended Invalidity Insurance Company? ',
                  '35': 'At What Company Should I Buy Disability Insurance? ',

                  '36': 'How Much Money Does A Life Insurance Salesman Make? ',
                  '37': 'How Much Cash Does A Life Protections Sales Representative Make? ',
                  '38': 'How Much Cash Does A Life Assurances Deals Agent Make? ',
                  '39': 'What Is The Amount Of Cash That A Life Protections Sales Make? '
                  }

for key, value in test_questions.items():
    print(f'Question: {value} ')
    print('--------question set-------------', int(key) // 4, '---------------')
    print('Variation: ', int(key) % 4)
    print(similar_questions(value, model))