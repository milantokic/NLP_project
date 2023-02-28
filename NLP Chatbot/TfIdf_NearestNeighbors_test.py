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

N = 25


def transform_vectorizer(new_q, vectorizer):
    return vectorizer.transform([new_q])


# find 10 nearest neighbors using ball tree algorythm
nbrs = NearestNeighbors(n_neighbors=N, algorithm='kd_tree').fit(X)


def similar_questions(new_q, model):
    vector_new = transform_vectorizer(new_q, vectorizer)
    return corpus_q.iloc[nbrs.kneighbors(vector_new)[1].flatten()]


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
    print('--------questionset-------------', int(key) // 4, '---------------')
    print('Variation: ', int(key) % 4)
    print(similar_questions(value, nbrs))