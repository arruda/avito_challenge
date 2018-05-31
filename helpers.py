
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from rake_nltk import Rake
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

import pandas as pd


def normalized_predict(y_predict):
    min_y_predict = min(y_predict)

    if min_y_predict < 0:
        neg_fix = (min_y_predict * -1)
        y_predict = y_predict + neg_fix

    max_y_predict = max(y_predict)

    return (y_predict - min_y_predict) / (max_y_predict - min_y_predict)


def limit_value(value):
    if value > 1:
        value = 1
    elif value < 0:
        value = 0
    return value


# trocando para que colunas com valores em russo sejam substituidas por valores de 0-N
# acho q isso vai facilitar na hora de acessar essas colunas
# ao inves de tentar pegar pelos diferentes valores de cada uma
def convert_columns_with_russian_values_to_numbers(df, cols):
    transform_dict = {}
    for col in cols:
        cats = pd.Categorical(df[col]).categories
        d = {}
        for i, cat in enumerate(cats):
            d[cat] = f'{i}'
        transform_dict[col] = d
    inverse_transform_dict = {}
    for col, d in transform_dict.items():
        inverse_transform_dict[col] = {v: k for k, v in d.items()}

    return df.replace(transform_dict), transform_dict, inverse_transform_dict


def convert_deal_probability_to_class(value):
    if value * 100 <= (100 / 3 * 1):
        return 'low'
    elif value * 100 <= (100 / 3 * 2):
        return 'average'
    else:
        return 'high'


nltk.download('stopwords')


def prepare_vectorizer_and_col(df, col_name):
    vectorizer = CountVectorizer(lowercase=True, stop_words=stopwords.words('russian'))
    corpus = df[col_name]
    col_vectorized = vectorizer.fit_transform(corpus)
    return col_vectorized, vectorizer


def return_description_reduced(df):
    # Uses stopwords for russian from NLTK, and all puntuation characters.
    r = Rake(language='russian')

    total = df['description'].shape[0]

    verbose_step_size = int(total / 10)
    description_reduced = []
    for i, description in enumerate(df['description']):
        r.extract_keywords_from_text(description)
        phrases_and_scores = r.get_ranked_phrases_with_scores()
        mean_score = np.mean([x[0] for x in phrases_and_scores])

        highest_phrases = [x[1] for x in phrases_and_scores if x[0] >= mean_score]
        reduced_description = ' '.join(highest_phrases)
        description_reduced.append(reduced_description)

        if i % verbose_step_size == 0:
            print(f'{int(i/total*100)}')

    return description_reduced


def save_description_linear_regression_model(pkl_filename, model):
    save_model(pkl_filename, model)


def load_description_linear_regression_model(pkl_filename):
    return load_model(pkl_filename)


def save_model(pkl_filename, model):
    with open(pkl_filename, 'wb') as f:
        pickle.dump(model, f)


def load_model(pkl_filename):
    with open(pkl_filename, 'rb') as f:
        pickle_model = pickle.load(f)
        return pickle_model


# train_frac == 0.34 ~= 510000 ~= total de exemplos no test_csv.zip
def get_train_test_samples(df, seed, train_frac=0.34):
    train_dfs = []
    test_dfs = []
    for klass in ['low', 'average', 'high']:
        klass_df = df[df['deal_prob_class'] == klass]
        klass_train, klass_test = train_test_split(
            klass_df, test_size=(1 - train_frac), random_state=seed
        )
        train_dfs.append(klass_train)
        test_dfs.append(klass_test)

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    return train_df, test_df
