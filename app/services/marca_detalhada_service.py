import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_test_marca_detalhada(dataframe_existing):

    vectorizer = TfidfVectorizer()
    # Preencher valores NaN com string vazia
    dataframe_existing['nome_produto'].fillna('', inplace=True)
    dataframe_existing['marca_detalhada'].fillna('', inplace=True)

    # Filtrar linhas com marca_detalhada igual a '-'
    dataframe_existing = dataframe_existing[dataframe_existing['marca_detalhada'] != '-']

    # NOME_PRODUTO PARA marca_detalhada
    X = vectorizer.fit_transform(dataframe_existing['nome_produto'])
    y = dataframe_existing['marca_detalhada']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)
    accuracy = random_forest.score(X_test, y_test)

    return vectorizer, random_forest, accuracy


def process_marca_detalhada(dataframe_new, vectorizer, random_forest):

    # Prever os 'marca_detalhadaS'
    X_new = vectorizer.transform(dataframe_new['nome_produto'])
    predict = random_forest.predict(X_new)

    # Atualizar coluna 'segmento'
    dataframe_new['marca_detalhada'] = predict

    return dataframe_new
