from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_test_setor_gerenciado(dataframe_existing):
    vectorizer = TfidfVectorizer()

    # Preencher valores NaN com string vazia
    dataframe_existing['setor_gerenciado'].fillna('', inplace=True)

    # MARCA PARA MARCA_VAREJISTA
    X = vectorizer.fit_transform(dataframe_existing['nome_produto'])
    y = dataframe_existing['setor_gerenciado']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)
    accuracy = random_forest.score(X_test, y_test)

    return vectorizer, random_forest, accuracy


def process_setor_gerenciado(dataframe_new, vectorizer, random_forest):

    # Prever o 'cod_categoria'
    X_new = vectorizer.transform(dataframe_new['nome_produto'])
    predict = random_forest.predict(X_new)

    # Atualizar coluna 'cod_categoria'
    dataframe_new['setor_gerenciado'] = predict

    return dataframe_new
