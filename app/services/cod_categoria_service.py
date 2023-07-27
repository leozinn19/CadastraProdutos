from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_test_cod_categoria(dataframe_existing):
    vectorizer = TfidfVectorizer()

    # Excluindo as linhas que não possuem valor para o teste:
    dataframe_existing.dropna(subset=['cod_categoria'], inplace=True)

    # MARCA PARA MARCA_VAREJISTA
    X = vectorizer.fit_transform(dataframe_existing['segmento'])
    y = dataframe_existing['cod_categoria']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)
    accuracy = random_forest.score(X_test, y_test)

    return vectorizer, random_forest, accuracy


def process_cod_categoria(dataframe_new, vectorizer, random_forest):
    # Preencher valores NaN com string vazia
    dataframe_new['segmento'].fillna('', inplace=True)

    # Prever o 'cod_categoria'
    X_new = vectorizer.transform(dataframe_new['segmento'])
    predict = random_forest.predict(X_new)

    # Atualizar coluna 'cod_categoria'
    dataframe_new['cod_categoria'] = predict

    return dataframe_new
