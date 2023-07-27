from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_test_marca(dataframe_existing):
    vectorizer = TfidfVectorizer()

    # MARCA PARA MARCA_VAREJISTA
    X = vectorizer.fit_transform(dataframe_existing['marca'])
    y = dataframe_existing['marca_verejista']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, y_train)
    accuracy = random_forest.score(X_test, y_test)

    return vectorizer, random_forest, accuracy;


def process_marca(dataframe_new, vectorizer, random_forest):

    # Prever as 'marcas_varejistas'
    X_new = vectorizer.transform(dataframe_new['marca'])
    predict = random_forest.predict(X_new)

    # Atualizar coluna 'marca_verejista'
    dataframe_new['marca_verejista'] = predict

    return dataframe_new