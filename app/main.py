import pandas as pd
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from termcolor import colored
from services.data_service import save_updated_dataframe, acurracy, xlsx_to_csv
from services.tamanho_service import train_test_tamanho, process_tamanho
from services.marca_service import train_test_marca, process_marca
from services.segmento_service import train_test_segmento, process_segmento
from services.cod_categoria_service import train_test_cod_categoria, process_cod_categoria
from services.sub_segmento_service import train_test_sub_segmento, process_sub_segmento
from services.setor_gerenciado_service import train_test_setor_gerenciado, process_setor_gerenciado
from services.setor_produto_service import train_test_setor_produto, process_setor_produto
from services.sub_categoria_service import train_test_sub_categoria, process_sub_categoria

# Converter a base xlsx para csv:

xlsx_to_csv('../bases/BASE.xlsx', '../bases/BASE_CONVERTIDA.csv')

# Ler os dados existentes do arquivo CSV
dataframe_existing = pd.read_csv('../bases/BASE_CONVERTIDA.csv')


# Pré-processar os dados existentes e treinar os classificadores
vectorizer_segmento, random_forest_segmento, accuracy_segmento = train_test_segmento(
    dataframe_existing)
vectorizer_sub_segmento, random_forest_sub_segmento, accuracy_sub_segmento = train_test_sub_segmento(
    dataframe_existing)
vectorizer_cod_categoria, random_forest_cod_categoria, accuracy_cod_categoria = train_test_cod_categoria(
    dataframe_existing)
vectorizer_marca, random_forest_marca, accuracy_marca = train_test_marca(
    dataframe_existing)
vectorizer_setor_gerenciado, random_forest_setor_gerenciado, accuracy_setor_gerenciado = train_test_setor_gerenciado(
    dataframe_existing)
vectorizer_setor_produto, random_forest_setor_produto, accuracy_setor_produto = train_test_setor_produto(
    dataframe_existing)
vectorizer_sub_categoria, random_forest_sub_categoria, accuracy_sub_categoria = train_test_sub_categoria(
    dataframe_existing)
vectorizer_tamanho, random_forest_tamanho, accuracy_tamanho = train_test_tamanho(
    dataframe_existing)

# Mostrar a acuracia dos classificadores
acurracy('SEGMENTO', accuracy_segmento)
acurracy('SUB_SEGMENTO', accuracy_sub_segmento)
acurracy('COD_CATEGORIA', accuracy_cod_categoria)
acurracy('MARCA_VAREJISTA', accuracy_marca)
acurracy('SETOR_GERENCIADO', accuracy_setor_gerenciado)
acurracy('SETOR_PRODUTO', accuracy_setor_produto)
acurracy('SUB_CATEGORIA', accuracy_sub_categoria)
acurracy('TAMANHO', accuracy_tamanho)

# Calcular a frequência das palavras
# frequencies = X.sum(axis=0)
# word_frequencies = {word: frequency for word, frequency in zip(
#    vectorizer_sub_segmento.get_feature_names_out(), frequencies.tolist()[0])}
# Criar a nuvem de palavras com max_words=500
# wordcloud = WordCloud(width=400, height=400, background_color='white',
#                      max_font_size=150,  max_words=300).generate_from_frequencies(word_frequencies)
# Exibir a nuvem de palavras
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()

# Converter os novos produtos de xlsx para csv:
xlsx_to_csv('../bases/NOVOS_PRODUTOS.xlsx',
            '../bases/NOVOS_PRODUTOS_CONVERTIDOS.csv')

# Ler os novos produtos a serem adicionados
dataframe_new = pd.read_csv('../bases/NOVOS_PRODUTOS_CONVERTIDOS.csv')

# Processar os novos produtos
processed_dataframe_new = process_segmento(
    dataframe_new, vectorizer_segmento, random_forest_segmento)
processed_dataframe_new = process_sub_segmento(
    dataframe_new, vectorizer_sub_segmento, random_forest_sub_segmento)
processed_dataframe_new = process_cod_categoria(
    dataframe_new, vectorizer_cod_categoria, random_forest_cod_categoria)
processed_dataframe_new = process_marca(
    dataframe_new, vectorizer_marca, random_forest_marca)
processed_dataframe_new = process_setor_gerenciado(
    dataframe_new, vectorizer_setor_gerenciado, random_forest_setor_gerenciado)
processed_dataframe_new = process_setor_produto(
    dataframe_new, vectorizer_setor_produto, random_forest_setor_produto)
processed_dataframe_new = process_sub_categoria(
    dataframe_new, vectorizer_sub_categoria, random_forest_sub_categoria)
processed_dataframe_new = process_tamanho(
    dataframe_new, vectorizer_tamanho, random_forest_tamanho)


# Salvar o DataFrame atualizado
save_updated_dataframe(processed_dataframe_new)
