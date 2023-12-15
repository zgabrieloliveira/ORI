import nltk
import os
import sys
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer

# 1. REMOÇÃO DE STOPWORDS, RADICAIS E TOKENIZAÇÃO

nltk.download('stopwords')
nltk.download('rslp')
nltk.download('punkt')

stop_words = set(stopwords.words('portuguese'))
stemmer = RSLPStemmer()
documents = []  # lista para armazenar documentos processados
inverted_index = {}

# util: tokenização das palavras, remoção de stopwords e extração de radicais
def preprocess_text(text):
    words = word_tokenize(text, language='portuguese')
    words = [stemmer.stem(word) for word in words if word not in stop_words and word.isalnum()]
    return words

# util: cálculo do Term Frequency-Inverse Document Frequency (TF-IDF)
def calculate_tf_idf(term, document, document_frequency, total_documents, inverted_index):
    term_frequency = document.count(term)
    tf = 1 + math.log(term_frequency, 10)
    idf = math.log(total_documents / (document_frequency[term]), 10)
    tf_idf = tf * idf
    return tf_idf

# 2. REGRAS DA CONSULTA

# obtenção dos nomes de arquivos da base e processamento dos documentos
current_directory = os.getcwd()
base_folder = "base1"
base_filename = "base.txt"
query_filename = "consulta.txt" # basta alterar aqui para alterar o arquivo de consulta

if len(sys.argv) != 3:
    print("\nUso correto: python modessa.py base.txt consulta.txt")
    sys.exit(1)

# caminhos para o arquivo da base de documentos e consulta
database_file_path = os.path.join(current_directory, base_folder, base_filename)
query_file_path = os.path.join(current_directory, base_folder, query_filename)

# verificação da existência dos arquivos
if not os.path.isfile(database_file_path):
    print(f"Arquivo de base não encontrado em {database_file_path}")
    sys.exit(1)

if not os.path.isfile(query_file_path):
    print(f"Arquivo de consulta não encontrado em {query_file_path}")
    sys.exit(1)

# leitura e processamento dos documentos da base
with open(database_file_path, 'r') as database_file:
    for file_line in database_file:
        doc_filename = file_line.strip().replace("\n", "")
        doc_path = os.path.join(base_folder, doc_filename)
        with open(doc_path, 'r', encoding='utf-8') as doc_file:
            text = doc_file.read()
            words = preprocess_text(text)
            doc_id = os.path.basename(doc_filename)
            documents.append((doc_id, words))

# 3. CONSTRUÇÃO DO ÍNDICE INVERTIDO

document_frequency = {}
total_documents = len(documents)

for doc_id, words in documents:
    unique_words = set(words)  # remove duplicatas para calcular o IDF
    for word in unique_words:
        if word not in document_frequency:
            document_frequency[word] = 1 
        else:
            document_frequency[word] += 1

for doc_id, words in documents:
    doc_length = len(words)
    for word in set(words):  # garantia termos únicos no documento
        tf_idf = calculate_tf_idf(word, words, document_frequency, total_documents, inverted_index)
        if tf_idf > 0:  # se tf-idf maior que 0, verifica índice invertido
            if word not in inverted_index:
                inverted_index[word] = {}
            inverted_index[word][doc_id] = tf_idf

print(inverted_index)

# 4. PROCESSAMENTO DA CONSULTA

with open(query_file_path, 'r', encoding='utf-8') as query_file:
    query = query_file.read()
    query_words = preprocess_text(query)

query_tf_idf = {}  # dicionário para armazenar os pesos TF-IDF dos termos da consulta

# cálculo dos pesos TF-IDF para a consulta
for term in query_words:
    if term in inverted_index:
        tf_idf = calculate_tf_idf(term, query_words, document_frequency, total_documents, inverted_index)
        query_tf_idf[term] = tf_idf

# 5. CÁLCULO DE SIMILARIDADE

document_similarities = {}  # dicionário para armazenar as similaridades entre a consulta e os documentos

# cálculo da similaridade entre a consulta e os documentos
for doc_id, words in documents:
    similarity = 0
    for term in query_words:
        if term in inverted_index and doc_id in inverted_index[term]:
            term_frequency = words.count(term)
            tf_doc = 1 + math.log(term_frequency, 10)
            idf_doc = math.log(total_documents / (len(inverted_index[term])), 10)
            tf_query = query_tf_idf.get(term, 0)
            similarity += tf_doc * tf_query  # calcula o produto TF-IDF da consulta e do documento
                    
    if similarity > 0:
        denominator_doc = 0  # calcula o denominador da similaridade para normalização
        for term in words:
            if term in inverted_index and doc_id in inverted_index[term]:
                term_frequency = words.count(term)
                tf_doc = 1 + math.log(term_frequency, 10)
                idf_doc = math.log(total_documents / (len(inverted_index[term])), 10)
                denominator_doc += (tf_doc * tf_doc)  # acumula o valor do denominador
        similarity /= (math.sqrt(denominator_doc))  # normaliza a similaridade
        document_similarities[doc_id] = similarity
        
# 6. SAÍDA DO PROGRAMA

# criação de arquivos de saída (pesos.txt e resposta.txt)
with open("pesos.txt", 'w', encoding='utf-8') as output_file:
    for doc_id, words in documents:
        output_file.write(f"{doc_id}: ")
        terms_processed = set()  # p/ evitar repetição de termos
        for term in words:
            if term in inverted_index and term not in terms_processed:
                terms_processed.add(term)
                output_file.write(f"{term}, {inverted_index[term][doc_id]:.10f}  ")
        output_file.write("\n")

with open("resposta.txt", 'w', encoding='utf-8') as output_file:
    output_file.write(f"{len(document_similarities)}\n")
    for doc_id, similarity in document_similarities.items():
        output_file.write(f"{doc_id}: {similarity:.10f}\n")

# leitura e impressão do conteúdo dos arquivos de saída
with open("pesos.txt", 'r', encoding='utf-8') as peso_file:
    peso_content = peso_file.read()
    print("\nConteúdo do arquivo pesos.txt:")
    print(peso_content)

with open("resposta.txt", 'r', encoding='utf-8') as resposta_file:
    resposta_content = resposta_file.read()
    print("Conteúdo do arquivo resposta.txt:")
    print(resposta_content)