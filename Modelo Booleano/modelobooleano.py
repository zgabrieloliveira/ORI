import nltk
import os
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer


# 1. STOPWORDS, EXTRAÇÃO DE RADICAIS E ÍNDICE INVERTIDO

nltk.download('stopwords')
nltk.download('rslp')
nltk.download('punkt')
nltk.download('mac_morpho')

stop_words = set(stopwords.words('portuguese')) # descarte de stopwords
stemmer = RSLPStemmer() # extração de radicais 
inverted_index = {} # "CHAVE: radical, VALOR: {documento em que aparece, quantidade de aparições}"


# 2. ENTRADA DO PROGRAMA 

if len(sys.argv) != 3:
    print("\nUso correto: python boolean_model.py base_samba/base.txt consulta.txt")
    sys.exit(1)

database_file_path = sys.argv[1] # argumento 1: arquivo da base de dados

if not os.path.isfile(database_file_path):
    print(f"\nArquivo não foi encontrado em {database_file_path}")
    sys.exit(1)

path_list = [] # lista dos caminhos dos arquivos na base de dados
with open(database_file_path, 'r') as database_file:
    for file_line in database_file:
        path_list.append(file_line.strip())

query_file_path = sys.argv[2] # argumento 2: arquivo da consulta

if not os.path.isfile(query_file_path):
    print(f"Arquivo de consulta não foi encontrado em {query_file_path}")
    sys.exit(1)


# 3. CRIANDO ÍNDICE INVERTIDO 

stems = [] # lista dos radicais de todos os arquivos
for path in path_list:
    with open(path, 'r') as file:
        for file_line in file:
            tokens = word_tokenize(file_line.lower()) # separando as palavras do arquivo atual
            for token in tokens:
                if token not in stop_words: # descartando stopwords
                    stems.append(stemmer.stem(token))  # extraindo radical e adicionando à lista

# preenchendo índice com frequência dos termos no documento atual 
for i, path in enumerate(path_list):
    with open(path, 'r') as file:
        term_count = {}  # dicionário para contagem de termos no documento
        for file_line in file:
            tokens = word_tokenize(file_line.lower()) # separando palavras
            for token in tokens:
                if token not in stop_words: # descartando stopwords
                    term = stemmer.stem(token)  # extraindo radical
                    
                    if term in term_count: # se o termo já estiver no dicionário, incrementa frequência
                        term_count[term] += 1
                    else:
                        term_count[term] = 1 # senão, ele é adicionado e contabiliza frequência 1
                        
        # adicionando contagem de ocorrência dos termos considerando todos os documentos da base
        for term, term_countage in term_count.items():
            
            # caso o termo já esteja no índice invertido
            if term in inverted_index:
                if i in inverted_index[term]: # se o termo já foi encontrado em outros documentos, incrementa contagem no documento atual
                    inverted_index[term][i] += term_countage
                else: # se é primeira ocorrência do termo no documento atual, então cria uma nova entrada 
                    inverted_index[term][i] = term_countage
                    
            # caso o termo não esteja no índice invertido, adiciona o termo ao índice com sua contagem no documento atual
            else:
                inverted_index[term] = {i: term_countage}


# criando arquivo indice.txt
with open('indice.txt', 'w') as index_file:
    for term, occurrences in inverted_index.items():
        # criando pares "documento:contagem"
        occurrences_str = " ".join([f"{doc + 1},{term_countage}" for doc, term_countage in occurrences.items()])
        index_file.write(f"{term}: {occurrences_str}\n")


# 4. CONSULTA 

# interpreta consulta, busca seus termos no índice invertido e retorna documentos correspondentes
def consult_index(query):
    query_terms = word_tokenize(query.lower())
    result = set() 
    logical_operator = None
    current_operand = set() # conjunto de termos do operador atual

    for term in query_terms:
        if term in {'&', '|', '!'}:
            logical_operator = term
        else:
            term = stemmer.stem(term)  # extrai o radical do termo que não é um operador
            term_docs = set(inverted_index.get(term, [])) # obtém os documentos onde ocorre o radical

            # None, atualiza o conjunto de resultados com term_docs
            if logical_operator is None:
                result |= term_docs
            # &, faz a interseção entre o operando atual e term_docs
            elif logical_operator == '&':
                current_operand &= term_docs
            # |, faz a união entre o operando atual e term_docs
            elif logical_operator == '|':
                current_operand |= term_docs
            # !, subtrai term_docs do operando atual
            elif logical_operator == '!':
                current_operand -= term_docs

    result |= current_operand  # atualiza o conjunto de resultados com o operando atual
    return result

# abrindo documento da consulta
with open(query_file_path, 'r') as query_file:
    query = query_file.readline().strip()

query_result = consult_index(query) # realizando busca através do índice invertido
recovered_docs = len(query_result)


# 5. SAÍDA DO PROGRAMA

print("\nÍndice Invertido:")
with open('indice.txt', 'r') as ii_file:
    content = ii_file.read()
    print(content)

# criando arquivo resposta.txt
with open('resposta.txt', 'w') as answer_file:
    answer_file.write(f"{recovered_docs}\n")
    for i in query_result:
        if i < len(path_list):
            answer_file.write(f"{path_list[i]}\n") # registra caminho do(s) documento(s) correspondente(s)

print(f"\nNúmero total de documentos que atendem à consulta: {recovered_docs}")

print("\nCaminhos dos documentos que atendem à consulta:")
for i in query_result:
    if i < len(path_list):
        print(path_list[i])
        
print("\n")