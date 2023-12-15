import sys
import matplotlib.pyplot as plt


# função que gere a entrada do programa, checando parâmetros necessários e lendo a entrada
def input_handler():

    # entrada recebe 2 argumentos
    if len(sys.argv) != 2:
        print("Uso: python3 avaliacao.py <arquivo_entrada>")
        sys.exit(1)

    input_file = sys.argv[1]

    with open(input_file, 'r') as file:
        # nº referente à quantidade de consultas
        query_number = int(file.readline().strip())
        # com base no nº de consultas, lê as linhas correspondentes e referencia como respostas ideais
        ideal_answers = [list(map(int, file.readline().strip().split())) for _ in range(query_number)]
        # linhas restantes se referem às respostas do sistema (1 linha = 1 resposta p/ consulta)
        system_answers = [list(map(int, file.readline().strip().split())) for _ in range(query_number)]

    return query_number, ideal_answers, system_answers

# função que realiza o cálculo de precisão e revocação
def precision_recall_handler(query_number, ideal_answers, system_answers):

    # lista para armazenar resultados de precisão e revocação de todas as consultas
    final_results = []

    # laço para iterar sobre cada query separadamente
    for current_query in range(query_number):

        # obtendo os dados da query atual do laço
        ideal_answer = ideal_answers[current_query]
        system_answer = system_answers[current_query]

        relevant_docs = set(ideal_answer)
        retrieved_docs = set()

        # lista para armazenar resultados de precisão e revocação para a consulta atual
        query_results = []

        # laço para iterar sobre os termos da query atual
        for doc in system_answer:

            retrieved_docs.add(doc) # adicionando em docs recuperados

            # calcula somente se o documento recuperado estiver no conjunto dos relevantes
            if (relevant_docs.__contains__(doc)):
                intersectionRA = relevant_docs.intersection(retrieved_docs)
                # (R ∩ A) / A - fração de documentos recuperados que é relevante
                precision = len(intersectionRA) / len(retrieved_docs) if len(retrieved_docs) > 0 else 0
                # (R ∩ A) / R - fração dos documentos relevantes que é recuperada
                recall = len(intersectionRA) / len(relevant_docs)

                query_results.append((precision, recall))

        # adiciona os resultados da consulta atual à lista principal
        final_results.append(query_results)

    return final_results

# função que realiza a interpolação do gráfico
def interpolation_handler(recall_levels, results):

    # lista para as precisões máximas obtidas pela regra de interpolação
    max_precisions = []

    # laço para iterar sobre cada resultado separadamente
    for idx, current_result in enumerate(results):
        max_precisions_current_result = [] # lista para as precisões máximas do resultado atual

        # realizando interpolação para cada nível de revocação (10% em 10%)
        for level in recall_levels:
            max_precision = 0 # precisão máxima iniciada em 0 por padrão
            added = False  # para rastrear se um valor já foi adicionado em max_precisions
            # acessando cada tupla de resultados separadamente e obter a maior precisão da tupla
            for tup in current_result:
                # só considera precisões maiores que o nível atual da revocação
                if tup[1] >= level:
                    max_precision = max(max_precision, tup[0])
                    added = True

            # adicionando na lista de precisões máximas do resultado atual
            # de acordo com bool added
            if added:
                max_precisions_current_result.append(max_precision)
            # adicionando valor padrão 0, caso contrário
            else:
                max_precisions_current_result.append(0)

        # adicionando resultados da iteração atual na lista final de precisões máximas
        max_precisions.append(max_precisions_current_result)

        # plotando gráfico através da função
        graphic_title = f'Gráficos de Precisão versus Revocação para Consulta {idx + 1}'
        plot_handler(graphic_title, recall_levels, max_precisions_current_result)

    return max_precisions

# função que realiza a plotagem de gráficos de acordo com os parâmetros recebidos
def plot_handler(graphic_title, recall_levels, precisions):

    plt.plot(recall_levels, precisions, marker='o')
    plt.xlabel('Revocação')
    plt.ylabel('Precisão')
    plt.title(graphic_title)
    plt.grid(True)
    plt.show()

# função que gere a saída do programa, gravando a precisão média de cada nível de revocação
def output_handler(recall_levels, max_precisions):

    # titulo do gráfico de média
    title = 'Gráfico de Precisão Média versus Revocação'
    # agrupa os elementos correspondentes à cada consulta
    grouped_precisions = zip(*max_precisions)
    # média da precisão entre cada elemento de cada consulta
    avg_precisions = [sum(group) / len(group) for group in grouped_precisions]

    # escrevendo resultados no arquivo
    with open('media.txt', 'w') as media_file:
        for avg in avg_precisions:
            media_file.write(f"{avg}\n")

    # lendo media.txt para obter seus valores do arquivo
    with open('media.txt', 'r') as file:
        averages = [float(line.strip()) for line in file]

    # plotando gráfico de acordo com os dados calculados
    plot_handler(title, recall_levels, averages)

def main():

    # 1. ENTRADA DO PROGRAMA
    query_number, ideal_answers, system_answers = input_handler()

    # 2. CÁLCULO DE PRECISÃO E REVOCAÇÃO
    results = precision_recall_handler(query_number, ideal_answers, system_answers)

    # 3. INTERPOLAÇÃO E PLOTAGEM DE GRÁFICOS
    recall_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    max_precisions = interpolation_handler(recall_levels, results)
    output_handler(recall_levels, max_precisions)

# executar a função principal se o script for executado diretamente
if __name__ == "__main__":
    main()