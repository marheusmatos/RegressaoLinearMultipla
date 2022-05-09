"""Respostas do trabalho 2."""
import pandas as pd
import matplotlib.pyplot as plt


def plot_graph(data: pd.DataFrame, col_x: str, col_y: str):
    """Plota em um gráfico a relação das duas variáveis."""
    x = data[col_x]
    y = data[col_y]
    print(f'    [ A ] Plotando o gráfico de {col_x} por {col_y}.')
    plt.scatter(x, y)
    plt.title(f'{col_x} x {col_y}')
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.show()


def get_correlation(data: pd.DataFrame, col_1: str, col_2: str):
    """Calcula a correlação de duas variáveis."""
    x = data[col_1]
    y = data[col_2]
    n = len(x)
    avg_x = sum(x) / len(x)
    avg_y = sum(y) / len(y)

    cov = sum([(x[i] - avg_x) * (y[i] - avg_y) for i in range(len(x))]) / n

    var_x = sum([(x[i] - avg_x) ** 2 for i in range(len(x))]) / (len(x) - 1)
    var_y = sum([(y[i] - avg_y) ** 2 for i in range(len(y))]) / (len(y) - 1)

    rho = round(cov / ((var_x * var_y) ** .5), 6)
    print(f'    [ B ] A correlação entre {col_1} e {col_2} é {rho}.')
    return rho


def exec_(alinea_a=False,
          alinea_b=False):
    """Executa todas as alineas do trabalho.
    aliena_a -> Caso True executa o solicitado na alinea A.
    aliena_b -> Caso True executa o solicitado na alinea B.
    """
    if alinea_a:
        # Alíena (A)
        # Plota o gráfico para a variável Volume
        plot_graph(data, 'Volume', 'CO2')
        # Plota o gráfico para a variável Peso
        plot_graph(data, 'Peso', 'CO2')
    # ======================================================================= #
    if alinea_b:
        # Alínea (B)
        # Estima a correlação entre a variavel Y (CO2)
        # e a variável regressora 1 (Volume)
        get_correlation(data, 'Volume', 'CO2')
        """
            Comentario sobre Relação entre Volume e CO2.
            (Variável Regressora 1 e variável Y)
            A correlação entre Volume e CO2 é 0.575635.
            Com isso, podemos afirmar que existe uma correlação positiva entre
            as variáveis, apesar de não ser muito forte, cerca de 57,6% de
            correlação se olharmos em questão de porcentagem.
        """
        # Estima a correlação entre a variavel Y (CO2)
        # e a variável regressora 2 (Peso)
        get_correlation(data, 'Peso', 'CO2')
        """
            Comentario sobre Relação entre Peso e CO2.
            (Variável Regressora 2 e variável Y)
            A correlação entre Peso e CO2 é 0.536813.
            Como na primeira variável regressora, podemos afirmar que existe
            uma correlação positiva entre asvariáveis, nesse caso um pouco mais
            fraca cerca de 53,7% de correlação se olharmos em questão de
            porcentagem.
        """
    # ======================================================================= #

    return


if __name__ == '__main__':
    data = pd.read_csv("cars.csv")
    executar_alinea_a = True
    executar_alinea_b = True

    exec_(
        executar_alinea_a,
        executar_alinea_b)
