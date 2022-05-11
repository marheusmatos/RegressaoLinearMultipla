"""Respostas do trabalho 2."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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


def mrlm(data: pd.DataFrame):
    """Função que realiza o metodo.
        data -> Base de dados.
    """
    x = pd.concat([
        pd.DataFrame([1 for _ in range(len(data))]),
        data.loc[:, ['Volume', 'Peso']]
    ], axis=1)
    y = data['CO2']

    x = np.matrix(x)
    y = np.reshape(np.matrix(y), (-1, 1))

    transp_x = x.transpose()

    beta_ = np.matmul(
        np.matmul(
            np.linalg.inv(np.matmul(transp_x, x)),
            transp_x),
        y)

    print(f'    [ D ] beta_0 = {round(float(beta_[0]), 6)}')
    print(f'    [ D ] beta_1 = {round(float(beta_[1]), 6)}')
    print(f'    [ D ] beta_2 = {round(float(beta_[2]), 6)}')
    print('    [ D ] O Hiperplano é: y = ' +
          f'{round(float(beta_[0]), 6)} + ' +
          f'{round(float(beta_[1]), 6)}*x1 + ' +
          f'{round(float(beta_[2]), 6)}*x2')
    return beta_


def get_y_(data: pd.DataFrame, beta_: np.matrix):
    """Calcula os valores estimados de y para o hiperplano (y chapéu)."""
    beta_0 = round(float(beta_[0]), 6)
    beta_1 = round(float(beta_[1]), 6)
    beta_2 = round(float(beta_[2]), 6)
    x1 = data['Volume'].values
    x2 = data['Peso'].values
    y_ = np.reshape(
        np.matrix([
            beta_0 + beta_1 * x1[i] + beta_2 * x2[i]
            for i in range(len(data))]),
        (-1, 1))
    y_list = [round(val[0], 6) for val in y_.tolist()]
    print(f'    [ E ] y^ = {y_list}')
    return y_


def get_res(data: pd.DataFrame, y_: np.matrix):
    """Calcula os resíduosde cada ponto."""
    y = data['CO2'].values
    res = np.reshape(
        np.matrix([y[i] - float(y_[i]) for i in range(len(y_))]),
        (-1, 1)
    )
    res_list = [round(val[0], 6) for val in res.tolist()]
    print(f'    [ F ] Os residuos são: {res_list}')
    return res


def exec_(data,
          alinea_a=False,
          alinea_b=False,
          alinea_d=False,
          alinea_e=False,
          alinea_f=False,
          alinea_g=False,
          ):
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

    if alinea_d:
        # Alínea D
        mrlm(data)
        """
            O Hiperplano é: y = 79.694719 + 0.007805*x1 + 0.007551*x2
        """
    # ======================================================================= #

    if alinea_e:
        # Alínea E
        beta_ = mrlm(data)
        get_y_(data, beta_)
    # ======================================================================= #

    if alinea_f:
        # Alínea F
        beta_ = mrlm(data)
        y_ = get_y_(data, beta_)
        get_res(data, y_)
    # ======================================================================= #

    if alinea_g:
        # Alínea D
        beta_ = mrlm(data)
        y_ = get_y_(data, beta_)
        res = get_res(data, y_)
        plt.hist(res)
        plt.show()
    # ======================================================================= #
    return


if __name__ == '__main__':
    data = pd.read_csv("cars.csv")
    executar_alinea_a = False
    executar_alinea_b = False
    executar_alinea_d = False
    executar_alinea_e = False
    executar_alinea_f = False
    executar_alinea_g = True

    exec_(data,
          executar_alinea_a,
          executar_alinea_b,
          executar_alinea_d,
          executar_alinea_e,
          executar_alinea_f,
          executar_alinea_g,
          )
