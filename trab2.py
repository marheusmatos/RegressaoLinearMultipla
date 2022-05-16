"""Respostas do trabalho 2."""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Funcao auxiliar para calcular o quadrado de um numero


def quad(x):
    return x*x


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
    """Estima os valores de β para o Hiperplano de Quadrados Mínimos.
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

    print(f'    [ D ] β0 = {round(float(beta_[0]), 6)}')
    print(f'    [ D ] β1 = {round(float(beta_[1]), 6)}')
    print(f'    [ D ] β2 = {round(float(beta_[2]), 6)}')
    print('    [ D ] O Hiperplano é: y = ' +
          f'{round(float(beta_[0]), 6)} + ' +
          f'{round(float(beta_[1]), 6)}*x1 + ' +
          f'{round(float(beta_[2]), 6)}*x2')
    return beta_


def get_estimated_y(data: pd.DataFrame, beta_: np.matrix):
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


def get_codetermination(data: pd.DataFrame, col_1: str, col_2: str):
    """Função que calcula o coeficiente de determinacao.
        data -> Base de dados.
        Fórmula:
            R² = {(1/N)*sum[((Xi-avg(X))*(Yi-avg(Y))]/(stdDev(X)*stdDev(Y))}²
    """
    x = data[col_1]
    y = data[col_2]
    n = len(x)

    # Cálculos matemáticos
    avg_x = sum(x)/len(x)
    avg_y = sum(y)/len(y)
    std_dev_x = np.std(x)  # desvio-padrão de X
    std_dev_y = np.std(y)  # desvio-padrão de Y
    std_product = std_dev_x * std_dev_y
    n = len(x)

    # List Comprehension com função de somatório para calcular a soma da
    # multiplicaçao entre a quantidade de variancia em x e quantidade de
    # variancia em y
    soma = sum([(x[i]-avg_x)*(y[i]-avg_y) for i in range(n)])

    # Calcula R², que será o quadrado da multiplicacao entre a razao
    # 1/tamanho vezes a variável temporaria calculada na linha acima
    r2 = quad((1/n)*(soma/std_product))

    print(f'    [ C ] A codeterminação entre {col_1} e {col_2} é {r2}.')
    return r2


def exec_(data,
          alinea_a=False,
          alinea_b=False,
          alinea_c=False,
          alinea_d=False,
          alinea_e=False,
          alinea_f=False,
          alinea_g=False,
          ):
    """Executa todas as alineas do trabalho.
    aliena_x -> Caso True executa o solicitado na alinea X.
    """
    if alinea_a:
        # Alíena (A)
        # Faça o gráfico de dispersão bidimensional (XY) entre a variável y e
        # cada uma das variáveis regressoras X1,..., Xp (p gráficos separados).

        # Plota o gráfico para a variável Volume
        plot_graph(data, 'Volume', 'CO2')
        # Plota o gráfico para a variável Peso
        plot_graph(data, 'Peso', 'CO2')
    # ======================================================================= #

    if alinea_b:
        # Alínea (B)
        # Estime a correlação entre a variável Y e cada uma das variáveis
        # regressoras (ρ(Y,Xi)). Comente.

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
            uma correlação positiva entre as variáveis, nesse caso um pouco
            mais fraca cerca de 53,7% de correlação se olharmos em questão de
            porcentagem.
        """
    # ======================================================================= #

    if alinea_c:
        # Alínea (C)
        # Calcule o coeficiente de determinacao R²(Y,Xi) entre a variavel
        # y e cada uma das variáveis regressoras. Comente o resultado.

        # Estima a codeterminacao entre a variavel Y (CO2)
        # e a variável regressora 1 (Volume)
        get_codetermination(data, 'Volume', 'CO2')
        """
            Comentario sobre a codeterminacao entre Volume e CO2:
            (Variável Regressora 1 e variável Y)
            A codeterminacao entre Volume e CO2 é 0.350560. Isto nos indica
            que se ocorrer um mudança de, aproximadamente, 35% em Y não será
            percebida muita mudança nos resultados finais aos quais se quer
            chegar
        """

        # Estima a codeterminacao entre a variavel Y (CO2)
        # e a variável regressora 2 (Peso)
        get_codetermination(data, 'Peso', 'CO2')
        """
            Comentario sobre Relação entre Peso e CO2.
            (Variável Regressora 2 e variável Y)
            A codeterminacao entre Peso e CO2 não é muito diferente do
            que se viu entre Volume e CO2. Aqui, temos um R² = 0.304869,
            aproximadamente 30%. Da mesma forma que o outro, temos um R²
            razoável e ele indica que uma mudança de 30% em Y não será
            percebida um brusca mudança nos resultados finais.
        """
    # ======================================================================= #

    if alinea_d:
        # Alínea (D)
        # Encontre o hiperplano de quadrados mínimos (estime β0, β1, ..., βp).

        # Estima os valores de β e o Hiperplano
        mrlm(data)
        """
            β0 = 79.694719
            β1 = 0.007805
            β2 = 0.007551
            O Hiperplano é: y = 79.694719 + 0.007805*x1 + 0.007551*x2
        """
    # ======================================================================= #

    if alinea_e:
        # Alínea (E)
        # Calcule os valores estimados de y (ou seja, ŷ) através do hiperplano
        # estimado no item anterior.

        # Calcula o hiperplano de quadrados mínimos
        beta_ = mrlm(data)
        # Calcula os valores estimados de y, baseados no hiperplano calculado
        get_estimated_y(data, beta_)
        """
        Retorno:
            y^ = [93.465009, 97.819879, 94.514598, 93.250834, 100.010359,
                  94.514598, 98.995778, 101.709334, 99.798931, 100.866369,
                  95.680199, 97.316709, 95.896431, 101.636571, 102.195345,
                  102.225549, 102.489834, 106.530999, 100.632288, 105.332447,
                  104.143503, 106.087547, 106.392334, 102.867384, 105.989384,
                  102.464434, 106.555709, 108.330194, 103.682892, 108.179174,
                  108.204574, 108.488765, 101.508204, 102.678609, 102.791874,
                  109.740864]
        """
    # ======================================================================= #

    if alinea_f:
        # Alínea (F)
        # Calcule os resíduos, ou seja, o erro de estimação y-ŷ.

        # Calcula o hiperplano de quadrados mínimos
        beta_ = mrlm(data)
        # Calcula os valores estimados de y, baseados no hiperplano calculado
        y_ = get_estimated_y(data, beta_)
        # Calcula os valores dos resíduos, baseado nos y estimados e nos
        # valores de y fornecidos como entrada
        get_res(data, y_)
        """
        Retorno:
            Os residuos são: [5.534991, -2.819879, 0.485402, -3.250834,
                              4.989641, 10.485402, -8.995778, -9.709334,
                              -1.798931, -1.866369, 3.319801, 3.683291,
                              3.103569, -7.636571, -5.195345, -5.225549,
                              -3.489834, -2.530999, 3.367712, -0.332447,
                              -10.143503, -7.087547, -7.392334, -3.867384,
                              -6.989384, -0.464434, -2.555709, 5.669806,
                              5.317108, 5.820826, 6.795426, 8.511235, 2.491796,
                              5.321391, 6.208126, 10.259136]
        """
    # ======================================================================= #

    if alinea_g:
        # Alínea (G)
        # Faça o histograma dos erros de estimação encontrados no item
        # anterior. Comente sobre sua característica.

        # Calcula o hiperplano de quadrados mínimos
        beta_ = mrlm(data)
        # Calcula os valores estimados de y, baseados no hiperplano calculado
        y_ = get_estimated_y(data, beta_)
        # Calcula os valores dos resíduos, baseado nos y estimados e nos
        # valores de y fornecidos como entrada
        res = get_res(data, y_)
        # Plota os resíduos
        plt.hist(res, bins=int(res.max() - res.min()), density=True)
        plt.show()
    # ======================================================================= #


if __name__ == '__main__':
    data = pd.read_csv("cars.csv")
    executar_alinea_a = False
    executar_alinea_b = False
    executar_alinea_c = False
    executar_alinea_d = False
    executar_alinea_e = False
    executar_alinea_f = False
    executar_alinea_g = True

    exec_(data,
          executar_alinea_a,
          executar_alinea_b,
          executar_alinea_c,
          executar_alinea_d,
          executar_alinea_e,
          executar_alinea_f,
          executar_alinea_g,
          )
