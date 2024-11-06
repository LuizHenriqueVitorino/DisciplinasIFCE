import numpy as np
import matplotlib.pyplot as plt
import imageio

# Carregar o arquivo de dados
data = np.loadtxt('artificial1d.csv', delimiter=',')

# Separar as coulunas
x = data[:, 0]
y = data[:, 1]

# Criar a Matrix X a partir da coluna x
n = len(x)
X = x.reshape((-1,1))

# Adicionar a coluna de 1s à matriz X
ones = np.ones((n, 1)) # Cria o vetor de 1s
X = np.hstack((ones, X)) # Concatena o vetor de 1s à matriz 


def calcular_mse(X, y, w):
    soma_dos_erros_quadrados = 0

    y_pred = X @ w
    for i in range (n):
        erro = y[i] - y_pred[i] # Calcular o erro para cada observação
        soma_dos_erros_quadrados += erro ** 2   # Somar os erros ao quadrado

    mse = soma_dos_erros_quadrados / n  # Calcula o mse
    return  mse

def gradiente_descendente(X, y, taxa_aprendizagem=0.01, iteracoes=50):
    # Inicializar os parâmetros
    alpha = taxa_aprendizagem # Taxa de aprendizado
    n_epochs = iteracoes # Número de iterações
    w = np.zeros((2, 1))    # w0 e w1 inicializados como zero
    mse_history = []    # Registros dos MSE de cada iteração

    imagens = []
    
    erro = y - X @ w
    for t in range (n_epochs):
        y_pred = X @ w
        
        

        # Atualizar os parâmetros
        w[0] += alpha * (1/n) * np.sum(erro)
        w[1] += alpha * (1/n) * np.sum(erro * X[:, 1])

        erro = y - y_pred

        # Calcular e registrar o mse
        mse = calcular_mse(X, y, w)
        mse_history.append(mse)

        if t%5 == 0:
            print(t)
            plt.figure(figsize=(8, 6))
            plt.scatter(x, y, color='blue', label='Dados')
            plt.plot(x, X @ w, color='red', label='Reta Ajustada')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Regressão Linear - Iteração {t + 1}')
            plt.legend()

            # Definir limites fixos para o eixo Y
            plt.ylim(-4, 4)  # Ajuste os limites conforme necessário

            caminho_imagem = f'graficos/itemB/iteracao_{t + 1}.png'
            plt.savefig(caminho_imagem)  # Salva a imagem da plotagem
            plt.close()  # Fecha a figura para liberar memória

            # Adiciona a imagem à lista
            imagens.append(caminho_imagem)

        # Criar o GIF a partir das imagens
    with imageio.get_writer('itemB.gif', mode='I', duration=0.5) as writer:
        for imagem in imagens:
            img = imageio.imread(imagem)
            writer.append_data(img)

gradiente_descendente(X, y)
    