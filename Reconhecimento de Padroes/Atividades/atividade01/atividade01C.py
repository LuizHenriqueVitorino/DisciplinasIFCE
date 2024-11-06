import numpy as np
import matplotlib.pyplot as plt
import imageio

# Carregar o arquivo de dados
data = np.loadtxt('artificial1d.csv', delimiter=',')
x = data[:, 0]
y = data[:, 1]

# Criar a Matrix X a partir da coluna x
n = len(x)
X = x.reshape(n, 1)

# Adicionar a coluna de 1s à matriz X
ones = np.ones((n, 1))  # Cria o vetor de 1s
X = np.hstack((ones, X))  # Concatena o vetor de 1s à matriz 

def calcular_mse(X, y, w):
    y_pred = X @ w
    erro = y - y_pred
    mse = np.mean(erro ** 2)  # Calcular o MSE diretamente
    return mse

def sgd(X, y, taxa_aprendizagem=0.01, n_epocas=50):
    alpha = taxa_aprendizagem  # Taxa de aprendizado
    w = np.zeros((2, 1))  # Inicializa w0 e w1 como zero
    mse_history = []  # Registros dos MSE de cada época
    imagens = []

    for epoch in range(n_epocas):
        # Embaralhar os dados para cada época
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(n):
            # Predição para o exemplo atual
            y_pred = X_shuffled[i] @ w
            erro = y_shuffled[i] - y_pred  # Cálculo do erro

            # Atualização dos parâmetros
            w[0] += alpha * erro  # Atualiza w0
            w[1] += alpha * erro * X_shuffled[i, 1]  # Atualiza w1

        # Calcular o MSE após cada época
        mse = calcular_mse(X, y, w)
        mse_history.append(mse)

        # Gravação da imagem a cada 50 épocas
        if epoch % 2 == 0:
            plt.figure(figsize=(8, 6))
            plt.scatter(x, y, color='blue', label='Dados')
            plt.plot(x, X @ w, color='red', label='Reta Ajustada')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Regressão Linear - Época {epoch + 1}')
            plt.legend()
            plt.ylim(-4, 4)
            caminho_imagem = f'graficos/itemC/epoca_{epoch + 1}.png'
            plt.savefig(caminho_imagem)  # Salva a imagem da plotagem
            plt.close()  # Fecha a figura para liberar memória
            imagens.append(caminho_imagem)

    # Criar o GIF a partir das imagens
    with imageio.get_writer('itemC.gif', mode='I', duration=0.5) as writer:
        for imagem in imagens:
            img = imageio.imread(imagem)
            writer.append_data(img)


    with imageio.get_writer('regressao_linear_sgd.mp4', fps=10, codec='libx264') as writer:
        for imagem in imagens:
            img = imageio.imread(imagem)
            writer.append_data(img)

    # Plotar a curva de aprendizagem (MSE ao longo das épocas)
    plt.figure(figsize=(8, 6))
    plt.plot(mse_history, color='purple')
    plt.xlabel('Épocas')
    plt.ylabel('MSE')
    plt.title('Curva de Aprendizado (MSE ao longo das épocas)')
    plt.show()

    return w, mse_history

# Executa o SGD e obtém os parâmetros finais e a curva de aprendizagem
w_final, mse_history = sgd(X, y, taxa_aprendizagem=0.01, n_epocas=80)
print(f'Parâmetros finais do modelo: w0 = {w_final[0][0]}, w1 = {w_final[1][0]}')
print(f'MSE final: {mse_history[-1]}')
