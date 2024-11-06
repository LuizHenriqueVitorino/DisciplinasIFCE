import numpy as np
import matplotlib.pyplot as plt

# Carregar o arquivo de dados
data = np.loadtxt('artificial1d.csv', delimiter=',')

# Separar as coulunas
x = data[:, 0]
y = data[:, 1]

# # Visualizar os dados em um gráfico
# plt.scatter(x, y, color='blue')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('VISUALIZAÇÃO DOS DADOS')
# plt.show()

# Criar a Matrix X a partir da coluna x
n = len(x)
X = x.reshape(n, 1)

# Adicionar a coluna de 1s à matriz X
ones = np.ones((n, 1)) # Cria o vetor de 1s

X = np.hstack((ones, X)) # Concatena o vetor de 1s à matriz 

# Calcular os parâmetros da regressão linear pelo método dois mínimos quadrados.
w = np.linalg.inv(X.T @ X) @ X.T @ y 

# Calcular as predições
y_pred = X @ w

# Calcular o MSE
soma_dos_erros_quadrados = 0
n = len(y) # Número de observações

for i in range (n):
    erro = y[i] - y_pred[i] # Calcular o erro para cada observação
    soma_dos_erros_quadrados += erro ** 2   #Somar os erros ao quadrado

mse = soma_dos_erros_quadrados / n  # Calcula o mse

print(f'MSE: {mse}')

plt.scatter(x, y, color='blue', label='Dados do arquivo')   # Dados Originais
plt.plot(x, y_pred, color='red', label='Reta de Regressão') # Reta Ajustada
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regressão Linear')
plt.show()

