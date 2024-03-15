'''
QUESTÃO 8: Dada as entradas x = [0, 3, -1, 4, 3, 5, 2, 10, 2, 4] e y =[0.2, 0.8, 2.4, 6.5, 7.1, 7.5, 7.7, 8.1, 8.9, 10.2].
a) Utilizando as funções polyfit e polyval da biblioteca NumPy, ajuste um polinômio aos dados fornecidos.
b) Como métricas de avaliação, utilize o Coeficiente de Determinação (R²) e o Erro Quadrático Médio (EQM) 
para determinar a qualidade do ajuste do polinômio aos dados.
c) Aplique o Z-score nos dados, garantindo que estejam padronizados com média zero e desvio padrão um, depois 
repita os passos dos itens "a" e "b" nesses dados normalizados.
'''

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# Constantes
DEGREE = 3

# Dados fornecidos
x = np.array([0, 3, -1, 4, 3, 5, 2, 10, 2, 4])
y = np.array([0.2, 0.8, 2.4, 6.5, 7.1, 7.5, 7.7, 8.1, 8.9, 10.2])

# a) Ajuste de um polinômio aos dados
coefficients = np.polyfit(x, y, DEGREE)
polynomial = np.poly1d(coefficients)

print('========== POLINÔMIO GERADO ===========')
print(f'{polynomial}\n')
print('=======================================\n')

# b) Avaliação do ajuste
y_pred = polynomial(x)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print("Coeficiente de Determinação (R²):", r2)
print("Erro Quadrático Médio (EQM):", mse)

# c) Aplicação do Z-score nos dados
x_normalized = (x - np.mean(x)) / np.std(x)
y_normalized = (y - np.mean(y)) / np.std(y)

# Ajuste de um polinômio aos dados normalizados
coefficients_norm = np.polyfit(x_normalized, y_normalized, 3)
polynomial_norm = np.poly1d(coefficients_norm)

# Avaliação do ajuste nos dados normalizados
y_pred_norm = polynomial_norm(x_normalized)
r2_norm = r2_score(y_normalized, y_pred_norm)
mse_norm = mean_squared_error(y_normalized, y_pred_norm)

print("\nApós normalização:")
print("Coeficiente de Determinação (R²):", r2_norm)
print("Erro Quadrático Médio (EQM):", mse_norm)