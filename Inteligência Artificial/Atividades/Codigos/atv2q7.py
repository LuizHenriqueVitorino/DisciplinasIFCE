'''
7 -  Usando a função linspace do NumPy, gere 500 dados igualmente espaçados no intervalo de 0 a 20. 
    
    a) Organização em Dataset: Depois de gerados, organize esses dados em um dataset (ou matricial) 
    com as dimensões 100x5. 
    
    b)  Utilize a técnica de Hold-Out com uma divisão de 80% para treinamento e 20% para testes.

*Pode fazer o upload do código comentado.  
'''
from numpy import linspace
import pandas as pd
from sklearn.model_selection import train_test_split

# ----- DECLARATIONS OF CONSTANTS ----- #
# LINSPACE
START  = 0   
END    = 20
SPACED = 500

# MATRIX
NUMBER_OF_LINES   = 100
NUMBER_OF_COLUMNS = 5

# ----- DECLARATIONS OF VARIABLES ----- #
dataset = []


# ----- DATA GENERATION ----- #
data = linspace(0, 20, 500) # <-- GERA 500 DADOS IGUALMENTE ESPAÇADOS ENTRE 0 E 20


# (ITEM A) - organize esses dados em um dataset (ou matricial) com as dimensões 100x5.

for line in range(0,SPACED,NUMBER_OF_COLUMNS):
    line_to_add = data[line:line + NUMBER_OF_COLUMNS]
    dataset.append(line_to_add)

# (ITEM B)  Utilize a técnica de Hold-Out com uma divisão de 80% para treinamento e 20% para testes.

dataframe = pd.DataFrame(dataset)
X_train, X_test = train_test_split(dataframe, test_size=0.2, random_state=42)

print(X_train)
print(X_test)