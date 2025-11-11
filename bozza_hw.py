# IMPORTANTE: Questo file contiene lo svolgimento fino al punto 2.a

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

csv_x1 = r"pressure_8714.csv"
csv_x2 = r"pressure_8644.csv"

# caricamento dati
df1 = pd.read_csv(csv_x1)
df2 = pd.read_csv(csv_x2)

# segnali di pressione
x1 = df1["pressure_value"].values
x2 = df2["pressure_value"].values

N1 = len(x1)
n1 = np.linspace(0, N1, N1)
N2 = len(x2)
n2 = np.linspace(0, N2, N2)


# ------ es 1

x1_energia = np.sum(x1**2)
x1_valor_medio = np.mean(x1)

plt.plot(n1, x1, marker='o')
plt.xlabel('Data')
plt.ylabel('Valore')
plt.title('Segnale x1')
plt.grid(True)
plt.figtext(0.7, 0.92, f'Energia segnale: {round(x1_energia,2)}', fontsize=10, ha='left', va='top')
plt.figtext(0.7, 0.97, f'Valor medio segnale: {round(x1_valor_medio,2)}', fontsize=10, ha='left', va='top')
plt.show()


# ------ es 2.a

filtro_sinc = np.sinc(n1 - (N1 - 1)/2)
uscita_filtro = np.convolve(x1, filtro_sinc, mode='same')

plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(n1, x1, color='gray')
plt.title("Segnale originale")
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(n1, filtro_sinc, color='purple')
plt.title("Risposta impulsiva")
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.plot(n1, uscita_filtro, color='purple')
plt.title("Uscita filtro")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


