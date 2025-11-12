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


# ------ es 2.b

uscita_filtro_valor_medio = np.mean(uscita_filtro)

x1N = x1 - x1_valor_medio
y1N = uscita_filtro - uscita_filtro_valor_medio

# Sulla traccia c'è scritto di usare correlate() di numpy
# https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]

autocorr_x1N = autocorr(x1N)
autocorr_y1N = autocorr(y1N)

plt.figure(figsize=(10,6))

plt.subplot(2, 1, 1)
plt.plot(n1, autocorr_x1N, color='gray')
plt.title("Autocorrelazione x1N")
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(n1, autocorr_y1N, color='purple')
plt.title("Autocorrelazione y1N")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ------ es 2.c

var_x1N = np.var(x1N)
var_y1N = np.var(y1N)

x1N_energia = np.sum(x1N**2)
y1N_energia = np.sum(y1N**2)

def larghezza_lobo_centrale(r):
    N = len(r)
    centro = N//2
    r_norm = r / np.max(np.abs(r))
    destra = np.where(r_norm[centro:] <= 0)[0]
    sinistra = np.where(r_norm[:centro] <= 0)[0]
    if len(destra) == 0:
        dx = N - centro
    else:
        dx = destra[0]
    if len(sinistra) == 0:
        sx = centro
    else:
        sx = centro - sinistra[-1]
    return dx + sx

lobo_autocorr_x1N = larghezza_lobo_centrale(autocorr_x1N)
lobo_autocorr_y1N = larghezza_lobo_centrale(autocorr_y1N)

print("varx =",var_x1N, "\nvary =", var_y1N, "\nenergiax1 =", x1N_energia, "\nenergiay1 =", y1N_energia )
print("lobox =", lobo_autocorr_x1N , "loboy =", lobo_autocorr_y1N )
