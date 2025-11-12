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

# ------ es 2.d — Commento e interpretazione
#questo lo ha fatto chatGPT quindi è solo per avere una base.

print("=== Esercizio 2d ===")
print("Analisi dell'effetto del filtro sinc sul segnale x1")

print("\n1) Varianza ed energia:")
print("- La varianza di y1N è maggiore rispetto a x1N, segno che il filtro sinc non è normalizzato,")
print("  e quindi amplifica leggermente le ampiezze.")
print(f"  Var(x1N) = {var_x1N:.4f}, Var(y1N) = {var_y1N:.4f}")
print(f"  Energia(x1N) = {x1N_energia:.2f}, Energia(y1N) = {y1N_energia:.2f}")

print("\n2) Larghezza del lobo centrale dell'autocorrelazione:")
print(f"  Larghezza r_xx = {lobo_autocorr_x1N} campioni,  Larghezza r_yy = {lobo_autocorr_y1N} campioni")
print("- Il lobo centrale di r_yy risulta più largo rispetto a r_xx,")
print("  come previsto per un segnale filtrato passa-basso: il filtro rimuove le alte frequenze,")
print("  rendendo il segnale più 'lento' e quindi più correlato nel tempo.")

print("\n3) Considerazioni generali:")
print("- Il filtro sinc agisce come un passa-basso ideale, smussando le variazioni rapide del segnale.")
print("- Dopo la rimozione del valor medio, si osserva che la correlazione a lungo termine diminuisce,")
print("  mentre la correlazione locale (attorno allo zero) si allarga.")
print("- In sintesi: il filtro riduce il contenuto ad alta frequenza, aumenta la regolarità del segnale")
print("  e modifica l’energia e la varianza per effetto della sua non normalizzazione.")

# ------ es 3.a

x2N = x2 - np.mean(x2)
delta_x = np.abs(x1N - x2N)

plt.figure(figsize=(10,6))
plt.subplot(3,1,1)
plt.plot(x1N, color='gray')
plt.title("x1N (senza valor medio)")

plt.subplot(3,1,2)
plt.plot(x2N, color='blue')
plt.title("x2N (senza valor medio)")

plt.subplot(3,1,3)
plt.plot(delta_x, color='red')
plt.title("Δx[n] = |x1N - x2N|")
plt.xlabel("Campione n")
plt.tight_layout()
plt.show()


# ------ es 3.b

def calc_corrcoeff(x,y):
    # IMPORTANTE: questa funzione usa la definizione della slide 15 (05_convfiltricorr.pdf) 
    # https://drive.google.com/drive/folders/1FscODSNyj1uR-GuXtY3VHb2KcW-G3BBE
    #
    # Il risultato deve essere compreso tra -1 e +1
    #  0 = segnali incorrelati
    # −1 = segnali anticorrelati
    #  1 = segnali correlati

    E_xy = sum(xi * yi for xi, yi in zip(x, y))
    E_x = np.sum(x**2)
    E_y = np.sum(y**2)

    return E_xy / (E_x * E_y)**0.5


l_finestra_x1 = N1 // 3
l_finestra_x2 = N2 // 3

x1N_K1 = x1N[:l_finestra_x1]
x2N_K1 = x2N[:l_finestra_x2]

x1N_K2 = x1N[l_finestra_x1:2*l_finestra_x1]
x2N_K2 = x2N[l_finestra_x2:2*l_finestra_x2]

x1N_K3 = x1N[2*l_finestra_x1:]
x2N_K3 = x2N[2*l_finestra_x2:]

print(f"Coefficiente di correlazione finestra K = 1 --> {calc_corrcoeff(x1N_K1, x2N_K1)}")
print(f"Coefficiente di correlazione finestra K = 2 --> {calc_corrcoeff(x1N_K2, x2N_K2)}")
print(f"Coefficiente di correlazione finestra K = 3 --> {calc_corrcoeff(x1N_K3, x2N_K3)}")