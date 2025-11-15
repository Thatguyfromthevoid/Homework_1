
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

csv_x1 = r"pressure_8714.csv"
csv_x2 = r"pressure_8644.csv"

df1 = pd.read_csv(csv_x1)
df2 = pd.read_csv(csv_x2)

x1 = df1["pressure_value"].values
x2 = df2["pressure_value"].values

N1 = len(x1)
n1 = np.linspace(0, N1, N1)

# === ESERCIZIO 1 ===

x1_energia = np.sum(x1**2)
x1_valor_medio = np.mean(x1)

plt.plot(n1, x1, marker='o')
plt.xlabel('Campione n')
plt.ylabel('Valore')
plt.title('Segnale x1')
plt.grid(True)
plt.figtext(0.7, 0.92, f'Energia segnale: {round(x1_energia,2)}')
plt.figtext(0.7, 0.97, f'Valor medio segnale: {round(x1_valor_medio,2)}')
plt.show()

# === ESERCIZIO 2.a ===

filtro_sinc = np.sinc(n1 - (N1 - 1)/2)
uscita_filtro = np.convolve(x1, filtro_sinc, mode='same')

plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(n1, x1, color='gray')
plt.title("Segnale x1")

plt.subplot(3, 1, 2)
plt.plot(n1, filtro_sinc, color='blue')
plt.title("Filtro sinc")

plt.subplot(3, 1, 3)
plt.plot(n1, uscita_filtro, color='red')
plt.title("Segnale filtrato y1 = x1 * h")
plt.tight_layout()
plt.show()

# === ESERCIZIO 2.b ===

auto_x1_raw = np.correlate(x1, x1, mode='full')
auto_y1_raw = np.correlate(uscita_filtro, uscita_filtro, mode='full')
lags_raw = np.arange(-len(x1)+1, len(x1))

plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(lags_raw, auto_x1_raw, color='gray')
plt.title("Autocorrelazione x1 (raw)")

plt.subplot(2,1,2)
plt.plot(lags_raw, auto_y1_raw, color='red')
plt.title("Autocorrelazione y1 (raw)")
plt.tight_layout()
plt.show()

# === ESERCIZIO 2.c ===

x1N = x1 - np.mean(x1)
y1N = uscita_filtro - np.mean(uscita_filtro)

rxx = np.correlate(x1N, x1N, mode='full')
ryy = np.correlate(y1N, y1N, mode='full')

var_x1N = np.var(x1N)
var_y1N = np.var(y1N)
energia_x1N = np.sum(x1N**2)
energia_y1N = np.sum(y1N**2)

def larghezza_lobo_centrale(r):
    N = len(r)
    centro = N // 2
    r_norm = r / abs(r[centro]) if r[centro] != 0 else r
    destra = next((i for i in range(centro + 1, N) if r_norm[i] <= 0), N - 1)
    sinistra = next((i for i in range(centro - 1, -1, -1) if r_norm[i] <= 0), 0)
    return destra - sinistra

lobo_x = larghezza_lobo_centrale(rxx)
lobo_y = larghezza_lobo_centrale(ryy)

print("=== Esercizio 2c ===")
print(f"Var(x1N) = {var_x1N:.4f}, Var(y1N) = {var_y1N:.4f}")
print(f"Energia(x1N) = {energia_x1N:.2f}, Energia(y1N) = {energia_y1N:.2f}")
print(f"Larghezza lobo centrale r_xx: {lobo_x} campioni")
print(f"Larghezza lobo centrale r_yy: {lobo_y} campioni")

# === ESERCIZIO 2.d ===

print("\n=== Esercizio 2d ===")
print("Il filtro sinc attenua le alte frequenze rendendo il segnale più lento e più correlato nel tempo.")
print(f"Larghezza lobo x1N = {lobo_x}, larghezza lobo y1N = {lobo_y}")

print("→ Il lobo centrale di y1N è più largo perché il filtro passa-basso aumenta la correlazione locale.")
print("→ Energia e varianza cambiano perché il filtro non è normalizzato.")

# === ESERCIZIO 3.a ===

x2N = x2 - np.mean(x2)
L = min(len(x1N), len(x2N))
x1N = x1N[:L]
x2N = x2N[:L]

delta_x = np.abs(x1N - x2N)

plt.figure(figsize=(10,7))
plt.subplot(3,1,1)
plt.plot(x1N, color='gray')
plt.title("x1N (senza valor medio)")

plt.subplot(3,1,2)
plt.plot(x2N, color='blue')
plt.title("x2N (senza valor medio)")

plt.subplot(3,1,3)
plt.plot(delta_x, color='red')
plt.title("Δx[n] = |x1N - x2N|")
plt.tight_layout()
plt.show()

# === ESERCIZIO 3.b ===

K = 3
N = len(x1N)
base = N // K
resto = N % K

idx = []
start = 0
for i in range(K):
    fine = start + base + (1 if i < resto else 0)
    idx.append((start, fine))
    start = fine

rho = []
lag_picco = []

for (a, b) in idx:
    seg1 = x1N[a:b]
    seg2 = x2N[a:b]
    rho_val = np.corrcoef(seg1, seg2)[0, 1] if np.std(seg1)*np.std(seg2) != 0 else np.nan
    rho.append(rho_val)

    rxy = np.correlate(seg1, seg2, mode='full')
    lag = np.arange(-len(seg1)+1, len(seg1))
    lag_max = lag[np.argmax(rxy)]
    lag_picco.append(lag_max)

print("\n=== Esercizio 3b ===")
for i in range(K):
    print(f"Finestra {i+1}: rho = {rho[i]:.4f}, lag = {lag_picco[i]}")

# === ESERCIZIO 3.c ===

print("\n=== Esercizio 3c ===")
print("Se i lag sono simili tra le finestre → ritardo costante.")
print("Se cambiano → ritardo variabile.")
