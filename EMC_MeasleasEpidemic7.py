# Codice Esame Metodi Computazionali 2026 - Niccolò Ruzza - Misleas Epidemic. 7
# Questo codice si propone di studiare l'evoluzione nel tempo del problema cercando regolarità.
# In particolare creando il ritratto di Poincarè a fase variabile.

# Si trova un'orbita caotica definita!!!

import numpy as np
import matplotlib.pyplot as plt
import time

N = 100  # Passi per anno
mu = 0.02
lam = 0.0279
ni = 0.01
beta0 = 1575.
a=0 # Valore Iniziale (Tempo)
b=1 # Valore Finale (Tempo)
h=(b-a)/(N) # Passo

# Funzione per il calcolo delle derivate
def f(t, y):
    beta = beta0 * (1 + np.cos(2 * np.pi * t))
    dy0 = mu - beta * y[0] * y[2]
    dy1 = beta * y[0] * y[2] - y[1] / lam
    dy2 = y[1] / lam - y[2] / ni
    return np.array([dy0, dy1, dy2])

# Metodo Runge-Kutta 4
def calc_y(t, y_curr):
    k1 = f(t, y_curr)
    k2 = f(t + h/2, y_curr + (h/2) * k1)
    k3 = f(t + h/2, y_curr + (h/2) * k2)
    k4 = f(t + h, y_curr + h * k3)
    return y_curr + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

def main():
    start_time = time.perf_counter()
    
    # --- Configurazione Simulazione Lunga ---
    anni_totali = 10000000    # Quanti anni simulare in totale ############################################
    anni_transitorio = 10   # Quanti anni scartare all'inizio (transitorio)
    
    # Condizioni iniziali
    y_curr = np.array([0.04, 0.1, 0.1])
    tempo_assoluto = 0.0
    
    # Liste per salvare SOLO i punti stroboscopici (1 punto all'anno)
    poincare_S = []  # Suscettibili
    poincare_E = []  # Latenti/Esposti
    poincare_I = []  # Infetti
    
    print(f"Simulazione di {anni_totali} anni in corso per la Mappa di Poincaré...")
    
    passo_campionamento = 50
    # Esecuzione del ciclo del tempo
    for anno in range(anni_totali):
        for passo_corrente in range(N):
            y_curr = calc_y(tempo_assoluto, y_curr)
            tempo_assoluto += h
            
            # Se abbiamo superato il transitorio E siamo nel "mese" scelto, salviamo
            if anno >= anni_transitorio and passo_corrente == passo_campionamento:
                poincare_S.append(y_curr[0] * 100)      # Convertito in %
                poincare_E.append(y_curr[1] * 100000)   # Scalato per visibilità
                poincare_I.append(y_curr[2] * 100000)   # Scalato per visibilità

    # Conversione in array NumPy
    poincare_S = np.array(poincare_S)
    poincare_E = np.array(poincare_E)
    poincare_I = np.array(poincare_I)
    
    # --- Creazione dei Grafici ---
    # Creiamo una figura grande per ospitare sia il 2D
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(poincare_S, poincare_I, s=1, color='crimson', alpha=0.5)
    plt.title("Proiezione in 2D della Mappa di Poincaré 3D")
    plt.xlabel("Suscettibili (%)")
    plt.ylabel("Infetti (x $10^{-3}$ %)")
    plt.grid(True, linestyle='--', alpha=0.6)
    end_time = time.perf_counter()
    print(f"Calcolo completato in {(end_time - start_time):.2f} secondi.")

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(poincare_S, poincare_I/1000, s=1, color='crimson', alpha=0.5)
    plt.yscale('log')
    plt.title("Proiezione in 2D della Mappa di Poincaré 3D (Scala Logaritmica)")
    plt.xlabel("Suscettibili (%)")
    plt.ylabel("Infetti (%)")
    plt.grid(True, which="both", linestyle='--', alpha=0.6) 

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(poincare_S, poincare_E, poincare_I, color='darkblue', alpha=0.5)
    ax.set_title("Mappa di Poincaré 3D")
    ax.set_xlabel("Suscettibili", fontsize=10)
    ax.set_ylabel("Latenti (x $10^{-3}$ %)", fontsize=10)
    ax.set_zlabel("Infetti (x $10^{-3}$ %)", fontsize=10)

    plt.show()

if __name__ == "__main__":
    main()