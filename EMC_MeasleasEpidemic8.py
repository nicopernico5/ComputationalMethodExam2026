# Codice Esame Metodi Computazionali 2026 - Niccolò Ruzza - Misleas Epidemic. 8
# Questo codice si propone di studiare l'evoluzione nel tempo del problema cercando regolarità.
# In particolare creando il ritratto di Poincarè con valori iniziali a scelta in schermo.

# Si nota che qualsiasi valore iniziale la soluzione cade nell'area conosciuta.
import numpy as np
import matplotlib.pyplot as plt
import time

# --- Parametri del Modello ---
N = 100         # Passi per anno
mu = 0.02
lam = 0.0279
ni = 0.01
beta0 = 1575.
a=0 # Valore Iniziale (Tempo)
b=1 # Valore Finale (Tempo)
h=(b-a)/(N) # Passo

def f(t, y):
    beta = beta0 * (1 + np.cos(2 * np.pi * t))
    dy0 = mu - beta * y[0] * y[2]
    dy1 = beta * y[0] * y[2] - y[1] / lam
    dy2 = y[1] / lam - y[2] / ni
    return np.array([dy0, dy1, dy2])

def calc_y(t, y_curr):
    k1 = f(t, y_curr)
    k2 = f(t + h/2, y_curr + (h/2) * k1)
    k3 = f(t + h/2, y_curr + (h/2) * k2)
    k4 = f(t + h, y_curr + h * k3)
    return y_curr + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

# Funzione per simulare e restituire una traiettoria di Poincaré (Lineare)
def simula_poincare(y0, anni_totali, scarta_transitorio=True):
    y_curr = np.array(y0)
    tempo_assoluto = 0.0
    anni_transitorio = 1 if scarta_transitorio else 0
    
    lin_S, lin_I = [], []
    
    for anno in range(anni_totali):
        for _ in range(N):
            y_curr = calc_y(tempo_assoluto, y_curr)
            tempo_assoluto += h
            
        if anno >= anni_transitorio:
            # Salviamo i valori in scala lineare (Suscettibili in %, Infetti x 100.000)
            lin_S.append(y_curr[0] * 100)
            lin_I.append(y_curr[2] * 100000)
            
    return lin_S, lin_I

def main():
    print("Calcolo dell'attrattore principale di riferimento in corso...")
    
    y0_base = [0.07, 0.0002, 0.0005]
    base_S, base_I = simula_poincare(y0_base, anni_totali=5000, scarta_transitorio=True)
    
    # Setup del grafico
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.15)
    
    # Disegniamo l'attrattore di sfondo
    ax.scatter(base_S, base_I, s=1, alpha=0.8, label='Attrattore Principale')
    
    ax.set_title("Esplorazione Interattiva - Scala Lineare (Clicca per lanciare un'orbita)")
    ax.set_xlabel("Suscettibili (%)")
    ax.set_ylabel("Infetti (x $10^{-3}$%)")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Limiti dinamici basati sull'attrattore per evitare salti del grafico
    pad_S = (max(base_S) - min(base_S)) * 0.15
    pad_I = (max(base_I) - min(base_I)) * 0.15
    ax.set_xlim(max(0, min(base_S) - pad_S), max(base_S) + pad_S)
    ax.set_ylim(max(0, min(base_I) - pad_I), max(base_I) + pad_I)
    
    colori = ['crimson', 'dodgerblue', 'darkorange', 'forestgreen', 'darkorchid', 'gold']
    conteggio_orbite = [0]
    
    # --- Funzione che gestisce il Click del Mouse ---
    def onclick(event):
        if event.inaxes != ax or event.button != 1:
            return
            
        # 1. Recupera le coordinate dal grafico (sono già nei fattori di scala scelti)
        x_click, y_click = event.xdata, event.ydata
        
        # 2. Inverte la scala per trovare le frazioni reali S0 e I0
        # Evitiamo valori negativi se si clicca leggermente sotto lo zero
        S0 = max(0.0001, x_click / 100)
        I0 = max(0.0000001, y_click / 100000)
        E0 = 0.0002 # Latenti costanti
        
        print(f"\nCliccato! Nuova condizione: S0={S0:.4f} ({x_click:.2f}%), I0={I0:.2e}")
        print("Calcolo della traiettoria (1000 anni, transitorio incluso)...")
        
        start_t = time.perf_counter()
        
        # 3. Simula l'orbita
        nuova_S, nuova_I = simula_poincare([S0, E0, I0], anni_totali=1000, scarta_transitorio=False)
        
        # 4. Scegli un colore e disegna la nuova orbita
        colore_corrente = colori[conteggio_orbite[0] % len(colori)]
        conteggio_orbite[0] += 1
        
        ax.plot(nuova_S, nuova_I, 'o', markersize=2, linewidth=0.5, color=colore_corrente, alpha=0.7)
        ax.scatter(x_click, y_click, color='black', marker='X', s=80, zorder=5)
        
        fig.canvas.draw()
        
        end_t = time.perf_counter()
        print(f"Fatto in {(end_t - start_t):.2f} secondi.")

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    print("Grafico pronto! Clicca in un punto qualsiasi dello spazio bianco per lanciare una condizione iniziale.")
    plt.show()

if __name__ == "__main__":
    main()