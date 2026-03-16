# Codice Esame Metodi Computazionali 2026 - Niccolò Ruzza - Misleas Epidemic. 6
# Questo codice si propone di studiare l'evoluzione nel tempo del problema cercando regolarità.
# In particolare facendo un diagramma delle fasi

# NESSUNA REGOLARITÀ DA EVIDENZIARE :(

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks

N = 200         # Passi per anno
mu = 0.02
lam = 0.0279
ni = 0.01
beta0 = 1575.
a=0 # Valore Iniziale (Tempo)
b=1 # Valore Finale (Tempo)
h=(b-a)/(N) # Passo

# Funzione vettorializzata per le derivate (supporta y come array 2D)
def f_vec(t, y):
    beta = beta0 * (1 + np.cos(2 * np.pi * t))
    dy0 = mu - beta * y[:, 0] * y[:, 2]
    dy1 = beta * y[:, 0] * y[:, 2] - y[:, 1] / lam
    dy2 = y[:, 1] / lam - y[:, 2] / ni
    return np.column_stack([dy0, dy1, dy2])

# RUNGE-KUTTA a 4
def calc_y_vec(t, y_curr, h):
    k1 = f_vec(t, y_curr)
    k2 = f_vec(t + h/2, y_curr + (h/2) * k1)
    k3 = f_vec(t + h/2, y_curr + (h/2) * k2)
    k4 = f_vec(t + h, y_curr + h * k3)
    return y_curr + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

# Funzione Principale Codice
def main():
    start_time = time.perf_counter()

    anni_transitorio = 50    # Anni da scartare per far stabilizzare il sistema
    anni_analisi = 200      # Anni effettivi di cui cercare i picchi #####################################
    anni_totali = anni_transitorio + anni_analisi
    iterazioni_totali = anni_totali * N
    
    # Creazione della griglia di Condizioni Iniziali
    res = 100 # Risoluzione della griglia ##################################################################
    s_iniziali = np.linspace(0.04, 0.12, res)       # Suscettibili dal 4% al 12%
    i_iniziali = np.linspace(0.00001, 0.001, res)   # Infetti da 0.001% a 0.1%
    
    SS, II = np.meshgrid(s_iniziali, i_iniziali)
    EE = np.full_like(SS, 0.0002) # Manteniamo gli Esposti iniziali costanti
    
    # Appiattiamo le griglie in un singolo array (1600 righe, 3 colonne)
    y_curr = np.column_stack([SS.ravel(), EE.ravel(), II.ravel()])
    num_sims = y_curr.shape[0]
    
    # Matrice per salvare solo la parte di analisi (Suscettibili)
    # Forma: (anni_analisi * N, num_sims)
    storia_suscettibili = np.zeros((anni_analisi * N, num_sims))
    storia_t = np.zeros(anni_analisi * N)
    
    tempo_assoluto = 0.0
    
    print(f"Esecuzione di {num_sims} simulazioni parallele in corso...")
    
    # Esecuzione del ciclo del tempo
    idx_salvataggio = 0
    for step in range(iterazioni_totali):
        y_curr = calc_y_vec(tempo_assoluto, y_curr, h)
        tempo_assoluto += h
        
        # Iniziamo a salvare i dati solo dopo il transitorio
        if step >= anni_transitorio * N:
            storia_suscettibili[idx_salvataggio, :] = y_curr[:, 0] * 100 # In %
            storia_t[idx_salvataggio] = tempo_assoluto
            idx_salvataggio += 1

    print("Simulazioni completate. Analisi dei picchi in corso...")
    
    # --- Analisi dei risultati per ogni punto della griglia ---
    # Inizializziamo le matrici dei risultati che mapperemo a colori
    mappa_numero_picchi = np.zeros(num_sims)
    mappa_distanza_media = np.zeros(num_sims)
    mappa_picco_susc = np.zeros(num_sims)
    
    for i in range(num_sims):
        susc = storia_suscettibili[:, i]
        # Troviamo i picchi (minimo 1 anno/N passi di distanza per ignorare rumore microscopico)
        picchi_idx, _ = find_peaks(susc, distance=N*0.8) 
        
        mappa_numero_picchi[i] = len(picchi_idx)
        
        if len(picchi_idx) > 1:
            tempi_picchi = storia_t[picchi_idx]
            valori_picchi = susc[picchi_idx]
            
            mappa_distanza_media[i] = np.mean(np.diff(tempi_picchi))
            mappa_picco_susc[i] = np.mean(valori_picchi)
        elif len(picchi_idx) == 1:
            mappa_distanza_media[i] = 0 # Solo un picco, impossibile fare la media
            mappa_picco_susc[i] = susc[picchi_idx[0]]
        else:
            mappa_distanza_media[i] = np.nan
            mappa_picco_susc[i] = np.nan

    # Rimettiamo i risultati in forma 2D per il plotting
    M_num = mappa_numero_picchi.reshape(res, res)
    M_dist = mappa_distanza_media.reshape(res, res)
    M_val = mappa_picco_susc.reshape(res, res)
    
    # --- Generazione dei Grafici delle Fasi ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Dettagli per il formato degli assi
    extent = [s_iniziali[0]*100, s_iniziali[-1]*100, i_iniziali[0]*1000, i_iniziali[-1]*1000]
    
    # Grafico 1: Distanza media temporale
    im1 = ax1.imshow(M_dist, origin='lower', extent=extent, aspect='auto', cmap='viridis')
    ax1.set_title("Distanza temporale media tra le stragi (Anni)")
    ax1.set_xlabel("Suscettibili iniziali (%)")
    ax1.set_ylabel("Infetti iniziali (x10^-3 %)")
    fig.colorbar(im1, ax=ax1)
    
    # Grafico 2: Numero totale di picchi registrati (nell'arco di 100 anni)
    im2 = ax2.imshow(M_num, origin='lower', extent=extent, aspect='auto', cmap='plasma')
    ax2.set_title(f"Numerosità dei Picchi (su {anni_analisi} anni)")
    ax2.set_xlabel("Suscettibili iniziali (%)")
    ax2.set_ylabel("Infetti iniziali (x10^-3 %)")
    fig.colorbar(im2, ax=ax2)
    
    # Grafico 3: Quantità media di suscettibili al momento del crollo
    im3 = ax3.imshow(M_val, origin='lower', extent=extent, aspect='auto', cmap='magma')
    ax3.set_title("Suscettibili al picco (%)")
    ax3.set_xlabel("Suscettibili iniziali (%)")
    ax3.set_ylabel("Infetti iniziali (x10^-3 %)")
    fig.colorbar(im3, ax=ax3)

    plt.tight_layout()
    end_time = time.perf_counter()
    print(f"Tempo totale di esecuzione: {(end_time - start_time):.2f} secondi")
    plt.show()

if __name__ == "__main__":
    main()