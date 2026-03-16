# Codice Esame Metodi Computazionali 2026 - Niccolò Ruzza - Misleas Epidemic. 5
# Questo codice si propone di studiare l'evoluzione nel tempo del problema cercando regolarità.
# Con la speranza di arrivare alla soluzione periodica

# Grafici complicati ma interessanti, c'è qualche regolarità successiva da studiare

import numpy as np
import matplotlib.pyplot as plt
import time

N= 100 # Numerosità intervallo
M=3 # Grandezza Sistema di Equazioni Differenziali
mu = 0.02 # Parametro associato alle nuove entrate nel sistema
lam = 0.0279 # Parametro associato alla velocità di incubazione
ni = 0.01 # Parametro associato alla mortalità
beta0 = 1575. # Parametro associato alla contagiosità della malattia
a=0 # Valore Iniziale (Tempo)
b=1 # Valore Finale (Tempo)
h=(b-a)/(N) # Passo

# Funzione per il calcolo della contagiosità ad ogni TEMPO associato
def beta(t):
    beta0 = 1575.
    beta=beta0*(1+ np.cos(2*np.pi*(t)))
    #beta= beta0
    return beta

# Funzione per il calcolo del vettore delle "velocità"
def f(t,y):
    dy0 = mu - beta(t) * y[0] * y[2]
    dy1 = beta(t) * y[0] * y[2] - y[1] / lam
    dy2 = y[1] / lam - y[2] / ni
    return np.array([dy0, dy1, dy2])

# Ho fatto un tentativo con un metodo Runge-Kutta più forte nella speranza di una periodicità
"""
def calc_y(t, y_curr):
    # Metodo di Runge-Kutta-Fehlberg (RKF5) - Calcolo di Ordine 5
    k1 = f(t, y_curr)
    k2 = f(t + h * (1/4), y_curr + h * (1/4) * k1)
    k3 = f(t + h * (3/8), y_curr + h * (3/32 * k1 + 9/32 * k2))
    k4 = f(t + h * (12/13), y_curr + h * (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3))
    k5 = f(t + h, y_curr + h * (439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4))
    k6 = f(t + h * (1/2), y_curr + h * (-8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5))
    y_next = y_curr + h * (16/135 * k1 + 0 * k2 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6)
    return y_next"""

# Calcolo di y successivo - RUNGE-KUTTA 
def calc_y(t, y_curr):
    k1 = f(t, y_curr)
    k2 = f(t + h/2, y_curr + (h/2) * k1)
    k3 = f(t + h/2, y_curr + (h/2) * k2)
    k4 = f(t + h, y_curr + h * k3)
    return y_curr + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

# Funzione Principale Codice
def main():
    start_time = time.perf_counter ()
    t=np.linspace(a,b,N) # Scala dei Tempi
    y0=np.array([0.07,0.0002, 0.0005])
    #y0=np.array([7.52303960e-02, 1.80055621e-05, 4.98195974e-06]) # Anche con valori "perfetti" non mantiene la periodicità
    yfin = np.copy(y0)

    storia_y = [y0] # Lista che conterrà tutti gli stati
    storia_t = [a]  # Lista che conterrà tutti i tempi continui
    tempo_assoluto = a # Variabile per il tempo che non si azzera mai

    iter=1000
    i=0
    for i in range(iter):
        y1 = np.copy(yfin)
        for j in range(N):
            # Usiamo tempo_assoluto invece di j*h per far avanzare l'orologio
            yfin = calc_y(tempo_assoluto, yfin) 
            tempo_assoluto += h
            # Salviamo il passo appena calcolato
            storia_y.append(yfin)
            storia_t.append(tempo_assoluto)
    
    # Convertiamo le liste in array
    storia_y = np.array(storia_y)
    storia_t = np.array(storia_t)
    suscettibili = storia_y[:, 0] * 100

    indici_massimi = np.where((suscettibili[1:-1] > suscettibili[:-2]) & (suscettibili[1:-1] > suscettibili[2:]))[0] + 1
    if len(indici_massimi) == 0:
        print("Nessun massimo locale trovato.")
    else:
        for idx in indici_massimi:
            tempo_picco = storia_t[idx]
            valore_picco = suscettibili[idx]

    print("--- Analisi Statistica delle Stragi (Picchi Epidemiologici) ---")
    if len(indici_massimi) > 1:
        # Estraiamo gli array dei tempi e dei valori esatti in cui avvengono i picchi
        tempi_picchi = storia_t[indici_massimi]
        valori_picchi = suscettibili[indici_massimi]
        sfasamento_tempi = tempi_picchi - np.round(tempi_picchi)

        intervalli_stragi = np.diff(tempi_picchi)
        tempo_medio = np.mean(intervalli_stragi)
        print(f"Numero totale di picchi registrati: {len(tempi_picchi)} in {iter} anni")
        print(f"Tempo medio tra una strage e l'altra: {tempo_medio:.3f} anni")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        # Istogramma 1: Distribuzione della fase (quando avviene il picco)
        ax1.hist(sfasamento_tempi, bins=60, color='teal', edgecolor='black', alpha=0.9)
        ax1.set_title("Tempismo rispetto al picco di contagio")
        ax1.set_xlabel("Sfasamento dal picco di contagiosità (Anni)")
        ax1.set_ylabel("Frequenza (N. di Stragi)")
        ax1.set_xlim(-0.3, 0.3)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Istogramma 2: Distribuzione dell'ampiezza (quanti suscettibili ci sono)
        ax2.hist(valori_picchi, bins=90, color='coral', edgecolor='black', alpha=0.9)
        ax2.set_title("Distribuzione per % di Suscettibili")
        ax2.set_xlabel("Suscettibili prima del crollo (%)")
        ax2.set_ylabel("Frequenza (N. di Stragi)")
        ax2.set_xlim(6, 15)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        # Istogramma 3: Distribuzione della distanza temporale tra le stragi
        ax3.hist(intervalli_stragi, bins=80, color='mediumpurple', edgecolor='black', alpha=0.7)
        ax3.set_title("Distanza temporale tra le stragi")
        ax3.set_xlabel("Tempo tra due picchi consecutivi (Anni)")
        ax3.set_ylabel("Frequenza (N. di Stragi)")
        ax3.set_xlim(0, 8)
        ax3.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        end_time = time.perf_counter ()
        print(f"Elapsed time : {(end_time - start_time):.2e}")
        plt.show()

    
    #plt.show()

if __name__ == "__main__":
    main()