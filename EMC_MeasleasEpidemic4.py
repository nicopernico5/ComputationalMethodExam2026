# Codice Esame Metodi Computazionali 2026 - Niccolò Ruzza - Misleas Epidemic. 4
# Questo codice si propone di risolvere con il metodo di rilassamento il problema,
# in particolare utilizza un metodo di integrazione in avanti (RUNGE-KUTTA)
# imponendo successivamente una correzione per arrivare alla soluzione periodica

# Non converge se non per valori iniziali perfetti

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

# Funzione per il calcolo della contagiosità ad ogni TEMPO associato - interessante studiare il problema con beta fisso
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
    y0=np.array([0.07,0.0002, 0.0005]) #########################################################
    #y0=np.array([7.52303960e-02, 1.80055621e-05, 4.98195974e-06]) #(Con valori perfetti converge)
    yfin = np.copy(y0)

    storia_y = [y0] # Lista che conterrà tutti gli stati
    storia_t = [a]  # Lista che conterrà tutti i tempi continui
    tempo_assoluto = a

    max_err= 1.
    toll = 1e-5
    iter=50 ####################
    i=0
    while max_err > toll and i<iter:
        y1 = np.copy(yfin)
        for j in range(N):
            yfin = calc_y(tempo_assoluto, yfin) 
            tempo_assoluto += h
            
            # Salviamo il passo appena calcolato
            storia_y.append(yfin)
            storia_t.append(tempo_assoluto)
        
        max_err = np.max(np.abs(y1-yfin))
        i+=1
    end_time = time.perf_counter ()
    print(f"Elapsed time : {(end_time - start_time):.2e}")

    # Convertiamo le liste in array
    storia_y = np.array(storia_y)
    storia_t = np.array(storia_t)

    # Estraiamo la storia dei Suscettibili (in percentuale)
    suscettibili = storia_y[:, 0] * 100

    indici_massimi = np.where((suscettibili[1:-1] > suscettibili[:-2]) & 
        (suscettibili[1:-1] > suscettibili[2:]))[0] + 1

    if len(indici_massimi) == 0:
        print("Nessun massimo trovato.")
    else:
        for idx in indici_massimi:
            tempo_picco = storia_t[idx]
            valore_picco = suscettibili[idx]

    plt.figure(figsize=(10, 6))
    plt.plot(storia_t, storia_y[:, 0] * 100, label="Suscettibili (Intera Storia)", alpha=0.8)
    plt.plot(storia_t, storia_y[:, 1] * 100, label="Latenti (Intera Storia)", alpha=0.8)
    plt.plot(storia_t, storia_y[:, 2] * 100, label="Malati (Intera Storia)", alpha=0.8)
    plt.title("Andamento temporale verso il ciclo limite")
    plt.xlabel("Anni simulati")
    plt.ylabel("Percentuale Popolazione")
    plt.legend()
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    main()