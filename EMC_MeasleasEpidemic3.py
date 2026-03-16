# Codice Esame Metodi Computazionali 2026 - Niccolò Ruzza - Measleas Epidemic. 3
# Questo codice si propone di risolvere con il metodo di shooting il problema,
# in particolare utilizza un metodo di integrazione in avanti (RUNGE-KUTTA),
# imponendo successivamente una correzione per arrivare alla soluzione periodica

# Notiamo felicemente che per qualsiasi valore sensato iniziale converge, 
# in meno tempo che il metodo matriciale

import numpy as np
import matplotlib.pyplot as plt
import time

N= 10000 # Numerosità intervallo ###################################################
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
    return beta0*(1+np.cos(2*np.pi*(t)))

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
    y0=np.array([0.3, 0.005,0.005]) ##############################################

    max_iter = 50
    eps= 1e-6
    toll = 1e-6
    for i in range(max_iter):
        yfin = np.copy(y0)
        for j in range(N):
            yfin= calc_y(j*h,yfin)
        
        J= np.zeros((M,M))
        for k in range(M):
            y0[k]+= eps
            y1 = np.copy(y0)
            for l in range(N):
                y1= calc_y(l*h,y1)
            J[:,k]=(y1-yfin)/eps
            y0[k]-=eps
        
        J-=np.eye(M)
        try:
            delta_y = np.linalg.solve(J, -(yfin-y0))
        except np.linalg.LinAlgError:
            print("Matrice singolare")
            break
        max_err = np.max(np.abs(delta_y))
        print(f"Iterazine {i+1},"f"Errore: {max_err:.2e}")
        if max_err < toll:
            print("Convergenza raggiunta!")
            print("Con valori al contorno:", y0)            
            break
        y0 += delta_y/1.5 # Correggiamo più leggermente il valore iniziale (Metodo di Smorzamento- Damped)

    trail = np.zeros((N, M))
    trail[0] = y0
    for j in range(N-1):
        trail[j+1] = calc_y(j*h, trail[j])

    end_time = time.perf_counter ()
    print(f"Elapsed time : {(end_time - start_time):.2e}")

    plt.figure(figsize=(6, 4))
    plt.title("Measles Epidemic - Andamenti")
    plt.plot(t, trail[:, 1] * 100, label="Latenti")
    plt.plot(t, trail[:, 2] * 100, label="Malati")
    plt.plot(t, trail[:, 0] * 100, label="Suscettibili")
    plt.legend()
    plt.grid(True)
    plt.xlabel("t")
    plt.ylabel("%")

    plt.figure(figsize=(6, 4))
    plt.title("Measles Epidemic - Focus Malati+Latenti")
    plt.plot(t, trail[:, 1] * 100, label="Latenti")
    plt.plot(t, trail[:, 2] * 100, label="Malati")
    plt.legend()
    plt.grid(True)
    plt.xlabel("t")
    plt.ylabel("%")

    plt.show()

if __name__ == "__main__":
    main()