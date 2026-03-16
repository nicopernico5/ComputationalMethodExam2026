# Codice Esame Metodi Computazionali 2026 - Niccolò Ruzza - Misleas Epidemic. 1
# Questo codice si propone di risolvere con il metodo matriciale di Newton il problema
# dato da un sistema di tre equazioni differenziali a termini misti con condizioni periodiche

import numpy as np # Per lavorare con le matrici
import matplotlib.pyplot as plt # Per la visualizzazione
import time # Per calcolare la durata del processo di calcolo

# Funzione per il calcolo della contagiosità ad ogni TEMPO associato
def beta(TEMPO):
    beta0 = 1575.
    beta=beta0*(1+np.cos(2*np.pi*(TEMPO)))
    return beta

# Funzione Principale Codice
def main():
    start_time = time.perf_counter ()
    N= 1000 # Numerosità intervallo ############################################
    M=3 # Grandezza Sistema di Equazioni Differenziali

    mu = 0.02 # Parametro associato alle nuove entrate nel sistema
    lam = 0.0279 # Parametro associato alla velocità di incubazione
    ni = 0.01 # Parametro associato alla mortalità
    beta0 = 1575. # Parametro associato alla contagiosità della malattia

    a=0 # Valore Iniziale (Tempo)
    b=1 # Valore Finale (Tempo)
    h=(b-a)/(N) # Passo

    t=np.linspace(a,b,N+1) # Scala dei Tempi
    y=np.zeros(M*(N+1)) # Vettore contenente tutti gli step delle equazioni differenziali, ordinati y1(j1), y2(j1), y3(j1), y1(j2)...

    # Inizializzazione con i valori dati
    y[0::3] = 0.1 ##############################################################
    """for i in range(N+1):
        y[1+M*i]= 0.005*(1-np.cos(2* np.pi * h * i))
        y[2+M*i]= 0.005*(1-np.cos(2* np.pi * h * i))"""
    y[1::3] = 0.005 # Tentativo con parametri iniziali "semplici"
    y[2::3] = 0.005

    max_iteration = 100 # Numero massimo di iterazioni per arrivare alla convergenza
    errore= 1e-6 # Differenza massima tra le stesse y_k tra le iterazioni affinchè il risultato sia accettabile

    for i in range(max_iteration):
        q=np.zeros(M*(N+1)) # Vettore degli errori
        J=np.zeros((M*(N+1),M*(N+1))) # Matrice Jacobiana (risolvente)
        
        for j in range(N):
            # Creazione degli indici per facile accesso al vettore q e S
            x_j0 = (j) * M
            x_j1 = (j+1) * M

            # Calcolo dei q-iesimi
            q[x_j0]= y[x_j1] - y[x_j0] - (h/2) * (2*mu - beta((j+1)*h)*y[x_j1]*y[x_j1+2] - beta(j*h)* y[x_j0]*y[x_j0+2])
            q[x_j0+1]= y[x_j1+1] - y[x_j0+1] - (h/2) * (beta((j+1)*h) * y[x_j1]*y[x_j1+2] + beta(j*h)*y[x_j0]*y[x_j0+2] - 1/lam*(y[x_j1+1] + y[x_j0+1]))
            q[x_j0+2]= y[x_j1+2] - y[x_j0+2] - (h/2) * ( 1/lam*(y[x_j1+1] + y[x_j0+1]) -  1/ni*(y[x_j1+2] + y[x_j0+2]) )

            # Inseriamo i blocchi Sj e Rj nella grande matrice Jacobiana J            
            J[x_j0 : x_j1, x_j0 : x_j1] = np.array([
                [-1 + h/2*beta(j*h)*y[x_j0+2],0, +h/2*beta(j*h)*y[x_j0]],
                [-h/2*beta(j*h)*y[x_j0+2],-1+h/(2*lam),- h/2*beta(j*h)*y[x_j0]],
                [0, -h/(2*lam),-1  +h/(2*ni)]
                ])
            J[x_j0 : x_j1, x_j1: x_j1+M] = np.array([
                [1 + h/2*beta((j+1)*h)*y[x_j1+2],0, h/2*beta((j+1)*h)*y[x_j1]],
                [-h/2*beta((j+1)*h)*y[x_j1+2],1+h/(2*lam), -h/2*beta((j+1)*h)*y[x_j1]],
                [0, -h/(2*lam),1 +h/(2*ni)]
                ])
            
        # Definisco gli ultimi 3 posti del vettore q (condizione al contorno)
        for k in range(M):
            q[-M+k]= y[k] - y[-M+k]
            
        # Inserisco le B_a e B_b
        J[-M : , 0 : M] = np.eye(M)
        J[-M : , -M : ] = -np.eye(M)
        
        # Risolviamo la Matrice Jacobiana per trovare il deltay con cui correggere i dati
        delta_y = np.linalg.solve(J, -q)
        y = y + delta_y
    
        # Calcoliamo l'errore massimo per vedere se abbiamo finito
        maxdeltay = np.max(np.abs(delta_y))
        print(f"Iterazione {i+1}, Errore: {maxdeltay:.2e}")
    
        # Stop iterazione se raggiunta la convergenza
        if maxdeltay < errore:
            print("È stata raggiunta la convergenza")
            break
    
    end_time = time.perf_counter ()
    print(f"Elapsed time : {(end_time - start_time):.2e}")
    # Grafici
    plt.figure(figsize=(6, 4))
    plt.title("Misles Epidemic - Andamenti")
    plt.plot(t, y[0::3]*100, marker='o',label="Suscettibili", color="green")
    plt.plot(t, y[1::3]*100, marker='o',label="Latenti", color="blue")
    plt.plot(t, y[2::3]*100, marker='o',label="Malati", color="red")
    plt.legend()
    plt.grid(True)
    plt.xlabel("t")
    plt.ylabel("%")

    # Grafico in risposta
    plt.figure(figsize=(6, 4))
    plt.title("Misles Epidemic - Focus Malati+Latenti")
    plt.plot(t, y[1::3]*100, marker='o',label="Latenti", color="blue")
    plt.plot(t, y[2::3]*100, marker='o',label="Malati", color="red")
    plt.legend()
    plt.grid(True)
    plt.xlabel("t")
    plt.ylabel("%")

    """# Parametro di Contagiosità
    plt.figure(figsize=(6, 4))
    plt.title("Misles Epidemic - Parametro di Contagiosità")
    plt.plot(t, beta(t), marker='o',label="Contagiosità")
    plt.legend()
    plt.grid(True)
    plt.xlabel("t")
    plt.ylabel("Beta")"""
    plt.show()


if __name__ == "__main__":
    main()