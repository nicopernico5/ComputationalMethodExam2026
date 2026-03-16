# Codice Esame Metodi Computazionali 2026 - Niccolò Ruzza - Misleas Epidemic. 2
# Questo codice si propone di risolvere con il metodo di shooting il problema,
# in particolare utilizza un metodo di integrazione in avanti (Eulero - naif)

###############################################################################################
# Sfortunatamento questo codice fallisce per la debole convergenza del metodo di integrazione #
###############################################################################################
import numpy as np
import matplotlib.pyplot as plt

N= 1000000 # Numerosità intervallo
M=3
mu = 0.02
lam = 0.0279
ni = 0.01
beta0 = 1575.
a=0
b=1
h=(b-a)/(N)

# Funzione per il calcolo della contagiosità ad ogni TEMPO associato
def beta(TEMPO):
    beta0 = 1575.
    beta=beta0*(1+np.cos(2*np.pi*(TEMPO)))
    return beta

# Calcolo di tutto y
def calc_y(v):
    y=np.zeros(M*(N))
    for i in range(2*M):
        y[-M+i]=v[i%M]
    for j in range(N-1):
        x_jp= (j-1) * M
        x_j = (j) * M
        x_jd = (j+1) * M

        y[x_jd] = y[x_jp] + 2*h*(mu-beta(h*j)*y[x_j]*y[x_j+2])
        y[x_jd+1] = y[x_jp+1] + 2*h*(beta(h*j)*y[x_j]*y[x_j+2]- y[x_j+1]/lam)
        y[x_jd+2] = y[x_jp+2] + 2*h*(y[x_j+1]/lam - y[x_j+2]/ni)
        """
        if y[x_jd] < 0: y[x_jd] = 1e-3 # Ho tentato di inserire condizioni correttive al metodo
        if y[x_jd+1] < 0: y[x_jd+1] = 1e-3
        if y[x_jd+2] < 0: y[x_jd+2] = 1e-3
        if y[x_jd] > 1.0: y[x_jd] = 0.1
        if y[x_jd+1] > 1.0: y[x_jd+1] = 0.1
        if y[x_jd+2] > 1.0: y[x_jd+2] = 0.1"""
    return y

# Funzione Principale Codice
def main():
    h=(b-a)/(N) # Passo
    t=np.linspace(a,b,N) # Scala dei Tempi

    # Inizializzazione con i valori dati
    v=[0.1, 0.005,0.005]
    max_iter = 100
    eps= 1e-4
    toll = 1e-4
    for i in range(max_iter):
        print("Inizio prova n ",i+1)
        J= np.zeros((M,M))
        y1= calc_y(v)
        for k in range(M):
            v[k]+= eps
            J[:,k]=(calc_y(v)[-3:]-y1[-3:])/eps
            v[k]-=eps
        print("Matrice J:\n", J)
        try:
            delta_v = np.linalg.solve(J, -y1[-M:])
        except np.linalg.LinAlgError:
            print("Matrice singolare, impossibile risolverla.")
            break
        v += delta_v/100
        max_err = np.max(np.abs(delta_v))
        if max_err < toll:
            print("Convergenza raggiunta!")
            break
    
    y_fin=calc_y(v)   

if __name__ == "__main__":
    main()