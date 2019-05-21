# -*- coding: utf-8 -*-
"""
Created on Mon Oct 02 12:31:15 2017

@author: dany
"""

from __future__ import print_function
import random
from simanneal import Annealer
import numpy as np
import scipy as sp

def crea_individuo(size=1242, datos=np.loadtxt("river_Honduras.csv", delimiter=";")):
    """ CONSTANTES """
    RHO  = 1000.0
    G    = 9.8
    F    = 2e-3
    DNOZ = 22e-3
    SNOZ = (np.pi*DNOZ**2)/4
    REND = 0.9
    "Nivel de refinado"
    k_rf = 2.0
    "Se elige un diámetro D aleatorio"
    D = random.uniform(0.10, 0.33)
    "Se remalla el dominio para asegurar que hay factibilidad al coger nodos contiguos"
    s  = datos[:,0]
    z  = datos[:,1]
    cs = sp.interpolate.PchipInterpolator(s, z)
    DZ = z[-1] - z[0]
    N_rf = int(k_rf * DZ/1.5)
    s_rf = np.linspace(s[0],s[-1],N_rf)
    "Calculo el punto máximo de x1  a partir del cual las soluciones son la misma"
    s_izq = s_rf[:-1]
    individuo = s_rf[-1]
    for i in range(1,len(s_izq)):
        "Añado un nodo por la izquierda"
        individuo   = np.append(s_izq[-i],individuo)
        individuo_z = cs(individuo)
        "Calculo la Potencia obtenida"
        L  = np.sum(np.sqrt((individuo[1:] - individuo[0:-1])**2 + (individuo_z[1:] - individuo_z[0:-1])**2))
        Hg = individuo_z[-1]-individuo_z[0]
        P  = REND * (RHO/(2*SNOZ**2))*(Hg/(1/(2*G*SNOZ**2)+F*L/(D**5)))**(3/2)
        "Compruebo si cumple Potencia, o si debo seguir añadiendo nodos"
        if P>8000:
            break
    x1max = individuo[0]
    x1 = random.uniform(0,x1max) 
    individuo = x1
    s_der = s_rf[s_rf>x1]
    for j in range(0,len(s_der)):
        individuo   = np.append(individuo,s_der[j])
        individuo_z = cs(individuo)
        L  = np.sum(np.sqrt((individuo[1:] - individuo[0:-1])**2 + (individuo_z[1:] - individuo_z[0:-1])**2))
        Hg = individuo_z[-1]-individuo_z[0]
        P  = REND * (RHO/(2*SNOZ**2))*(Hg/(1/(2*G*SNOZ**2)+F*L/(D**5)))**(3/2)
        if P>8000:
            break
    individuo = np.append(individuo, D*1e2)
    return individuo
    
def mut_gaussiana(individuo, indpb=0.1, pmin=0, pmax= 1140, p_pop=1, sigma=0.05):
    
    ind_aux = individuo[0:-1] # copiamos todo menos el ultimo
    
    if random.random() <= indpb:
        diametro = random.gauss(individuo[-1], individuo[-1]*sigma) # desviación de 0.3
    else:
        diametro = individuo[-1]

    #eliminar un punto de manera aleatoria
    reduce_longitud = 0
    if random.random() <= p_pop and len(ind_aux) > 1:
        reduce_longitud = 1
        indice = [random.randint(0, len(ind_aux)-1)] 
        ind_aux = np.delete(ind_aux, indice)
        
    # mutación gaussiana    
    for i in range(len(ind_aux)-2): # el último no se tiene en cuenta porque es el diametro
        if random.random() <= indpb:
            old = ind_aux[i]
            try:
                ind_aux[i] = random.gauss(ind_aux[i], float(ind_aux[i]*sigma)) # desviación de 100
            except:
                print(i)
                print(ind_aux[i])
                print(sigma)
            if ind_aux[i] < pmin or ind_aux[i] > pmax:
                ind_aux[i] = old
                
    # ordenar los individuos
    ind_aux_ordenado = np.sort(ind_aux)
    ind_aux_ordenado_d = np.append(ind_aux_ordenado, diametro) 
    if reduce_longitud == 1: # ha menguado
        individuo = [random.uniform(0,1) for i in range(len(ind_aux_ordenado_d))]
        individuo[:] = ind_aux_ordenado_d[:]
    else:
        individuo[:] = ind_aux_ordenado_d[:] 
    return individuo


def fitness_function_single(individual, datos):
    """ función de fitness con un único objetivo, el individio
    representa las x y éstas están siempre ordenadas,
    datos, son los datos del río"""
    x = datos[:,0]
    z = datos[:,1]
    sred = np.array(individual[:-1]) # puntos de nuestra solución
    
    if (len(sred) < 2): # individuo no válido porque es muy pequeño        coste = 1000000,
        coste = 1000000
        #print("el tamaño no es valido")
        return coste
    D = individual[-1]*10**-2 # diámetro
    #D = 20*10**-2
    cs = sp.interpolate.PchipInterpolator(x, z)
    #cs = sp.interpolate.CubicSpline(x, z) # interpolación con splines
    zred = cs(sred) # alturas de la solución

    f_interpola = sp.interpolate.interp1d(sred, zred) # interpolación lineal
    # sirve para saber si se cumples las restricciones

    x_intermedio = np.linspace(min(sred), max(sred), 100)
    z_intermedio = cs(x_intermedio)

    comprueba_superior = f_interpola(x_intermedio) - z_intermedio
    comprueba_inferior = -1 * comprueba_superior        
    
    if (all(comprueba_superior <= 1.5) == False):
        #print("No cumple superior")
        coste = 1000000
        return coste
    if (all(comprueba_inferior <= 1.5) == False):
        #print("No cumple inferior")
        coste = 1000000
        return coste
    
    Hg = zred[-1] - zred[0] # diferencia de altura
    L= np.sum(np.sqrt((sred[1:] - sred[0:-1])**2 + (zred[1:] - zred[0:-1])**2))

    RHO  = 1000
    G    = 9.8
    F    = 2e-3
    DNOZ = 22e-3
    SNOZ = (np.pi*DNOZ**2)/4
    REND = 0.9
    
    potencia = REND * (RHO/(2*SNOZ**2))*(Hg/(1/(2*G*SNOZ**2)+F*L/(D**5)))**(3/2)
    caudal = (Hg/(1/(2*G*SNOZ**2)+F*L/D**5))**(1/2)
    # ahora el número de codos es len(sred) -1, hablar con Ale
    coste = (L + 50 * (len(sred)-1)) * D**2 
#    print("potencia", potencia)
#    print("longitud ", L)
#    print("caudal ", caudal)
#    print("Altura Hg ", Hg)
#    print("Sred", sred)
#    print("Zred", zred)
    if potencia < 8e3:
        coste = 1000000
        return coste
    if caudal > 35e-3:
        coste = 1000000
        return coste
    
    return coste  


class Turbina(Annealer):

    """Test annealer with a travelling salesman problem.
    """

    # pass extra data (the distance matrix) into the constructor
    def __init__(self, state):
        super(Turbina, self).__init__(state)  # important!

    def move(self):
        """Swaps two cities in the route."""
        #self.state = mutacion(self.state)
        self.state = mut_gaussiana(self.state)

    def energy(self):
        """Calculates the length of the route."""
        e = 0
        datos=np.loadtxt("river_Honduras.csv", delimiter=";")
        e = fitness_function_single(self.state, datos)
        return e


if __name__ == '__main__':
    res_individuos = open("individuos.txt", "w")
    res_fitness = open("fitness.txt", "w")
    # initial state, a randomly-ordered itinerary
    for i in range(30):
        init_state = crea_individuo()
        tur = Turbina(init_state)
        #auto_schedule = tur.auto(minutes=1) 
        #tur.set_schedule(auto_schedule)
        tur.steps = 100000
        # since our state is just a list, slice is the fastest way to copy
        tur.copy_strategy = "slice"
        state, e = tur.anneal()
        res_individuos.write(str(state))
        res_individuos.write("\n")
        res_fitness.write(str(e))
        res_fitness.write("\n")
        del(tur)
    res_fitness.close()
    res_individuos.close()
        
        