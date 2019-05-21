"""
Python: Machine Learning, Optimización y Aplicaciones, 2017
ga1: primer problema de optimización con algoritmo genético.
rellenar una lista con 1
"""

# módulos de Python que vamos a utilizar
import random
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import scipy as sp

def crea_individuo_random(size= 6):
    """ Creamos un individuo, las primeras posiciones son
    de los puntos de los codos, la última el diámetro"""
    max_x = 1140
    individuo = [random.uniform(0, max_x) for n in range(size-1)]
    individuo.sort()
    individuo.append(random.uniform(1, 33)) # diametro en cm
    return individuo

#def crea_individuo(size= 1140, datos= np.loadtxt("river_Honduras.csv", delimiter=";")):
#    """ Crea individuo a partir de los valores discretos
#    del perfil del río"""
#    
#    x = datos[:,0]
#    x1 = random.uniform(0, size)
#    x2 = random.uniform(0, size)
#    if x1 > x2:
#        x3 = x1
#        x1 = x2
#        x2 = x3
#    elif x1 == x2:
#        if x2 < size - 2:
#            x2 = x2 + 2
#        else:
#            x1 = x1 - 2
#            
#    if (x2 == (x1 + 1)):
#        if x2 < size -2:
#            x2 = x2 + 1
#        else:
#            x1 = x1 - 1
#    
#    ind = x[x > x1]
#    individuo = ind[ind < x2]
#    individuo = np.append(individuo, (random.uniform(1, 33)))
#       
#    return individuo

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
    
def mut_gaussiana(individuo, indpb, pmin, pmax, p_pop, toolbox, sigma):
    
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
        individuo = toolbox.individual_dummy(n = len(ind_aux_ordenado_d))
        individuo[:] = ind_aux_ordenado_d[:]
    else:
        individuo[:] = ind_aux_ordenado_d[:] 
    return individuo,

def cxBlend(ind1, ind2, alpha):
    """Executes a blend crossover that modify in-place the input individuals.
    The blend crossover expects :term:`sequence` individuals of floating point
    numbers.
    """
    ind1_nd = ind1[0:-1] # puntos del rio
    ind2_nd = ind2[0:-1]
    
    D1 = ind1[-1] # diámetros
    D2 = ind2[-1]
    
    #print("diametros", D1, D2)
    
    l1 = len(ind1_nd) # longitudes de las soluciones
    l2 = len(ind2_nd)
    
    if l1 <= l2:
        ind1_aux = ind1_nd.copy()
        ind2_aux = ind2_nd.copy()
        # añadimos ceros
        ind1_aux = np.append(ind1_nd, np.zeros(l2-l1))
    else:
        ind1_aux = ind1_nd.copy()
        ind2_aux = ind2_nd.copy()
        # añadimos ceros
        ind2_aux = np.append(ind2_nd, np.zeros(l1-l2))
    
    # operaciones de blend    
    for i, (x1, x2) in enumerate(zip(ind1_aux, ind2_aux)):
        gamma = (1. + 2. * alpha) * random.random() - alpha
        ind1_aux[i] = abs((1. - gamma) * x1 + gamma * x2)
        ind2_aux[i] = abs(gamma * x1 + (1. - gamma) * x2)
    
    # recuperamos las longitudes originales
    ind1_aux = ind1_aux[0:l1]
    ind2_aux = ind2_aux[0:l2]
        
    ind1_ordenado = np.sort(ind1_aux) # siempre tenemos que ordenar las soluciones
    ind2_ordenado = np.sort(ind2_aux)
    
    gamma = (1. + 2. * alpha) * random.random() - alpha
    ind1_ordenado = np.append(ind1_ordenado, [abs((1. - gamma) * D1 + gamma * D2)])
    ind2_ordenado = np.append(ind2_ordenado, [abs(gamma * D1 + (1. - gamma) * D2)])
    
    ind1[:] = ind1_ordenado[:]
    ind2[:] = ind2_ordenado[:]
    
    return ind1, ind2

def fitness_function_multiobjective(individual, datos):
    x = datos[:,0]
    z = datos[:,1]
    sred = np.array(individual[:-1]) # puntos de nuestra solución
    
    if (len(sred) < 2): # individuo no válido porque es muy pequeño        coste = 1000000,
        coste = 1000000,
        #print("el tamaño no es valido")
        return coste
    D = individual[-1]*10**-2 # diámetro
    
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
        potencia = -1000000
        return coste, potencia
    if (all(comprueba_inferior <= 1.5) == False):
        #print("No cumple inferior")
        coste = 1000000
        potencia = -1000000
        return coste, potencia
    
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

    if potencia < 8e3:
        coste = 1000000
        potencia = -1000000
        return coste, potencia
    if caudal > 35e-3:
        coste = 1000000
        potencia = -1000000
        return coste, potencia
    
    return coste, potencia   

def fitness_function_single(individual, datos):
    """ función de fitness con un único objetivo, el individio
    representa las x y éstas están siempre ordenadas,
    datos, son los datos del río"""
    x = datos[:,0]
    z = datos[:,1]
    sred = np.array(individual[:-1]) # puntos de nuestra solución
    
    if (len(sred) < 2): # individuo no válido porque es muy pequeño        coste = 1000000,
        coste = 1000000,
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
        coste = 1000000,
        return coste
    if (all(comprueba_inferior <= 1.5) == False):
        #print("No cumple inferior")
        coste = 1000000,
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
        coste = 1000000,
        return coste
    if caudal > 35e-3:
        coste = 1000000,
        return coste
    
    return coste,  

def configura_toolbox(sigma, p_cut):
    # paso1: creación del problema
    #creator.create("Problema1", base.Fitness, weights=(1.0,-1,-1))
    creator.create("Problema1", base.Fitness, weights=(-1,))

    # paso2: creación del individuo
    creator.create("Individual", np.ndarray, fitness=creator.Problema1)

    toolbox = base.Toolbox() # creamos la caja de herramientas
    
    # Registramos nuevas funciones

    toolbox.register("inicio_booleano", random.uniform, 0, 1)

    toolbox.register("individual", tools.initIterate, creator.Individual, crea_individuo)
    toolbox.register("individual_dummy", tools.initRepeat, creator.Individual, toolbox.inicio_booleano)

    #toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.inicio_booleano, 200)

    #toolbox.register("individual", tools.initIterate, creator.Individual, crea_individuo)
    toolbox.register("ini_poblacion", tools.initRepeat, list, toolbox.individual)

    # Operaciones genéticas

    #toolbox.register("evaluate", fitness_function, datos = np.loadtxt("river.csv", delimiter=";"))
    toolbox.register("evaluate", fitness_function_single, datos = np.loadtxt("river_Honduras.csv", delimiter=";"))
    #toolbox.register("evaluate", fitness_function_multiobjective, datos = np.loadtxt("river_Honduras.csv", delimiter=";"))
    toolbox.register("mate", cxBlend, alpha= 0.5)
    toolbox.register("mutate", mut_gaussiana, indpb=0.1, pmin=0, pmax= 1140, p_pop = p_cut, toolbox = toolbox, sigma = sigma)
    #toolbox.register("select", tools.selNSGA2)
    toolbox.register("select", tools.selTournament, tournsize = 3)
    return toolbox

def unico_objetivo_ga(c, m, i, sigma, p_cut):
    NGEN = 200
    MU = 2000
    LAMBDA = 2000
    CXPB = c
    MUTPB = m
    random.seed(i) # actualizamos la semilla cada vez que hacemos una simulación
    toolbox = configura_toolbox(sigma, p_cut)
    
    pop = toolbox.ini_poblacion(n=MU)
    hof = tools.HallOfFame(1, similar=np.array_equal)
    
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    
    #stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_fit.register("avg", np.mean)
    stats_fit.register("std", np.std)
    stats_fit.register("min", np.min)
    stats_fit.register("max", np.max)
    
    stats_size.register("avg", np.mean)
    stats_size.register("std", np.std)
    stats_size.register("min", np.min)
    stats_size.register("max", np.max)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "fitness", "size"
    logbook.chapters["fitness"].header = "min", "avg", "max"
    logbook.chapters["size"].header = "min", "avg", "max"
    
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN,
                              stats= mstats, halloffame=hof, verbose = False)
    
    return pop, hof, logbook


def multi_objetivo_ga():
    NGEN = 200
    MU = 2000
    LAMBDA = 2000
    CXPB = 0.5
    MUTPB = 0.5
    sigma = 0.05
    p_cut = 1
    toolbox = configura_toolbox(sigma, p_cut)
    
    pop = toolbox.ini_poblacion(n=MU)
    hof = tools.ParetoFront(similar=np.array_equal)
    random.seed(64) # semilla del generador de números aleatorios
    
    
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN,
                              halloffame=hof)
    
    return pop, hof

#toolbox = configura_toolbox(sigma = 0.05)
#individuo = toolbox.individual()
#individuo2= toolbox.mutate(individuo)
#ind1, ind2 = toolbox.mate(individuo, individuo2)


#%%
def plot(log):
    gen = log.select("gen")
    fit_mins = log.chapters["fitness"].select("min")
    fit_maxs = log.chapters["fitness"].select("max")
    fit_ave = log.chapters["fitness"].select("avg")
    
    length_mins = log.chapters["size"].select("min")
    length_maxs = log.chapters["size"].select("max")
    length_ave = log.chapters["size"].select("avg")
    
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()
    ax1.plot(gen, fit_mins, "b")
    #ax1.plot(gen, fit_maxs, "r")
    #ax1.plot(gen, fit_ave, "--k")
    #ax1.fill_between(gen, fit_mins, fit_maxs, where=fit_maxs >= fit_mins, facecolor='g', alpha = 0.2)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.legend(["Min", "Max", "Avg"])
    ax1.set_ylim([0, 50])
    ax1.set_xlim([0, 210])
    plt.grid(True)
    
    print(length_mins)
    fig, ax2 = plt.subplots()
    ax2.plot(gen, length_mins, "b")
    ax2.plot(gen, length_maxs, "r")
    ax2.plot(gen, length_ave, "--k")
    ax2.fill_between(gen, length_mins, length_maxs, where=length_maxs >= length_mins, facecolor='g', alpha = 0.2)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Length")
    ax2.legend(["Min", "Max", "Avg"])
    ax2.set_ylim([0, 100])
    ax2.set_xlim([0, 210])
    plt.grid(True)

if __name__ == "__main__":
    multi_objetivo = False
    if multi_objetivo == True:
        res_individuos = open("individuos.txt", "w")
        res_fitness = open("fitness.txt", "w")
        pop_new, pareto_new = multi_objetivo_ga()
        for ide, ind in enumerate(pareto_new):
            res_individuos.write(str(ind))
            res_individuos.write("\n")
            res_fitness.write(str(ind.fitness.values[0]))
            res_fitness.write(",")
            res_fitness.write(str(ind.fitness.values[1]))
            #res_fitness.write(",")
            #res_fitness.write(str(ind.fitness.values[2]))
            res_fitness.write("\n")
        res_fitness.close()
        res_individuos.close()
    else:    
        #parameters= [(0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]
        parameters= [1]
        for p_cut in parameters:
            c = 0.5
            m = 0.5
            sigma = 0.05
            for i in range(0, 1):
                res_individuos = open("individuos.txt", "a")
                res_fitness = open("fitness.txt", "a")
                pop_new, pareto_new, log = unico_objetivo_ga(c, m, int(i), sigma, p_cut)
                for ide, ind in enumerate(pareto_new):
                    res_individuos.write(str(i))
                    res_individuos.write(",")
                    res_individuos.write(str(c))
                    res_individuos.write(",")
                    res_individuos.write(str(m))
                    res_individuos.write(",")
                    res_individuos.write(str(sigma))
                    res_individuos.write(",")
                    res_individuos.write(str(p_cut))
                    res_individuos.write(",")
                    res_individuos.write(str(ind))
                    res_individuos.write("\n")
                    res_fitness.write(str(i))
                    res_fitness.write(",")
                    res_fitness.write(str(c))
                    res_fitness.write(",")
                    res_fitness.write(str(m))
                    res_fitness.write(",")
                    res_fitness.write(str(sigma))
                    res_fitness.write(",")
                    res_fitness.write(str(p_cut))
                    res_fitness.write(",")
                    res_fitness.write(str(ind.fitness.values[0]))
                    res_fitness.write("\n")
                del(pop_new)
                del(pareto_new)
                res_fitness.close()
                res_individuos.close()
        plot(log)
