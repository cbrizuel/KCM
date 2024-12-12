import copy
import numpy as np
from scipy.stats import entropy
"funcion para determinar las probabilidades segun los valores guardados en la poblacion"
def calcula_prob_tot(poblacion,nodos_continuan):
    prob_pob=copy.deepcopy(poblacion)
    for i in range(len(poblacion)):
        for j in range(len(poblacion[i])):
            for k in nodos_continuan:
                prob_pob[i][j][k]=[elemento / sum(poblacion[i][j][k]) for elemento in poblacion[i][j][k]]
    return prob_pob

"funcion para determinar una ejecucion segun un individuo de la poblacion"
def ejecuta_uno(prob_pob,posicion,n,M_ady, indicador, indicador_amino):
    ejecucion=[]
    for i in range(n):
        termina=0
        j=0
        while termina==0:
            j=int(np.random.choice(M_ady[j], size=1, p=prob_pob[posicion][i][j]))
            if indicador[j]==1:
                ejecucion.append(indicador_amino[j])
                termina=1
    return ejecucion
            
def ejecuta_uno_red(prob_pob,posicion,n):
    ejecucion=[]
    pos=0
    for i in range(n):
        pos=int(np.random.choice(list(range(20)), size=1, p=prob_pob[posicion][i][pos]))
        ejecucion.append(pos)
    return ejecucion
            

"no filtra"
def filtro1(tot_ececution,posicion,det_fitness,n_fit,n_rep):
    det_fitness[posicion]=list(range(n_fit))
    return det_fitness
      
    
"determina la similitud entre dos matrices de contactos, utiliza el error cuadratico medio"   
def similitud_MC(MC1,MC2):
    similitud=0
    for i in range(len(MC1)):
        for j in range(i+1,len(MC1[i])):
            similitud+=(MC1[i][j]-MC2[i][j])**2
    return similitud/(len(MC1)*(len(MC1)+1)/2)
               

"Determina la entropia si los datos son observaciones"
def entropia_descrip_val(descriptor_ref, descriptor_temp,lugar):
    data1=descriptor_ref[lugar]
    data2=descriptor_temp[lugar]
    prob1, _ = np.histogram(data1, bins=20, range=[min([min(data1),min(data2)]),max([max(data1),max(data2)])],density=False)
    prob2, _ = np.histogram(data2, bins=20, range=[min([min(data1),min(data2)]),max([max(data1),max(data2)])],density=False)
    prob1=[x + 0.5 for x in prob1]
    prob2=[x + 0.5 for x in prob2]
    kl_divergence = entropy(prob1, prob2)
    return kl_divergence

"Determina la entropia si los datos son probabilidades"
def entropia_descrip_prob(descriptor_ref, descriptor_temp,lugar):
    data1=descriptor_ref[lugar]
    data2=descriptor_temp[lugar]
    data1=[x + 0.01 for x in data1]
    data2=[x + 0.01 for x in data2]
    kl_divergence = entropy(data1, data2)
    return kl_divergence


"ordena de mayor a menor las variables que almacenan las n_best mejores ejecuciones "
def ordena_mejores_energia(best_fitness, best_execution, pdb_select,best_fitness_energ,best_execution_dist):
    ordenada = all(best_fitness[i] >= best_fitness[i + 1] for i in range(len(best_fitness) - 1))
    if not ordenada:
        indices_ordenados = np.argsort(-np.array(best_fitness))
        best_fitness[:] = [best_fitness[idx] for idx in indices_ordenados]
        best_fitness_energ[:] = [best_fitness_energ[idx] for idx in indices_ordenados]
        best_execution[:] = [best_execution[idx] for idx in indices_ordenados]
        best_execution_dist[:] = [best_execution_dist[idx] for idx in indices_ordenados]
        pdb_select[:] = [pdb_select[idx] for idx in indices_ordenados]
    return best_fitness, best_execution, pdb_select, best_fitness_energ,best_execution_dist

def ordena_mejores_all(best_fitness, best_execution):
    ordenada = all(best_fitness[i] >= best_fitness[i + 1] for i in range(len(best_fitness) - 1))
    if not ordenada:
        indices_ordenados = np.argsort(-np.array(best_fitness))
        best_fitness[:] = [best_fitness[idx] for idx in indices_ordenados]
        best_execution[:] = [best_execution[idx] for idx in indices_ordenados]
    return best_fitness, best_execution

"determina la matriz de actualualizacion para cada distribucion donde los elementos en "
"best_execution se toman proporcional al tamanos"
def W_act1(posicion,best_execution,act_glob,tot_ececution,act_prop,det_fitness,fitness1,best_fitness,best_execution_dist,tot_ececution_dist):
    total_fitness = sum(best_fitness)
    probabilidades = [iji / total_fitness for iji in best_fitness]
    if any(p11 == 0 for p11 in probabilidades):
        probabilidades = np.array([0.1 if p44 == 0 else p44 for p44 in probabilidades])
        probabilidades /= sum(probabilidades)
    muestra = np.random.choice(list(range(len(best_fitness))), size=act_glob, p=probabilidades,replace=False)
    muestra_enteros = [int(i) for i in muestra]
    w_act = [best_execution[i] for i in muestra_enteros]
    w_act_dist = [best_execution_dist[i] for i in muestra_enteros]
    temporal=tot_ececution[posicion]
    temporall=tot_ececution_dist[posicion]
    temporal1 = [temporal[idx] for idx in det_fitness[posicion]]
    temporall1 = [temporall[idx] for idx in det_fitness[posicion]]
    indices_ordenados=np.argsort(-np.array(fitness1[posicion]))
    indices_ordenados=indices_ordenados[:act_prop]
    temporal = [temporal1[idx] for idx in indices_ordenados]
    temporall = [temporall1[idx] for idx in indices_ordenados]
    w_act += temporal
    w_act_dist+=temporall
    return w_act,w_act_dist


"actualiza los individuos de la poblacion"
def actualiza_individuo(n,poblacion,posicion,w_act,w_act_dist,indicador_entrada,indicador_entrada_pos,nodo_terminal):
  for i in range(len(w_act)):
    for j in range(n):
      temptemp=nodo_terminal[w_act[i][j]]
      while temptemp!=0:
        poblacion[posicion][j][indicador_entrada[temptemp]][indicador_entrada_pos[temptemp]]+=max(1-w_act_dist[i][j]/10,0)
        temptemp=indicador_entrada[temptemp]
  return poblacion


"actualiza los individuos de la poblacion"
def actualiza_individuo_red(poblacion,all_best_execution,tot_act_red,n):
    for elementos in range(tot_act_red):
        for elementos1 in range(n):
            if elementos1==0:
                poblacion[0][elementos1][elementos1][all_best_execution[elementos][elementos1]]+=1
            else:
                poblacion[0][elementos1][all_best_execution[elementos][elementos1-1]][all_best_execution[elementos][elementos1]]+=1   
    for i in range(1,len(poblacion)):
        poblacion[i]=poblacion[0]
    return poblacion

def calcula_prob_red(poblacion,n):
    prob_salida=[[[0 for _ in range(20)] for _ in range(20)]for _ in range(n)] 
    for i in range(len(poblacion[0])):
        for j in range(20):
            total=sum(poblacion[0][i][j])
            prob_salida[i][j]=[elemento/total for elemento in poblacion[0][i][j]]
    return prob_salida



def calcula_prob_red_M(poblacion,n):
    prob_salida=[[[0 for _ in range(20)] for _ in range(20)]for _ in range(n)] 
    for i in range(len(poblacion[0])):
        for j in range(20):
            total=sum(poblacion[0][i][j])
            prob_salida[i][j]=[elemento/total for elemento in poblacion[0][i][j]]
    pos=0
    constants=[0]*20
    constants[pos]=1
    for i in range(1,n):
        coefficients = copy.deepcopy(prob_salida[i])
        for iii in range(20):
            for jjj in range(20):
                coefficients[iii][jjj] = -coefficients[iii][jjj]
        for j in range(20):
            coefficients[j][j] =coefficients[j][j]+ 1
        coefficients=np.array(coefficients)
        coefficients=coefficients.transpose()
        coefficients[pos]=np.array([1]*20)
        solution=np.linalg.solve(coefficients, constants)
        if i ==1:
            prob_salida[0][0]=solution.tolist()
            for ff in range(20):
                prob_salida[i][ff]=solution.tolist()
        else:
            for ff in range(20):
                prob_salida[i][ff]=solution.tolist()
    return prob_salida


