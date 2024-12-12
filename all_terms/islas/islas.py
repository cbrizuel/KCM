import copy
from ..pdb_seq_tools.pdb_seq_tools import extract_amino_acid_sequence_pdb,extract_backbone_atoms,det_sec,extract_backbone_atoms_str,extract_amino_acid_sequence
from ..Fitness.fitness import descriptores,mapa_contacto,fitness_gdt_rmsd_mc_fisquim,fitness_gdt_rmsd_mc,agrega_rmsd_gdt_E_MC_divKl,agrega_rmsd_gdt_E_MC,ESM2_desc
from ..EDA_tools.EDAtools import calcula_prob_tot, ejecuta_uno, filtro1,ordena_mejores_energia,W_act1,actualiza_individuo,ejecuta_uno_red
from ..getDstructure.get_structure import _request_esmfold_prediction,parse_output,get_hash,genera
import pyrosetta
import numpy as np


def inicializa_isla_1(n,n_pop,amino,n_best):
    global best_execution_1
    global best_execution_dist_1
    global pdb_select_1
    global best_fitness_1
    global best_fitness_energ_1
    nodo_terminal_1=[7,10,13,14,15,17,18,19,20,22,24,25,26,27,28,29,30,31,32,33]
    amino_nodo_1=['F','C','K','R','H','I','M','Y','W','P','D','E','N','Q','V','L','S','T','A','G'] 
    rango_numeros_1 = set(range(34))  
    "nodos que continuan"
    nodos_continuan_1 = list(rango_numeros_1 - set(nodo_terminal_1))
    "indica si el nodo es terminal=1 o si continua=0"
    indicador_1 = [0] * 34
    for posicion in nodo_terminal_1:
        indicador_1[posicion] = 1
    "reorganiza las posiciones de los aminoaciodos segun el; nuevo orden dado por amino_nodo" 
    letra_a_indice_1 = {letra: indice for indice, letra in enumerate(amino)}
    num_amino_new_1 = [letra_a_indice_1[letra] for letra in amino_nodo_1]
    "indica segun el numero de nodo el numero de aminoacido que corresponde"
    indicador_amino_1=[0] * 34
    for i in range(20):
        indicador_amino_1[nodo_terminal_1[i]] = num_amino_new_1[i]
    "se construye la matriz de adyasencia del arbol y la matriz de probabilidades"
    M_ady_1=[0]*24
    M_prob_1=[0]*24
    M_ady_1[0]=[1,2,3]
    M_prob_1[0]=[7/20,5/20,8/20]
    M_ady_1[1]=[4,5]
    M_prob_1[1]=[4/7,3/7]
    M_ady_1[2]=[6,7]
    M_prob_1[2]=[4/5,1/5]
    M_ady_1[3]=[8,9,10]
    M_prob_1[3]=[2/8,5/8,1/8]
    M_ady_1[4]=[11,12]
    M_prob_1[4]=[2/4,2/4]
    M_ady_1[5]=[13,14,15]
    M_prob_1[5]=[1/3,1/3,1/3]
    M_ady_1[6]=[16,17,18]
    M_prob_1[6]=[2/4,1/4,1/4]
    M_ady_1[8]=[19,20]
    M_prob_1[8]=[1/2,1/2]
    M_ady_1[9]=[21,22,23]
    M_prob_1[9]=[2/5,1/5,2/5]
    M_ady_1[11]=[24,25]
    M_prob_1[11]=[1/2,1/2]
    M_ady_1[12]=[26,27]
    M_prob_1[12]=[1/2,1/2]
    M_ady_1[16]=[28,29]
    M_prob_1[16]=[1/2,1/2]
    M_ady_1[21]=[30,31]
    M_prob_1[21]=[1/2,1/2]
    M_ady_1[23]=[32,33]
    M_prob_1[23]=[1/2,1/2]
    "indica el padre del nodo"
    indicador_entrada_1=[0]*34
    indicador_entrada_pos_1=[0]*34
    for i in range(len(M_ady_1)):
      if isinstance(M_ady_1[i], int):
        "puede ser entero"
      else:
        for j in range(len(M_ady_1[i])):
            indicador_entrada_1[M_ady_1[i][j]]=i
            indicador_entrada_pos_1[M_ady_1[i][j]]=j
    "construlle la poblacion inicial"
    poblacion_1 = [[copy.deepcopy(M_prob_1) for _ in range(n)] for _ in range(n_pop)]
    "determina las probabilidades iniciales"
    prob_pob_1=calcula_prob_tot(poblacion_1,nodos_continuan_1)
    "almacena el mejor valor de fitmess"
    best_fitness_1=[0]*n_best
    "almacena los valores de energia"
    best_fitness_energ_1=[20000]*n_best
    "almacena las n_best mejores ejecuciones"
    "best_execution = [[random.randint(0, 19) for _ in range(n)] for _ in range(n_best)]"
    best_execution_1 = [[0 for _ in range(n)] for _ in range(n_best)]
    
    "almacena las distancias de las n_best mejores ejecuciones"
    best_execution_dist_1 = [[0 for _ in range(n)] for _ in range(n_best)]
    "almacena los PDB de las n_best mejores ejecuciones"
    pdb_select_1 = ['a' for _ in range(n_best)]
    
    return nodo_terminal_1,amino_nodo_1,rango_numeros_1,nodos_continuan_1,indicador_1,num_amino_new_1,indicador_amino_1,M_ady_1,M_prob_1,indicador_entrada_1,indicador_entrada_pos_1,poblacion_1,prob_pob_1,best_fitness_1,best_fitness_energ_1,best_execution_1,best_execution_dist_1,pdb_select_1
    



def isla_1(indicador_entrada_1,indicador_entrada_pos_1,nodo_terminal_1,act_prop,act_glob,best_fitness_1, best_execution_1, pdb_select_1, best_fitness_energ_1,best_execution_dist_1,a,b,MC_BB,corte,
           descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
           n_rep,tot_ececution,nodos_continuan_1,indicador_1,indicador_amino_1,M_ady_1,poblacion_1,prob_pob_1,generacion):
    max_gdt_sal=0
    "i sera el individuo de la poblacion que se esta trabajando"
    for i in range(n_pop):
        "realiza las ejecuciones para el individuo de la poblacion"
        for kkk in range(n_rep):
            tot_ececution[i][kkk]=ejecuta_uno(prob_pob_1,i,n,M_ady_1, indicador_1, indicador_amino_1)
        "realiza el filtro correspondiente, por el momento sera las proimeras ejecuciones, no hay filtro"
        det_fitness=filtro1(tot_ececution,i,det_fitness,n_fit,n_rep)
        k=0
        for j in det_fitness[i]:
            print([i,j])
            "ejecucion j del elemento i de la poblacion a trabajar"
            posiciones_seq = tot_ececution[i][j]
            "Secuencia de aminoacidos de la ejecucion"
            secuencia = det_sec(posiciones_seq, amino)
            "PDB de la secuencia"
            if desc_si_no==1:
                descriptor_temp=descriptores(secuencia,tokenizer,model_ESM2) 
            if local==1:
                """inputs = tokenizer([secuencia], return_tensors="pt", add_special_tokens=False)
                outputs = model(**inputs)
                PDB_pdredict = [outputs.positions]"""
                pdb_str, output,ptm,plddt=genera(secuencia,model)
                PDB_pdredict = [pdb_str]
            else:
                PDB_pdredict = [_request_esmfold_prediction(secuencia)]
            "Backbone de la secuencia"
            back_temp=extract_backbone_atoms_str(PDB_pdredict[0])
            "asigna a p le estructura determinada"
            pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(p,PDB_pdredict[0]) 
            "calcula la energia"
            energia_desing=my_scorefxn(p)
            # PDB_pdredict = EMSFold_function(tokenizer,model,secuencia)
            "pdb_select[i][j] = PDB_pdredict"
            "rms y gdt de la secuencia"
            if desc_si_no==1:
                rms,gdt,MC_similitud, divKl,distancias_sal,tms=fitness_gdt_rmsd_mc_fisquim(BB, back_temp,MC_BB,corte,descriptor_ref,descriptor_temp)
                print([rms,gdt,energia_desing,MC_similitud,np.mean(divKl)])
                "se agregan los valores"
                fitness1[i][k]=agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b,tms)
                tot_ececution_dist[i][j]=distancias_sal
            else:
                rms,gdt,MC_similitud,distancias_sal,tms=fitness_gdt_rmsd_mc(BB, back_temp,MC_BB,corte)
                print([rms,gdt,energia_desing,MC_similitud])
                fitness1[i][k]=agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b,tms)  
                tot_ececution_dist[i][j]=distancias_sal
            
            "si la ejecucion tiene mayor fitnes que alguno en best_fitness entonces se actualiza la lista de los mejores"
            if fitness1[i][k]>best_fitness_1[pos_min]:
                best_fitness_1[pos_min]=fitness1[i][k]
                best_fitness_energ_1[pos_min]=[energia_desing,1,copy.deepcopy(generacion)]
                best_execution_1[pos_min]=list(tot_ececution[i][j])
                pdb_select_1[pos_min] = PDB_pdredict[0][:]
                best_execution_dist_1[pos_min]=distancias_sal
                best_fitness_1, best_execution_1, pdb_select_1, best_fitness_energ_1,best_execution_dist_1=ordena_mejores_energia(best_fitness_1, best_execution_1, pdb_select_1, best_fitness_energ_1,best_execution_dist_1)
                max_gdt_sal=max(max_gdt_sal,gdt)
            k+=1
        w_act,w_act_dist=W_act1(i, best_execution_1, act_glob, tot_ececution, act_prop, det_fitness, fitness1,best_fitness_1,best_execution_dist_1,tot_ececution_dist)
        poblacion_1=actualiza_individuo(n,poblacion_1,i,w_act,w_act_dist,indicador_entrada_1,indicador_entrada_pos_1,nodo_terminal_1)

    prob_pob_1=calcula_prob_tot(poblacion_1,nodos_continuan_1)       
    return max_gdt_sal,best_fitness_1, best_execution_1, pdb_select_1, best_fitness_energ_1,best_execution_dist_1,poblacion_1,prob_pob_1
    





def inicializa_isla_2(n,n_pop,amino,n_best):
    global best_execution_2
    global best_execution_dist_2
    global pdb_select_2
    global best_fitness_2
    global best_fitness_energ_2
    nodo_terminal_2=[7,10,13,14,15,17,18,19,20,22,24,25,26,27,28,29,30,31,32,33]
    amino_nodo_2=['F','C','K','R','H','I','M','Y','W','P','D','E','N','Q','V','L','S','T','A','G'] 
    rango_numeros_2 = set(range(34))  
    "nodos que continuan"
    nodos_continuan_2 = list(rango_numeros_2 - set(nodo_terminal_2))
    "indica si el nodo es terminal=1 o si continua=0"
    indicador_2 = [0] * 34
    for posicion in nodo_terminal_2:
        indicador_2[posicion] = 1
    "reorganiza las posiciones de los aminoaciodos segun el; nuevo orden dado por amino_nodo" 
    letra_a_indice_2 = {letra: indice for indice, letra in enumerate(amino)}
    num_amino_new_2 = [letra_a_indice_2[letra] for letra in amino_nodo_2]
    "indica segun el numero de nodo el numero de aminoacido que corresponde"
    indicador_amino_2=[0] * 34
    for i in range(20):
        indicador_amino_2[nodo_terminal_2[i]] = num_amino_new_2[i]
    "se construye la matriz de adyasencia del arbol y la matriz de probabilidades"
    M_ady_2=[0]*24
    M_prob_2=[0]*24
    M_ady_2[0]=[1,2,3]
    M_prob_2[0]=[7/20,5/20,8/20]
    M_ady_2[1]=[4,5]
    M_prob_2[1]=[4/7,3/7]
    M_ady_2[2]=[6,7]
    M_prob_2[2]=[4/5,1/5]
    M_ady_2[3]=[8,9,10]
    M_prob_2[3]=[2/8,5/8,1/8]
    M_ady_2[4]=[11,12]
    M_prob_2[4]=[2/4,2/4]
    M_ady_2[5]=[13,14,15]
    M_prob_2[5]=[1/3,1/3,1/3]
    M_ady_2[6]=[16,17,18]
    M_prob_2[6]=[2/4,1/4,1/4]
    M_ady_2[8]=[19,20]
    M_prob_2[8]=[1/2,1/2]
    M_ady_2[9]=[21,22,23]
    M_prob_2[9]=[2/5,1/5,2/5]
    M_ady_2[11]=[24,25]
    M_prob_2[11]=[1/2,1/2]
    M_ady_2[12]=[26,27]
    M_prob_2[12]=[1/2,1/2]
    M_ady_2[16]=[28,29]
    M_prob_2[16]=[1/2,1/2]
    M_ady_2[21]=[30,31]
    M_prob_2[21]=[1/2,1/2]
    M_ady_2[23]=[32,33]
    M_prob_2[23]=[1/2,1/2]
    "indica el padre del nodo"
    indicador_entrada_2=[0]*34
    indicador_entrada_pos_2=[0]*34
    for i in range(len(M_ady_2)):
      if isinstance(M_ady_2[i], int):
        "puede ser entero"
      else:
        for j in range(len(M_ady_2[i])):
            indicador_entrada_2[M_ady_2[i][j]]=i
            indicador_entrada_pos_2[M_ady_2[i][j]]=j
    "construlle la poblacion inicial"
    poblacion_2 = [[copy.deepcopy(M_prob_2) for _ in range(n)] for _ in range(n_pop)]
    "determina las probabilidades iniciales"
    prob_pob_2=calcula_prob_tot(poblacion_2,nodos_continuan_2)
    "almacena el mejor valor de fitmess"
    best_fitness_2=[0]*n_best
    "almacena los valores de energia"
    best_fitness_energ_2=[20000]*n_best
    "almacena las n_best mejores ejecuciones"
    "best_execution = [[random.randint(0, 19) for _ in range(n)] for _ in range(n_best)]"
    best_execution_2 = [[0 for _ in range(n)] for _ in range(n_best)]
    
    "almacena las distancias de las n_best mejores ejecuciones"
    best_execution_dist_2 = [[0 for _ in range(n)] for _ in range(n_best)]
    "almacena los PDB de las n_best mejores ejecuciones"
    pdb_select_2 = ['a' for _ in range(n_best)]
    
    return nodo_terminal_2,amino_nodo_2,rango_numeros_2,nodos_continuan_2,indicador_2,num_amino_new_2,indicador_amino_2,M_ady_2,M_prob_2,indicador_entrada_2,indicador_entrada_pos_2,poblacion_2,prob_pob_2,best_fitness_2,best_fitness_energ_2,best_execution_2,best_execution_dist_2,pdb_select_2
    



def isla_2(indicador_entrada_2,indicador_entrada_pos_2,nodo_terminal_2,act_prop,act_glob,best_fitness_2, best_execution_2, pdb_select_2, best_fitness_energ_2,best_execution_dist_2,a,b,MC_BB,corte,
           descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
           n_rep,tot_ececution,nodos_continuan_2,indicador_2,indicador_amino_2,M_ady_2,poblacion_2,prob_pob_2,generacion):
    "para utilizar en las actualizaciones"
    best_execution_dist_2_1 = [[1 for _ in range(n)] for _ in range(len(best_execution_dist_2))]
    max_gdt_sal=0
    "i sera el individuo de la poblacion que se esta trabajando"
    for i in range(n_pop):
        "realiza las ejecuciones para el individuo de la poblacion"
        for kkk in range(n_rep):
            tot_ececution[i][kkk]=ejecuta_uno(prob_pob_2,i,n,M_ady_2, indicador_2, indicador_amino_2)
        "realiza el filtro correspondiente, por el momento sera las proimeras ejecuciones, no hay filtro"
        det_fitness=filtro1(tot_ececution,i,det_fitness,n_fit,n_rep)
        k=0
        for j in det_fitness[i]:
            print([i,j])
            "ejecucion j del elemento i de la poblacion a trabajar"
            posiciones_seq = tot_ececution[i][j]
            "Secuencia de aminoacidos de la ejecucion"
            secuencia = det_sec(posiciones_seq, amino)
            "PDB de la secuencia"
            if desc_si_no==1:
                descriptor_temp=descriptores(secuencia,tokenizer,model_ESM2) 
            if local==1:
                """inputs = tokenizer([secuencia], return_tensors="pt", add_special_tokens=False)
                outputs = model(**inputs)
                PDB_pdredict = [outputs.positions]"""
                pdb_str, output,ptm,plddt=genera(secuencia,model)
                PDB_pdredict = [pdb_str]
            else:
                PDB_pdredict = [_request_esmfold_prediction(secuencia)]
            "Backbone de la secuencia"
            back_temp=extract_backbone_atoms_str(PDB_pdredict[0])
            "asigna a p le estructura determinada"
            pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(p,PDB_pdredict[0]) 
            "calcula la energia"
            energia_desing=my_scorefxn(p)
            # PDB_pdredict = EMSFold_function(tokenizer,model,secuencia)
            "pdb_select[i][j] = PDB_pdredict"
            "rms y gdt de la secuencia"
            if desc_si_no==1:
                rms,gdt,MC_similitud, divKl,distancias_sal,tms=fitness_gdt_rmsd_mc_fisquim(BB, back_temp,MC_BB,corte,descriptor_ref,descriptor_temp)
                print([rms,gdt,energia_desing,MC_similitud,np.mean(divKl)])
                "se agregan los valores"
                fitness1[i][k]=agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b,tms)
                tot_ececution_dist[i][j]= [1] * len(distancias_sal)
            else:
                rms,gdt,MC_similitud,distancias_sal,tms=fitness_gdt_rmsd_mc(BB, back_temp,MC_BB,corte)
                print([rms,gdt,energia_desing,MC_similitud])
                fitness1[i][k]=agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b,tms)  
                tot_ececution_dist[i][j]=[1] * len(distancias_sal)
            
            "si la ejecucion tiene mayor fitnes que alguno en best_fitness entonces se actualiza la lista de los mejores"
            if fitness1[i][k]>best_fitness_2[pos_min]:
                best_fitness_2[pos_min]=fitness1[i][k]
                best_fitness_energ_2[pos_min]=[energia_desing,2,copy.deepcopy(generacion)]
                best_execution_2[pos_min]=list(tot_ececution[i][j])
                pdb_select_2[pos_min] = PDB_pdredict[0][:]
                best_execution_dist_2[pos_min]=distancias_sal
                best_fitness_2, best_execution_2, pdb_select_2, best_fitness_energ_2,best_execution_dist_2=ordena_mejores_energia(best_fitness_2, best_execution_2, pdb_select_2, best_fitness_energ_2,best_execution_dist_2)
                max_gdt_sal=max(max_gdt_sal,gdt)
            k+=1
        w_act,w_act_dist=W_act1(i, best_execution_2, act_glob, tot_ececution, act_prop, det_fitness, fitness1,best_fitness_2,best_execution_dist_2_1,tot_ececution_dist)
        poblacion_2=actualiza_individuo(n,poblacion_2,i,w_act,w_act_dist,indicador_entrada_2,indicador_entrada_pos_2,nodo_terminal_2)
    prob_pob_2=calcula_prob_tot(poblacion_2,nodos_continuan_2)       
    return max_gdt_sal,best_fitness_2, best_execution_2, pdb_select_2, best_fitness_energ_2,best_execution_dist_2,poblacion_2,prob_pob_2
    




def inicializa_isla_3(n,n_pop,amino,n_best):
    global best_execution_3
    global best_execution_dist_3
    global pdb_select_3
    global best_fitness_3
    global best_fitness_energ_3
    nodo_terminal_3=list(range(1, 21))
    amino_nodo_3=amino
    rango_numeros_3 = set(range(21))  
    "nodos que continuan"
    nodos_continuan_3 = list(rango_numeros_3 - set(nodo_terminal_3))
    "indica si el nodo es terminal=1 o si continua=0"
    indicador_3 = [0] * 21
    for posicion in nodo_terminal_3:
        indicador_3[posicion] = 1
    "reorganiza las posiciones de los aminoaciodos segun el; nuevo orden dado por amino_nodo" 
    letra_a_indice_3 = {letra: indice for indice, letra in enumerate(amino)}
    num_amino_new_3 = [letra_a_indice_3[letra] for letra in amino_nodo_3]
    "indica segun el numero de nodo el numero de aminoacido que corresponde"
    indicador_amino_3=[0] * 21
    for i in range(20):
        indicador_amino_3[nodo_terminal_3[i]] = num_amino_new_3[i]
    "se construye la matriz de adyasencia del arbol y la matriz de probabilidades"
    M_ady_3=[0]*1
    M_prob_3=[0]*1
    
    M_ady_3[0]=list(range(1, 21))
    M_prob_3[0]=[1/20] * 20
   
    "indica el padre del nodo"
    indicador_entrada_3=[0]*21
    indicador_entrada_pos_3=[0]*21
    for i in range(len(M_ady_3)):
      if isinstance(M_ady_3[i], int):
        "puede ser entero"
      else:
        for j in range(len(M_ady_3[i])):
            indicador_entrada_3[M_ady_3[i][j]]=i
            indicador_entrada_pos_3[M_ady_3[i][j]]=j
    "construlle la poblacion inicial"
    poblacion_3 = [[copy.deepcopy(M_prob_3) for _ in range(n)] for _ in range(n_pop)]
    "determina las probabilidades iniciales"
    prob_pob_3=calcula_prob_tot(poblacion_3,nodos_continuan_3)
    "almacena el mejor valor de fitmess"
    best_fitness_3=[0]*n_best
    "almacena los valores de energia"
    best_fitness_energ_3=[20000]*n_best
    "almacena las n_best mejores ejecuciones"
    "best_execution = [[random.randint(0, 19) for _ in range(n)] for _ in range(n_best)]"
    best_execution_3 = [[0 for _ in range(n)] for _ in range(n_best)]
    
    "almacena las distancias de las n_best mejores ejecuciones"
    best_execution_dist_3 = [[0 for _ in range(n)] for _ in range(n_best)]
    "almacena los PDB de las n_best mejores ejecuciones"
    pdb_select_3 = ['a' for _ in range(n_best)]
    
    return nodo_terminal_3,amino_nodo_3,rango_numeros_3,nodos_continuan_3,indicador_3,num_amino_new_3,indicador_amino_3,M_ady_3,M_prob_3,indicador_entrada_3,indicador_entrada_pos_3,poblacion_3,prob_pob_3,best_fitness_3,best_fitness_energ_3,best_execution_3,best_execution_dist_3,pdb_select_3
    



def isla_3(indicador_entrada_3,indicador_entrada_pos_3,nodo_terminal_3,act_prop,act_glob,best_fitness_3, best_execution_3, pdb_select_3, best_fitness_energ_3,best_execution_dist_3,a,b,MC_BB,corte,
           descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
           n_rep,tot_ececution,nodos_continuan_3,indicador_3,indicador_amino_3,M_ady_3,poblacion_3,prob_pob_3,generacion):
    max_gdt_sal=0
    "i sera el individuo de la poblacion que se esta trabajando"
    for i in range(n_pop):
        "realiza las ejecuciones para el individuo de la poblacion"
        for kkk in range(n_rep):
            tot_ececution[i][kkk]=ejecuta_uno(prob_pob_3,i,n,M_ady_3, indicador_3, indicador_amino_3)
        "realiza el filtro correspondiente, por el momento sera las proimeras ejecuciones, no hay filtro"
        det_fitness=filtro1(tot_ececution,i,det_fitness,n_fit,n_rep)
        k=0
        for j in det_fitness[i]:
            print([i,j])
            "ejecucion j del elemento i de la poblacion a trabajar"
            posiciones_seq = tot_ececution[i][j]
            "Secuencia de aminoacidos de la ejecucion"
            secuencia = det_sec(posiciones_seq, amino)
            "PDB de la secuencia"
            if desc_si_no==1:
                descriptor_temp=descriptores(secuencia,tokenizer,model_ESM2) 
            if local==1:
                """inputs = tokenizer([secuencia], return_tensors="pt", add_special_tokens=False)
                outputs = model(**inputs)
                PDB_pdredict = [outputs.positions]"""
                pdb_str, output,ptm,plddt=genera(secuencia,model)
                PDB_pdredict = [pdb_str]
            else:
                PDB_pdredict = [_request_esmfold_prediction(secuencia)]
            "Backbone de la secuencia"
            back_temp=extract_backbone_atoms_str(PDB_pdredict[0])
            "asigna a p le estructura determinada"
            pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(p,PDB_pdredict[0]) 
            "calcula la energia"
            energia_desing=my_scorefxn(p)
            # PDB_pdredict = EMSFold_function(tokenizer,model,secuencia)
            "pdb_select[i][j] = PDB_pdredict"
            "rms y gdt de la secuencia"
            if desc_si_no==1:
                rms,gdt,MC_similitud, divKl,distancias_sal,tms=fitness_gdt_rmsd_mc_fisquim(BB, back_temp,MC_BB,corte,descriptor_ref,descriptor_temp)
                print([rms,gdt,energia_desing,MC_similitud,np.mean(divKl)])
                "se agregan los valores"
                fitness1[i][k]=agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b,tms)
                tot_ececution_dist[i][j]=distancias_sal
            else:
                rms,gdt,MC_similitud,distancias_sal,tms=fitness_gdt_rmsd_mc(BB, back_temp,MC_BB,corte)
                print([rms,gdt,energia_desing,MC_similitud])
                fitness1[i][k]=agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b,tms)  
                tot_ececution_dist[i][j]=distancias_sal
            
            "si la ejecucion tiene mayor fitnes que alguno en best_fitness entonces se actualiza la lista de los mejores"
            if fitness1[i][k]>best_fitness_3[pos_min]:
                best_fitness_3[pos_min]=fitness1[i][k]
                best_fitness_energ_3[pos_min]=[energia_desing,3,copy.deepcopy(generacion)]
                best_execution_3[pos_min]=list(tot_ececution[i][j])
                pdb_select_3[pos_min] = PDB_pdredict[0][:]
                best_execution_dist_3[pos_min]=distancias_sal
                best_fitness_3, best_execution_3, pdb_select_3, best_fitness_energ_3,best_execution_dist_3=ordena_mejores_energia(best_fitness_3, best_execution_3, pdb_select_3, best_fitness_energ_3,best_execution_dist_3)
                max_gdt_sal=max(max_gdt_sal,gdt)
            k+=1
        w_act,w_act_dist=W_act1(i, best_execution_3, act_glob, tot_ececution, act_prop, det_fitness, fitness1,best_fitness_3,best_execution_dist_3,tot_ececution_dist)
        poblacion_3=actualiza_individuo(n,poblacion_3,i,w_act,w_act_dist,indicador_entrada_3,indicador_entrada_pos_3,nodo_terminal_3)
    prob_pob_3=calcula_prob_tot(poblacion_3,nodos_continuan_3)       
    return max_gdt_sal,best_fitness_3, best_execution_3, pdb_select_3, best_fitness_energ_3,best_execution_dist_3,poblacion_3,prob_pob_3
    


def inicializa_isla_4(n,n_pop,amino,n_best):
    global best_execution_4
    global best_execution_dist_4
    global pdb_select_4
    global best_fitness_4
    global best_fitness_energ_4
    nodo_terminal_4=list(range(1, 21))
    amino_nodo_4=amino
    rango_numeros_4 = set(range(21))  
    "nodos que continuan"
    nodos_continuan_4 = list(rango_numeros_4 - set(nodo_terminal_4))
    "indica si el nodo es terminal=1 o si continua=0"
    indicador_4 = [0] * 21
    for posicion in nodo_terminal_4:
        indicador_4[posicion] = 1
    "reorganiza las posiciones de los aminoaciodos segun el; nuevo orden dado por amino_nodo" 
    letra_a_indice_4 = {letra: indice for indice, letra in enumerate(amino)}
    num_amino_new_4 = [letra_a_indice_4[letra] for letra in amino_nodo_4]
    "indica segun el numero de nodo el numero de aminoacido que corresponde"
    indicador_amino_4=[0] * 21
    for i in range(20):
        indicador_amino_4[nodo_terminal_4[i]] = num_amino_new_4[i]
    "se construye la matriz de adyasencia del arbol y la matriz de probabilidades"
    M_ady_4=[0]*1
    M_prob_4=[0]*1
    
    M_ady_4[0]=list(range(1, 21))
    M_prob_4[0]=[1/20] * 20
   
    "indica el padre del nodo"
    indicador_entrada_4=[0]*21
    indicador_entrada_pos_4=[0]*21
    for i in range(len(M_ady_4)):
      if isinstance(M_ady_4[i], int):
        "puede ser entero"
      else:
        for j in range(len(M_ady_4[i])):
            indicador_entrada_4[M_ady_4[i][j]]=i
            indicador_entrada_pos_4[M_ady_4[i][j]]=j
    "construlle la poblacion inicial"
    poblacion_4 = [[copy.deepcopy(M_prob_4) for _ in range(n)] for _ in range(n_pop)]
    "determina las probabilidades iniciales"
    prob_pob_4=calcula_prob_tot(poblacion_4,nodos_continuan_4)
    "almacena el mejor valor de fitmess"
    best_fitness_4=[0]*n_best
    "almacena los valores de energia"
    best_fitness_energ_4=[20000]*n_best
    "almacena las n_best mejores ejecuciones"
    "best_execution = [[random.randint(0, 19) for _ in range(n)] for _ in range(n_best)]"
    best_execution_4 = [[0 for _ in range(n)] for _ in range(n_best)]
    
    "almacena las distancias de las n_best mejores ejecuciones"
    best_execution_dist_4 = [[0 for _ in range(n)] for _ in range(n_best)]
    "almacena los PDB de las n_best mejores ejecuciones"
    pdb_select_4 = ['a' for _ in range(n_best)]
    
    return nodo_terminal_4,amino_nodo_4,rango_numeros_4,nodos_continuan_4,indicador_4,num_amino_new_4,indicador_amino_4,M_ady_4,M_prob_4,indicador_entrada_4,indicador_entrada_pos_4,poblacion_4,prob_pob_4,best_fitness_4,best_fitness_energ_4,best_execution_4,best_execution_dist_4,pdb_select_4
    



def isla_4(indicador_entrada_4,indicador_entrada_pos_4,nodo_terminal_4,act_prop,act_glob,best_fitness_4, best_execution_4, pdb_select_4, best_fitness_energ_4,best_execution_dist_4,a,b,MC_BB,corte,
           descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
           n_rep,tot_ececution,nodos_continuan_4,indicador_4,indicador_amino_4,M_ady_4,poblacion_4,prob_pob_4,generacion):
    "para utilizar en las actualizaciones"
    best_execution_dist_4_1 = [[1 for _ in range(n)] for _ in range(len(best_execution_dist_4))]
    max_gdt_sal=0
    "i sera el individuo de la poblacion que se esta trabajando"
    for i in range(n_pop):
        "realiza las ejecuciones para el individuo de la poblacion"
        for kkk in range(n_rep):
            tot_ececution[i][kkk]=ejecuta_uno(prob_pob_4,i,n,M_ady_4, indicador_4, indicador_amino_4)
        "realiza el filtro correspondiente, por el momento sera las proimeras ejecuciones, no hay filtro"
        det_fitness=filtro1(tot_ececution,i,det_fitness,n_fit,n_rep)
        k=0
        for j in det_fitness[i]:
            print([i,j])
            "ejecucion j del elemento i de la poblacion a trabajar"
            posiciones_seq = tot_ececution[i][j]
            "Secuencia de aminoacidos de la ejecucion"
            secuencia = det_sec(posiciones_seq, amino)
            "PDB de la secuencia"
            if desc_si_no==1:
                descriptor_temp=descriptores(secuencia,tokenizer,model_ESM2) 
            if local==1:
                """inputs = tokenizer([secuencia], return_tensors="pt", add_special_tokens=False)
                outputs = model(**inputs)
                PDB_pdredict = [outputs.positions]"""
                pdb_str, output,ptm,plddt=genera(secuencia,model)
                PDB_pdredict = [pdb_str]
            else:
                PDB_pdredict = [_request_esmfold_prediction(secuencia)]
            "Backbone de la secuencia"
            back_temp=extract_backbone_atoms_str(PDB_pdredict[0])
            "asigna a p le estructura determinada"
            pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(p,PDB_pdredict[0]) 
            "calcula la energia"
            energia_desing=my_scorefxn(p)
            # PDB_pdredict = EMSFold_function(tokenizer,model,secuencia)
            "pdb_select[i][j] = PDB_pdredict"
            "rms y gdt de la secuencia"
            if desc_si_no==1:
                rms,gdt,MC_similitud, divKl,distancias_sal,tms=fitness_gdt_rmsd_mc_fisquim(BB, back_temp,MC_BB,corte,descriptor_ref,descriptor_temp)
                print([rms,gdt,energia_desing,MC_similitud,np.mean(divKl)])
                "se agregan los valores"
                fitness1[i][k]=agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b,tms)
                tot_ececution_dist[i][j]=[1] * len(distancias_sal)
            else:
                rms,gdt,MC_similitud,distancias_sal,tms=fitness_gdt_rmsd_mc(BB, back_temp,MC_BB,corte)
                print([rms,gdt,energia_desing,MC_similitud])
                fitness1[i][k]=agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b,tms)  
                tot_ececution_dist[i][j]=[1] * len(distancias_sal)
            
            "si la ejecucion tiene mayor fitnes que alguno en best_fitness entonces se actualiza la lista de los mejores"
            if fitness1[i][k]>best_fitness_4[pos_min]:
                best_fitness_4[pos_min]=fitness1[i][k]
                best_fitness_energ_4[pos_min]=[energia_desing,4,copy.deepcopy(generacion)]
                best_execution_4[pos_min]=list(tot_ececution[i][j])
                pdb_select_4[pos_min] = PDB_pdredict[0][:]
                best_execution_dist_4[pos_min]=distancias_sal
                best_fitness_4, best_execution_4, pdb_select_4, best_fitness_energ_4,best_execution_dist_4=ordena_mejores_energia(best_fitness_4, best_execution_4, pdb_select_4, best_fitness_energ_4,best_execution_dist_4)
                max_gdt_sal=max(max_gdt_sal,gdt)
            k+=1
        w_act,w_act_dist=W_act1(i, best_execution_4, act_glob, tot_ececution, act_prop, det_fitness, fitness1,best_fitness_4,best_execution_dist_4_1,tot_ececution_dist)
        poblacion_4=actualiza_individuo(n,poblacion_4,i,w_act,w_act_dist,indicador_entrada_4,indicador_entrada_pos_4,nodo_terminal_4)
    prob_pob_4=calcula_prob_tot(poblacion_4,nodos_continuan_4)       
    return max_gdt_sal,best_fitness_4, best_execution_4, pdb_select_4, best_fitness_energ_4,best_execution_dist_4,poblacion_4,prob_pob_4
    


def inicializa_isla_5(n,n_pop,amino,n_best):
    global best_execution_5
    global best_execution_dist_5
    global pdb_select_5
    global best_fitness_5
    global best_fitness_energ_5
    nodo_terminal_5=[7,10,13,14,15,17,18,19,20,22,24,25,26,27,28,29,30,31,32,33]
    amino_nodo_5=['F','C','K','R','H','I','M','Y','W','P','D','E','N','Q','V','L','S','T','A','G'] 
    rango_numeros_5 = set(range(34))  
    "nodos que continuan"
    nodos_continuan_5 = list(rango_numeros_5 - set(nodo_terminal_5))
    "indica si el nodo es terminal=1 o si continua=0"
    indicador_5 = [0] * 34
    for posicion in nodo_terminal_5:
        indicador_5[posicion] = 1
    "reorganiza las posiciones de los aminoaciodos segun el; nuevo orden dado por amino_nodo" 
    letra_a_indice_5 = {letra: indice for indice, letra in enumerate(amino)}
    num_amino_new_5 = [letra_a_indice_5[letra] for letra in amino_nodo_5]
    "indica segun el numero de nodo el numero de aminoacido que corresponde"
    indicador_amino_5=[0] * 34
    for i in range(20):
        indicador_amino_5[nodo_terminal_5[i]] = num_amino_new_5[i]
    "se construye la matriz de adyasencia del arbol y la matriz de probabilidades"
    M_ady_5=[0]*24
    M_prob_5=[0]*24
    M_ady_5[0]=[1,2,3]
    M_prob_5[0]=[7/20,5/20,8/20]
    M_ady_5[1]=[4,5]
    M_prob_5[1]=[4/7,3/7]
    M_ady_5[2]=[6,7]
    M_prob_5[2]=[4/5,1/5]
    M_ady_5[3]=[8,9,10]
    M_prob_5[3]=[2/8,5/8,1/8]
    M_ady_5[4]=[11,12]
    M_prob_5[4]=[2/4,2/4]
    M_ady_5[5]=[13,14,15]
    M_prob_5[5]=[1/3,1/3,1/3]
    M_ady_5[6]=[16,17,18]
    M_prob_5[6]=[2/4,1/4,1/4]
    M_ady_5[8]=[19,20]
    M_prob_5[8]=[1/2,1/2]
    M_ady_5[9]=[21,22,23]
    M_prob_5[9]=[2/5,1/5,2/5]
    M_ady_5[11]=[24,25]
    M_prob_5[11]=[1/2,1/2]
    M_ady_5[12]=[26,27]
    M_prob_5[12]=[1/2,1/2]
    M_ady_5[16]=[28,29]
    M_prob_5[16]=[1/2,1/2]
    M_ady_5[21]=[30,31]
    M_prob_5[21]=[1/2,1/2]
    M_ady_5[23]=[32,33]
    M_prob_5[23]=[1/2,1/2]
    "indica el padre del nodo"
    indicador_entrada_5=[0]*34
    indicador_entrada_pos_5=[0]*34
    for i in range(len(M_ady_5)):
      if isinstance(M_ady_5[i], int):
        "puede ser entero"
      else:
        for j in range(len(M_ady_5[i])):
            indicador_entrada_5[M_ady_5[i][j]]=i
            indicador_entrada_pos_5[M_ady_5[i][j]]=j
    "construlle la poblacion inicial"
    poblacion_5 = [[copy.deepcopy(M_prob_5) for _ in range(n)] for _ in range(n_pop)]
    "determina las probabilidades iniciales"
    prob_pob_5=calcula_prob_tot(poblacion_5,nodos_continuan_5)
    "almacena el mejor valor de fitmess"
    best_fitness_5=[0]*n_best
    "almacena los valores de energia"
    best_fitness_energ_5=[20000]*n_best
    "almacena las n_best mejores ejecuciones"
    "best_execution = [[random.randint(0, 19) for _ in range(n)] for _ in range(n_best)]"
    best_execution_5 = [[0 for _ in range(n)] for _ in range(n_best)]
    
    "almacena las distancias de las n_best mejores ejecuciones"
    best_execution_dist_5 = [[0 for _ in range(n)] for _ in range(n_best)]
    "almacena los PDB de las n_best mejores ejecuciones"
    pdb_select_5 = ['a' for _ in range(n_best)]
    
    return nodo_terminal_5,amino_nodo_5,rango_numeros_5,nodos_continuan_5,indicador_5,num_amino_new_5,indicador_amino_5,M_ady_5,M_prob_5,indicador_entrada_5,indicador_entrada_pos_5,poblacion_5,prob_pob_5,best_fitness_5,best_fitness_energ_5,best_execution_5,best_execution_dist_5,pdb_select_5
    



def isla_5(generacion,utt_reinicio1,indicador_entrada_5,indicador_entrada_pos_5,nodo_terminal_5,act_prop,act_glob,best_fitness_5, best_execution_5, pdb_select_5, best_fitness_energ_5,best_execution_dist_5,a,b,MC_BB,corte,
           descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
           n_rep,tot_ececution,nodos_continuan_5,indicador_5,indicador_amino_5,M_ady_5,poblacion_5,prob_pob_5,M_prob_5):
    max_gdt_sal=0
    "i sera el individuo de la poblacion que se esta trabajando"
    for i in range(n_pop):
        "realiza las ejecuciones para el individuo de la poblacion"
        for kkk in range(n_rep):
            tot_ececution[i][kkk]=ejecuta_uno(prob_pob_5,i,n,M_ady_5, indicador_5, indicador_amino_5)
        "realiza el filtro correspondiente, por el momento sera las proimeras ejecuciones, no hay filtro"
        det_fitness=filtro1(tot_ececution,i,det_fitness,n_fit,n_rep)
        k=0
        for j in det_fitness[i]:
            print([generacion,i,j])
            "ejecucion j del elemento i de la poblacion a trabajar"
            posiciones_seq = tot_ececution[i][j]
            "Secuencia de aminoacidos de la ejecucion"
            secuencia = det_sec(posiciones_seq, amino)
            "PDB de la secuencia"
            if desc_si_no==1:
                descriptor_temp=descriptores(secuencia,tokenizer,model_ESM2) 
            if local==1:
                """inputs = tokenizer([secuencia], return_tensors="pt", add_special_tokens=False)
                outputs = model(**inputs)
                PDB_pdredict = [outputs.positions]"""
                pdb_str, output,ptm,plddt=genera(secuencia,model)
                PDB_pdredict = [pdb_str]
            else:
                PDB_pdredict = [_request_esmfold_prediction(secuencia)]
            "Backbone de la secuencia"
            back_temp=extract_backbone_atoms_str(PDB_pdredict[0])
            "asigna a p le estructura determinada"
            pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(p,PDB_pdredict[0]) 
            "calcula la energia"
            energia_desing=my_scorefxn(p)
            # PDB_pdredict = EMSFold_function(tokenizer,model,secuencia)
            "pdb_select[i][j] = PDB_pdredict"
            "rms y gdt de la secuencia"
            if desc_si_no==1:
                rms,gdt,MC_similitud, divKl,distancias_sal,tms=fitness_gdt_rmsd_mc_fisquim(BB, back_temp,MC_BB,corte,descriptor_ref,descriptor_temp)
                print([rms,gdt,energia_desing,MC_similitud,np.mean(divKl)])
                "se agregan los valores"
                fitness1[i][k]=agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b,tms)
                tot_ececution_dist[i][j]=distancias_sal
            else:
                rms,gdt,MC_similitud,distancias_sal,tms=fitness_gdt_rmsd_mc(BB, back_temp,MC_BB,corte)
                print([rms,gdt,energia_desing,MC_similitud])
                fitness1[i][k]=agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b,tms)  
                tot_ececution_dist[i][j]=distancias_sal
            
            "si la ejecucion tiene mayor fitnes que alguno en best_fitness entonces se actualiza la lista de los mejores"
            if fitness1[i][k]>best_fitness_5[pos_min]:
                best_fitness_5[pos_min]=fitness1[i][k]
                best_fitness_energ_5[pos_min]=[energia_desing,5,copy.deepcopy(generacion)]
                best_execution_5[pos_min]=list(tot_ececution[i][j])
                pdb_select_5[pos_min] = PDB_pdredict[0][:]
                best_execution_dist_5[pos_min]=distancias_sal
                best_fitness_5, best_execution_5, pdb_select_5, best_fitness_energ_5,best_execution_dist_5=ordena_mejores_energia(best_fitness_5, best_execution_5, pdb_select_5, best_fitness_energ_5,best_execution_dist_5)
                max_gdt_sal=max(max_gdt_sal,gdt)
            k+=1
        if generacion in utt_reinicio1 :
            w_act,w_act_dist=W_act1(i, best_execution_5, len(best_execution_5), tot_ececution, act_prop, det_fitness, fitness1,best_fitness_5,best_execution_dist_5,tot_ececution_dist)
            poblacion_5 = [[copy.deepcopy(M_prob_5) for _ in range(n)] for _ in range(n_pop)]
            poblacion_5=actualiza_individuo(n,poblacion_5,i,w_act,w_act_dist,indicador_entrada_5,indicador_entrada_pos_5,nodo_terminal_5)
        else:
            w_act,w_act_dist=W_act1(i, best_execution_5, act_glob, tot_ececution, act_prop, det_fitness, fitness1,best_fitness_5,best_execution_dist_5,tot_ececution_dist)
            poblacion_5=actualiza_individuo(n,poblacion_5,i,w_act,w_act_dist,indicador_entrada_5,indicador_entrada_pos_5,nodo_terminal_5)

    prob_pob_5=calcula_prob_tot(poblacion_5,nodos_continuan_5)       
    return max_gdt_sal,best_fitness_5, best_execution_5, pdb_select_5, best_fitness_energ_5,best_execution_dist_5,poblacion_5,prob_pob_5




def inicializa_isla_6(n,n_pop,amino,n_best):
    global best_execution_6
    global best_execution_dist_6
    global pdb_select_6
    global best_fitness_6
    global best_fitness_energ_6
    nodo_terminal_6=[7,10,13,14,15,17,18,19,20,22,24,25,26,27,28,29,30,31,32,33]
    amino_nodo_6=['F','C','K','R','H','I','M','Y','W','P','D','E','N','Q','V','L','S','T','A','G'] 
    rango_numeros_6 = set(range(34))  
    "nodos que continuan"
    nodos_continuan_6 = list(rango_numeros_6 - set(nodo_terminal_6))
    "indica si el nodo es terminal=1 o si continua=0"
    indicador_6 = [0] * 34
    for posicion in nodo_terminal_6:
        indicador_6[posicion] = 1
    "reorganiza las posiciones de los aminoaciodos segun el; nuevo orden dado por amino_nodo" 
    letra_a_indice_6 = {letra: indice for indice, letra in enumerate(amino)}
    num_amino_new_6 = [letra_a_indice_6[letra] for letra in amino_nodo_6]
    "indica segun el numero de nodo el numero de aminoacido que corresponde"
    indicador_amino_6=[0] * 34
    for i in range(20):
        indicador_amino_6[nodo_terminal_6[i]] = num_amino_new_6[i]
    "se construye la matriz de adyasencia del arbol y la matriz de probabilidades"
    M_ady_6=[0]*24
    M_prob_6=[0]*24
    M_ady_6[0]=[1,2,3]
    M_prob_6[0]=[7/20,5/20,8/20]
    M_ady_6[1]=[4,5]
    M_prob_6[1]=[4/7,3/7]
    M_ady_6[2]=[6,7]
    M_prob_6[2]=[4/5,1/5]
    M_ady_6[3]=[8,9,10]
    M_prob_6[3]=[2/8,5/8,1/8]
    M_ady_6[4]=[11,12]
    M_prob_6[4]=[2/4,2/4]
    M_ady_6[5]=[13,14,15]
    M_prob_6[5]=[1/3,1/3,1/3]
    M_ady_6[6]=[16,17,18]
    M_prob_6[6]=[2/4,1/4,1/4]
    M_ady_6[8]=[19,20]
    M_prob_6[8]=[1/2,1/2]
    M_ady_6[9]=[21,22,23]
    M_prob_6[9]=[2/5,1/5,2/5]
    M_ady_6[11]=[24,25]
    M_prob_6[11]=[1/2,1/2]
    M_ady_6[12]=[26,27]
    M_prob_6[12]=[1/2,1/2]
    M_ady_6[16]=[28,29]
    M_prob_6[16]=[1/2,1/2]
    M_ady_6[21]=[30,31]
    M_prob_6[21]=[1/2,1/2]
    M_ady_6[23]=[32,33]
    M_prob_6[23]=[1/2,1/2]
    "indica el padre del nodo"
    indicador_entrada_6=[0]*34
    indicador_entrada_pos_6=[0]*34
    for i in range(len(M_ady_6)):
      if isinstance(M_ady_6[i], int):
        "puede ser entero"
      else:
        for j in range(len(M_ady_6[i])):
            indicador_entrada_6[M_ady_6[i][j]]=i
            indicador_entrada_pos_6[M_ady_6[i][j]]=j
    "construlle la poblacion inicial"
    poblacion_6 = [[copy.deepcopy(M_prob_6) for _ in range(n)] for _ in range(n_pop)]
    "determina las probabilidades iniciales"
    prob_pob_6=calcula_prob_tot(poblacion_6,nodos_continuan_6)
    "almacena el mejor valor de fitmess"
    best_fitness_6=[0]*n_best
    "almacena los valores de energia"
    best_fitness_energ_6=[20000]*n_best
    "almacena las n_best mejores ejecuciones"
    "best_execution = [[random.randint(0, 19) for _ in range(n)] for _ in range(n_best)]"
    best_execution_6 = [[0 for _ in range(n)] for _ in range(n_best)]
    
    "almacena las distancias de las n_best mejores ejecuciones"
    best_execution_dist_6 = [[0 for _ in range(n)] for _ in range(n_best)]
    "almacena los PDB de las n_best mejores ejecuciones"
    pdb_select_6 = ['a' for _ in range(n_best)]
    
    return nodo_terminal_6,amino_nodo_6,rango_numeros_6,nodos_continuan_6,indicador_6,num_amino_new_6,indicador_amino_6,M_ady_6,M_prob_6,indicador_entrada_6,indicador_entrada_pos_6,poblacion_6,prob_pob_6,best_fitness_6,best_fitness_energ_6,best_execution_6,best_execution_dist_6,pdb_select_6
    



def isla_6(generacion,utt_reinicio1,indicador_entrada_6,indicador_entrada_pos_6,nodo_terminal_6,act_prop,act_glob,best_fitness_6, best_execution_6, pdb_select_6, best_fitness_energ_6,best_execution_dist_6,a,b,MC_BB,corte,
           descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
           n_rep,tot_ececution,nodos_continuan_6,indicador_6,indicador_amino_6,M_ady_6,poblacion_6,prob_pob_6,M_prob_6):
    "para utilizar en las actualizaciones"
    best_execution_dist_6_1 = [[1 for _ in range(n)] for _ in range(len(best_execution_dist_6))]
    max_gdt_sal=0
    "i sera el individuo de la poblacion que se esta trabajando"
    for i in range(n_pop):
        "realiza las ejecuciones para el individuo de la poblacion"
        for kkk in range(n_rep):
            tot_ececution[i][kkk]=ejecuta_uno(prob_pob_6,i,n,M_ady_6, indicador_6, indicador_amino_6)
        "realiza el filtro correspondiente, por el momento sera las proimeras ejecuciones, no hay filtro"
        det_fitness=filtro1(tot_ececution,i,det_fitness,n_fit,n_rep)
        k=0
        for j in det_fitness[i]:
            print([generacion,i,j])
            "ejecucion j del elemento i de la poblacion a trabajar"
            posiciones_seq = tot_ececution[i][j]
            "Secuencia de aminoacidos de la ejecucion"
            secuencia = det_sec(posiciones_seq, amino)
            "PDB de la secuencia"
            if desc_si_no==1:
                descriptor_temp=descriptores(secuencia,tokenizer,model_ESM2) 
            if local==1:
                """inputs = tokenizer([secuencia], return_tensors="pt", add_special_tokens=False)
                outputs = model(**inputs)
                PDB_pdredict = [outputs.positions]"""
                pdb_str, output,ptm,plddt=genera(secuencia,model)
                PDB_pdredict = [pdb_str]
            else:
                PDB_pdredict = [_request_esmfold_prediction(secuencia)]
            "Backbone de la secuencia"
            back_temp=extract_backbone_atoms_str(PDB_pdredict[0])
            "asigna a p le estructura determinada"
            pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(p,PDB_pdredict[0]) 
            "calcula la energia"
            energia_desing=my_scorefxn(p)
            # PDB_pdredict = EMSFold_function(tokenizer,model,secuencia)
            "pdb_select[i][j] = PDB_pdredict"
            "rms y gdt de la secuencia"
            if desc_si_no==1:
                rms,gdt,MC_similitud, divKl,distancias_sal,tms=fitness_gdt_rmsd_mc_fisquim(BB, back_temp,MC_BB,corte,descriptor_ref,descriptor_temp)
                print([rms,gdt,energia_desing,MC_similitud,np.mean(divKl)])
                "se agregan los valores"
                fitness1[i][k]=agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b,tms)
                tot_ececution_dist[i][j]= [1] * len(distancias_sal)
            else:
                rms,gdt,MC_similitud,distancias_sal,tms=fitness_gdt_rmsd_mc(BB, back_temp,MC_BB,corte)
                print([rms,gdt,energia_desing,MC_similitud])
                fitness1[i][k]=agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b,tms)  
                tot_ececution_dist[i][j]=[1] * len(distancias_sal)
            
            "si la ejecucion tiene mayor fitnes que alguno en best_fitness entonces se actualiza la lista de los mejores"
            if fitness1[i][k]>best_fitness_6[pos_min]:
                best_fitness_6[pos_min]=fitness1[i][k]
                best_fitness_energ_6[pos_min]=[energia_desing,6,copy.deepcopy(generacion)]
                best_execution_6[pos_min]=list(tot_ececution[i][j])
                pdb_select_6[pos_min] = PDB_pdredict[0][:]
                best_execution_dist_6[pos_min]=distancias_sal
                best_fitness_6, best_execution_6, pdb_select_6, best_fitness_energ_6,best_execution_dist_6=ordena_mejores_energia(best_fitness_6, best_execution_6, pdb_select_6, best_fitness_energ_6,best_execution_dist_6)
                max_gdt_sal=max(max_gdt_sal,gdt)
            k+=1
        if generacion in utt_reinicio1 :
            w_act,w_act_dist=W_act1(i, best_execution_6, len(best_execution_6), tot_ececution, act_prop, det_fitness, fitness1,best_fitness_6,best_execution_dist_6_1,tot_ececution_dist)
            poblacion_6 = [[copy.deepcopy(M_prob_6) for _ in range(n)] for _ in range(n_pop)]
            poblacion_6=actualiza_individuo(n,poblacion_6,i,w_act,w_act_dist,indicador_entrada_6,indicador_entrada_pos_6,nodo_terminal_6)
        else:
            w_act,w_act_dist=W_act1(i, best_execution_6, act_glob, tot_ececution, act_prop, det_fitness, fitness1,best_fitness_6,best_execution_dist_6_1,tot_ececution_dist)
            poblacion_6=actualiza_individuo(n,poblacion_6,i,w_act,w_act_dist,indicador_entrada_6,indicador_entrada_pos_6,nodo_terminal_6)

    prob_pob_6=calcula_prob_tot(poblacion_6,nodos_continuan_6)       
    return max_gdt_sal,best_fitness_6, best_execution_6, pdb_select_6, best_fitness_energ_6,best_execution_dist_6,poblacion_6,prob_pob_6
    


def inicializa_isla_7(n,n_pop,amino,n_best):
    global best_execution_7
    global best_execution_dist_7
    global pdb_select_7
    global best_fitness_7
    global best_fitness_energ_7
    nodo_terminal_7=list(range(1, 21))
    amino_nodo_7=amino
    rango_numeros_7 = set(range(21))  
    "nodos que continuan"
    nodos_continuan_7 = list(rango_numeros_7 - set(nodo_terminal_7))
    "indica si el nodo es terminal=1 o si continua=0"
    indicador_7 = [0] * 21
    for posicion in nodo_terminal_7:
        indicador_7[posicion] = 1
    "reorganiza las posiciones de los aminoaciodos segun el; nuevo orden dado por amino_nodo" 
    letra_a_indice_7 = {letra: indice for indice, letra in enumerate(amino)}
    num_amino_new_7 = [letra_a_indice_7[letra] for letra in amino_nodo_7]
    "indica segun el numero de nodo el numero de aminoacido que corresponde"
    indicador_amino_7=[0] * 21
    for i in range(20):
        indicador_amino_7[nodo_terminal_7[i]] = num_amino_new_7[i]
    "se construye la matriz de adyasencia del arbol y la matriz de probabilidades"
    M_ady_7=[0]*1
    M_prob_7=[0]*1
    
    M_ady_7[0]=list(range(1, 21))
    M_prob_7[0]=[1/20] * 20
   
    "indica el padre del nodo"
    indicador_entrada_7=[0]*21
    indicador_entrada_pos_7=[0]*21
    for i in range(len(M_ady_7)):
      if isinstance(M_ady_7[i], int):
        "puede ser entero"
      else:
        for j in range(len(M_ady_7[i])):
            indicador_entrada_7[M_ady_7[i][j]]=i
            indicador_entrada_pos_7[M_ady_7[i][j]]=j
    "construlle la poblacion inicial"
    poblacion_7 = [[copy.deepcopy(M_prob_7) for _ in range(n)] for _ in range(n_pop)]
    "determina las probabilidades iniciales"
    prob_pob_7=calcula_prob_tot(poblacion_7,nodos_continuan_7)
    "almacena el mejor valor de fitmess"
    best_fitness_7=[0]*n_best
    "almacena los valores de energia"
    best_fitness_energ_7=[20000]*n_best
    "almacena las n_best mejores ejecuciones"
    "best_execution = [[random.randint(0, 19) for _ in range(n)] for _ in range(n_best)]"
    best_execution_7 = [[0 for _ in range(n)] for _ in range(n_best)]
    
    "almacena las distancias de las n_best mejores ejecuciones"
    best_execution_dist_7 = [[0 for _ in range(n)] for _ in range(n_best)]
    "almacena los PDB de las n_best mejores ejecuciones"
    pdb_select_7 = ['a' for _ in range(n_best)]
    
    return nodo_terminal_7,amino_nodo_7,rango_numeros_7,nodos_continuan_7,indicador_7,num_amino_new_7,indicador_amino_7,M_ady_7,M_prob_7,indicador_entrada_7,indicador_entrada_pos_7,poblacion_7,prob_pob_7,best_fitness_7,best_fitness_energ_7,best_execution_7,best_execution_dist_7,pdb_select_7
    
                
def isla_7(generacion,utt_reinicio1,indicador_entrada_7,indicador_entrada_pos_7,nodo_terminal_7,act_prop,act_glob,best_fitness_7, best_execution_7, pdb_select_7, best_fitness_energ_7,best_execution_dist_7,a,b,MC_BB,corte,
           descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
           n_rep,tot_ececution,nodos_continuan_7,indicador_7,indicador_amino_7,M_ady_7,poblacion_7,prob_pob_7,M_prob_7):
    max_gdt_sal=0
    "i sera el individuo de la poblacion que se esta trabajando"
    for i in range(n_pop):
        "realiza las ejecuciones para el individuo de la poblacion"
        for kkk in range(n_rep):
            tot_ececution[i][kkk]=ejecuta_uno(prob_pob_7,i,n,M_ady_7, indicador_7, indicador_amino_7)
        "realiza el filtro correspondiente, por el momento sera las proimeras ejecuciones, no hay filtro"
        det_fitness=filtro1(tot_ececution,i,det_fitness,n_fit,n_rep)
        k=0
        for j in det_fitness[i]:
            print([generacion,i,j])
            "ejecucion j del elemento i de la poblacion a trabajar"
            posiciones_seq = tot_ececution[i][j]
            "Secuencia de aminoacidos de la ejecucion"
            secuencia = det_sec(posiciones_seq, amino)
            "PDB de la secuencia"
            if desc_si_no==1:
                descriptor_temp=descriptores(secuencia,tokenizer,model_ESM2) 
            if local==1:
                """inputs = tokenizer([secuencia], return_tensors="pt", add_special_tokens=False)
                outputs = model(**inputs)
                PDB_pdredict = [outputs.positions]"""
                pdb_str, output,ptm,plddt=genera(secuencia,model)
                PDB_pdredict = [pdb_str]
            else:
                PDB_pdredict = [_request_esmfold_prediction(secuencia)]
            "Backbone de la secuencia"
            back_temp=extract_backbone_atoms_str(PDB_pdredict[0])
            "asigna a p le estructura determinada"
            pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(p,PDB_pdredict[0]) 
            "calcula la energia"
            energia_desing=my_scorefxn(p)
            # PDB_pdredict = EMSFold_function(tokenizer,model,secuencia)
            "pdb_select[i][j] = PDB_pdredict"
            "rms y gdt de la secuencia"
            if desc_si_no==1:
                rms,gdt,MC_similitud, divKl,distancias_sal,tms=fitness_gdt_rmsd_mc_fisquim(BB, back_temp,MC_BB,corte,descriptor_ref,descriptor_temp)
                print([rms,gdt,energia_desing,MC_similitud,np.mean(divKl)])
                "se agregan los valores"
                fitness1[i][k]=agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b,tms)
                tot_ececution_dist[i][j]=distancias_sal
            else:
                rms,gdt,MC_similitud,distancias_sal,tms=fitness_gdt_rmsd_mc(BB, back_temp,MC_BB,corte)
                print([rms,gdt,energia_desing,MC_similitud])
                fitness1[i][k]=agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b,tms)  
                tot_ececution_dist[i][j]=distancias_sal
            
            "si la ejecucion tiene mayor fitnes que alguno en best_fitness entonces se actualiza la lista de los mejores"
            if fitness1[i][k]>best_fitness_7[pos_min]:
                best_fitness_7[pos_min]=fitness1[i][k]
                best_fitness_energ_7[pos_min]=[energia_desing,7,copy.deepcopy(generacion)]
                best_execution_7[pos_min]=list(tot_ececution[i][j])
                pdb_select_7[pos_min] = PDB_pdredict[0][:]
                best_execution_dist_7[pos_min]=distancias_sal
                best_fitness_7, best_execution_7, pdb_select_7, best_fitness_energ_7,best_execution_dist_7=ordena_mejores_energia(best_fitness_7, best_execution_7, pdb_select_7, best_fitness_energ_7,best_execution_dist_7)
                max_gdt_sal=max(max_gdt_sal,gdt)
            k+=1
        if generacion in utt_reinicio1 :
            w_act,w_act_dist=W_act1(i, best_execution_7, len(best_execution_7), tot_ececution, act_prop, det_fitness, fitness1,best_fitness_7,best_execution_dist_7,tot_ececution_dist)
            poblacion_7 = [[copy.deepcopy(M_prob_7) for _ in range(n)] for _ in range(n_pop)]
            poblacion_7=actualiza_individuo(n,poblacion_7,i,w_act,w_act_dist,indicador_entrada_7,indicador_entrada_pos_7,nodo_terminal_7)
        else:
            w_act,w_act_dist=W_act1(i, best_execution_7, act_glob, tot_ececution, act_prop, det_fitness, fitness1,best_fitness_7,best_execution_dist_7,tot_ececution_dist)
            poblacion_7=actualiza_individuo(n,poblacion_7,i,w_act,w_act_dist,indicador_entrada_7,indicador_entrada_pos_7,nodo_terminal_7)

    prob_pob_7=calcula_prob_tot(poblacion_7,nodos_continuan_7)       
    return max_gdt_sal,best_fitness_7, best_execution_7, pdb_select_7, best_fitness_energ_7,best_execution_dist_7,poblacion_7,prob_pob_7
    



def inicializa_isla_8(n,n_pop,amino,n_best):
    global best_execution_8
    global best_execution_dist_8
    global pdb_select_8
    global best_fitness_8
    global best_fitness_energ_8
    nodo_terminal_8=list(range(1, 21))
    amino_nodo_8=amino
    rango_numeros_8 = set(range(21))  
    "nodos que continuan"
    nodos_continuan_8 = list(rango_numeros_8 - set(nodo_terminal_8))
    "indica si el nodo es terminal=1 o si continua=0"
    indicador_8 = [0] * 21
    for posicion in nodo_terminal_8:
        indicador_8[posicion] = 1
    "reorganiza las posiciones de los aminoaciodos segun el; nuevo orden dado por amino_nodo" 
    letra_a_indice_8 = {letra: indice for indice, letra in enumerate(amino)}
    num_amino_new_8 = [letra_a_indice_8[letra] for letra in amino_nodo_8]
    "indica segun el numero de nodo el numero de aminoacido que corresponde"
    indicador_amino_8=[0] * 21
    for i in range(20):
        indicador_amino_8[nodo_terminal_8[i]] = num_amino_new_8[i]
    "se construye la matriz de adyasencia del arbol y la matriz de probabilidades"
    M_ady_8=[0]*1
    M_prob_8=[0]*1
    
    M_ady_8[0]=list(range(1, 21))
    M_prob_8[0]=[1/20] * 20
   
    "indica el padre del nodo"
    indicador_entrada_8=[0]*21
    indicador_entrada_pos_8=[0]*21
    for i in range(len(M_ady_8)):
      if isinstance(M_ady_8[i], int):
        "puede ser entero"
      else:
        for j in range(len(M_ady_8[i])):
            indicador_entrada_8[M_ady_8[i][j]]=i
            indicador_entrada_pos_8[M_ady_8[i][j]]=j
    "construlle la poblacion inicial"
    poblacion_8 = [[copy.deepcopy(M_prob_8) for _ in range(n)] for _ in range(n_pop)]
    "determina las probabilidades iniciales"
    prob_pob_8=calcula_prob_tot(poblacion_8,nodos_continuan_8)
    "almacena el mejor valor de fitmess"
    best_fitness_8=[0]*n_best
    "almacena los valores de energia"
    best_fitness_energ_8=[20000]*n_best
    "almacena las n_best mejores ejecuciones"
    "best_execution = [[random.randint(0, 19) for _ in range(n)] for _ in range(n_best)]"
    best_execution_8 = [[0 for _ in range(n)] for _ in range(n_best)]
    
    "almacena las distancias de las n_best mejores ejecuciones"
    best_execution_dist_8 = [[0 for _ in range(n)] for _ in range(n_best)]
    "almacena los PDB de las n_best mejores ejecuciones"
    pdb_select_8 = ['a' for _ in range(n_best)]
    
    return nodo_terminal_8,amino_nodo_8,rango_numeros_8,nodos_continuan_8,indicador_8,num_amino_new_8,indicador_amino_8,M_ady_8,M_prob_8,indicador_entrada_8,indicador_entrada_pos_8,poblacion_8,prob_pob_8,best_fitness_8,best_fitness_energ_8,best_execution_8,best_execution_dist_8,pdb_select_8
    



def isla_8(generacion,utt_reinicio1,indicador_entrada_8,indicador_entrada_pos_8,nodo_terminal_8,act_prop,act_glob,best_fitness_8, best_execution_8, pdb_select_8, best_fitness_energ_8,best_execution_dist_8,a,b,MC_BB,corte,
           descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
           n_rep,tot_ececution,nodos_continuan_8,indicador_8,indicador_amino_8,M_ady_8,poblacion_8,prob_pob_8,M_prob_8):
    "para utilizar en las actualizaciones"
    best_execution_dist_8_1 = [[1 for _ in range(n)] for _ in range(len(best_execution_dist_8))]
    max_gdt_sal=0
    "i sera el individuo de la poblacion que se esta trabajando"
    for i in range(n_pop):
        "realiza las ejecuciones para el individuo de la poblacion"
        for kkk in range(n_rep):
            tot_ececution[i][kkk]=ejecuta_uno(prob_pob_8,i,n,M_ady_8, indicador_8, indicador_amino_8)
        "realiza el filtro correspondiente, por el momento sera las proimeras ejecuciones, no hay filtro"
        det_fitness=filtro1(tot_ececution,i,det_fitness,n_fit,n_rep)
        k=0
        for j in det_fitness[i]:
            print([generacion,i,j])
            "ejecucion j del elemento i de la poblacion a trabajar"
            posiciones_seq = tot_ececution[i][j]
            "Secuencia de aminoacidos de la ejecucion"
            secuencia = det_sec(posiciones_seq, amino)
            "PDB de la secuencia"
            if desc_si_no==1:
                descriptor_temp=descriptores(secuencia,tokenizer,model_ESM2) 
            if local==1:
                """inputs = tokenizer([secuencia], return_tensors="pt", add_special_tokens=False)
                outputs = model(**inputs)
                PDB_pdredict = [outputs.positions]"""
                pdb_str, output,ptm,plddt=genera(secuencia,model)
                PDB_pdredict = [pdb_str]
            else:
                PDB_pdredict = [_request_esmfold_prediction(secuencia)]
            "Backbone de la secuencia"
            back_temp=extract_backbone_atoms_str(PDB_pdredict[0])
            "asigna a p le estructura determinada"
            pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(p,PDB_pdredict[0]) 
            "calcula la energia"
            energia_desing=my_scorefxn(p)
            # PDB_pdredict = EMSFold_function(tokenizer,model,secuencia)
            "pdb_select[i][j] = PDB_pdredict"
            "rms y gdt de la secuencia"
            if desc_si_no==1:
                rms,gdt,MC_similitud, divKl,distancias_sal,tms=fitness_gdt_rmsd_mc_fisquim(BB, back_temp,MC_BB,corte,descriptor_ref,descriptor_temp)
                print([rms,gdt,energia_desing,MC_similitud,np.mean(divKl)])
                "se agregan los valores"
                fitness1[i][k]=agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b,tms)
                tot_ececution_dist[i][j]=[1] * len(distancias_sal)
            else:
                rms,gdt,MC_similitud,distancias_sal,tms=fitness_gdt_rmsd_mc(BB, back_temp,MC_BB,corte)
                print([rms,gdt,energia_desing,MC_similitud])
                fitness1[i][k]=agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b,tms)  
                tot_ececution_dist[i][j]=[1] * len(distancias_sal)
            
            "si la ejecucion tiene mayor fitnes que alguno en best_fitness entonces se actualiza la lista de los mejores"
            if fitness1[i][k]>best_fitness_8[pos_min]:
                best_fitness_8[pos_min]=fitness1[i][k]
                best_fitness_energ_8[pos_min]=[energia_desing,8,copy.deepcopy(generacion)]
                best_execution_8[pos_min]=list(tot_ececution[i][j])
                pdb_select_8[pos_min] = PDB_pdredict[0][:]
                best_execution_dist_8[pos_min]=distancias_sal
                best_fitness_8, best_execution_8, pdb_select_8, best_fitness_energ_8,best_execution_dist_8=ordena_mejores_energia(best_fitness_8, best_execution_8, pdb_select_8, best_fitness_energ_8,best_execution_dist_8)
                max_gdt_sal=max(max_gdt_sal,gdt)
            k+=1
        if generacion in utt_reinicio1 :
            w_act,w_act_dist=W_act1(i, best_execution_8, len(best_execution_8), tot_ececution, act_prop, det_fitness, fitness1,best_fitness_8,best_execution_dist_8_1,tot_ececution_dist)
            poblacion_8 = [[copy.deepcopy(M_prob_8) for _ in range(n)] for _ in range(n_pop)]
            poblacion_8=actualiza_individuo(n,poblacion_8,i,w_act,w_act_dist,indicador_entrada_8,indicador_entrada_pos_8,nodo_terminal_8)
        else:
            w_act,w_act_dist=W_act1(i, best_execution_8, act_glob, tot_ececution, act_prop, det_fitness, fitness1,best_fitness_8,best_execution_dist_8_1,tot_ececution_dist)
            poblacion_8=actualiza_individuo(n,poblacion_8,i,w_act,w_act_dist,indicador_entrada_8,indicador_entrada_pos_8,nodo_terminal_8)
    prob_pob_8=calcula_prob_tot(poblacion_8,nodos_continuan_8)       
    return max_gdt_sal,best_fitness_8, best_execution_8, pdb_select_8, best_fitness_energ_8,best_execution_dist_8,poblacion_8,prob_pob_8
    


def inicializa_isla_9(n,n_pop,amino,n_best):
    global best_execution_9
    global best_execution_dist_9
    global pdb_select_9
    global best_fitness_9
    global best_fitness_energ_9
    nodo_terminal_9=[8,11,12,14,15,16,19,20,22,23,24,25,26,27,28,29,30,31,32,33]
    amino_nodo_9=['H','C','A','F','L','M','D','N','T','G','P','S','V','I','K','R','E','Q','Y','W'] 
    rango_numeros_9 = set(range(34))  
    "nodos que continuan"
    nodos_continuan_9 = list(rango_numeros_9 - set(nodo_terminal_9))
    "indica si el nodo es terminal=1 o si continua=0"
    indicador_9 = [0] * 34
    for posicion in nodo_terminal_9:
        indicador_9[posicion] = 1
    "reorganiza las posiciones de los aminoaciodos segun el; nuevo orden dado por amino_nodo" 
    letra_a_indice_9 = {letra: indice for indice, letra in enumerate(amino)}
    num_amino_new_9 = [letra_a_indice_9[letra] for letra in amino_nodo_9]
    "indica segun el numero de nodo el numero de aminoacido que corresponde"
    indicador_amino_9=[0] * 34
    for i in range(20):
        indicador_amino_9[nodo_terminal_9[i]] = num_amino_new_9[i]
    "se construye la matriz de adyasencia del arbol y la matriz de probabilidades"
    M_ady_9=[0]*22
    M_prob_9=[0]*22
    M_ady_9[0]=[1,2,3]
    M_prob_9[0]=[5/20,7/20,8/20]
    M_ady_9[1]=[4,5]
    M_prob_9[1]=[3/5,2/5]
    M_ady_9[2]=[6,7,8]
    M_prob_9[2]=[4/7,2/7,1/7]
    M_ady_9[3]=[9,10,11,12]
    M_prob_9[3]=[3/8,3/8,1/8,1/8] 
    M_ady_9[4]=[13,14]
    M_prob_9[4]=[2/3,1/3]
    M_ady_9[5]=[15,16]
    M_prob_9[5]=[1/2,1/2]
    M_ady_9[6]=[17,18]
    M_prob_9[6]=[2/4,2/4]
    M_ady_9[7]=[19,20]
    M_prob_9[7]=[1/2,1/2]
    M_ady_9[9]=[21,22]
    M_prob_9[9]=[2/3,1/3]
    M_ady_9[10]=[23,24,25]
    M_prob_9[10]=[1/3,1/3,1/3]
    M_ady_9[13]=[26,27]
    M_prob_9[13]=[1/2,1/2]
    M_ady_9[17]=[28,29]
    M_prob_9[17]=[1/2,1/2]
    M_ady_9[18]=[30,31]
    M_prob_9[18]=[1/2,1/2]
    M_ady_9[21]=[32,33]
    M_prob_9[21]=[1/2,1/2]
    "indica el padre del nodo"
    indicador_entrada_9=[0]*34
    indicador_entrada_pos_9=[0]*34
    for i in range(len(M_ady_9)):
      if isinstance(M_ady_9[i], int):
        "puede ser entero"
      else:
        for j in range(len(M_ady_9[i])):
            indicador_entrada_9[M_ady_9[i][j]]=i
            indicador_entrada_pos_9[M_ady_9[i][j]]=j
    "construlle la poblacion inicial"
    poblacion_9 = [[copy.deepcopy(M_prob_9) for _ in range(n)] for _ in range(n_pop)]
    "determina las probabilidades iniciales"
    prob_pob_9=calcula_prob_tot(poblacion_9,nodos_continuan_9)
    "almacena el mejor valor de fitmess"
    best_fitness_9=[0]*n_best
    "almacena los valores de energia"
    best_fitness_energ_9=[20000]*n_best
    "almacena las n_best mejores ejecuciones"
    "best_execution = [[random.randint(0, 19) for _ in range(n)] for _ in range(n_best)]"
    best_execution_9 = [[0 for _ in range(n)] for _ in range(n_best)]
    
    "almacena las distancias de las n_best mejores ejecuciones"
    best_execution_dist_9 = [[0 for _ in range(n)] for _ in range(n_best)]
    "almacena los PDB de las n_best mejores ejecuciones"
    pdb_select_9 = ['a' for _ in range(n_best)]
    
    return nodo_terminal_9,amino_nodo_9,rango_numeros_9,nodos_continuan_9,indicador_9,num_amino_new_9,indicador_amino_9,M_ady_9,M_prob_9,indicador_entrada_9,indicador_entrada_pos_9,poblacion_9,prob_pob_9,best_fitness_9,best_fitness_energ_9,best_execution_9,best_execution_dist_9,pdb_select_9
    



def isla_9(indicador_entrada_9,indicador_entrada_pos_9,nodo_terminal_9,act_prop,act_glob,best_fitness_9, best_execution_9, pdb_select_9, best_fitness_energ_9,best_execution_dist_9,a,b,MC_BB,corte,
           descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
           n_rep,tot_ececution,nodos_continuan_9,indicador_9,indicador_amino_9,M_ady_9,poblacion_9,prob_pob_9,generacion):
    max_gdt_sal=0
    "i sera el individuo de la poblacion que se esta trabajando"
    for i in range(n_pop):
        "realiza las ejecuciones para el individuo de la poblacion"
        for kkk in range(n_rep):
            tot_ececution[i][kkk]=ejecuta_uno(prob_pob_9,i,n,M_ady_9, indicador_9, indicador_amino_9)
        "realiza el filtro correspondiente, por el momento sera las proimeras ejecuciones, no hay filtro"
        det_fitness=filtro1(tot_ececution,i,det_fitness,n_fit,n_rep)
        k=0
        for j in det_fitness[i]:
            print([i,j])
            "ejecucion j del elemento i de la poblacion a trabajar"
            posiciones_seq = tot_ececution[i][j]
            "Secuencia de aminoacidos de la ejecucion"
            secuencia = det_sec(posiciones_seq, amino)
            "PDB de la secuencia"
            if desc_si_no==1:
                descriptor_temp=descriptores(secuencia,tokenizer,model_ESM2) 
            if local==1:
                """inputs = tokenizer([secuencia], return_tensors="pt", add_special_tokens=False)
                outputs = model(**inputs)
                PDB_pdredict = [outputs.positions]"""
                pdb_str, output,ptm,plddt=genera(secuencia,model)
                PDB_pdredict = [pdb_str]
            else:
                PDB_pdredict = [_request_esmfold_prediction(secuencia)]
            "Backbone de la secuencia"
            back_temp=extract_backbone_atoms_str(PDB_pdredict[0])
            "asigna a p le estructura determinada"
            pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(p,PDB_pdredict[0]) 
            "calcula la energia"
            energia_desing=my_scorefxn(p)
            # PDB_pdredict = EMSFold_function(tokenizer,model,secuencia)
            "pdb_select[i][j] = PDB_pdredict"
            "rms y gdt de la secuencia"
            if desc_si_no==1:
                rms,gdt,MC_similitud, divKl,distancias_sal,tms=fitness_gdt_rmsd_mc_fisquim(BB, back_temp,MC_BB,corte,descriptor_ref,descriptor_temp)
                print([rms,gdt,energia_desing,MC_similitud,np.mean(divKl)])
                "se agregan los valores"
                fitness1[i][k]=agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b,tms)
                tot_ececution_dist[i][j]=distancias_sal
            else:
                rms,gdt,MC_similitud,distancias_sal,tms=fitness_gdt_rmsd_mc(BB, back_temp,MC_BB,corte)
                print([rms,gdt,energia_desing,MC_similitud])
                fitness1[i][k]=agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b,tms)  
                tot_ececution_dist[i][j]=distancias_sal
            
            "si la ejecucion tiene mayor fitnes que alguno en best_fitness entonces se actualiza la lista de los mejores"
            if fitness1[i][k]>best_fitness_9[pos_min]:
                best_fitness_9[pos_min]=fitness1[i][k]
                best_fitness_energ_9[pos_min]=[energia_desing,9,copy.deepcopy(generacion)]
                best_execution_9[pos_min]=list(tot_ececution[i][j])
                pdb_select_9[pos_min] = PDB_pdredict[0][:]
                best_execution_dist_9[pos_min]=distancias_sal
                best_fitness_9, best_execution_9, pdb_select_9, best_fitness_energ_9,best_execution_dist_9=ordena_mejores_energia(best_fitness_9, best_execution_9, pdb_select_9, best_fitness_energ_9,best_execution_dist_9)
                max_gdt_sal=max(max_gdt_sal,gdt)
            k+=1
        w_act,w_act_dist=W_act1(i, best_execution_9, act_glob, tot_ececution, act_prop, det_fitness, fitness1,best_fitness_9,best_execution_dist_9,tot_ececution_dist)
        poblacion_9=actualiza_individuo(n,poblacion_9,i,w_act,w_act_dist,indicador_entrada_9,indicador_entrada_pos_9,nodo_terminal_9)

    prob_pob_9=calcula_prob_tot(poblacion_9,nodos_continuan_9)       
    return max_gdt_sal,best_fitness_9, best_execution_9, pdb_select_9, best_fitness_energ_9,best_execution_dist_9,poblacion_9,prob_pob_9
    



def inicializa_isla_10(n,n_pop,amino,n_best):
    global best_execution_10
    global best_execution_dist_10
    global pdb_select_10
    global best_fitness_10
    global best_fitness_energ_10
    nodo_terminal_10=[8,11,12,14,15,16,19,20,22,23,24,25,26,27,28,29,30,31,32,33]
    amino_nodo_10=['H','C','A','F','L','M','D','N','T','G','P','S','V','I','K','R','E','Q','Y','W'] 
    rango_numeros_10 = set(range(34))  
    "nodos que continuan"
    nodos_continuan_10 = list(rango_numeros_10 - set(nodo_terminal_10))
    "indica si el nodo es terminal=1 o si continua=0"
    indicador_10 = [0] * 34
    for posicion in nodo_terminal_10:
        indicador_10[posicion] = 1
    "reorganiza las posiciones de los aminoaciodos segun el; nuevo orden dado por amino_nodo" 
    letra_a_indice_10 = {letra: indice for indice, letra in enumerate(amino)}
    num_amino_new_10 = [letra_a_indice_10[letra] for letra in amino_nodo_10]
    "indica segun el numero de nodo el numero de aminoacido que corresponde"
    indicador_amino_10=[0] * 34
    for i in range(20):
        indicador_amino_10[nodo_terminal_10[i]] = num_amino_new_10[i]
    "se construye la matriz de adyasencia del arbol y la matriz de probabilidades"
    M_ady_10=[0]*22
    M_prob_10=[0]*22
    M_ady_10[0]=[1,2,3]
    M_prob_10[0]=[5/20,7/20,8/20]
    M_ady_10[1]=[4,5]
    M_prob_10[1]=[3/5,2/5]
    M_ady_10[2]=[6,7,8]
    M_prob_10[2]=[4/7,2/7,1/7]
    M_ady_10[3]=[9,10,11,12]
    M_prob_10[3]=[3/8,3/8,1/8,1/8] 
    M_ady_10[4]=[13,14]
    M_prob_10[4]=[2/3,1/3]
    M_ady_10[5]=[15,16]
    M_prob_10[5]=[1/2,1/2]
    M_ady_10[6]=[17,18]
    M_prob_10[6]=[2/4,2/4]
    M_ady_10[7]=[19,20]
    M_prob_10[7]=[1/2,1/2]
    M_ady_10[9]=[21,22]
    M_prob_10[9]=[2/3,1/3]
    M_ady_10[10]=[23,24,25]
    M_prob_10[10]=[1/3,1/3,1/3]
    M_ady_10[13]=[26,27]
    M_prob_10[13]=[1/2,1/2]
    M_ady_10[17]=[28,29]
    M_prob_10[17]=[1/2,1/2]
    M_ady_10[18]=[30,31]
    M_prob_10[18]=[1/2,1/2]
    M_ady_10[21]=[32,33]
    M_prob_10[21]=[1/2,1/2]
    "indica el padre del nodo"
    indicador_entrada_10=[0]*34
    indicador_entrada_pos_10=[0]*34
    for i in range(len(M_ady_10)):
      if isinstance(M_ady_10[i], int):
        "puede ser entero"
      else:
        for j in range(len(M_ady_10[i])):
            indicador_entrada_10[M_ady_10[i][j]]=i
            indicador_entrada_pos_10[M_ady_10[i][j]]=j
    "construlle la poblacion inicial"
    poblacion_10 = [[copy.deepcopy(M_prob_10) for _ in range(n)] for _ in range(n_pop)]
    "determina las probabilidades iniciales"
    prob_pob_10=calcula_prob_tot(poblacion_10,nodos_continuan_10)
    "almacena el mejor valor de fitmess"
    best_fitness_10=[0]*n_best
    "almacena los valores de energia"
    best_fitness_energ_10=[20000]*n_best
    "almacena las n_best mejores ejecuciones"
    "best_execution = [[random.randint(0, 19) for _ in range(n)] for _ in range(n_best)]"
    best_execution_10 = [[0 for _ in range(n)] for _ in range(n_best)]
    
    "almacena las distancias de las n_best mejores ejecuciones"
    best_execution_dist_10 = [[0 for _ in range(n)] for _ in range(n_best)]
    "almacena los PDB de las n_best mejores ejecuciones"
    pdb_select_10 = ['a' for _ in range(n_best)]
    
    return nodo_terminal_10,amino_nodo_10,rango_numeros_10,nodos_continuan_10,indicador_10,num_amino_new_10,indicador_amino_10,M_ady_10,M_prob_10,indicador_entrada_10,indicador_entrada_pos_10,poblacion_10,prob_pob_10,best_fitness_10,best_fitness_energ_10,best_execution_10,best_execution_dist_10,pdb_select_10
    



def isla_10(indicador_entrada_10,indicador_entrada_pos_10,nodo_terminal_10,act_prop,act_glob,best_fitness_10, best_execution_10, pdb_select_10, best_fitness_energ_10,best_execution_dist_10,a,b,MC_BB,corte,
           descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
           n_rep,tot_ececution,nodos_continuan_10,indicador_10,indicador_amino_10,M_ady_10,poblacion_10,prob_pob_10,generacion):
    "para utilizar en las actualizaciones"
    best_execution_dist_10_1 = [[1 for _ in range(n)] for _ in range(len(best_execution_dist_10))]
    max_gdt_sal=0
    "i sera el individuo de la poblacion que se esta trabajando"
    for i in range(n_pop):
        "realiza las ejecuciones para el individuo de la poblacion"
        for kkk in range(n_rep):
            tot_ececution[i][kkk]=ejecuta_uno(prob_pob_10,i,n,M_ady_10, indicador_10, indicador_amino_10)
        "realiza el filtro correspondiente, por el momento sera las proimeras ejecuciones, no hay filtro"
        det_fitness=filtro1(tot_ececution,i,det_fitness,n_fit,n_rep)
        k=0
        for j in det_fitness[i]:
            print([i,j])
            "ejecucion j del elemento i de la poblacion a trabajar"
            posiciones_seq = tot_ececution[i][j]
            "Secuencia de aminoacidos de la ejecucion"
            secuencia = det_sec(posiciones_seq, amino)
            "PDB de la secuencia"
            if desc_si_no==1:
                descriptor_temp=descriptores(secuencia,tokenizer,model_ESM2) 
            if local==1:
                """inputs = tokenizer([secuencia], return_tensors="pt", add_special_tokens=False)
                outputs = model(**inputs)
                PDB_pdredict = [outputs.positions]"""
                pdb_str, output,ptm,plddt=genera(secuencia,model)
                PDB_pdredict = [pdb_str]
            else:
                PDB_pdredict = [_request_esmfold_prediction(secuencia)]
            "Backbone de la secuencia"
            back_temp=extract_backbone_atoms_str(PDB_pdredict[0])
            "asigna a p le estructura determinada"
            pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(p,PDB_pdredict[0]) 
            "calcula la energia"
            energia_desing=my_scorefxn(p)
            # PDB_pdredict = EMSFold_function(tokenizer,model,secuencia)
            "pdb_select[i][j] = PDB_pdredict"
            "rms y gdt de la secuencia"
            if desc_si_no==1:
                rms,gdt,MC_similitud, divKl,distancias_sal,tms=fitness_gdt_rmsd_mc_fisquim(BB, back_temp,MC_BB,corte,descriptor_ref,descriptor_temp)
                print([rms,gdt,energia_desing,MC_similitud,np.mean(divKl)])
                "se agregan los valores"
                fitness1[i][k]=agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b,tms)
                tot_ececution_dist[i][j]=[1] * len(distancias_sal)
            else:
                rms,gdt,MC_similitud,distancias_sal,tms=fitness_gdt_rmsd_mc(BB, back_temp,MC_BB,corte)
                print([rms,gdt,energia_desing,MC_similitud])
                fitness1[i][k]=agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b,tms)  
                tot_ececution_dist[i][j]=[1] * len(distancias_sal)
            
            "si la ejecucion tiene mayor fitnes que alguno en best_fitness entonces se actualiza la lista de los mejores"
            if fitness1[i][k]>best_fitness_10[pos_min]:
                best_fitness_10[pos_min]=fitness1[i][k]
                best_fitness_energ_10[pos_min]=[energia_desing,10,copy.deepcopy(generacion)]
                best_execution_10[pos_min]=list(tot_ececution[i][j])
                pdb_select_10[pos_min] = PDB_pdredict[0][:]
                best_execution_dist_10[pos_min]=distancias_sal
                best_fitness_10, best_execution_10, pdb_select_10, best_fitness_energ_10,best_execution_dist_10=ordena_mejores_energia(best_fitness_10, best_execution_10, pdb_select_10, best_fitness_energ_10,best_execution_dist_10)
                max_gdt_sal=max(max_gdt_sal,gdt)
            k+=1
        w_act,w_act_dist=W_act1(i, best_execution_10, act_glob, tot_ececution, act_prop, det_fitness, fitness1,best_fitness_10,best_execution_dist_10_1,tot_ececution_dist)
        poblacion_10=actualiza_individuo(n,poblacion_10,i,w_act,w_act_dist,indicador_entrada_10,indicador_entrada_pos_10,nodo_terminal_10)

    prob_pob_10=calcula_prob_tot(poblacion_10,nodos_continuan_10)       
    return max_gdt_sal,best_fitness_10, best_execution_10, pdb_select_10, best_fitness_energ_10,best_execution_dist_10,poblacion_10,prob_pob_10
    


def inicializa_isla_11(n,n_pop,amino,n_best):
    global best_execution_11
    global best_execution_dist_11
    global pdb_select_11
    global best_fitness_11
    global best_fitness_energ_11
    nodo_terminal_11=[8,11,12,14,15,16,19,20,22,23,24,25,26,27,28,29,30,31,32,33]
    amino_nodo_11=['H','C','A','F','L','M','D','N','T','G','P','S','V','I','K','R','E','Q','Y','W'] 
    rango_numeros_11 = set(range(34))  
    "nodos que continuan"
    nodos_continuan_11 = list(rango_numeros_11 - set(nodo_terminal_11))
    "indica si el nodo es terminal=1 o si continua=0"
    indicador_11 = [0] * 34
    for posicion in nodo_terminal_11:
        indicador_11[posicion] = 1
    "reorganiza las posiciones de los aminoaciodos segun el; nuevo orden dado por amino_nodo" 
    letra_a_indice_11 = {letra: indice for indice, letra in enumerate(amino)}
    num_amino_new_11 = [letra_a_indice_11[letra] for letra in amino_nodo_11]
    "indica segun el numero de nodo el numero de aminoacido que corresponde"
    indicador_amino_11=[0] * 34
    for i in range(20):
        indicador_amino_11[nodo_terminal_11[i]] = num_amino_new_11[i]
    "se construye la matriz de adyasencia del arbol y la matriz de probabilidades"
    M_ady_11=[0]*22
    M_prob_11=[0]*22
    M_ady_11[0]=[1,2,3]
    M_prob_11[0]=[5/20,7/20,8/20]
    M_ady_11[1]=[4,5]
    M_prob_11[1]=[3/5,2/5]
    M_ady_11[2]=[6,7,8]
    M_prob_11[2]=[4/7,2/7,1/7]
    M_ady_11[3]=[9,10,11,12]
    M_prob_11[3]=[3/8,3/8,1/8,1/8] 
    M_ady_11[4]=[13,14]
    M_prob_11[4]=[2/3,1/3]
    M_ady_11[5]=[15,16]
    M_prob_11[5]=[1/2,1/2]
    M_ady_11[6]=[17,18]
    M_prob_11[6]=[2/4,2/4]
    M_ady_11[7]=[19,20]
    M_prob_11[7]=[1/2,1/2]
    M_ady_11[9]=[21,22]
    M_prob_11[9]=[2/3,1/3]
    M_ady_11[10]=[23,24,25]
    M_prob_11[10]=[1/3,1/3,1/3]
    M_ady_11[13]=[26,27]
    M_prob_11[13]=[1/2,1/2]
    M_ady_11[17]=[28,29]
    M_prob_11[17]=[1/2,1/2]
    M_ady_11[18]=[30,31]
    M_prob_11[18]=[1/2,1/2]
    M_ady_11[21]=[32,33]
    M_prob_11[21]=[1/2,1/2]
    "indica el padre del nodo"
    indicador_entrada_11=[0]*34
    indicador_entrada_pos_11=[0]*34
    for i in range(len(M_ady_11)):
      if isinstance(M_ady_11[i], int):
        "puede ser entero"
      else:
        for j in range(len(M_ady_11[i])):
            indicador_entrada_11[M_ady_11[i][j]]=i
            indicador_entrada_pos_11[M_ady_11[i][j]]=j
    "construlle la poblacion inicial"
    poblacion_11 = [[copy.deepcopy(M_prob_11) for _ in range(n)] for _ in range(n_pop)]
    "determina las probabilidades iniciales"
    prob_pob_11=calcula_prob_tot(poblacion_11,nodos_continuan_11)
    "almacena el mejor valor de fitmess"
    best_fitness_11=[0]*n_best
    "almacena los valores de energia"
    best_fitness_energ_11=[20000]*n_best
    "almacena las n_best mejores ejecuciones"
    "best_execution = [[random.randint(0, 19) for _ in range(n)] for _ in range(n_best)]"
    best_execution_11 = [[0 for _ in range(n)] for _ in range(n_best)]
    
    "almacena las distancias de las n_best mejores ejecuciones"
    best_execution_dist_11 = [[0 for _ in range(n)] for _ in range(n_best)]
    "almacena los PDB de las n_best mejores ejecuciones"
    pdb_select_11 = ['a' for _ in range(n_best)]
    
    return nodo_terminal_11,amino_nodo_11,rango_numeros_11,nodos_continuan_11,indicador_11,num_amino_new_11,indicador_amino_11,M_ady_11,M_prob_11,indicador_entrada_11,indicador_entrada_pos_11,poblacion_11,prob_pob_11,best_fitness_11,best_fitness_energ_11,best_execution_11,best_execution_dist_11,pdb_select_11
    



def isla_11(generacion,utt_reinicio1,indicador_entrada_11,indicador_entrada_pos_11,nodo_terminal_11,act_prop,act_glob,best_fitness_11, best_execution_11, pdb_select_11, best_fitness_energ_11,best_execution_dist_11,a,b,MC_BB,corte,
           descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
           n_rep,tot_ececution,nodos_continuan_11,indicador_11,indicador_amino_11,M_ady_11,poblacion_11,prob_pob_11,M_prob_11):
    max_gdt_sal=0
    "i sera el individuo de la poblacion que se esta trabajando"
    for i in range(n_pop):
        "realiza las ejecuciones para el individuo de la poblacion"
        for kkk in range(n_rep):
            tot_ececution[i][kkk]=ejecuta_uno(prob_pob_11,i,n,M_ady_11, indicador_11, indicador_amino_11)
        "realiza el filtro correspondiente, por el momento sera las proimeras ejecuciones, no hay filtro"
        det_fitness=filtro1(tot_ececution,i,det_fitness,n_fit,n_rep)
        k=0
        for j in det_fitness[i]:
            print([i,j])
            "ejecucion j del elemento i de la poblacion a trabajar"
            posiciones_seq = tot_ececution[i][j]
            "Secuencia de aminoacidos de la ejecucion"
            secuencia = det_sec(posiciones_seq, amino)
            "PDB de la secuencia"
            if desc_si_no==1:
                descriptor_temp=descriptores(secuencia,tokenizer,model_ESM2) 
            if local==1:
                """inputs = tokenizer([secuencia], return_tensors="pt", add_special_tokens=False)
                outputs = model(**inputs)
                PDB_pdredict = [outputs.positions]"""
                pdb_str, output,ptm,plddt=genera(secuencia,model)
                PDB_pdredict = [pdb_str]
            else:
                PDB_pdredict = [_request_esmfold_prediction(secuencia)]
            "Backbone de la secuencia"
            back_temp=extract_backbone_atoms_str(PDB_pdredict[0])
            "asigna a p le estructura determinada"
            pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(p,PDB_pdredict[0]) 
            "calcula la energia"
            energia_desing=my_scorefxn(p)
            # PDB_pdredict = EMSFold_function(tokenizer,model,secuencia)
            "pdb_select[i][j] = PDB_pdredict"
            "rms y gdt de la secuencia"
            if desc_si_no==1:
                rms,gdt,MC_similitud, divKl,distancias_sal,tms=fitness_gdt_rmsd_mc_fisquim(BB, back_temp,MC_BB,corte,descriptor_ref,descriptor_temp)
                print([rms,gdt,energia_desing,MC_similitud,np.mean(divKl)])
                "se agregan los valores"
                fitness1[i][k]=agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b,tms)
                tot_ececution_dist[i][j]=distancias_sal
            else:
                rms,gdt,MC_similitud,distancias_sal,tms=fitness_gdt_rmsd_mc(BB, back_temp,MC_BB,corte)
                print([rms,gdt,energia_desing,MC_similitud])
                fitness1[i][k]=agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b,tms)  
                tot_ececution_dist[i][j]=distancias_sal
            
            "si la ejecucion tiene mayor fitnes que alguno en best_fitness entonces se actualiza la lista de los mejores"
            if fitness1[i][k]>best_fitness_11[pos_min]:
                best_fitness_11[pos_min]=fitness1[i][k]
                best_fitness_energ_11[pos_min]=[energia_desing,11,copy.deepcopy(generacion)]
                best_execution_11[pos_min]=list(tot_ececution[i][j])
                pdb_select_11[pos_min] = PDB_pdredict[0][:]
                best_execution_dist_11[pos_min]=distancias_sal
                best_fitness_11, best_execution_11, pdb_select_11, best_fitness_energ_11,best_execution_dist_11=ordena_mejores_energia(best_fitness_11, best_execution_11, pdb_select_11, best_fitness_energ_11,best_execution_dist_11)
                max_gdt_sal=max(max_gdt_sal,gdt)
            k+=1
        if generacion in utt_reinicio1 :
            w_act,w_act_dist=W_act1(i, best_execution_11, len(best_execution_11), tot_ececution, act_prop, det_fitness, fitness1,best_fitness_11,best_execution_dist_11,tot_ececution_dist)
            poblacion_11 = [[copy.deepcopy(M_prob_11) for _ in range(n)] for _ in range(n_pop)]
            poblacion_11=actualiza_individuo(n,poblacion_11,i,w_act,w_act_dist,indicador_entrada_11,indicador_entrada_pos_11,nodo_terminal_11)
        else:
            w_act,w_act_dist=W_act1(i, best_execution_11, act_glob, tot_ececution, act_prop, det_fitness, fitness1,best_fitness_11,best_execution_dist_11,tot_ececution_dist)
            poblacion_11=actualiza_individuo(n,poblacion_11,i,w_act,w_act_dist,indicador_entrada_11,indicador_entrada_pos_11,nodo_terminal_11)
    prob_pob_11=calcula_prob_tot(poblacion_11,nodos_continuan_11)       
    return max_gdt_sal,best_fitness_11, best_execution_11, pdb_select_11, best_fitness_energ_11,best_execution_dist_11,poblacion_11,prob_pob_11
    



def inicializa_isla_12(n,n_pop,amino,n_best):
    global best_execution_12
    global best_execution_dist_12
    global pdb_select_12
    global best_fitness_12
    global best_fitness_energ_12
    nodo_terminal_12=[8,11,12,14,15,16,19,20,22,23,24,25,26,27,28,29,30,31,32,33]
    amino_nodo_12=['H','C','A','F','L','M','D','N','T','G','P','S','V','I','K','R','E','Q','Y','W'] 
    rango_numeros_12 = set(range(34))  
    "nodos que continuan"
    nodos_continuan_12 = list(rango_numeros_12 - set(nodo_terminal_12))
    "indica si el nodo es terminal=1 o si continua=0"
    indicador_12 = [0] * 34
    for posicion in nodo_terminal_12:
        indicador_12[posicion] = 1
    "reorganiza las posiciones de los aminoaciodos segun el; nuevo orden dado por amino_nodo" 
    letra_a_indice_12 = {letra: indice for indice, letra in enumerate(amino)}
    num_amino_new_12 = [letra_a_indice_12[letra] for letra in amino_nodo_12]
    "indica segun el numero de nodo el numero de aminoacido que corresponde"
    indicador_amino_12=[0] * 34
    for i in range(20):
        indicador_amino_12[nodo_terminal_12[i]] = num_amino_new_12[i]
    "se construye la matriz de adyasencia del arbol y la matriz de probabilidades"
    M_ady_12=[0]*22
    M_prob_12=[0]*22
    M_ady_12[0]=[1,2,3]
    M_prob_12[0]=[5/20,7/20,8/20]
    M_ady_12[1]=[4,5]
    M_prob_12[1]=[3/5,2/5]
    M_ady_12[2]=[6,7,8]
    M_prob_12[2]=[4/7,2/7,1/7]
    M_ady_12[3]=[9,10,11,12]
    M_prob_12[3]=[3/8,3/8,1/8,1/8] 
    M_ady_12[4]=[13,14]
    M_prob_12[4]=[2/3,1/3]
    M_ady_12[5]=[15,16]
    M_prob_12[5]=[1/2,1/2]
    M_ady_12[6]=[17,18]
    M_prob_12[6]=[2/4,2/4]
    M_ady_12[7]=[19,20]
    M_prob_12[7]=[1/2,1/2]
    M_ady_12[9]=[21,22]
    M_prob_12[9]=[2/3,1/3]
    M_ady_12[10]=[23,24,25]
    M_prob_12[10]=[1/3,1/3,1/3]
    M_ady_12[13]=[26,27]
    M_prob_12[13]=[1/2,1/2]
    M_ady_12[17]=[28,29]
    M_prob_12[17]=[1/2,1/2]
    M_ady_12[18]=[30,31]
    M_prob_12[18]=[1/2,1/2]
    M_ady_12[21]=[32,33]
    M_prob_12[21]=[1/2,1/2]
    "indica el padre del nodo"
    indicador_entrada_12=[0]*34
    indicador_entrada_pos_12=[0]*34
    for i in range(len(M_ady_12)):
      if isinstance(M_ady_12[i], int):
        "puede ser entero"
      else:
        for j in range(len(M_ady_12[i])):
            indicador_entrada_12[M_ady_12[i][j]]=i
            indicador_entrada_pos_12[M_ady_12[i][j]]=j
    "construlle la poblacion inicial"
    poblacion_12 = [[copy.deepcopy(M_prob_12) for _ in range(n)] for _ in range(n_pop)]
    "determina las probabilidades iniciales"
    prob_pob_12=calcula_prob_tot(poblacion_12,nodos_continuan_12)
    "almacena el mejor valor de fitmess"
    best_fitness_12=[0]*n_best
    "almacena los valores de energia"
    best_fitness_energ_12=[20000]*n_best
    "almacena las n_best mejores ejecuciones"
    "best_execution = [[random.randint(0, 19) for _ in range(n)] for _ in range(n_best)]"
    best_execution_12 = [[0 for _ in range(n)] for _ in range(n_best)]
    
    "almacena las distancias de las n_best mejores ejecuciones"
    best_execution_dist_12 = [[0 for _ in range(n)] for _ in range(n_best)]
    "almacena los PDB de las n_best mejores ejecuciones"
    pdb_select_12 = ['a' for _ in range(n_best)]
    
    return nodo_terminal_12,amino_nodo_12,rango_numeros_12,nodos_continuan_12,indicador_12,num_amino_new_12,indicador_amino_12,M_ady_12,M_prob_12,indicador_entrada_12,indicador_entrada_pos_12,poblacion_12,prob_pob_12,best_fitness_12,best_fitness_energ_12,best_execution_12,best_execution_dist_12,pdb_select_12
    



def isla_12(generacion,utt_reinicio1,indicador_entrada_12,indicador_entrada_pos_12,nodo_terminal_12,act_prop,act_glob,best_fitness_12, best_execution_12, pdb_select_12, best_fitness_energ_12,best_execution_dist_12,a,b,MC_BB,corte,
           descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
           n_rep,tot_ececution,nodos_continuan_12,indicador_12,indicador_amino_12,M_ady_12,poblacion_12,prob_pob_12,M_prob_12):
    "para utilizar en las actualizaciones"
    best_execution_dist_12_1 = [[1 for _ in range(n)] for _ in range(len(best_execution_dist_12))]
    max_gdt_sal=0
    "i sera el individuo de la poblacion que se esta trabajando"
    for i in range(n_pop):
        "realiza las ejecuciones para el individuo de la poblacion"
        for kkk in range(n_rep):
            tot_ececution[i][kkk]=ejecuta_uno(prob_pob_12,i,n,M_ady_12, indicador_12, indicador_amino_12)
        "realiza el filtro correspondiente, por el momento sera las proimeras ejecuciones, no hay filtro"
        det_fitness=filtro1(tot_ececution,i,det_fitness,n_fit,n_rep)
        k=0
        for j in det_fitness[i]:
            print([i,j])
            "ejecucion j del elemento i de la poblacion a trabajar"
            posiciones_seq = tot_ececution[i][j]
            "Secuencia de aminoacidos de la ejecucion"
            secuencia = det_sec(posiciones_seq, amino)
            "PDB de la secuencia"
            if desc_si_no==1:
                descriptor_temp=descriptores(secuencia,tokenizer,model_ESM2) 
            if local==1:
                """inputs = tokenizer([secuencia], return_tensors="pt", add_special_tokens=False)
                outputs = model(**inputs)
                PDB_pdredict = [outputs.positions]"""
                pdb_str, output,ptm,plddt=genera(secuencia,model)
                PDB_pdredict = [pdb_str]
            else:
                PDB_pdredict = [_request_esmfold_prediction(secuencia)]
            "Backbone de la secuencia"
            back_temp=extract_backbone_atoms_str(PDB_pdredict[0])
            "asigna a p le estructura determinada"
            pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(p,PDB_pdredict[0]) 
            "calcula la energia"
            energia_desing=my_scorefxn(p)
            # PDB_pdredict = EMSFold_function(tokenizer,model,secuencia)
            "pdb_select[i][j] = PDB_pdredict"
            "rms y gdt de la secuencia"
            if desc_si_no==1:
                rms,gdt,MC_similitud, divKl,distancias_sal,tms=fitness_gdt_rmsd_mc_fisquim(BB, back_temp,MC_BB,corte,descriptor_ref,descriptor_temp)
                print([rms,gdt,energia_desing,MC_similitud,np.mean(divKl)])
                "se agregan los valores"
                fitness1[i][k]=agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b,tms)
                tot_ececution_dist[i][j]=[1] * len(distancias_sal)
            else:
                rms,gdt,MC_similitud,distancias_sal,tms=fitness_gdt_rmsd_mc(BB, back_temp,MC_BB,corte)
                print([rms,gdt,energia_desing,MC_similitud])
                fitness1[i][k]=agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b,tms)  
                tot_ececution_dist[i][j]=[1] * len(distancias_sal)
            
            "si la ejecucion tiene mayor fitnes que alguno en best_fitness entonces se actualiza la lista de los mejores"
            if fitness1[i][k]>best_fitness_12[pos_min]:
                best_fitness_12[pos_min]=fitness1[i][k]
                best_fitness_energ_12[pos_min]=[energia_desing,12,copy.deepcopy(generacion)]
                best_execution_12[pos_min]=list(tot_ececution[i][j])
                pdb_select_12[pos_min] = PDB_pdredict[0][:]
                best_execution_dist_12[pos_min]=distancias_sal
                best_fitness_12, best_execution_12, pdb_select_12, best_fitness_energ_12,best_execution_dist_12=ordena_mejores_energia(best_fitness_12, best_execution_12, pdb_select_12, best_fitness_energ_12,best_execution_dist_12)
                max_gdt_sal=max(max_gdt_sal,gdt)
            k+=1
        if generacion in utt_reinicio1 :
            w_act,w_act_dist=W_act1(i, best_execution_12, len(best_execution_12), tot_ececution, act_prop, det_fitness, fitness1,best_fitness_12,best_execution_dist_12_1,tot_ececution_dist)
            poblacion_12 = [[copy.deepcopy(M_prob_12) for _ in range(n)] for _ in range(n_pop)]
            poblacion_12=actualiza_individuo(n,poblacion_12,i,w_act,w_act_dist,indicador_entrada_12,indicador_entrada_pos_12,nodo_terminal_12)
        else:
            w_act,w_act_dist=W_act1(i, best_execution_12, act_glob, tot_ececution, act_prop, det_fitness, fitness1,best_fitness_12,best_execution_dist_12_1,tot_ececution_dist)
            poblacion_12=actualiza_individuo(n,poblacion_12,i,w_act,w_act_dist,indicador_entrada_12,indicador_entrada_pos_12,nodo_terminal_12)
    prob_pob_12=calcula_prob_tot(poblacion_12,nodos_continuan_12)       
    return max_gdt_sal,best_fitness_12, best_execution_12, pdb_select_12, best_fitness_energ_12,best_execution_dist_12,poblacion_12,prob_pob_12
    


def inicializa_isla_13(n,n_pop,amino,n_best):
    global best_execution_13
    global best_execution_dist_13
    global pdb_select_13
    global best_fitness_13
    global best_fitness_energ_13
    nodo_terminal_13=[5,12,13,14,15,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
    amino_nodo_13=['A','C','H','L','M','F','T','G','P','S','N','D','E','Q','R','K','V','I','Y','W'] 
    rango_numeros_13 = set(range(35))  
    "nodos que continuan"
    nodos_continuan_13 = list(rango_numeros_13 - set(nodo_terminal_13))
    "indica si el nodo es terminal=1 o si continua=0"
    indicador_13 = [0] * 35
    for posicion in nodo_terminal_13:
        indicador_13[posicion] = 1
    "reorganiza las posiciones de los aminoaciodos segun el; nuevo orden dado por amino_nodo" 
    letra_a_indice_13 = {letra: indice for indice, letra in enumerate(amino)}
    num_amino_new_13 = [letra_a_indice_13[letra] for letra in amino_nodo_13]
    "indica segun el numero de nodo el numero de aminoacido que corresponde"
    indicador_amino_13=[0] * 35
    for i in range(20):
        indicador_amino_13[nodo_terminal_13[i]] = num_amino_new_13[i]
    "se construye la matriz de adyasencia del arbol y la matriz de probabilidades"
    M_ady_13=[0]*21
    M_prob_13=[0]*21
    M_ady_13[0]=[1,2,3,4]
    M_prob_13[0]=[7/20,6/20,5/20,2/20]
    M_ady_13[1]=[5,6,7]
    M_prob_13[1]=[1/7,2/7,4/7]
    M_ady_13[2]=[8,9]
    M_prob_13[2]=[3/6,3/6]
    M_ady_13[3]=[10,11]
    M_prob_13[3]=[3/5,2/5]
    M_ady_13[4]=[12,13]
    M_prob_13[4]=[1/2,1/2]
    M_ady_13[6]=[14,15]
    M_prob_13[6]=[1/2,1/2]
    M_ady_13[7]=[16,17]
    M_prob_13[7]=[2/4,2/4]
    M_ady_13[8]=[18,19]
    M_prob_13[8]=[2/3,1/3]
    M_ady_13[9]=[20,21]
    M_prob_13[9]=[2/3,1/3]
    M_ady_13[10]=[22,23,24]
    M_prob_13[10]=[1/3,1/3,1/3]
    M_ady_13[11]=[25,26]
    M_prob_13[11]=[1/2,1/2]
    M_ady_13[16]=[27,28]
    M_prob_13[16]=[1/2,1/2]
    M_ady_13[17]=[29,30]
    M_prob_13[17]=[1/2,1/2]
    M_ady_13[18]=[31,32]
    M_prob_13[18]=[1/2,1/2]
    M_ady_13[20]=[33,34]
    M_prob_13[20]=[1/2,1/2]
    "indica el padre del nodo"
    indicador_entrada_13=[0]*35
    indicador_entrada_pos_13=[0]*35
    for i in range(len(M_ady_13)):
      if isinstance(M_ady_13[i], int):
        "puede ser entero"
      else:
        for j in range(len(M_ady_13[i])):
            indicador_entrada_13[M_ady_13[i][j]]=i
            indicador_entrada_pos_13[M_ady_13[i][j]]=j
    "construlle la poblacion inicial"
    poblacion_13 = [[copy.deepcopy(M_prob_13) for _ in range(n)] for _ in range(n_pop)]
    "determina las probabilidades iniciales"
    prob_pob_13=calcula_prob_tot(poblacion_13,nodos_continuan_13)
    "almacena el mejor valor de fitmess"
    best_fitness_13=[0]*n_best
    "almacena los valores de energia"
    best_fitness_energ_13=[20000]*n_best
    "almacena las n_best mejores ejecuciones"
    "best_execution = [[random.randint(0, 19) for _ in range(n)] for _ in range(n_best)]"
    best_execution_13 = [[0 for _ in range(n)] for _ in range(n_best)]
    
    "almacena las distancias de las n_best mejores ejecuciones"
    best_execution_dist_13 = [[0 for _ in range(n)] for _ in range(n_best)]
    "almacena los PDB de las n_best mejores ejecuciones"
    pdb_select_13 = ['a' for _ in range(n_best)]
    
    return nodo_terminal_13,amino_nodo_13,rango_numeros_13,nodos_continuan_13,indicador_13,num_amino_new_13,indicador_amino_13,M_ady_13,M_prob_13,indicador_entrada_13,indicador_entrada_pos_13,poblacion_13,prob_pob_13,best_fitness_13,best_fitness_energ_13,best_execution_13,best_execution_dist_13,pdb_select_13
    



def isla_13(indicador_entrada_13,indicador_entrada_pos_13,nodo_terminal_13,act_prop,act_glob,best_fitness_13, best_execution_13, pdb_select_13, best_fitness_energ_13,best_execution_dist_13,a,b,MC_BB,corte,
           descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
           n_rep,tot_ececution,nodos_continuan_13,indicador_13,indicador_amino_13,M_ady_13,poblacion_13,prob_pob_13,generacion):
    max_gdt_sal=0
    "i sera el individuo de la poblacion que se esta trabajando"
    for i in range(n_pop):
        "realiza las ejecuciones para el individuo de la poblacion"
        for kkk in range(n_rep):
            tot_ececution[i][kkk]=ejecuta_uno(prob_pob_13,i,n,M_ady_13, indicador_13, indicador_amino_13)
        "realiza el filtro correspondiente, por el momento sera las proimeras ejecuciones, no hay filtro"
        det_fitness=filtro1(tot_ececution,i,det_fitness,n_fit,n_rep)
        k=0
        for j in det_fitness[i]:
            print([i,j])
            "ejecucion j del elemento i de la poblacion a trabajar"
            posiciones_seq = tot_ececution[i][j]
            "Secuencia de aminoacidos de la ejecucion"
            secuencia = det_sec(posiciones_seq, amino)
            "PDB de la secuencia"
            if desc_si_no==1:
                descriptor_temp=descriptores(secuencia,tokenizer,model_ESM2) 
            if local==1:
                """inputs = tokenizer([secuencia], return_tensors="pt", add_special_tokens=False)
                outputs = model(**inputs)
                PDB_pdredict = [outputs.positions]"""
                pdb_str, output,ptm,plddt=genera(secuencia,model)
                PDB_pdredict = [pdb_str]
            else:
                PDB_pdredict = [_request_esmfold_prediction(secuencia)]
            "Backbone de la secuencia"
            back_temp=extract_backbone_atoms_str(PDB_pdredict[0])
            "asigna a p le estructura determinada"
            pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(p,PDB_pdredict[0]) 
            "calcula la energia"
            energia_desing=my_scorefxn(p)
            # PDB_pdredict = EMSFold_function(tokenizer,model,secuencia)
            "pdb_select[i][j] = PDB_pdredict"
            "rms y gdt de la secuencia"
            if desc_si_no==1:
                rms,gdt,MC_similitud, divKl,distancias_sal,tms=fitness_gdt_rmsd_mc_fisquim(BB, back_temp,MC_BB,corte,descriptor_ref,descriptor_temp)
                print([rms,gdt,energia_desing,MC_similitud,np.mean(divKl)])
                "se agregan los valores"
                fitness1[i][k]=agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b,tms)
                tot_ececution_dist[i][j]=distancias_sal
            else:
                rms,gdt,MC_similitud,distancias_sal,tms=fitness_gdt_rmsd_mc(BB, back_temp,MC_BB,corte)
                print([rms,gdt,energia_desing,MC_similitud])
                fitness1[i][k]=agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b,tms)  
                tot_ececution_dist[i][j]=distancias_sal
            
            "si la ejecucion tiene mayor fitnes que alguno en best_fitness entonces se actualiza la lista de los mejores"
            if fitness1[i][k]>best_fitness_13[pos_min]:
                best_fitness_13[pos_min]=fitness1[i][k]
                best_fitness_energ_13[pos_min]=[energia_desing,13,copy.deepcopy(generacion)]
                best_execution_13[pos_min]=list(tot_ececution[i][j])
                pdb_select_13[pos_min] = PDB_pdredict[0][:]
                best_execution_dist_13[pos_min]=distancias_sal
                best_fitness_13, best_execution_13, pdb_select_13, best_fitness_energ_13,best_execution_dist_13=ordena_mejores_energia(best_fitness_13, best_execution_13, pdb_select_13, best_fitness_energ_13,best_execution_dist_13)
                max_gdt_sal=max(max_gdt_sal,gdt)
            k+=1
        w_act,w_act_dist=W_act1(i, best_execution_13, act_glob, tot_ececution, act_prop, det_fitness, fitness1,best_fitness_13,best_execution_dist_13,tot_ececution_dist)
        poblacion_13=actualiza_individuo(n,poblacion_13,i,w_act,w_act_dist,indicador_entrada_13,indicador_entrada_pos_13,nodo_terminal_13)

    prob_pob_13=calcula_prob_tot(poblacion_13,nodos_continuan_13)       
    return max_gdt_sal,best_fitness_13, best_execution_13, pdb_select_13, best_fitness_energ_13,best_execution_dist_13,poblacion_13,prob_pob_13
    

def inicializa_isla_14(n,n_pop,amino,n_best):
    global best_execution_14
    global best_execution_dist_14
    global pdb_select_14
    global best_fitness_14
    global best_fitness_energ_14
    nodo_terminal_14=[5,12,13,14,15,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
    amino_nodo_14=['A','C','H','L','M','F','T','G','P','S','N','D','E','Q','R','K','V','I','Y','W'] 
    rango_numeros_14 = set(range(35))  
    "nodos que continuan"
    nodos_continuan_14 = list(rango_numeros_14 - set(nodo_terminal_14))
    "indica si el nodo es terminal=1 o si continua=0"
    indicador_14 = [0] * 35
    for posicion in nodo_terminal_14:
        indicador_14[posicion] = 1
    "reorganiza las posiciones de los aminoaciodos segun el; nuevo orden dado por amino_nodo" 
    letra_a_indice_14 = {letra: indice for indice, letra in enumerate(amino)}
    num_amino_new_14 = [letra_a_indice_14[letra] for letra in amino_nodo_14]
    "indica segun el numero de nodo el numero de aminoacido que corresponde"
    indicador_amino_14=[0] * 35
    for i in range(20):
        indicador_amino_14[nodo_terminal_14[i]] = num_amino_new_14[i]
    "se construye la matriz de adyasencia del arbol y la matriz de probabilidades"
    M_ady_14=[0]*21
    M_prob_14=[0]*21
    M_ady_14[0]=[1,2,3,4]
    M_prob_14[0]=[7/20,6/20,5/20,2/20]
    M_ady_14[1]=[5,6,7]
    M_prob_14[1]=[1/7,2/7,4/7]
    M_ady_14[2]=[8,9]
    M_prob_14[2]=[3/6,3/6]
    M_ady_14[3]=[10,11]
    M_prob_14[3]=[3/5,2/5]
    M_ady_14[4]=[12,13]
    M_prob_14[4]=[1/2,1/2]
    M_ady_14[6]=[14,15]
    M_prob_14[6]=[1/2,1/2]
    M_ady_14[7]=[16,17]
    M_prob_14[7]=[2/4,2/4]
    M_ady_14[8]=[18,19]
    M_prob_14[8]=[2/3,1/3]
    M_ady_14[9]=[20,21]
    M_prob_14[9]=[2/3,1/3]
    M_ady_14[10]=[22,23,24]
    M_prob_14[10]=[1/3,1/3,1/3]
    M_ady_14[11]=[25,26]
    M_prob_14[11]=[1/2,1/2]
    M_ady_14[16]=[27,28]
    M_prob_14[16]=[1/2,1/2]
    M_ady_14[17]=[29,30]
    M_prob_14[17]=[1/2,1/2]
    M_ady_14[18]=[31,32]
    M_prob_14[18]=[1/2,1/2]
    M_ady_14[20]=[33,34]
    M_prob_14[20]=[1/2,1/2]
    "indica el padre del nodo"
    indicador_entrada_14=[0]*35
    indicador_entrada_pos_14=[0]*35
    for i in range(len(M_ady_14)):
      if isinstance(M_ady_14[i], int):
        "puede ser entero"
      else:
        for j in range(len(M_ady_14[i])):
            indicador_entrada_14[M_ady_14[i][j]]=i
            indicador_entrada_pos_14[M_ady_14[i][j]]=j
    "construlle la poblacion inicial"
    poblacion_14 = [[copy.deepcopy(M_prob_14) for _ in range(n)] for _ in range(n_pop)]
    "determina las probabilidades iniciales"
    prob_pob_14=calcula_prob_tot(poblacion_14,nodos_continuan_14)
    "almacena el mejor valor de fitmess"
    best_fitness_14=[0]*n_best
    "almacena los valores de energia"
    best_fitness_energ_14=[20000]*n_best
    "almacena las n_best mejores ejecuciones"
    "best_execution = [[random.randint(0, 19) for _ in range(n)] for _ in range(n_best)]"
    best_execution_14 = [[0 for _ in range(n)] for _ in range(n_best)]
    
    "almacena las distancias de las n_best mejores ejecuciones"
    best_execution_dist_14 = [[0 for _ in range(n)] for _ in range(n_best)]
    "almacena los PDB de las n_best mejores ejecuciones"
    pdb_select_14 = ['a' for _ in range(n_best)]
    
    return nodo_terminal_14,amino_nodo_14,rango_numeros_14,nodos_continuan_14,indicador_14,num_amino_new_14,indicador_amino_14,M_ady_14,M_prob_14,indicador_entrada_14,indicador_entrada_pos_14,poblacion_14,prob_pob_14,best_fitness_14,best_fitness_energ_14,best_execution_14,best_execution_dist_14,pdb_select_14
    



def isla_14(indicador_entrada_14,indicador_entrada_pos_14,nodo_terminal_14,act_prop,act_glob,best_fitness_14, best_execution_14, pdb_select_14, best_fitness_energ_14,best_execution_dist_14,a,b,MC_BB,corte,
           descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
           n_rep,tot_ececution,nodos_continuan_14,indicador_14,indicador_amino_14,M_ady_14,poblacion_14,prob_pob_14,generacion):
    "para utilizar en las actualizaciones"
    best_execution_dist_14_14 = [[1 for _ in range(n)] for _ in range(len(best_execution_dist_14))]
    max_gdt_sal=0
    "i sera el individuo de la poblacion que se esta trabajando"
    for i in range(n_pop):
        "realiza las ejecuciones para el individuo de la poblacion"
        for kkk in range(n_rep):
            tot_ececution[i][kkk]=ejecuta_uno(prob_pob_14,i,n,M_ady_14, indicador_14, indicador_amino_14)
        "realiza el filtro correspondiente, por el momento sera las proimeras ejecuciones, no hay filtro"
        det_fitness=filtro1(tot_ececution,i,det_fitness,n_fit,n_rep)
        k=0
        for j in det_fitness[i]:
            print([i,j])
            "ejecucion j del elemento i de la poblacion a trabajar"
            posiciones_seq = tot_ececution[i][j]
            "Secuencia de aminoacidos de la ejecucion"
            secuencia = det_sec(posiciones_seq, amino)
            "PDB de la secuencia"
            if desc_si_no==1:
                descriptor_temp=descriptores(secuencia,tokenizer,model_ESM2) 
            if local==1:
                """inputs = tokenizer([secuencia], return_tensors="pt", add_special_tokens=False)
                outputs = model(**inputs)
                PDB_pdredict = [outputs.positions]"""
                pdb_str, output,ptm,plddt=genera(secuencia,model)
                PDB_pdredict = [pdb_str]
            else:
                PDB_pdredict = [_request_esmfold_prediction(secuencia)]
            "Backbone de la secuencia"
            back_temp=extract_backbone_atoms_str(PDB_pdredict[0])
            "asigna a p le estructura determinada"
            pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(p,PDB_pdredict[0]) 
            "calcula la energia"
            energia_desing=my_scorefxn(p)
            # PDB_pdredict = EMSFold_function(tokenizer,model,secuencia)
            "pdb_select[i][j] = PDB_pdredict"
            "rms y gdt de la secuencia"
            if desc_si_no==1:
                rms,gdt,MC_similitud, divKl,distancias_sal,tms=fitness_gdt_rmsd_mc_fisquim(BB, back_temp,MC_BB,corte,descriptor_ref,descriptor_temp)
                print([rms,gdt,energia_desing,MC_similitud,np.mean(divKl)])
                "se agregan los valores"
                fitness1[i][k]=agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b,tms)
                tot_ececution_dist[i][j]= [1] * len(distancias_sal)
            else:
                rms,gdt,MC_similitud,distancias_sal,tms=fitness_gdt_rmsd_mc(BB, back_temp,MC_BB,corte)
                print([rms,gdt,energia_desing,MC_similitud])
                fitness1[i][k]=agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b,tms)  
                tot_ececution_dist[i][j]=[1] * len(distancias_sal)
            
            "si la ejecucion tiene mayor fitnes que alguno en best_fitness entonces se actualiza la lista de los mejores"
            if fitness1[i][k]>best_fitness_14[pos_min]:
                best_fitness_14[pos_min]=fitness1[i][k]
                best_fitness_energ_14[pos_min]=[energia_desing,14,copy.deepcopy(generacion)]
                best_execution_14[pos_min]=list(tot_ececution[i][j])
                pdb_select_14[pos_min] = PDB_pdredict[0][:]
                best_execution_dist_14[pos_min]=distancias_sal
                best_fitness_14, best_execution_14, pdb_select_14, best_fitness_energ_14,best_execution_dist_14=ordena_mejores_energia(best_fitness_14, best_execution_14, pdb_select_14, best_fitness_energ_14,best_execution_dist_14)
                max_gdt_sal=max(max_gdt_sal,gdt)
            k+=1
        w_act,w_act_dist=W_act1(i, best_execution_14, act_glob, tot_ececution, act_prop, det_fitness, fitness1,best_fitness_14,best_execution_dist_14_14,tot_ececution_dist)
        poblacion_14=actualiza_individuo(n,poblacion_14,i,w_act,w_act_dist,indicador_entrada_14,indicador_entrada_pos_14,nodo_terminal_14)
    prob_pob_14=calcula_prob_tot(poblacion_14,nodos_continuan_14)       
    return max_gdt_sal,best_fitness_14, best_execution_14, pdb_select_14, best_fitness_energ_14,best_execution_dist_14,poblacion_14,prob_pob_14
    



def inicializa_isla_15(n,n_pop,amino,n_best):
    global best_execution_15
    global best_execution_dist_15
    global pdb_select_15
    global best_fitness_15
    global best_fitness_energ_15
    nodo_terminal_15=[5,12,13,14,15,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
    amino_nodo_15=['A','C','H','L','M','F','T','G','P','S','N','D','E','Q','R','K','V','I','Y','W'] 
    rango_numeros_15 = set(range(35))  
    "nodos que continuan"
    nodos_continuan_15 = list(rango_numeros_15 - set(nodo_terminal_15))
    "indica si el nodo es terminal=1 o si continua=0"
    indicador_15 = [0] * 35
    for posicion in nodo_terminal_15:
        indicador_15[posicion] = 1
    "reorganiza las posiciones de los aminoaciodos segun el; nuevo orden dado por amino_nodo" 
    letra_a_indice_15 = {letra: indice for indice, letra in enumerate(amino)}
    num_amino_new_15 = [letra_a_indice_15[letra] for letra in amino_nodo_15]
    "indica segun el numero de nodo el numero de aminoacido que corresponde"
    indicador_amino_15=[0] * 35
    for i in range(20):
        indicador_amino_15[nodo_terminal_15[i]] = num_amino_new_15[i]
    "se construye la matriz de adyasencia del arbol y la matriz de probabilidades"
    M_ady_15=[0]*21
    M_prob_15=[0]*21
    M_ady_15[0]=[1,2,3,4]
    M_prob_15[0]=[7/20,6/20,5/20,2/20]
    M_ady_15[1]=[5,6,7]
    M_prob_15[1]=[1/7,2/7,4/7]
    M_ady_15[2]=[8,9]
    M_prob_15[2]=[3/6,3/6]
    M_ady_15[3]=[10,11]
    M_prob_15[3]=[3/5,2/5]
    M_ady_15[4]=[12,13]
    M_prob_15[4]=[1/2,1/2]
    M_ady_15[6]=[14,15]
    M_prob_15[6]=[1/2,1/2]
    M_ady_15[7]=[16,17]
    M_prob_15[7]=[2/4,2/4]
    M_ady_15[8]=[18,19]
    M_prob_15[8]=[2/3,1/3]
    M_ady_15[9]=[20,21]
    M_prob_15[9]=[2/3,1/3]
    M_ady_15[10]=[22,23,24]
    M_prob_15[10]=[1/3,1/3,1/3]
    M_ady_15[11]=[25,26]
    M_prob_15[11]=[1/2,1/2]
    M_ady_15[16]=[27,28]
    M_prob_15[16]=[1/2,1/2]
    M_ady_15[17]=[29,30]
    M_prob_15[17]=[1/2,1/2]
    M_ady_15[18]=[31,32]
    M_prob_15[18]=[1/2,1/2]
    M_ady_15[20]=[33,34]
    M_prob_15[20]=[1/2,1/2]
    "indica el padre del nodo"
    indicador_entrada_15=[0]*35
    indicador_entrada_pos_15=[0]*35
    for i in range(len(M_ady_15)):
      if isinstance(M_ady_15[i], int):
        "puede ser entero"
      else:
        for j in range(len(M_ady_15[i])):
            indicador_entrada_15[M_ady_15[i][j]]=i
            indicador_entrada_pos_15[M_ady_15[i][j]]=j
    "construlle la poblacion inicial"
    poblacion_15 = [[copy.deepcopy(M_prob_15) for _ in range(n)] for _ in range(n_pop)]
    "determina las probabilidades iniciales"
    prob_pob_15=calcula_prob_tot(poblacion_15,nodos_continuan_15)
    "almacena el mejor valor de fitmess"
    best_fitness_15=[0]*n_best
    "almacena los valores de energia"
    best_fitness_energ_15=[20000]*n_best
    "almacena las n_best mejores ejecuciones"
    "best_execution = [[random.randint(0, 19) for _ in range(n)] for _ in range(n_best)]"
    best_execution_15 = [[0 for _ in range(n)] for _ in range(n_best)]
    
    "almacena las distancias de las n_best mejores ejecuciones"
    best_execution_dist_15 = [[0 for _ in range(n)] for _ in range(n_best)]
    "almacena los PDB de las n_best mejores ejecuciones"
    pdb_select_15 = ['a' for _ in range(n_best)]
    
    return nodo_terminal_15,amino_nodo_15,rango_numeros_15,nodos_continuan_15,indicador_15,num_amino_new_15,indicador_amino_15,M_ady_15,M_prob_15,indicador_entrada_15,indicador_entrada_pos_15,poblacion_15,prob_pob_15,best_fitness_15,best_fitness_energ_15,best_execution_15,best_execution_dist_15,pdb_select_15
    



def isla_15(generacion,utt_reinicio1,indicador_entrada_15,indicador_entrada_pos_15,nodo_terminal_15,act_prop,act_glob,best_fitness_15, best_execution_15, pdb_select_15, best_fitness_energ_15,best_execution_dist_15,a,b,MC_BB,corte,
           descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
           n_rep,tot_ececution,nodos_continuan_15,indicador_15,indicador_amino_15,M_ady_15,poblacion_15,prob_pob_15,M_prob_15):
    max_gdt_sal=0
    "i sera el individuo de la poblacion que se esta trabajando"
    for i in range(n_pop):
        "realiza las ejecuciones para el individuo de la poblacion"
        for kkk in range(n_rep):
            tot_ececution[i][kkk]=ejecuta_uno(prob_pob_15,i,n,M_ady_15, indicador_15, indicador_amino_15)
        "realiza el filtro correspondiente, por el momento sera las proimeras ejecuciones, no hay filtro"
        det_fitness=filtro1(tot_ececution,i,det_fitness,n_fit,n_rep)
        k=0
        for j in det_fitness[i]:
            print([i,j])
            "ejecucion j del elemento i de la poblacion a trabajar"
            posiciones_seq = tot_ececution[i][j]
            "Secuencia de aminoacidos de la ejecucion"
            secuencia = det_sec(posiciones_seq, amino)
            "PDB de la secuencia"
            if desc_si_no==1:
                descriptor_temp=descriptores(secuencia,tokenizer,model_ESM2) 
            if local==1:
                """inputs = tokenizer([secuencia], return_tensors="pt", add_special_tokens=False)
                outputs = model(**inputs)
                PDB_pdredict = [outputs.positions]"""
                pdb_str, output,ptm,plddt=genera(secuencia,model)
                PDB_pdredict = [pdb_str]
            else:
                PDB_pdredict = [_request_esmfold_prediction(secuencia)]
            "Backbone de la secuencia"
            back_temp=extract_backbone_atoms_str(PDB_pdredict[0])
            "asigna a p le estructura determinada"
            pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(p,PDB_pdredict[0]) 
            "calcula la energia"
            energia_desing=my_scorefxn(p)
            # PDB_pdredict = EMSFold_function(tokenizer,model,secuencia)
            "pdb_select[i][j] = PDB_pdredict"
            "rms y gdt de la secuencia"
            if desc_si_no==1:
                rms,gdt,MC_similitud, divKl,distancias_sal,tms=fitness_gdt_rmsd_mc_fisquim(BB, back_temp,MC_BB,corte,descriptor_ref,descriptor_temp)
                print([rms,gdt,energia_desing,MC_similitud,np.mean(divKl)])
                "se agregan los valores"
                fitness1[i][k]=agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b,tms)
                tot_ececution_dist[i][j]=distancias_sal
            else:
                rms,gdt,MC_similitud,distancias_sal,tms=fitness_gdt_rmsd_mc(BB, back_temp,MC_BB,corte)
                print([rms,gdt,energia_desing,MC_similitud])
                fitness1[i][k]=agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b,tms)  
                tot_ececution_dist[i][j]=distancias_sal
            
            "si la ejecucion tiene mayor fitnes que alguno en best_fitness entonces se actualiza la lista de los mejores"
            if fitness1[i][k]>best_fitness_15[pos_min]:
                best_fitness_15[pos_min]=fitness1[i][k]
                best_fitness_energ_15[pos_min]=[energia_desing,15,copy.deepcopy(generacion)]
                best_execution_15[pos_min]=list(tot_ececution[i][j])
                pdb_select_15[pos_min] = PDB_pdredict[0][:]
                best_execution_dist_15[pos_min]=distancias_sal
                best_fitness_15, best_execution_15, pdb_select_15, best_fitness_energ_15,best_execution_dist_15=ordena_mejores_energia(best_fitness_15, best_execution_15, pdb_select_15, best_fitness_energ_15,best_execution_dist_15)
                max_gdt_sal=max(max_gdt_sal,gdt)
            k+=1
        if generacion in utt_reinicio1 :
            w_act,w_act_dist=W_act1(i, best_execution_15, len(best_execution_15), tot_ececution, act_prop, det_fitness, fitness1,best_fitness_15,best_execution_dist_15,tot_ececution_dist)
            poblacion_15 = [[copy.deepcopy(M_prob_15) for _ in range(n)] for _ in range(n_pop)]
            poblacion_15=actualiza_individuo(n,poblacion_15,i,w_act,w_act_dist,indicador_entrada_15,indicador_entrada_pos_15,nodo_terminal_15)
        else:
            w_act,w_act_dist=W_act1(i, best_execution_15, act_glob, tot_ececution, act_prop, det_fitness, fitness1,best_fitness_15,best_execution_dist_15,tot_ececution_dist)
            poblacion_15=actualiza_individuo(n,poblacion_15,i,w_act,w_act_dist,indicador_entrada_15,indicador_entrada_pos_15,nodo_terminal_15)
    prob_pob_15=calcula_prob_tot(poblacion_15,nodos_continuan_15)       
    return max_gdt_sal,best_fitness_15, best_execution_15, pdb_select_15, best_fitness_energ_15,best_execution_dist_15,poblacion_15,prob_pob_15
    



def inicializa_isla_16(n,n_pop,amino,n_best):
    global best_execution_16
    global best_execution_dist_16
    global pdb_select_16
    global best_fitness_16
    global best_fitness_energ_16
    nodo_terminal_16=[5,12,13,14,15,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
    amino_nodo_16=['A','C','H','L','M','F','T','G','P','S','N','D','E','Q','R','K','V','I','Y','W'] 
    rango_numeros_16 = set(range(35))  
    "nodos que continuan"
    nodos_continuan_16 = list(rango_numeros_16 - set(nodo_terminal_16))
    "indica si el nodo es terminal=1 o si continua=0"
    indicador_16 = [0] * 35
    for posicion in nodo_terminal_16:
        indicador_16[posicion] = 1
    "reorganiza las posiciones de los aminoaciodos segun el; nuevo orden dado por amino_nodo" 
    letra_a_indice_16 = {letra: indice for indice, letra in enumerate(amino)}
    num_amino_new_16 = [letra_a_indice_16[letra] for letra in amino_nodo_16]
    "indica segun el numero de nodo el numero de aminoacido que corresponde"
    indicador_amino_16=[0] * 35
    for i in range(20):
        indicador_amino_16[nodo_terminal_16[i]] = num_amino_new_16[i]
    "se construye la matriz de adyasencia del arbol y la matriz de probabilidades"
    M_ady_16=[0]*21
    M_prob_16=[0]*21
    M_ady_16[0]=[1,2,3,4]
    M_prob_16[0]=[7/20,6/20,5/20,2/20]
    M_ady_16[1]=[5,6,7]
    M_prob_16[1]=[1/7,2/7,4/7]
    M_ady_16[2]=[8,9]
    M_prob_16[2]=[3/6,3/6]
    M_ady_16[3]=[10,11]
    M_prob_16[3]=[3/5,2/5]
    M_ady_16[4]=[12,13]
    M_prob_16[4]=[1/2,1/2]
    M_ady_16[6]=[14,15]
    M_prob_16[6]=[1/2,1/2]
    M_ady_16[7]=[16,17]
    M_prob_16[7]=[2/4,2/4]
    M_ady_16[8]=[18,19]
    M_prob_16[8]=[2/3,1/3]
    M_ady_16[9]=[20,21]
    M_prob_16[9]=[2/3,1/3]
    M_ady_16[10]=[22,23,24]
    M_prob_16[10]=[1/3,1/3,1/3]
    M_ady_16[11]=[25,26]
    M_prob_16[11]=[1/2,1/2]
    M_ady_16[16]=[27,28]
    M_prob_16[16]=[1/2,1/2]
    M_ady_16[17]=[29,30]
    M_prob_16[17]=[1/2,1/2]
    M_ady_16[18]=[31,32]
    M_prob_16[18]=[1/2,1/2]
    M_ady_16[20]=[33,34]
    M_prob_16[20]=[1/2,1/2]
    "indica el padre del nodo"
    indicador_entrada_16=[0]*35
    indicador_entrada_pos_16=[0]*35
    for i in range(len(M_ady_16)):
      if isinstance(M_ady_16[i], int):
        "puede ser entero"
      else:
        for j in range(len(M_ady_16[i])):
            indicador_entrada_16[M_ady_16[i][j]]=i
            indicador_entrada_pos_16[M_ady_16[i][j]]=j
    "construlle la poblacion inicial"
    poblacion_16 = [[copy.deepcopy(M_prob_16) for _ in range(n)] for _ in range(n_pop)]
    "determina las probabilidades iniciales"
    prob_pob_16=calcula_prob_tot(poblacion_16,nodos_continuan_16)
    "almacena el mejor valor de fitmess"
    best_fitness_16=[0]*n_best
    "almacena los valores de energia"
    best_fitness_energ_16=[20000]*n_best
    "almacena las n_best mejores ejecuciones"
    "best_execution = [[random.randint(0, 19) for _ in range(n)] for _ in range(n_best)]"
    best_execution_16 = [[0 for _ in range(n)] for _ in range(n_best)]
    
    "almacena las distancias de las n_best mejores ejecuciones"
    best_execution_dist_16 = [[0 for _ in range(n)] for _ in range(n_best)]
    "almacena los PDB de las n_best mejores ejecuciones"
    pdb_select_16 = ['a' for _ in range(n_best)]
    
    return nodo_terminal_16,amino_nodo_16,rango_numeros_16,nodos_continuan_16,indicador_16,num_amino_new_16,indicador_amino_16,M_ady_16,M_prob_16,indicador_entrada_16,indicador_entrada_pos_16,poblacion_16,prob_pob_16,best_fitness_16,best_fitness_energ_16,best_execution_16,best_execution_dist_16,pdb_select_16
    



def isla_16(generacion,utt_reinicio1,indicador_entrada_16,indicador_entrada_pos_16,nodo_terminal_16,act_prop,act_glob,best_fitness_16, best_execution_16, pdb_select_16, best_fitness_energ_16,best_execution_dist_16,a,b,MC_BB,corte,
           descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
           n_rep,tot_ececution,nodos_continuan_16,indicador_16,indicador_amino_16,M_ady_16,poblacion_16,prob_pob_16,M_prob_16):
    "para utilizar en las actualizaciones"
    best_execution_dist_16_1 = [[1 for _ in range(n)] for _ in range(len(best_execution_dist_16))]
    max_gdt_sal=0
    "i sera el individuo de la poblacion que se esta trabajando"
    for i in range(n_pop):
        "realiza las ejecuciones para el individuo de la poblacion"
        for kkk in range(n_rep):
            tot_ececution[i][kkk]=ejecuta_uno(prob_pob_16,i,n,M_ady_16, indicador_16, indicador_amino_16)
        "realiza el filtro correspondiente, por el momento sera las proimeras ejecuciones, no hay filtro"
        det_fitness=filtro1(tot_ececution,i,det_fitness,n_fit,n_rep)
        k=0
        for j in det_fitness[i]:
            print([i,j])
            "ejecucion j del elemento i de la poblacion a trabajar"
            posiciones_seq = tot_ececution[i][j]
            "Secuencia de aminoacidos de la ejecucion"
            secuencia = det_sec(posiciones_seq, amino)
            "PDB de la secuencia"
            if desc_si_no==1:
                descriptor_temp=descriptores(secuencia,tokenizer,model_ESM2) 
            if local==1:
                """inputs = tokenizer([secuencia], return_tensors="pt", add_special_tokens=False)
                outputs = model(**inputs)
                PDB_pdredict = [outputs.positions]"""
                pdb_str, output,ptm,plddt=genera(secuencia,model)
                PDB_pdredict = [pdb_str]
            else:
                PDB_pdredict = [_request_esmfold_prediction(secuencia)]
            "Backbone de la secuencia"
            back_temp=extract_backbone_atoms_str(PDB_pdredict[0])
            "asigna a p le estructura determinada"
            pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(p,PDB_pdredict[0]) 
            "calcula la energia"
            energia_desing=my_scorefxn(p)
            # PDB_pdredict = EMSFold_function(tokenizer,model,secuencia)
            "pdb_select[i][j] = PDB_pdredict"
            "rms y gdt de la secuencia"
            if desc_si_no==1:
                rms,gdt,MC_similitud, divKl,distancias_sal,tms=fitness_gdt_rmsd_mc_fisquim(BB, back_temp,MC_BB,corte,descriptor_ref,descriptor_temp)
                print([rms,gdt,energia_desing,MC_similitud,np.mean(divKl)])
                "se agregan los valores"
                fitness1[i][k]=agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b,tms)
                tot_ececution_dist[i][j]=[1] * len(distancias_sal)
            else:
                rms,gdt,MC_similitud,distancias_sal,tms=fitness_gdt_rmsd_mc(BB, back_temp,MC_BB,corte)
                print([rms,gdt,energia_desing,MC_similitud])
                fitness1[i][k]=agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b,tms)  
                tot_ececution_dist[i][j]=[1] * len(distancias_sal)
            
            "si la ejecucion tiene mayor fitnes que alguno en best_fitness entonces se actualiza la lista de los mejores"
            if fitness1[i][k]>best_fitness_16[pos_min]:
                best_fitness_16[pos_min]=fitness1[i][k]
                best_fitness_energ_16[pos_min]=[energia_desing,16,copy.deepcopy(generacion)]
                best_execution_16[pos_min]=list(tot_ececution[i][j])
                pdb_select_16[pos_min] = PDB_pdredict[0][:]
                best_execution_dist_16[pos_min]=distancias_sal
                best_fitness_16, best_execution_16, pdb_select_16, best_fitness_energ_16,best_execution_dist_16=ordena_mejores_energia(best_fitness_16, best_execution_16, pdb_select_16, best_fitness_energ_16,best_execution_dist_16)
                max_gdt_sal=max(max_gdt_sal,gdt)
            k+=1
        if generacion in utt_reinicio1 :
            w_act,w_act_dist=W_act1(i, best_execution_16, len(best_execution_16), tot_ececution, act_prop, det_fitness, fitness1,best_fitness_16,best_execution_dist_16_1,tot_ececution_dist)
            poblacion_16 = [[copy.deepcopy(M_prob_16) for _ in range(n)] for _ in range(n_pop)]
            poblacion_16=actualiza_individuo(n,poblacion_16,i,w_act,w_act_dist,indicador_entrada_16,indicador_entrada_pos_16,nodo_terminal_16)
        else:
            w_act,w_act_dist=W_act1(i, best_execution_16, act_glob, tot_ececution, act_prop, det_fitness, fitness1,best_fitness_16,best_execution_dist_16_1,tot_ececution_dist)
            poblacion_16=actualiza_individuo(n,poblacion_16,i,w_act,w_act_dist,indicador_entrada_16,indicador_entrada_pos_16,nodo_terminal_16)
    prob_pob_16=calcula_prob_tot(poblacion_16,nodos_continuan_16)       
    return max_gdt_sal,best_fitness_16, best_execution_16, pdb_select_16, best_fitness_energ_16,best_execution_dist_16,poblacion_16,prob_pob_16
    


def inicializa_isla_17(n,n_pop,amino,n_best):
    global best_execution_17
    global best_execution_dist_17
    global pdb_select_17
    global best_fitness_17
    global best_fitness_energ_17
    
    "se construye la matriz de adyasencia del arbol y la matriz de probabilidades"
    M_ady_17=[[1]*20]*20
    M_prob_17=[[1/20]*20]*20
    
    "construlle la poblacion inicial"
    poblacion_17 = [[[[1 for _ in range(20)] for _ in range(20)]for _ in range(n)] for _ in range(n_pop)]
    "determina las probabilidades iniciales"
    prob_pob_17=[[[[1/20 for _ in range(20)] for _ in range(20)]for _ in range(n)] for _ in range(n_pop)]
    "almacena el mejor valor de fitmess"
    best_fitness_17=[0]*n_best
    "almacena los valores de energia"
    best_fitness_energ_17=[20000]*n_best
    "almacena las n_best mejores ejecuciones"
    "best_execution = [[random.randint(0, 19) for _ in range(n)] for _ in range(n_best)]"
    best_execution_17 = [[0 for _ in range(n)] for _ in range(n_best)]
    
    "almacena las distancias de las n_best mejores ejecuciones"
    best_execution_dist_17 = [[0 for _ in range(n)] for _ in range(n_best)]
    "almacena los PDB de las n_best mejores ejecuciones"
    pdb_select_17 = ['a' for _ in range(n_best)]
    
    return M_ady_17,M_prob_17,poblacion_17,prob_pob_17,best_fitness_17,best_fitness_energ_17,best_execution_17,best_execution_dist_17,pdb_select_17
    



def isla_17(utt_reinicio1,best_fitness_17, best_execution_17, pdb_select_17, best_fitness_energ_17,best_execution_dist_17,a,b,MC_BB,corte,
           descriptor_ref,my_scorefxn,p,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
           n_rep,tot_ececution,prob_pob_17,det_fitness,generacion):
    max_gdt_sal=0
    "i sera el individuo de la poblacion que se esta trabajando"
    for i in range(n_pop):
        "realiza las ejecuciones para el individuo de la poblacion"
        for kkk in range(n_rep):
            tot_ececution[i][kkk]=ejecuta_uno_red(prob_pob_17,i,n)
        "realiza el filtro correspondiente, por el momento sera las proimeras ejecuciones, no hay filtro"
        det_fitness=filtro1(tot_ececution,i,det_fitness,n_fit,n_rep)
        k=0
        for j in det_fitness[i]:
            print([i,j])
            "ejecucion j del elemento i de la poblacion a trabajar"
            posiciones_seq = tot_ececution[i][j]
            "Secuencia de aminoacidos de la ejecucion"
            secuencia = det_sec(posiciones_seq, amino)
            "PDB de la secuencia"
            if desc_si_no==1:
                descriptor_temp=descriptores(secuencia,tokenizer,model_ESM2) 
            if local==1:
                """inputs = tokenizer([secuencia], return_tensors="pt", add_special_tokens=False)
                outputs = model(**inputs)
                PDB_pdredict = [outputs.positions]"""
                pdb_str, output,ptm,plddt=genera(secuencia,model)
                PDB_pdredict = [pdb_str]
            else:
                PDB_pdredict = [_request_esmfold_prediction(secuencia)]
            "Backbone de la secuencia"
            back_temp=extract_backbone_atoms_str(PDB_pdredict[0])
            "asigna a p le estructura determinada"
            pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(p,PDB_pdredict[0]) 
            "calcula la energia"
            energia_desing=my_scorefxn(p)
            # PDB_pdredict = EMSFold_function(tokenizer,model,secuencia)
            "pdb_select[i][j] = PDB_pdredict"
            "rms y gdt de la secuencia"
            if desc_si_no==1:
                rms,gdt,MC_similitud, divKl,distancias_sal,tms=fitness_gdt_rmsd_mc_fisquim(BB, back_temp,MC_BB,corte,descriptor_ref,descriptor_temp)
                print([rms,gdt,energia_desing,MC_similitud,np.mean(divKl)])
                "se agregan los valores"
                fitness1[i][k]=agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b,tms)
                tot_ececution_dist[i][j]=[1] * len(distancias_sal)
            else:
                rms,gdt,MC_similitud,distancias_sal,tms=fitness_gdt_rmsd_mc(BB, back_temp,MC_BB,corte)
                print([rms,gdt,energia_desing,MC_similitud])
                fitness1[i][k]=agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b,tms)  
                tot_ececution_dist[i][j]=[1] * len(distancias_sal)
            
            "si la ejecucion tiene mayor fitnes que alguno en best_fitness entonces se actualiza la lista de los mejores"
            if fitness1[i][k]>best_fitness_17[pos_min]:
                best_fitness_17[pos_min]=fitness1[i][k]
                best_fitness_energ_17[pos_min]=[energia_desing,17,copy.deepcopy(generacion)]
                best_execution_17[pos_min]=list(tot_ececution[i][j])
                pdb_select_17[pos_min] = PDB_pdredict[0][:]
                best_execution_dist_17[pos_min]=distancias_sal
                best_fitness_17, best_execution_17, pdb_select_17, best_fitness_energ_17,best_execution_dist_17=ordena_mejores_energia(best_fitness_17, best_execution_17, pdb_select_17, best_fitness_energ_17,best_execution_dist_17)
                max_gdt_sal=max(max_gdt_sal,gdt)
            k+=1
    #     if generacion in utt_reinicio1 :
    #         w_act,w_act_dist=W_act1(i, best_execution_17, len(best_execution_17), tot_ececution, act_prop, det_fitness, fitness1,best_fitness_17,best_execution_dist_17_1,tot_ececution_dist)
    #         poblacion_17 = [[copy.deepcopy(M_prob_17) for _ in range(n)] for _ in range(n_pop)]
    #         poblacion_17=actualiza_individuo(n,poblacion_17,i,w_act,w_act_dist,indicador_entrada_17,indicador_entrada_pos_17,nodo_terminal_17)
    #     else:
    #         w_act,w_act_dist=W_act1(i, best_execution_17, act_glob, tot_ececution, act_prop, det_fitness, fitness1,best_fitness_17,best_execution_dist_17_1,tot_ececution_dist)
    #         poblacion_17=actualiza_individuo(n,poblacion_17,i,w_act,w_act_dist,indicador_entrada_17,indicador_entrada_pos_17,nodo_terminal_17)
    # prob_pob_17=calcula_prob_tot(poblacion_17,nodos_continuan_17)       
    return max_gdt_sal,best_fitness_17, best_execution_17, pdb_select_17, best_fitness_energ_17,best_execution_dist_17
    


def inicializa_isla_18(n,n_pop,amino,n_best):
    global best_execution_18
    global best_execution_dist_18
    global pdb_select_18
    global best_fitness_18
    global best_fitness_energ_18
    
    "se construye la matriz de adyasencia del arbol y la matriz de probabilidades"
    M_ady_18=[[1]*20]*20
    M_prob_18=[[1/20]*20]*20
    
    "construlle la poblacion inicial"
    poblacion_18 = [[[[1 for _ in range(20)] for _ in range(20)]for _ in range(n)] for _ in range(n_pop)]
    "determina las probabilidades iniciales"
    prob_pob_18=[[[[1/20 for _ in range(20)] for _ in range(20)]for _ in range(n)] for _ in range(n_pop)]
    "almacena el mejor valor de fitmess"
    best_fitness_18=[0]*n_best
    "almacena los valores de energia"
    best_fitness_energ_18=[20000]*n_best
    "almacena las n_best mejores ejecuciones"
    "best_execution = [[random.randint(0, 19) for _ in range(n)] for _ in range(n_best)]"
    best_execution_18 = [[0 for _ in range(n)] for _ in range(n_best)]
    
    "almacena las distancias de las n_best mejores ejecuciones"
    best_execution_dist_18 = [[0 for _ in range(n)] for _ in range(n_best)]
    "almacena los PDB de las n_best mejores ejecuciones"
    pdb_select_18 = ['a' for _ in range(n_best)]
    
    return M_ady_18,M_prob_18,poblacion_18,prob_pob_18,best_fitness_18,best_fitness_energ_18,best_execution_18,best_execution_dist_18,pdb_select_18
    



def isla_18(utt_reinicio1,best_fitness_18, best_execution_18, pdb_select_18, best_fitness_energ_18,best_execution_dist_18,a,b,MC_BB,corte,
           descriptor_ref,my_scorefxn,p,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
           n_rep,tot_ececution,prob_pob_18,det_fitness,generacion):
    max_gdt_sal=0
    "i sera el individuo de la poblacion que se esta trabajando"
    for i in range(n_pop):
        "realiza las ejecuciones para el individuo de la poblacion"
        for kkk in range(n_rep):
            tot_ececution[i][kkk]=ejecuta_uno_red(prob_pob_18,i,n)
        "realiza el filtro correspondiente, por el momento sera las proimeras ejecuciones, no hay filtro"
        det_fitness=filtro1(tot_ececution,i,det_fitness,n_fit,n_rep)
        k=0
        for j in det_fitness[i]:
            print([i,j])
            "ejecucion j del elemento i de la poblacion a trabajar"
            posiciones_seq = tot_ececution[i][j]
            "Secuencia de aminoacidos de la ejecucion"
            secuencia = det_sec(posiciones_seq, amino)
            "PDB de la secuencia"
            if desc_si_no==1:
                descriptor_temp=descriptores(secuencia,tokenizer,model_ESM2) 
            if local==1:
                """inputs = tokenizer([secuencia], return_tensors="pt", add_special_tokens=False)
                outputs = model(**inputs)
                PDB_pdredict = [outputs.positions]"""
                pdb_str, output,ptm,plddt=genera(secuencia,model)
                PDB_pdredict = [pdb_str]
            else:
                PDB_pdredict = [_request_esmfold_prediction(secuencia)]
            "Backbone de la secuencia"
            back_temp=extract_backbone_atoms_str(PDB_pdredict[0])
            "asigna a p le estructura determinada"
            pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(p,PDB_pdredict[0]) 
            "calcula la energia"
            energia_desing=my_scorefxn(p)
            # PDB_pdredict = EMSFold_function(tokenizer,model,secuencia)
            "pdb_select[i][j] = PDB_pdredict"
            "rms y gdt de la secuencia"
            if desc_si_no==1:
                rms,gdt,MC_similitud, divKl,distancias_sal,tms=fitness_gdt_rmsd_mc_fisquim(BB, back_temp,MC_BB,corte,descriptor_ref,descriptor_temp)
                print([rms,gdt,energia_desing,MC_similitud,np.mean(divKl)])
                "se agregan los valores"
                fitness1[i][k]=agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b,tms)
                tot_ececution_dist[i][j]=[1] * len(distancias_sal)
            else:
                rms,gdt,MC_similitud,distancias_sal,tms=fitness_gdt_rmsd_mc(BB, back_temp,MC_BB,corte)
                print([rms,gdt,energia_desing,MC_similitud])
                fitness1[i][k]=agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b,tms)  
                tot_ececution_dist[i][j]=[1] * len(distancias_sal)
            
            "si la ejecucion tiene mayor fitnes que alguno en best_fitness entonces se actualiza la lista de los mejores"
            if fitness1[i][k]>best_fitness_18[pos_min]:
                best_fitness_18[pos_min]=fitness1[i][k]
                best_fitness_energ_18[pos_min]=[energia_desing,18,copy.deepcopy(generacion)]
                best_execution_18[pos_min]=list(tot_ececution[i][j])
                pdb_select_18[pos_min] = PDB_pdredict[0][:]
                best_execution_dist_18[pos_min]=distancias_sal
                best_fitness_18, best_execution_18, pdb_select_18, best_fitness_energ_18,best_execution_dist_18=ordena_mejores_energia(best_fitness_18, best_execution_18, pdb_select_18, best_fitness_energ_18,best_execution_dist_18)
                max_gdt_sal=max(max_gdt_sal,gdt)
            k+=1
    #     if generacion in utt_reinicio1 :
    #         w_act,w_act_dist=W_act1(i, best_execution_18, len(best_execution_18), tot_ececution, act_prop, det_fitness, fitness1,best_fitness_18,best_execution_dist_18_1,tot_ececution_dist)
    #         poblacion_18 = [[copy.deepcopy(M_prob_18) for _ in range(n)] for _ in range(n_pop)]
    #         poblacion_18=actualiza_individuo(n,poblacion_18,i,w_act,w_act_dist,indicador_entrada_18,indicador_entrada_pos_18,nodo_terminal_18)
    #     else:
    #         w_act,w_act_dist=W_act1(i, best_execution_18, act_glob, tot_ececution, act_prop, det_fitness, fitness1,best_fitness_18,best_execution_dist_18_1,tot_ececution_dist)
    #         poblacion_18=actualiza_individuo(n,poblacion_18,i,w_act,w_act_dist,indicador_entrada_18,indicador_entrada_pos_18,nodo_terminal_18)
    # prob_pob_18=calcula_prob_tot(poblacion_18,nodos_continuan_18)       
    return max_gdt_sal,best_fitness_18, best_execution_18, pdb_select_18, best_fitness_energ_18,best_execution_dist_18
    

def inicializa_isla_19(n,n_pop,amino,n_best):
    global best_execution_19
    global best_execution_dist_19
    global pdb_select_19
    global best_fitness_19
    global best_fitness_energ_19
    
    "se construye la matriz de adyasencia del arbol y la matriz de probabilidades"
    M_ady_19=[[1]*20]*20
    M_prob_19=[[1/20]*20]*20
    
    "construlle la poblacion inicial"
    poblacion_19 = [[[[1 for _ in range(20)] for _ in range(20)]for _ in range(n)] for _ in range(n_pop)]
    "determina las probabilidades iniciales"
    prob_pob_19=[[[[1/20 for _ in range(20)] for _ in range(20)]for _ in range(n)] for _ in range(n_pop)]
    "almacena el mejor valor de fitmess"
    best_fitness_19=[0]*n_best
    "almacena los valores de energia"
    best_fitness_energ_19=[20000]*n_best
    "almacena las n_best mejores ejecuciones"
    "best_execution = [[random.randint(0, 19) for _ in range(n)] for _ in range(n_best)]"
    best_execution_19 = [[0 for _ in range(n)] for _ in range(n_best)]
    
    "almacena las distancias de las n_best mejores ejecuciones"
    best_execution_dist_19 = [[0 for _ in range(n)] for _ in range(n_best)]
    "almacena los PDB de las n_best mejores ejecuciones"
    pdb_select_19 = ['a' for _ in range(n_best)]
    
    return M_ady_19,M_prob_19,poblacion_19,prob_pob_19,best_fitness_19,best_fitness_energ_19,best_execution_19,best_execution_dist_19,pdb_select_19
    



def isla_19(utt_reinicio1,best_fitness_19, best_execution_19, pdb_select_19, best_fitness_energ_19,best_execution_dist_19,a,b,MC_BB,corte,
           descriptor_ref,my_scorefxn,p,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
           n_rep,tot_ececution,prob_pob_19,det_fitness,generacion):
    max_gdt_sal=0
    "i sera el individuo de la poblacion que se esta trabajando"
    for i in range(n_pop):
        "realiza las ejecuciones para el individuo de la poblacion"
        for kkk in range(n_rep):
            tot_ececution[i][kkk]=ejecuta_uno_red(prob_pob_19,i,n)
        "realiza el filtro correspondiente, por el momento sera las proimeras ejecuciones, no hay filtro"
        det_fitness=filtro1(tot_ececution,i,det_fitness,n_fit,n_rep)
        k=0
        for j in det_fitness[i]:
            print([i,j])
            "ejecucion j del elemento i de la poblacion a trabajar"
            posiciones_seq = tot_ececution[i][j]
            "Secuencia de aminoacidos de la ejecucion"
            secuencia = det_sec(posiciones_seq, amino)
            "PDB de la secuencia"
            if desc_si_no==1:
                descriptor_temp=descriptores(secuencia,tokenizer,model_ESM2) 
            if local==1:
                """inputs = tokenizer([secuencia], return_tensors="pt", add_special_tokens=False)
                outputs = model(**inputs)
                PDB_pdredict = [outputs.positions]"""
                pdb_str, output,ptm,plddt=genera(secuencia,model)
                PDB_pdredict = [pdb_str]
            else:
                PDB_pdredict = [_request_esmfold_prediction(secuencia)]
            "Backbone de la secuencia"
            back_temp=extract_backbone_atoms_str(PDB_pdredict[0])
            "asigna a p le estructura determinada"
            pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(p,PDB_pdredict[0]) 
            "calcula la energia"
            energia_desing=my_scorefxn(p)
            # PDB_pdredict = EMSFold_function(tokenizer,model,secuencia)
            "pdb_select[i][j] = PDB_pdredict"
            "rms y gdt de la secuencia"
            if desc_si_no==1:
                rms,gdt,MC_similitud, divKl,distancias_sal,tms=fitness_gdt_rmsd_mc_fisquim(BB, back_temp,MC_BB,corte,descriptor_ref,descriptor_temp)
                print([rms,gdt,energia_desing,MC_similitud,np.mean(divKl)])
                "se agregan los valores"
                fitness1[i][k]=agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b,tms)
                tot_ececution_dist[i][j]=[1] * len(distancias_sal)
            else:
                rms,gdt,MC_similitud,distancias_sal,tms=fitness_gdt_rmsd_mc(BB, back_temp,MC_BB,corte)
                print([rms,gdt,energia_desing,MC_similitud])
                fitness1[i][k]=agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b,tms)  
                tot_ececution_dist[i][j]=[1] * len(distancias_sal)
            
            "si la ejecucion tiene mayor fitnes que alguno en best_fitness entonces se actualiza la lista de los mejores"
            if fitness1[i][k]>best_fitness_19[pos_min]:
                best_fitness_19[pos_min]=fitness1[i][k]
                best_fitness_energ_19[pos_min]=[energia_desing,19,copy.deepcopy(generacion)]
                best_execution_19[pos_min]=list(tot_ececution[i][j])
                pdb_select_19[pos_min] = PDB_pdredict[0][:]
                best_execution_dist_19[pos_min]=distancias_sal
                best_fitness_19, best_execution_19, pdb_select_19, best_fitness_energ_19,best_execution_dist_19=ordena_mejores_energia(best_fitness_19, best_execution_19, pdb_select_19, best_fitness_energ_19,best_execution_dist_19)
                max_gdt_sal=max(max_gdt_sal,gdt)
            k+=1
    #     if generacion in utt_reinicio1 :
    #         w_act,w_act_dist=W_act1(i, best_execution_19, len(best_execution_19), tot_ececution, act_prop, det_fitness, fitness1,best_fitness_19,best_execution_dist_19_1,tot_ececution_dist)
    #         poblacion_19 = [[copy.deepcopy(M_prob_19) for _ in range(n)] for _ in range(n_pop)]
    #         poblacion_19=actualiza_individuo(n,poblacion_19,i,w_act,w_act_dist,indicador_entrada_19,indicador_entrada_pos_19,nodo_terminal_19)
    #     else:
    #         w_act,w_act_dist=W_act1(i, best_execution_19, act_glob, tot_ececution, act_prop, det_fitness, fitness1,best_fitness_19,best_execution_dist_19_1,tot_ececution_dist)
    #         poblacion_19=actualiza_individuo(n,poblacion_19,i,w_act,w_act_dist,indicador_entrada_19,indicador_entrada_pos_19,nodo_terminal_19)
    # prob_pob_19=calcula_prob_tot(poblacion_19,nodos_continuan_19)       
    return max_gdt_sal,best_fitness_19, best_execution_19, pdb_select_19, best_fitness_energ_19,best_execution_dist_19
    

def inicializa_isla_20(n,n_pop,amino,n_best):
    global best_execution_20
    global best_execution_dist_20
    global pdb_select_20
    global best_fitness_20
    global best_fitness_energ_20
    
    "se construye la matriz de adyasencia del arbol y la matriz de probabilidades"
    M_ady_20=[[1]*20]*20
    M_prob_20=[[1/20]*20]*20
    
    "construlle la poblacion inicial"
    poblacion_20 = [[[[1 for _ in range(20)] for _ in range(20)]for _ in range(n)] for _ in range(n_pop)]
    "determina las probabilidades iniciales"
    prob_pob_20=[[[[1/20 for _ in range(20)] for _ in range(20)]for _ in range(n)] for _ in range(n_pop)]
    "almacena el mejor valor de fitmess"
    best_fitness_20=[0]*n_best
    "almacena los valores de energia"
    best_fitness_energ_20=[20000]*n_best
    "almacena las n_best mejores ejecuciones"
    "best_execution = [[random.randint(0, 19) for _ in range(n)] for _ in range(n_best)]"
    best_execution_20 = [[0 for _ in range(n)] for _ in range(n_best)]
    
    "almacena las distancias de las n_best mejores ejecuciones"
    best_execution_dist_20 = [[0 for _ in range(n)] for _ in range(n_best)]
    "almacena los PDB de las n_best mejores ejecuciones"
    pdb_select_20 = ['a' for _ in range(n_best)]
    
    return M_ady_20,M_prob_20,poblacion_20,prob_pob_20,best_fitness_20,best_fitness_energ_20,best_execution_20,best_execution_dist_20,pdb_select_20
    



def isla_20(utt_reinicio1,best_fitness_20, best_execution_20, pdb_select_20, best_fitness_energ_20,best_execution_dist_20,a,b,MC_BB,corte,
           descriptor_ref,my_scorefxn,p,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
           n_rep,tot_ececution,prob_pob_20,det_fitness,generacion):
    max_gdt_sal=0
    "i sera el individuo de la poblacion que se esta trabajando"
    for i in range(n_pop):
        "realiza las ejecuciones para el individuo de la poblacion"
        for kkk in range(n_rep):
            tot_ececution[i][kkk]=ejecuta_uno_red(prob_pob_20,i,n)
        "realiza el filtro correspondiente, por el momento sera las proimeras ejecuciones, no hay filtro"
        det_fitness=filtro1(tot_ececution,i,det_fitness,n_fit,n_rep)
        k=0
        for j in det_fitness[i]:
            print([i,j])
            "ejecucion j del elemento i de la poblacion a trabajar"
            posiciones_seq = tot_ececution[i][j]
            "Secuencia de aminoacidos de la ejecucion"
            secuencia = det_sec(posiciones_seq, amino)
            "PDB de la secuencia"
            if desc_si_no==1:
                descriptor_temp=descriptores(secuencia,tokenizer,model_ESM2) 
            if local==1:
                """inputs = tokenizer([secuencia], return_tensors="pt", add_special_tokens=False)
                outputs = model(**inputs)
                PDB_pdredict = [outputs.positions]"""
                pdb_str, output,ptm,plddt=genera(secuencia,model)
                PDB_pdredict = [pdb_str]
            else:
                PDB_pdredict = [_request_esmfold_prediction(secuencia)]
            "Backbone de la secuencia"
            back_temp=extract_backbone_atoms_str(PDB_pdredict[0])
            "asigna a p le estructura determinada"
            pyrosetta.rosetta.core.import_pose.pose_from_pdbstring(p,PDB_pdredict[0]) 
            "calcula la energia"
            energia_desing=my_scorefxn(p)
            # PDB_pdredict = EMSFold_function(tokenizer,model,secuencia)
            "pdb_select[i][j] = PDB_pdredict"
            "rms y gdt de la secuencia"
            if desc_si_no==1:
                rms,gdt,MC_similitud, divKl,distancias_sal,tms=fitness_gdt_rmsd_mc_fisquim(BB, back_temp,MC_BB,corte,descriptor_ref,descriptor_temp)
                print([rms,gdt,energia_desing,MC_similitud,np.mean(divKl)])
                "se agregan los valores"
                fitness1[i][k]=agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b,tms)
                tot_ececution_dist[i][j]=[1] * len(distancias_sal)
            else:
                rms,gdt,MC_similitud,distancias_sal,tms=fitness_gdt_rmsd_mc(BB, back_temp,MC_BB,corte)
                print([rms,gdt,energia_desing,MC_similitud])
                fitness1[i][k]=agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b,tms)  
                tot_ececution_dist[i][j]=[1] * len(distancias_sal)
            
            "si la ejecucion tiene mayor fitnes que alguno en best_fitness entonces se actualiza la lista de los mejores"
            if fitness1[i][k]>best_fitness_20[pos_min]:
                best_fitness_20[pos_min]=fitness1[i][k]
                best_fitness_energ_20[pos_min]=[energia_desing,20,copy.deepcopy(generacion)]
                best_execution_20[pos_min]=list(tot_ececution[i][j])
                pdb_select_20[pos_min] = PDB_pdredict[0][:]
                best_execution_dist_20[pos_min]=distancias_sal
                best_fitness_20, best_execution_20, pdb_select_20, best_fitness_energ_20,best_execution_dist_20=ordena_mejores_energia(best_fitness_20, best_execution_20, pdb_select_20, best_fitness_energ_20,best_execution_dist_20)
                max_gdt_sal=max(max_gdt_sal,gdt)
            k+=1
    #     if generacion in utt_reinicio1 :
    #         w_act,w_act_dist=W_act1(i, best_execution_20, len(best_execution_20), tot_ececution, act_prop, det_fitness, fitness1,best_fitness_20,best_execution_dist_20_1,tot_ececution_dist)
    #         poblacion_20 = [[copy.deepcopy(M_prob_20) for _ in range(n)] for _ in range(n_pop)]
    #         poblacion_20=actualiza_individuo(n,poblacion_20,i,w_act,w_act_dist,indicador_entrada_20,indicador_entrada_pos_20,nodo_terminal_20)
    #     else:
    #         w_act,w_act_dist=W_act1(i, best_execution_20, act_glob, tot_ececution, act_prop, det_fitness, fitness1,best_fitness_20,best_execution_dist_20_1,tot_ececution_dist)
    #         poblacion_20=actualiza_individuo(n,poblacion_20,i,w_act,w_act_dist,indicador_entrada_20,indicador_entrada_pos_20,nodo_terminal_20)
    # prob_pob_20=calcula_prob_tot(poblacion_20,nodos_continuan_20)       
    return max_gdt_sal,best_fitness_20, best_execution_20, pdb_select_20, best_fitness_energ_20,best_execution_dist_20
    