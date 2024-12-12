from ..pdb_seq_tools.pdb_seq_tools import extract_amino_acid_sequence_pdb,extract_backbone_atoms,det_sec,extract_backbone_atoms_str,extract_amino_acid_sequence
from ..Fitness.fitness import descriptores,mapa_contacto,fitness_gdt_rmsd_mc_fisquim,fitness_gdt_rmsd_mc,agrega_rmsd_gdt_E_MC_divKl,agrega_rmsd_gdt_E_MC,ESM2_desc
from ..EDA_tools.EDAtools import calcula_prob_tot, ejecuta_uno, filtro1,ordena_mejores_energia,W_act1,actualiza_individuo,actualiza_individuo_red,ordena_mejores_all,calcula_prob_red,calcula_prob_red_M
from ..getDstructure.get_structure import _request_esmfold_prediction,parse_output,get_hash,genera,esmfold_predict_structure
from ..islas.islas import inicializa_isla_1,isla_1,inicializa_isla_2,isla_2,inicializa_isla_3,isla_3,inicializa_isla_4,isla_4,inicializa_isla_5,isla_5
from ..islas.islas import inicializa_isla_6,isla_6,inicializa_isla_7,isla_7,inicializa_isla_8,isla_8,inicializa_isla_9,isla_9,inicializa_isla_10,isla_10
from ..islas.islas import inicializa_isla_11,isla_11,inicializa_isla_12,isla_12,inicializa_isla_13,isla_13,inicializa_isla_14,isla_14,inicializa_isla_15
from ..islas.islas import isla_15,inicializa_isla_16,isla_16,inicializa_isla_17,isla_17,inicializa_isla_18,isla_18
from ..islas.islas import isla_19,inicializa_isla_19,isla_20,inicializa_isla_20
import os
import copy
import pyrosetta
import numpy as np
from transformers import EsmTokenizer, EsmModel
import torch
import random

"""import torch
from google.colab import drive"""
"from transformers import AutoTokenizer, EsmForProteinFolding"
"import torch"

"se escribe la funcion asociada al algoritmo evolutivo"
def EDA_tres_capas(pdb_file_path,desc_si_no,continua,numnum,n_pop,n_rep,n_fit,act_prop,act_glob,n_best,pos_min,n_generaciones,n_reinicia,n_genra_reinicio,corte,amino,a,b,local,nueva_ruta):
    "cada aminoacido en amino se hace coincidir con su correspondiente numero en"
    num_amino = list(range(0, 20))
    "secuencia del peptido, si procede"
    if desc_si_no==1:
        tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        model_ESM2 = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        secuencia_ref=extract_amino_acid_sequence_pdb(pdb_file_path)
        descriptor_ref=descriptores(secuencia_ref,tokenizer,model_ESM2) 
    "Extraer el backbone del primer modelo y de la cadena principal"
    BB=extract_backbone_atoms(pdb_file_path)
    "n longitud de la secuencia"
    n=int(len(BB)/4)
    "nombre de la proteina sin .pdb"
    nombre_sin_extension = os.path.splitext(pdb_file_path)[0]
    "mapa de contactos del BB"
    MC_BB=mapa_contacto(BB)
    "Generacion"
    generacion=-1
    "matriz para guardar las actualizaciones de las distribuciones y son 4 reprecentaciones"

    
    """""""""""""inicializacion del algoritmo"
    "nodos terminales"
    nodo_terminal=[5,10,12,13,14,15,16,19,20,22,24,25,26,27,28,29,30,31,32,33]
    amino_nodo=['F','C','I','M','K','R','H','Y','W','P','V','L','D','E','N','Q','S','T','A','G']
    rango_numeros = set(range(34))  
    "nodos que continuan"
    nodos_continuan = list(rango_numeros - set(nodo_terminal))
    "indica si el nodo es terminal=1 o si continua=0"
    indicador = [0] * 34
    for posicion in nodo_terminal:
        indicador[posicion] = 1
    "reorganiza las posiciones de los aminoaciodos segun el; nuevo orden dado por amino_nodo" 
    letra_a_indice = {letra: indice for indice, letra in enumerate(amino)}
    num_amino_new = [letra_a_indice[letra] for letra in amino_nodo]
    "indica segun el numero de nodo el numero de aminoacido que corresponde"
    indicador_amino=[0] * 34
    for i in range(20):
        indicador_amino[nodo_terminal[i]] = num_amino_new[i]
    
    "se construye la matriz de adyasencia del arbol y la matriz de probabilidades"
    M_ady=[0]*24
    M_prob=[0]*24
    M_ady[0]=[1,2,3]
    M_prob[0]=[5/20,7/20,8/20]
    M_ady[1]=[4,5]
    M_prob[1]=[4/5,1/5]
    M_ady[2]=[6,7]
    M_prob[2]=[3/7,4/7]
    M_ady[3]=[8,9,10]
    M_prob[3]=[2/8,5/8,1/8]
    M_ady[4]=[11,12,13]
    M_prob[4]=[2/4,1/4,1/4]
    M_ady[6]=[14,15,16]
    M_prob[6]=[1/3,1/3,1/3]
    M_ady[7]=[17,18]
    M_prob[7]=[1/2,1/2]
    M_ady[8]=[19,20]
    M_prob[8]=[1/2,1/2]
    M_ady[9]=[21,22,23]
    M_prob[9]=[2/5,1/5,2/5]
    M_ady[11]=[24,25]
    M_prob[11]=[1/2,1/2]
    M_ady[17]=[26,27]
    M_prob[17]=[1/2,1/2]
    M_ady[18]=[28,29]
    M_prob[18]=[1/2,1/2]
    M_ady[21]=[30,31]
    M_prob[21]=[1/2,1/2]
    M_ady[23]=[32,33]
    M_prob[23]=[1/2,1/2]
    
    "indica el padre del nodo"
    indicador_entrada=[0]*34
    indicador_entrada_pos=[0]*34
    for i in range(len(M_ady)):
      if isinstance(M_ady[i], int):
        "puede ser entero"
      else:
        for j in range(len(M_ady[i])):
            indicador_entrada[M_ady[i][j]]=i
            indicador_entrada_pos[M_ady[i][j]]=j
    "construlle la poblacion inicial"
    poblacion = [[copy.deepcopy(M_prob) for _ in range(n)] for _ in range(n_pop)]
    "determina las probabilidades iniciales"
    prob_pob=calcula_prob_tot(poblacion,nodos_continuan)
    
    "almacena lo que se va a utilizar aleatorio cada cierta cantidad de ejecuciones"
    poblacion_aleatoria= copy.deepcopy(poblacion[0:n_reinicia])
    prob_pob_aleatoria= copy.deepcopy(prob_pob[0:n_reinicia])
    

    "almacena el mejor valor de fitmess"
    best_fitness=[0]*n_best
    "almacena los valores de energia"
    best_fitness_energ=[20000]*n_best
    "almacena las n_best mejores ejecuciones"
    "best_execution = [[random.randint(0, 19) for _ in range(n)] for _ in range(n_best)]"
    best_execution = [[0 for _ in range(n)] for _ in range(n_best)]
    
    "almacena las distancias de las n_best mejores ejecuciones"
    best_execution_dist = [[0 for _ in range(n)] for _ in range(n_best)]
    "almacena los PDB de las n_best mejores ejecuciones"
    pdb_select = ['a' for _ in range(n_best)]
    
    
    "crea las primeras ejecuciones"
    tot_ececution=[[[0 for _ in range(n)] for _ in range(n_rep)] for _ in range(n_pop)]
    
    "almacena las distancia de las ejecuciones a las cuales se le calculo el fitness"
    tot_ececution_dist=[[[0 for _ in range(n)] for _ in range(n_rep)] for _ in range(n_pop)]
    
    "for i in range(n_pop*n_rep):"
    "tot_ececution[i]=ejecuta_uno(prob_pob,0,n,M_ady, indicador, indicador_amino)"
    "almacena las n_fit posiciones segun el filtro aplicado"
    det_fitness=[[None]*n_fit]*n_pop
    "almacena las n_fit fitness de las n_fit posiciones segun el filtro aplicado"
    fitness1=[[None]*n_fit]*n_pop
    "indica cuando hay que hacer copias"
    uutt=list(range(0, n_generaciones, 10))
    uutt_reinicio=list(range(0, n_generaciones, n_genra_reinicio))
    utt_reinicio1=[x + 1 for x in uutt_reinicio]
    reiniciados=list(range(0,n_reinicia))
    "funcion de energia"
    my_scorefxn = pyrosetta.get_score_function(True)
    p = pyrosetta.Pose() 
    energia_desing=0
    version=1
    model_name = "esmfold_v0.model" if version == "0" else "esmfold.model"
    "modelos para ESMFold utilizando transformer"
    if local==1:
        "model = EsmForProteinFolding.from_pretrained('facebook/esmfold_v1')"
        "tokenizer = AutoTokenizer.from_pretrained('facebook/esmfold_v1')"
        if "model" not in dir() or model_name != model_name_:
            if "model" in dir():
              # delete old model from memory
              del model
              gc.collect()
              if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
            model = torch.load("esmfold.model")
            model.eval().cuda().requires_grad_(False)
            model_name_ = "esmfold.model"
        # optimized for Tesla T4
        model.set_chunk_size(128)

    
    
    "continuar aqui este parra el final"

    if continua==1:
        "ruta donde estan los datos almacenados para continuar con la ejecucion"
        ruta_archivo=os.getcwd()+'/datos_guardados'+str(numnum)+'.txt'
        with open(ruta_archivo, "r") as archivo:
            # Leer cada línea del archivo
            for linea in archivo:
                # Dividir la línea en etiqueta y valor
                etiqueta, valor = linea.strip().split(": ")
                # Evaluar y almacenar el valor correspondiente
                if etiqueta == "best_execution":
                    best_execution = eval(valor)
                elif etiqueta == "best_execution_dist":
                    best_execution_dist = eval(valor)            
                elif etiqueta == "best_fitness":
                    best_fitness = eval(valor)
                elif etiqueta == "poblacion":
                    poblacion1 = eval(valor)
                elif etiqueta == "best_fitness_energ":
                    best_fitness_energ= eval(valor)
                elif etiqueta == "pdb_select":
                    pdb_select= eval(valor)
        generacion=numnum
        for i in range(len(poblacion)):
            for j in range(len(poblacion[i])):
                poblacion[i][j]=poblacion1[i][j]
        for i in range(len(best_execution)):
            back_temp=extract_backbone_atoms_str(pdb_select[i])
            secuencia=extract_amino_acid_sequence(pdb_select[i])
            if desc_si_no==1:
                descriptor_temp=descriptores(secuencia,tokenizer,model_ESM2) 
                rms,gdt,MC_similitud, divKl,distancias_sal=fitness_gdt_rmsd_mc_fisquim(BB, back_temp,MC_BB,corte,descriptor_ref,descriptor_temp)
                print([rms,gdt,energia_desing,MC_similitud,np.mean(divKl)])
                "se agregan los valores"
                best_fitness[i]=agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b)
            else:
                rms,gdt,MC_similitud,distancias_sal=fitness_gdt_rmsd_mc(BB, back_temp,MC_BB,corte)
                print([rms,gdt,energia_desing,MC_similitud])
                best_fitness[i]=agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b)            
            best_fitness, best_execution, pdb_select, best_fitness_energ,best_execution_dist=ordena_mejores_energia(best_fitness, best_execution, pdb_select, best_fitness_energ,best_execution_dist)
            
    
    "Algoritmo"
    while generacion<n_generaciones:
        """drive.mount('/content/drive')
        os.chdir(nueva_ruta)"""
        "Determina la cantidad por la que se deven dividir la cantidad almacenada "
        "en cada indice de cada elemento de la poblacion"
        generacion=generacion+1
        print(generacion)
        "i sera el individuo de la poblacion que se esta trabajando"
        for i in range(n_pop):
            "realiza las ejecuciones para el individuo de la poblacion"
            for kkk in range(n_rep):
                tot_ececution[i][kkk]=ejecuta_uno(prob_pob,i,n,M_ady, indicador, indicador_amino)
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
                    rms,gdt,MC_similitud, divKl,distancias_sal=fitness_gdt_rmsd_mc_fisquim(BB, back_temp,MC_BB,corte,descriptor_ref,descriptor_temp)
                    print([rms,gdt,energia_desing,MC_similitud,np.mean(divKl)])
                    "se agregan los valores"
                    fitness1[i][k]=agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b)
                    tot_ececution_dist[i][j]=distancias_sal
                else:
                    rms,gdt,MC_similitud,distancias_sal=fitness_gdt_rmsd_mc(BB, back_temp,MC_BB,corte)
                    print([rms,gdt,energia_desing,MC_similitud])
                    fitness1[i][k]=agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b)  
                    tot_ececution_dist[i][j]=distancias_sal
                
                "si la ejecucion tiene mayor fitnes que alguno en best_fitness entonces se actualiza la lista de los mejores"
                if fitness1[i][k]>best_fitness[pos_min]:
                    best_fitness[pos_min]=fitness1[i][k]
                    best_fitness_energ[pos_min]=energia_desing
                    best_execution[pos_min]=list(tot_ececution[i][j])
                    pdb_select[pos_min] = PDB_pdredict[0][:]
                    best_execution_dist[pos_min]=distancias_sal
                    best_fitness, best_execution, pdb_select, best_fitness_energ,best_execution_dist=ordena_mejores_energia(best_fitness, best_execution, pdb_select, best_fitness_energ,best_execution_dist)
                    print(best_fitness)
                k+=1

            "Actualiza la distribucion con las mejores act_prop ejecuciones  de la distribucion "
            "y las mejores act_glob ejecuciones globales"
            if generacion in utt_reinicio1 and i in reiniciados:
                w_act,w_act_dist=W_act1(i, best_execution, len(best_execution), tot_ececution, act_prop, det_fitness, fitness1,best_fitness,best_execution_dist,tot_ececution_dist)
                poblacion=actualiza_individuo(n,poblacion,i,w_act,w_act_dist,indicador_entrada,indicador_entrada_pos,nodo_terminal)
            else:
                w_act,w_act_dist=W_act1(i, best_execution, act_glob, tot_ececution, act_prop, det_fitness, fitness1,best_fitness,best_execution_dist,tot_ececution_dist)
                poblacion=actualiza_individuo(n,poblacion,i,w_act,w_act_dist,indicador_entrada,indicador_entrada_pos,nodo_terminal)
            	
        prob_pob=calcula_prob_tot(poblacion,nodos_continuan)            
    
        "[5,20,30, 40, 50, 60, 70,80,90,100,120,140,160,190,220,250,280,310,340,370,400,430,460,490,500]:"
        if generacion in uutt:
            for iii in range(n_best):
                nombre=nombre_sin_extension+'_sol4_'+str(iii)+'_'+str(generacion)+'.pdb'
                with open(nombre, 'w') as archivo_pdb:
                    archivo_pdb.write(pdb_select[iii])
            ruta_archivo = "datos_guardados"+str(generacion)+".txt"
            with open(ruta_archivo, 'w') as archivo:
                archivo.write(f"best_execution: {best_execution}\n")
                archivo.write(f"best_execution_dist: {best_execution_dist}\n")
                archivo.write(f"best_fitness: {best_fitness}\n")
                archivo.write(f"pdb_select: {pdb_select}\n")
                archivo.write(f"poblacion: {poblacion}\n")
                archivo.write(f"best_fitness_energ: {best_fitness_energ}\n") 
        if generacion in uutt_reinicio:
            poblacion[0:n_reinicia]= copy.deepcopy(poblacion_aleatoria)
            prob_pob[0:n_reinicia]= copy.deepcopy(prob_pob_aleatoria)
    "salida"
    return poblacion, best_execution,best_fitness,pdb_select,best_fitness_energ            



def EDA_isla(pdb_file_path,desc_si_no,continua,numnum,n_pop,n_rep,n_fit,act_prop,act_glob,n_best,pos_min,n_generaciones,
             n_reinicia,n_genra_reinicio,corte,amino,a,b,local,nueva_ruta, islas,prob_copera,n_colab,tot_act_red,max_gdt):
    global best_execution_1
    global best_execution_dist_1
    global pdb_select_1
    global best_fitness_1
    global best_fitness_energ_1
    
    global best_execution_2
    global best_execution_dist_2
    global pdb_select_2
    global best_fitness_2
    global best_fitness_energ_2
    
    global best_execution_3
    global best_execution_dist_3
    global pdb_select_3
    global best_fitness_3
    global best_fitness_energ_3
    
    global best_execution_4
    global best_execution_dist_4
    global pdb_select_4
    global best_fitness_4
    global best_fitness_energ_4
    
    global best_execution_5
    global best_execution_dist_5
    global pdb_select_5
    global best_fitness_5
    global best_fitness_energ_5
    
    global best_execution_6
    global best_execution_dist_6
    global pdb_select_6
    global best_fitness_6
    global best_fitness_energ_6
    
    global best_execution_7
    global best_execution_dist_7
    global pdb_select_7
    global best_fitness_7
    global best_fitness_energ_7
    
    global best_execution_8
    global best_execution_dist_8
    global pdb_select_8
    global best_fitness_8
    global best_fitness_energ_8
    
    global best_execution_9
    global best_execution_dist_9
    global pdb_select_9
    global best_fitness_9
    global best_fitness_energ_9
    
    global best_execution_10
    global best_execution_dist_10
    global pdb_select_10
    global best_fitness_10
    global best_fitness_energ_10
    
    global best_execution_11
    global best_execution_dist_11
    global pdb_select_11
    global best_fitness_11
    global best_fitness_energ_11
    
    global best_execution_12
    global best_execution_dist_12
    global pdb_select_12
    global best_fitness_12
    global best_fitness_energ_12
    
    global best_execution_13
    global best_execution_dist_13
    global pdb_select_13
    global best_fitness_13
    global best_fitness_energ_13
    
    global best_execution_14
    global best_execution_dist_14
    global pdb_select_14
    global best_fitness_14
    global best_fitness_energ_14
    
    global best_execution_15
    global best_execution_dist_15
    global pdb_select_15
    global best_fitness_15
    global best_fitness_energ_15
    
    global best_execution_16
    global best_execution_dist_16
    global pdb_select_16
    global best_fitness_16
    global best_fitness_energ_16
    
    global best_execution_17
    global best_execution_dist_17
    global pdb_select_17
    global best_fitness_17
    global best_fitness_energ_17
    
    global best_execution_18
    global best_execution_dist_18
    global pdb_select_18
    global best_fitness_18
    global best_fitness_energ_18
    
    global best_execution_19
    global best_execution_dist_19
    global pdb_select_19
    global best_fitness_19
    global best_fitness_energ_19
    
    global best_execution_20
    global best_execution_dist_20
    global pdb_select_20
    global best_fitness_20
    global best_fitness_energ_20
    
    global max_gdt_sal_1
    global max_gdt_sal_2
    global max_gdt_sal_3
    global max_gdt_sal_4
    global max_gdt_sal_5
    global max_gdt_sal_6
    global max_gdt_sal_7
    global max_gdt_sal_8
    global max_gdt_sal_9
    global max_gdt_sal_10
    global max_gdt_sal_11
    global max_gdt_sal_12
    global max_gdt_sal_13
    global max_gdt_sal_14
    global max_gdt_sal_15
    global max_gdt_sal_16
    global max_gdt_sal_17
    global max_gdt_sal_18
    global max_gdt_sal_19
    global max_gdt_sal_20
    
    model_3d = None
    prediction_3d=''
    #model_3d,prediction_3d=esmfold_predict_structure('ACDEFGH','prediction.pdb',model_3d)
    print(prediction_3d)
    print(type(prediction_3d))
    max_gdt_global=0
    "cada aminoacido en amino se hace coincidir con su correspondiente numero en"
    num_amino = list(range(0, 20))
    "secuencia del peptido, si procede"
    if desc_si_no==0:
        tokenizer = 0
        model_ESM2 = 0
        secuencia_ref=extract_amino_acid_sequence_pdb(pdb_file_path)
        descriptor_ref=0
    if desc_si_no==1:
        tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        model_ESM2 = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
        secuencia_ref=extract_amino_acid_sequence_pdb(pdb_file_path)
        descriptor_ref=descriptores(secuencia_ref,tokenizer,model_ESM2) 
    "Extraer el backbone del primer modelo y de la cadena principal"
    BB=extract_backbone_atoms(pdb_file_path)
    "n longitud de la secuencia"
    n=int(len(BB)/4)
    "nombre de la proteina sin .pdb"
    nombre_sin_extension = os.path.splitext(pdb_file_path)[0]
    "mapa de contactos del BB"
    MC_BB=mapa_contacto(BB)
    "Generacion"
    generacion=-1
    "matriz para guardar las actualizaciones de las distribuciones y son 4 reprecentaciones"
    if 1 in islas:
        nodo_terminal_1,amino_nodo_1,rango_numeros_1,nodos_continuan_1,indicador_1,num_amino_new_1,indicador_amino_1,M_ady_1,M_prob_1,indicador_entrada_1,indicador_entrada_pos_1,poblacion_1,prob_pob_1,best_fitness_1,best_fitness_energ_1,best_execution_1,best_execution_dist_1,pdb_select_1=inicializa_isla_1(n,n_pop,amino,n_best)
    if 2 in islas:
        nodo_terminal_2,amino_nodo_2,rango_numeros_2,nodos_continuan_2,indicador_2,num_amino_new_2,indicador_amino_2,M_ady_2,M_prob_2,indicador_entrada_2,indicador_entrada_pos_2,poblacion_2,prob_pob_2,best_fitness_2,best_fitness_energ_2,best_execution_2,best_execution_dist_2,pdb_select_2=inicializa_isla_2(n,n_pop,amino,n_best)
    if 3 in islas:
        nodo_terminal_3,amino_nodo_3,rango_numeros_3,nodos_continuan_3,indicador_3,num_amino_new_3,indicador_amino_3,M_ady_3,M_prob_3,indicador_entrada_3,indicador_entrada_pos_3,poblacion_3,prob_pob_3,best_fitness_3,best_fitness_energ_3,best_execution_3,best_execution_dist_3,pdb_select_3=inicializa_isla_3(n,n_pop,amino,n_best)
    if 4 in islas:
        nodo_terminal_4,amino_nodo_4,rango_numeros_4,nodos_continuan_4,indicador_4,num_amino_new_4,indicador_amino_4,M_ady_4,M_prob_4,indicador_entrada_4,indicador_entrada_pos_4,poblacion_4,prob_pob_4,best_fitness_4,best_fitness_energ_4,best_execution_4,best_execution_dist_4,pdb_select_4=inicializa_isla_4(n,n_pop,amino,n_best)
    if 5 in islas:
        nodo_terminal_5,amino_nodo_5,rango_numeros_5,nodos_continuan_5,indicador_5,num_amino_new_5,indicador_amino_5,M_ady_5,M_prob_5,indicador_entrada_5,indicador_entrada_pos_5,poblacion_5,prob_pob_5,best_fitness_5,best_fitness_energ_5,best_execution_5,best_execution_dist_5,pdb_select_5=inicializa_isla_5(n,n_pop,amino,n_best)
    if 6 in islas:
        nodo_terminal_6,amino_nodo_6,rango_numeros_6,nodos_continuan_6,indicador_6,num_amino_new_6,indicador_amino_6,M_ady_6,M_prob_6,indicador_entrada_6,indicador_entrada_pos_6,poblacion_6,prob_pob_6,best_fitness_6,best_fitness_energ_6,best_execution_6,best_execution_dist_6,pdb_select_6=inicializa_isla_6(n,n_pop,amino,n_best)
    if 7 in islas:
        nodo_terminal_7,amino_nodo_7,rango_numeros_7,nodos_continuan_7,indicador_7,num_amino_new_7,indicador_amino_7,M_ady_7,M_prob_7,indicador_entrada_7,indicador_entrada_pos_7,poblacion_7,prob_pob_7,best_fitness_7,best_fitness_energ_7,best_execution_7,best_execution_dist_7,pdb_select_7=inicializa_isla_7(n,n_pop,amino,n_best)
    if 8 in islas:
        nodo_terminal_8,amino_nodo_8,rango_numeros_8,nodos_continuan_8,indicador_8,num_amino_new_8,indicador_amino_8,M_ady_8,M_prob_8,indicador_entrada_8,indicador_entrada_pos_8,poblacion_8,prob_pob_8,best_fitness_8,best_fitness_energ_8,best_execution_8,best_execution_dist_8,pdb_select_8=inicializa_isla_8(n,n_pop,amino,n_best)
    if 9 in islas:
        nodo_terminal_9,amino_nodo_9,rango_numeros_9,nodos_continuan_9,indicador_9,num_amino_new_9,indicador_amino_9,M_ady_9,M_prob_9,indicador_entrada_9,indicador_entrada_pos_9,poblacion_9,prob_pob_9,best_fitness_9,best_fitness_energ_9,best_execution_9,best_execution_dist_9,pdb_select_9=inicializa_isla_9(n,n_pop,amino,n_best)
    if 10 in islas:
        nodo_terminal_10,amino_nodo_10,rango_numeros_10,nodos_continuan_10,indicador_10,num_amino_new_10,indicador_amino_10,M_ady_10,M_prob_10,indicador_entrada_10,indicador_entrada_pos_10,poblacion_10,prob_pob_10,best_fitness_10,best_fitness_energ_10,best_execution_10,best_execution_dist_10,pdb_select_10=inicializa_isla_10(n,n_pop,amino,n_best)
    if 11 in islas:
        nodo_terminal_11,amino_nodo_11,rango_numeros_11,nodos_continuan_11,indicador_11,num_amino_new_11,indicador_amino_11,M_ady_11,M_prob_11,indicador_entrada_11,indicador_entrada_pos_11,poblacion_11,prob_pob_11,best_fitness_11,best_fitness_energ_11,best_execution_11,best_execution_dist_11,pdb_select_11=inicializa_isla_11(n,n_pop,amino,n_best)
    if 12 in islas:
        nodo_terminal_12,amino_nodo_12,rango_numeros_12,nodos_continuan_12,indicador_12,num_amino_new_12,indicador_amino_12,M_ady_12,M_prob_12,indicador_entrada_12,indicador_entrada_pos_12,poblacion_12,prob_pob_12,best_fitness_12,best_fitness_energ_12,best_execution_12,best_execution_dist_12,pdb_select_12=inicializa_isla_12(n,n_pop,amino,n_best)
    if 13 in islas:
        nodo_terminal_13,amino_nodo_13,rango_numeros_13,nodos_continuan_13,indicador_13,num_amino_new_13,indicador_amino_13,M_ady_13,M_prob_13,indicador_entrada_13,indicador_entrada_pos_13,poblacion_13,prob_pob_13,best_fitness_13,best_fitness_energ_13,best_execution_13,best_execution_dist_13,pdb_select_13=inicializa_isla_13(n,n_pop,amino,n_best)
    if 14 in islas:
        nodo_terminal_14,amino_nodo_14,rango_numeros_14,nodos_continuan_14,indicador_14,num_amino_new_14,indicador_amino_14,M_ady_14,M_prob_14,indicador_entrada_14,indicador_entrada_pos_14,poblacion_14,prob_pob_14,best_fitness_14,best_fitness_energ_14,best_execution_14,best_execution_dist_14,pdb_select_14=inicializa_isla_14(n,n_pop,amino,n_best)
    if 15 in islas:
        nodo_terminal_15,amino_nodo_15,rango_numeros_15,nodos_continuan_15,indicador_15,num_amino_new_15,indicador_amino_15,M_ady_15,M_prob_15,indicador_entrada_15,indicador_entrada_pos_15,poblacion_15,prob_pob_15,best_fitness_15,best_fitness_energ_15,best_execution_15,best_execution_dist_15,pdb_select_15=inicializa_isla_15(n,n_pop,amino,n_best)

    if 16 in islas:
        nodo_terminal_16,amino_nodo_16,rango_numeros_16,nodos_continuan_16,indicador_16,num_amino_new_16,indicador_amino_16,M_ady_16,M_prob_16,indicador_entrada_16,indicador_entrada_pos_16,poblacion_16,prob_pob_16,best_fitness_16,best_fitness_energ_16,best_execution_16,best_execution_dist_16,pdb_select_16=inicializa_isla_16(n,n_pop,amino,n_best)
    if 17 in islas:
        M_ady_17,M_prob_17,poblacion_17,prob_pob_17,best_fitness_17,best_fitness_energ_17,best_execution_17,best_execution_dist_17,pdb_select_17=inicializa_isla_17(n,n_pop,amino,n_best)
    if 18 in islas:
        M_ady_18,M_prob_18,poblacion_18,prob_pob_18,best_fitness_18,best_fitness_energ_18,best_execution_18,best_execution_dist_18,pdb_select_18=inicializa_isla_18(n,n_pop,amino,n_best)
    if 19 in islas:
        M_ady_19,M_prob_19,poblacion_19,prob_pob_19,best_fitness_19,best_fitness_energ_19,best_execution_19,best_execution_dist_19,pdb_select_19=inicializa_isla_19(n,n_pop,amino,n_best)
    if 20 in islas:
        M_ady_20,M_prob_20,poblacion_20,prob_pob_20,best_fitness_20,best_fitness_energ_20,best_execution_20,best_execution_dist_20,pdb_select_20=inicializa_isla_20(n,n_pop,amino,n_best)


    "crea las primeras ejecuciones"
    tot_ececution=[[[0 for _ in range(n)] for _ in range(n_rep)] for _ in range(n_pop)]
    
    "almacena las distancia de las ejecuciones a las cuales se le calculo el fitness"
    tot_ececution_dist=[[[0 for _ in range(n)] for _ in range(n_rep)] for _ in range(n_pop)]
    
    "for i in range(n_pop*n_rep):"
    "tot_ececution[i]=ejecuta_uno(prob_pob,0,n,M_ady, indicador, indicador_amino)"
    "almacena las n_fit posiciones segun el filtro aplicado"
    det_fitness=[[None]*n_fit]*n_pop
    "almacena las n_fit fitness de las n_fit posiciones segun el filtro aplicado"
    fitness1=[[None]*n_fit]*n_pop
    "indica cuando hay que hacer copias"
    uutt=list(range(0, n_generaciones, 10))
    uutt_reinicio=list(range(0, n_generaciones, n_genra_reinicio))
    utt_reinicio1=[x + 1 for x in uutt_reinicio]
    reiniciados=list(range(0,n_reinicia))
    "funcion de energia"
    my_scorefxn = pyrosetta.get_score_function(True)
    p = pyrosetta.Pose() 
    energia_desing=0
    version=1
    model_name = "esmfold_v0.model" if version == "0" else "esmfold.model"
    "modelos para ESMFold utilizando transformer"
    model=1
    if local==1:
        "model = EsmForProteinFolding.from_pretrained('facebook/esmfold_v1')"
        "tokenizer = AutoTokenizer.from_pretrained('facebook/esmfold_v1')"
        if "model" not in dir() or model_name != model_name_:
            if "model" in dir():
              # delete old model from memory
              del model
              gc.collect()
              if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
            model = torch.load("esmfold.model")
            model.eval().cuda().requires_grad_(False)
            model_name_ = "esmfold.model"
        # optimized for Tesla T4
        model.set_chunk_size(128)
    
    if continua==1:
        "ruta donde estan los datos almacenados para continuar con la ejecucion"
        ruta_archivo=os.getcwd()+'/'+str(pdb_file_path[-8:-4])+'_datos_guardados'+str(numnum)+'.txt'
        with open(ruta_archivo, "r") as archivo:
            for linea in archivo:
                etiqueta, valor = linea.strip().split(": ")
                if etiqueta == "best_execution_1":
                    best_execution_1 = eval(valor)
                elif etiqueta == "best_execution_dist_1":
                    best_execution_dist_1 = eval(valor)            
                elif etiqueta == "best_fitness_1":
                    best_fitness_1 = eval(valor)
                elif etiqueta == "poblacion_1":
                    poblacion_1 = eval(valor)
                elif etiqueta == "best_fitness_energ_1":
                    best_fitness_energ_1= eval(valor)
                elif etiqueta == "pdb_select_1":
                    pdb_select_1= eval(valor)
                elif etiqueta == "prob_pob_1":
                    prob_pob_1= eval(valor)
                elif etiqueta == "best_execution_2":
                    best_execution_2 = eval(valor)
                elif etiqueta == "best_execution_dist_2":
                    best_execution_dist_2 = eval(valor)            
                elif etiqueta == "best_fitness_2":
                    best_fitness_2 = eval(valor)
                elif etiqueta == "poblacion_2":
                    poblacion_2 = eval(valor)
                elif etiqueta == "best_fitness_energ_2":
                    best_fitness_energ_2= eval(valor)
                elif etiqueta == "pdb_select_2":
                    pdb_select_2= eval(valor)
                elif etiqueta == "prob_pob_2":
                    prob_pob_2= eval(valor)           
                elif etiqueta == "best_execution_3":
                    best_execution_3 = eval(valor)
                elif etiqueta == "best_execution_dist_3":
                    best_execution_dist_3 = eval(valor)            
                elif etiqueta == "best_fitness_3":
                    best_fitness_3 = eval(valor)
                elif etiqueta == "poblacion_3":
                    poblacion_3 = eval(valor)
                elif etiqueta == "best_fitness_energ_3":
                    best_fitness_energ_3= eval(valor)
                elif etiqueta == "pdb_select_3":
                    pdb_select_3= eval(valor)
                elif etiqueta == "prob_pob_3":
                    prob_pob_3= eval(valor)
                elif etiqueta == "best_execution_4":
                    best_execution_4 = eval(valor)
                elif etiqueta == "best_execution_dist_4":
                    best_execution_dist_4 = eval(valor)            
                elif etiqueta == "best_fitness_4":
                    best_fitness_4 = eval(valor)
                elif etiqueta == "poblacion_4":
                    poblacion_4 = eval(valor)
                elif etiqueta == "best_fitness_energ_4":
                    best_fitness_energ_4= eval(valor)
                elif etiqueta == "pdb_select_4":
                    pdb_select_4= eval(valor)
                elif etiqueta == "prob_pob_4":
                    prob_pob_4= eval(valor)
                elif etiqueta == "best_execution_5":
                    best_execution_5 = eval(valor)
                elif etiqueta == "best_execution_dist_5":
                    best_execution_dist_5 = eval(valor)            
                elif etiqueta == "best_fitness_5":
                    best_fitness_5 = eval(valor)
                elif etiqueta == "poblacion_5":
                    poblacion_5 = eval(valor)
                elif etiqueta == "best_fitness_energ_5":
                    best_fitness_energ_5= eval(valor)
                elif etiqueta == "pdb_select_5":
                    pdb_select_5= eval(valor)
                elif etiqueta == "prob_pob_5":
                    prob_pob_5= eval(valor)
                elif etiqueta == "best_execution_6":
                    best_execution_6 = eval(valor)
                elif etiqueta == "best_execution_dist_6":
                    best_execution_dist_6 = eval(valor)            
                elif etiqueta == "best_fitness_6":
                    best_fitness_6 = eval(valor)
                elif etiqueta == "poblacion_6":
                    poblacion_6 = eval(valor)
                elif etiqueta == "best_fitness_energ_6":
                    best_fitness_energ_6= eval(valor)
                elif etiqueta == "pdb_select_6":
                    pdb_select_6= eval(valor)
                elif etiqueta == "prob_pob_6":
                    prob_pob_6= eval(valor)
                elif etiqueta == "best_execution_7":
                    best_execution_7 = eval(valor)
                elif etiqueta == "best_execution_dist_7":
                    best_execution_dist_7 = eval(valor)            
                elif etiqueta == "best_fitness_7":
                    best_fitness_7 = eval(valor)
                elif etiqueta == "poblacion_7":
                    poblacion_7 = eval(valor)
                elif etiqueta == "best_fitness_energ_7":
                    best_fitness_energ_7= eval(valor)
                elif etiqueta == "pdb_select_7":
                    pdb_select_7= eval(valor)
                elif etiqueta == "prob_pob_7":
                    prob_pob_7= eval(valor)
                elif etiqueta == "best_execution_8":
                    best_execution_8 = eval(valor)
                elif etiqueta == "best_execution_dist_8":
                    best_execution_dist_8 = eval(valor)            
                elif etiqueta == "best_fitness_8":
                    best_fitness_8 = eval(valor)
                elif etiqueta == "poblacion_8":
                    poblacion_8 = eval(valor)
                elif etiqueta == "best_fitness_energ_8":
                    best_fitness_energ_8= eval(valor)
                elif etiqueta == "pdb_select_8":
                    pdb_select_8= eval(valor)
                elif etiqueta == "prob_pob_8":
                    prob_pob_8= eval(valor)
                elif etiqueta == "best_execution_9":
                    best_execution_9 = eval(valor)
                elif etiqueta == "best_execution_dist_9":
                    best_execution_dist_9 = eval(valor)            
                elif etiqueta == "best_fitness_9":
                    best_fitness_9 = eval(valor)
                elif etiqueta == "poblacion_9":
                    poblacion_9 = eval(valor)
                elif etiqueta == "best_fitness_energ_9":
                    best_fitness_energ_9= eval(valor)
                elif etiqueta == "pdb_select_9":
                    pdb_select_9= eval(valor)
                elif etiqueta == "prob_pob_9":
                    prob_pob_9= eval(valor)
                elif etiqueta == "best_execution_10":
                    best_execution_10 = eval(valor)
                elif etiqueta == "best_execution_dist_10":
                    best_execution_dist_10 = eval(valor)            
                elif etiqueta == "best_fitness_10":
                    best_fitness_10 = eval(valor)
                elif etiqueta == "poblacion_10":
                    poblacion_10 = eval(valor)
                elif etiqueta == "best_fitness_energ_10":
                    best_fitness_energ_10= eval(valor)
                elif etiqueta == "pdb_select_10":
                    pdb_select_10= eval(valor)
                elif etiqueta == "prob_pob_10":
                    prob_pob_10= eval(valor)
                elif etiqueta == "best_execution_11":
                    best_execution_11 = eval(valor)
                elif etiqueta == "best_execution_dist_11":
                    best_execution_dist_11 = eval(valor)            
                elif etiqueta == "best_fitness_11":
                    best_fitness_11 = eval(valor)
                elif etiqueta == "poblacion_11":
                    poblacion_11 = eval(valor)
                elif etiqueta == "best_fitness_energ_11":
                    best_fitness_energ_11= eval(valor)
                elif etiqueta == "pdb_select_11":
                    pdb_select_11= eval(valor)
                elif etiqueta == "prob_pob_11":
                    prob_pob_11= eval(valor)
                elif etiqueta == "best_execution_12":
                    best_execution_12 = eval(valor)
                elif etiqueta == "best_execution_dist_12":
                    best_execution_dist_12 = eval(valor)            
                elif etiqueta == "best_fitness_12":
                    best_fitness_12 = eval(valor)
                elif etiqueta == "poblacion_12":
                    poblacion_12 = eval(valor)
                elif etiqueta == "best_fitness_energ_12":
                    best_fitness_energ_12= eval(valor)
                elif etiqueta == "pdb_select_12":
                    pdb_select_12= eval(valor)
                elif etiqueta == "prob_pob_12":
                    prob_pob_12= eval(valor)
                elif etiqueta == "best_execution_13":
                    best_execution_13 = eval(valor)
                elif etiqueta == "best_execution_dist_13":
                    best_execution_dist_13 = eval(valor)            
                elif etiqueta == "best_fitness_13":
                    best_fitness_13 = eval(valor)
                elif etiqueta == "poblacion_13":
                    poblacion_13 = eval(valor)
                elif etiqueta == "best_fitness_energ_13":
                    best_fitness_energ_13= eval(valor)
                elif etiqueta == "pdb_select_13":
                    pdb_select_13= eval(valor)
                elif etiqueta == "prob_pob_13":
                    prob_pob_13= eval(valor)
                elif etiqueta == "best_execution_14":
                    best_execution_14 = eval(valor)
                elif etiqueta == "best_execution_dist_14":
                    best_execution_dist_14 = eval(valor)            
                elif etiqueta == "best_fitness_14":
                    best_fitness_14 = eval(valor)
                elif etiqueta == "poblacion_14":
                    poblacion_14 = eval(valor)
                elif etiqueta == "best_fitness_energ_14":
                    best_fitness_energ_14= eval(valor)
                elif etiqueta == "pdb_select_14":
                    pdb_select_14= eval(valor)
                elif etiqueta == "prob_pob_14":
                    prob_pob_14= eval(valor)                    
                elif etiqueta == "best_execution_15":
                    best_execution_15 = eval(valor)
                elif etiqueta == "best_execution_dist_15":
                    best_execution_dist_15 = eval(valor)            
                elif etiqueta == "best_fitness_15":
                    best_fitness_15 = eval(valor)
                elif etiqueta == "poblacion_15":
                    poblacion_15 = eval(valor)
                elif etiqueta == "best_fitness_energ_15":
                    best_fitness_energ_15= eval(valor)
                elif etiqueta == "pdb_select_15":
                    pdb_select_15= eval(valor)
                elif etiqueta == "prob_pob_15":
                    prob_pob_15= eval(valor)
                elif etiqueta == "best_execution_16":
                    best_execution_16 = eval(valor)
                elif etiqueta == "best_execution_dist_16":
                    best_execution_dist_16 = eval(valor)            
                elif etiqueta == "best_fitness_16":
                    best_fitness_16 = eval(valor)
                elif etiqueta == "poblacion_16":
                    poblacion_16 = eval(valor)
                elif etiqueta == "best_fitness_energ_16":
                    best_fitness_energ_16= eval(valor)
                elif etiqueta == "pdb_select_16":
                    pdb_select_16= eval(valor)
                elif etiqueta == "prob_pob_16":
                    prob_pob_16= eval(valor)
                elif etiqueta == "best_execution_17":
                    best_execution_17 = eval(valor)
                elif etiqueta == "best_execution_dist_17":
                    best_execution_dist_17 = eval(valor)            
                elif etiqueta == "best_fitness_17":
                    best_fitness_17 = eval(valor)
                elif etiqueta == "poblacion_17":
                    poblacion_17 = eval(valor)
                elif etiqueta == "best_fitness_energ_17":
                    best_fitness_energ_17= eval(valor)
                elif etiqueta == "pdb_select_17":
                    pdb_select_17= eval(valor)
                elif etiqueta == "prob_pob_17":
                    prob_pob_17= eval(valor)
                elif etiqueta == "best_execution_18":
                    best_execution_18 = eval(valor)
                elif etiqueta == "best_execution_dist_18":
                    best_execution_dist_18 = eval(valor)            
                elif etiqueta == "best_fitness_18":
                    best_fitness_18 = eval(valor)
                elif etiqueta == "poblacion_18":
                    poblacion_18 = eval(valor)
                elif etiqueta == "best_fitness_energ_18":
                    best_fitness_energ_18= eval(valor)
                elif etiqueta == "pdb_select_18":
                    pdb_select_18= eval(valor)
                elif etiqueta == "prob_pob_18":
                    prob_pob_18= eval(valor)
                elif etiqueta == "best_execution_19":
                    best_execution_19 = eval(valor)
                elif etiqueta == "best_execution_dist_19":
                    best_execution_dist_19 = eval(valor)            
                elif etiqueta == "best_fitness_19":
                    best_fitness_19 = eval(valor)
                elif etiqueta == "poblacion_19":
                    poblacion_19 = eval(valor)
                elif etiqueta == "best_fitness_energ_19":
                    best_fitness_energ_19= eval(valor)
                elif etiqueta == "pdb_select_19":
                    pdb_select_19= eval(valor)
                elif etiqueta == "prob_pob_19":
                    prob_pob_19= eval(valor)
                elif etiqueta == "best_execution_20":
                    best_execution_20 = eval(valor)
                elif etiqueta == "best_execution_dist_20":
                    best_execution_dist_20 = eval(valor)            
                elif etiqueta == "best_fitness_20":
                    best_fitness_20 = eval(valor)
                elif etiqueta == "poblacion_20":
                    poblacion_20 = eval(valor)
                elif etiqueta == "best_fitness_energ_20":
                    best_fitness_energ_20= eval(valor)
                elif etiqueta == "pdb_select_20":
                    pdb_select_20= eval(valor)
                elif etiqueta == "prob_pob_20":
                    prob_pob_20= eval(valor)
        generacion=numnum
    
    "Algoritmo"
    while generacion<n_generaciones and max_gdt_global<max_gdt:
        """drive.mount('/content/drive')
        os.chdir(nueva_ruta)"""
        "Determina la cantidad por la que se deven dividir la cantidad almacenada "
        "en cada indice de cada elemento de la poblacion"
        if 17 in islas or 18 in islas:
            all_best_execution=[]
            all_best_fitnes=[]
            for isla in islas: 
                nombre_variable = 'best_execution_' + str(isla)
                best_execution_ = globals()[nombre_variable]
                nombre_variable = 'best_fitness_' + str(isla)
                best_fitness_ = globals()[nombre_variable]
                for item in range(len(best_execution_)):
                    all_best_execution.append(copy.deepcopy(best_execution_[item]))
                    all_best_fitnes.append(copy.deepcopy(best_fitness_[item]))
            all_best_fitnes, all_best_execution=ordena_mejores_all(all_best_fitnes, all_best_execution)
            if 17 in islas:
                poblacion_17=actualiza_individuo_red(poblacion_17,all_best_execution,tot_act_red,n)
                prob_pob_17[0]= calcula_prob_red(poblacion_17,n)
                for iikkpp in range(1,len(prob_pob_17)):
                    prob_pob_17[iikkpp]=prob_pob_17[0]
            if 18 in islas:
                if generacion in utt_reinicio1:
                    poblacion_18 = [[[[1 for _ in range(20)] for _ in range(20)]for _ in range(n)] for _ in range(n_pop)]
                    poblacion_18=actualiza_individuo_red(poblacion_18,all_best_execution,tot_act_red,n)
                    prob_pob_18[0]= calcula_prob_red(poblacion_18,n)
                    for iikkpp in range(1,len(prob_pob_18)):
                        prob_pob_18[iikkpp]=prob_pob_18[0]
                else:
                    poblacion_18=actualiza_individuo_red(poblacion_18,all_best_execution,tot_act_red,n)
                    prob_pob_18[0]= calcula_prob_red(poblacion_18,n)
                    for iikkpp in range(1,len(prob_pob_18)):
                        prob_pob_18[iikkpp]=prob_pob_18[0]
            if 19 in islas:
                poblacion_19=actualiza_individuo_red(poblacion_19,all_best_execution,tot_act_red,n)
                prob_pob_19[0]= calcula_prob_red_M(poblacion_19,n)
                for iikkpp in range(1,len(prob_pob_19)):
                    prob_pob_19[iikkpp]=prob_pob_19[0]
            if 20 in islas:
                if generacion in utt_reinicio1:
                    poblacion_20 = [[[[1 for _ in range(20)] for _ in range(20)]for _ in range(n)] for _ in range(n_pop)]
                    poblacion_20=actualiza_individuo_red(poblacion_20,all_best_execution,tot_act_red,n)
                    prob_pob_20[0]= calcula_prob_red_M(poblacion_20,n)
                    for iikkpp in range(1,len(prob_pob_20)):
                        prob_pob_20[iikkpp]=prob_pob_20[0]
                else:
                    poblacion_20=actualiza_individuo_red(poblacion_20,all_best_execution,tot_act_red,n)
                    prob_pob_20[0]= calcula_prob_red_M(poblacion_20,n)
                    for iikkpp in range(1,len(prob_pob_20)):
                        prob_pob_20[iikkpp]=prob_pob_20[0]
                    
        generacion=generacion+1
        if 1 in islas:
            print([1,generacion])
            max_gdt_sal_1,best_fitness_1, best_execution_1, pdb_select_1, best_fitness_energ_1,best_execution_dist_1,poblacion_1,prob_pob_1=isla_1(indicador_entrada_1,indicador_entrada_pos_1,nodo_terminal_1,act_prop,act_glob,best_fitness_1, best_execution_1, pdb_select_1, best_fitness_energ_1,best_execution_dist_1,a,b,MC_BB,corte,
                       descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
                       n_rep,tot_ececution,nodos_continuan_1,indicador_1,indicador_amino_1,M_ady_1,poblacion_1,prob_pob_1,model_3d)
        if 2 in islas:
            print([2,generacion])
            max_gdt_sal_2,best_fitness_2, best_execution_2, pdb_select_2, best_fitness_energ_2,best_execution_dist_2,poblacion_2,prob_pob_2=isla_2(indicador_entrada_2,indicador_entrada_pos_2,nodo_terminal_2,act_prop,act_glob,best_fitness_2, best_execution_2, pdb_select_2, best_fitness_energ_2,best_execution_dist_2,a,b,MC_BB,corte,
                   descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
                   n_rep,tot_ececution,nodos_continuan_2,indicador_2,indicador_amino_2,M_ady_2,poblacion_2,prob_pob_2,model_3d)
        if 3 in islas:
            print([3,generacion])
            max_gdt_sal_3,best_fitness_3, best_execution_3, pdb_select_3, best_fitness_energ_3,best_execution_dist_3,poblacion_3,prob_pob_3=isla_3(indicador_entrada_3,indicador_entrada_pos_3,nodo_terminal_3,act_prop,act_glob,best_fitness_3, best_execution_3, pdb_select_3, best_fitness_energ_3,best_execution_dist_3,a,b,MC_BB,corte,
                   descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
                   n_rep,tot_ececution,nodos_continuan_3,indicador_3,indicador_amino_3,M_ady_3,poblacion_3,prob_pob_3,model_3d)
        if 4 in islas:
            print([4,generacion])
            max_gdt_sal_4,best_fitness_4, best_execution_4, pdb_select_4, best_fitness_energ_4,best_execution_dist_4,poblacion_4,prob_pob_4=isla_4(indicador_entrada_4,indicador_entrada_pos_4,nodo_terminal_4,act_prop,act_glob,best_fitness_4, best_execution_4, pdb_select_4, best_fitness_energ_4,best_execution_dist_4,a,b,MC_BB,corte,
                   descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
                   n_rep,tot_ececution,nodos_continuan_4,indicador_4,indicador_amino_4,M_ady_4,poblacion_4,prob_pob_4,model_3d)
        if 5 in islas:
            print([5,generacion])
            max_gdt_sal_5,best_fitness_5, best_execution_5, pdb_select_5, best_fitness_energ_5,best_execution_dist_5,poblacion_5,prob_pob_5=isla_5(generacion,utt_reinicio1,indicador_entrada_5,indicador_entrada_pos_5,nodo_terminal_5,act_prop,act_glob,best_fitness_5, best_execution_5, pdb_select_5, best_fitness_energ_5,best_execution_dist_5,a,b,MC_BB,corte,
                       descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
                       n_rep,tot_ececution,nodos_continuan_5,indicador_5,indicador_amino_5,M_ady_5,poblacion_5,prob_pob_5,M_prob_5,model_3d)
        if 6 in islas:
            print([6,generacion])
            max_gdt_sal_6,best_fitness_6, best_execution_6, pdb_select_6, best_fitness_energ_6,best_execution_dist_6,poblacion_6,prob_pob_6=isla_6(generacion,utt_reinicio1,indicador_entrada_6,indicador_entrada_pos_6,nodo_terminal_6,act_prop,act_glob,best_fitness_6, best_execution_6, pdb_select_6, best_fitness_energ_6,best_execution_dist_6,a,b,MC_BB,corte,
                       descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
                       n_rep,tot_ececution,nodos_continuan_6,indicador_6,indicador_amino_6,M_ady_6,poblacion_6,prob_pob_6,M_prob_6,model_3d)
        if 7 in islas:
            print([7,generacion])
            max_gdt_sal_7,best_fitness_7, best_execution_7, pdb_select_7, best_fitness_energ_7,best_execution_dist_7,poblacion_7,prob_pob_7=isla_7(generacion,utt_reinicio1,indicador_entrada_7,indicador_entrada_pos_7,nodo_terminal_7,act_prop,act_glob,best_fitness_7, best_execution_7, pdb_select_7, best_fitness_energ_7,best_execution_dist_7,a,b,MC_BB,corte,
                       descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
                       n_rep,tot_ececution,nodos_continuan_7,indicador_7,indicador_amino_7,M_ady_7,poblacion_7,prob_pob_7,M_prob_7,model_3d)
        if 8 in islas:
            print([8,generacion])
            max_gdt_sal_8,best_fitness_8, best_execution_8, pdb_select_8, best_fitness_energ_8,best_execution_dist_8,poblacion_8,prob_pob_8=isla_8(generacion,utt_reinicio1,indicador_entrada_8,indicador_entrada_pos_8,nodo_terminal_8,act_prop,act_glob,best_fitness_8, best_execution_8, pdb_select_8, best_fitness_energ_8,best_execution_dist_8,a,b,MC_BB,corte,
                       descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
                       n_rep,tot_ececution,nodos_continuan_8,indicador_8,indicador_amino_8,M_ady_8,poblacion_8,prob_pob_8,M_prob_8,model_3d)
        if 9 in islas:
            print([9,generacion])
            max_gdt_sal_9,best_fitness_9, best_execution_9, pdb_select_9, best_fitness_energ_9,best_execution_dist_9,poblacion_9,prob_pob_9=isla_9(indicador_entrada_9,indicador_entrada_pos_9,nodo_terminal_9,act_prop,act_glob,best_fitness_9, best_execution_9, pdb_select_9, best_fitness_energ_9,best_execution_dist_9,a,b,MC_BB,corte,
                       descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
                       n_rep,tot_ececution,nodos_continuan_9,indicador_9,indicador_amino_9,M_ady_9,poblacion_9,prob_pob_9,model_3d)
        if 10 in islas:
            print([10,generacion])
            max_gdt_sal_10,best_fitness_10, best_execution_10, pdb_select_10, best_fitness_energ_10,best_execution_dist_10,poblacion_10,prob_pob_10=isla_10(indicador_entrada_10,indicador_entrada_pos_10,nodo_terminal_10,act_prop,act_glob,best_fitness_10, best_execution_10, pdb_select_10, best_fitness_energ_10,best_execution_dist_10,a,b,MC_BB,corte,
                       descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
                       n_rep,tot_ececution,nodos_continuan_10,indicador_10,indicador_amino_10,M_ady_10,poblacion_10,prob_pob_10,model_3d)
        if 11 in islas:
            print([11,generacion])
            max_gdt_sal_11,best_fitness_11, best_execution_11, pdb_select_11, best_fitness_energ_11,best_execution_dist_11,poblacion_11,prob_pob_11=isla_11(generacion,utt_reinicio1,indicador_entrada_11,indicador_entrada_pos_11,nodo_terminal_11,act_prop,act_glob,best_fitness_11, best_execution_11, pdb_select_11, best_fitness_energ_11,best_execution_dist_11,a,b,MC_BB,corte,
                       descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
                       n_rep,tot_ececution,nodos_continuan_11,indicador_11,indicador_amino_11,M_ady_11,poblacion_11,prob_pob_11,M_prob_11,model_3d)
        if 12 in islas:
            print([12,generacion])
            max_gdt_sal_12,best_fitness_12, best_execution_12, pdb_select_12, best_fitness_energ_12,best_execution_dist_12,poblacion_12,prob_pob_12=isla_12(generacion,utt_reinicio1,indicador_entrada_12,indicador_entrada_pos_12,nodo_terminal_12,act_prop,act_glob,best_fitness_12, best_execution_12, pdb_select_12, best_fitness_energ_12,best_execution_dist_12,a,b,MC_BB,corte,
                       descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
                       n_rep,tot_ececution,nodos_continuan_12,indicador_12,indicador_amino_12,M_ady_12,poblacion_12,prob_pob_12,M_prob_12,model_3d)
        if 13 in islas:
            print([13,generacion])
            max_gdt_sal_13,best_fitness_13, best_execution_13, pdb_select_13, best_fitness_energ_13,best_execution_dist_13,poblacion_13,prob_pob_13=isla_13(indicador_entrada_13,indicador_entrada_pos_13,nodo_terminal_13,act_prop,act_glob,best_fitness_13, best_execution_13, pdb_select_13, best_fitness_energ_13,best_execution_dist_13,a,b,MC_BB,corte,
                       descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
                       n_rep,tot_ececution,nodos_continuan_13,indicador_13,indicador_amino_13,M_ady_13,poblacion_13,prob_pob_13,model_3d)
        if 14 in islas:
            print([14,generacion])
            max_gdt_sal_14,best_fitness_14, best_execution_14, pdb_select_14, best_fitness_energ_14,best_execution_dist_14,poblacion_14,prob_pob_14=isla_14(indicador_entrada_14,indicador_entrada_pos_14,nodo_terminal_14,act_prop,act_glob,best_fitness_14, best_execution_14, pdb_select_14, best_fitness_energ_14,best_execution_dist_14,a,b,MC_BB,corte,
                   descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
                   n_rep,tot_ececution,nodos_continuan_14,indicador_14,indicador_amino_14,M_ady_14,poblacion_14,prob_pob_14,model_3d)
        if 15 in islas:
            print([15,generacion])
            max_gdt_sal_15,best_fitness_15, best_execution_15, pdb_select_15, best_fitness_energ_15,best_execution_dist_15,poblacion_15,prob_pob_15=isla_15(generacion,utt_reinicio1,indicador_entrada_15,indicador_entrada_pos_15,nodo_terminal_15,act_prop,act_glob,best_fitness_15, best_execution_15, pdb_select_15, best_fitness_energ_15,best_execution_dist_15,a,b,MC_BB,corte,
                       descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
                       n_rep,tot_ececution,nodos_continuan_15,indicador_15,indicador_amino_15,M_ady_15,poblacion_15,prob_pob_15,M_prob_15,model_3d)
        if 16 in islas:
            print([16,generacion])
            max_gdt_sal_16,best_fitness_16, best_execution_16, pdb_select_16, best_fitness_energ_16,best_execution_dist_16,poblacion_16,prob_pob_16=isla_16(generacion,utt_reinicio1,indicador_entrada_16,indicador_entrada_pos_16,nodo_terminal_16,act_prop,act_glob,best_fitness_16, best_execution_16, pdb_select_16, best_fitness_energ_16,best_execution_dist_16,a,b,MC_BB,corte,
                       descriptor_ref,my_scorefxn,p,det_fitness,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
                       n_rep,tot_ececution,nodos_continuan_16,indicador_16,indicador_amino_16,M_ady_16,poblacion_16,prob_pob_16,M_prob_16,model_3d)
        if 17 in islas:
            print([17,generacion])
            max_gdt_sal_17,best_fitness_17, best_execution_17, pdb_select_17, best_fitness_energ_17,best_execution_dist_17=isla_17(utt_reinicio1,best_fitness_17, best_execution_17, pdb_select_17, best_fitness_energ_17,best_execution_dist_17,a,b,MC_BB,corte,
                       descriptor_ref,my_scorefxn,p,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
                       n_rep,tot_ececution,prob_pob_17,det_fitness,model_3d)
        if 18 in islas:
            print([18,generacion])
            max_gdt_sal_18,best_fitness_18, best_execution_18, pdb_select_18, best_fitness_energ_18,best_execution_dist_18=isla_18(utt_reinicio1,best_fitness_18, best_execution_18, pdb_select_18, best_fitness_energ_18,best_execution_dist_18,a,b,MC_BB,corte,
                       descriptor_ref,my_scorefxn,p,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
                       n_rep,tot_ececution,prob_pob_18,det_fitness,model_3d)
        if 19 in islas:
            print([19,generacion])
            max_gdt_sal_19,best_fitness_19, best_execution_19, pdb_select_19, best_fitness_energ_19,best_execution_dist_19=isla_19(utt_reinicio1,best_fitness_19, best_execution_19, pdb_select_19, best_fitness_energ_19,best_execution_dist_19,a,b,MC_BB,corte,
                       descriptor_ref,my_scorefxn,p,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
                       n_rep,tot_ececution,prob_pob_19,det_fitness,model_3d)
        if 20 in islas:
            print([20,generacion])
            max_gdt_sal_20,best_fitness_20, best_execution_20, pdb_select_20, best_fitness_energ_20,best_execution_dist_20=isla_20(utt_reinicio1,best_fitness_20, best_execution_20, pdb_select_20, best_fitness_energ_20,best_execution_dist_20,a,b,MC_BB,corte,
                       descriptor_ref,my_scorefxn,p,BB,pos_min,tot_ececution_dist,fitness1,model,tokenizer,model_ESM2,desc_si_no,local,amino,n,n_fit,n_pop,
                       n_rep,tot_ececution,prob_pob_20,det_fitness,model_3d)
        "actualiza el mayor gdt encontrado"
        for isla in islas: 
            nombre_variable = 'max_gdt_sal_' + str(isla)
            max_gdt_sal_ = globals()[nombre_variable]
            max_gdt_global=max(max_gdt_sal_,max_gdt_global)
        if generacion in uutt or max_gdt<=max_gdt_global:
            ruta_archivo = str(pdb_file_path[-8:-4])+'_datos_guardados'+str(generacion)+".txt"
            with open(ruta_archivo, 'w') as archivo:
                if 1 in islas:
                    archivo.write(f"best_execution_1: {best_execution_1}\n")
                    archivo.write(f"best_execution_dist_1: {best_execution_dist_1}\n")
                    archivo.write(f"best_fitness_1: {best_fitness_1}\n")
                    archivo.write(f"pdb_select_1: {pdb_select_1}\n")
                    archivo.write(f"poblacion_1: {poblacion_1}\n")
                    archivo.write(f"best_fitness_energ_1: {best_fitness_energ_1}\n") 
                    archivo.write(f"prob_pob_1: {prob_pob_1}\n") 
                if 2 in islas:
                    archivo.write(f"best_execution_2: {best_execution_2}\n")
                    archivo.write(f"best_execution_dist_2: {best_execution_dist_2}\n")
                    archivo.write(f"best_fitness_2: {best_fitness_2}\n")
                    archivo.write(f"pdb_select_2: {pdb_select_2}\n")
                    archivo.write(f"poblacion_2: {poblacion_2}\n")
                    archivo.write(f"best_fitness_energ_2: {best_fitness_energ_2}\n") 
                    archivo.write(f"prob_pob_2: {prob_pob_2}\n") 
                if 3 in islas:
                    archivo.write(f"best_execution_3: {best_execution_3}\n")
                    archivo.write(f"best_execution_dist_3: {best_execution_dist_3}\n")
                    archivo.write(f"best_fitness_3: {best_fitness_3}\n")
                    archivo.write(f"pdb_select_3: {pdb_select_3}\n")
                    archivo.write(f"poblacion_3: {poblacion_3}\n")
                    archivo.write(f"best_fitness_energ_3: {best_fitness_energ_3}\n") 
                    archivo.write(f"prob_pob_3: {prob_pob_3}\n") 
                if 4 in islas:
                    archivo.write(f"best_execution_4: {best_execution_4}\n")
                    archivo.write(f"best_execution_dist_4: {best_execution_dist_4}\n")
                    archivo.write(f"best_fitness_4: {best_fitness_4}\n")
                    archivo.write(f"pdb_select_4: {pdb_select_4}\n")
                    archivo.write(f"poblacion_4: {poblacion_4}\n")
                    archivo.write(f"best_fitness_energ_4: {best_fitness_energ_4}\n") 
                    archivo.write(f"prob_pob_4: {prob_pob_4}\n") 
                if 5 in islas:
                    archivo.write(f"best_execution_5: {best_execution_5}\n")
                    archivo.write(f"best_execution_dist_5: {best_execution_dist_5}\n")
                    archivo.write(f"best_fitness_5: {best_fitness_5}\n")
                    archivo.write(f"pdb_select_5: {pdb_select_5}\n")
                    archivo.write(f"poblacion_5: {poblacion_5}\n")
                    archivo.write(f"best_fitness_energ_5: {best_fitness_energ_5}\n") 
                    archivo.write(f"prob_pob_5: {prob_pob_5}\n") 
                if 6 in islas:
                    archivo.write(f"best_execution_6: {best_execution_6}\n")
                    archivo.write(f"best_execution_dist_6: {best_execution_dist_6}\n")
                    archivo.write(f"best_fitness_6: {best_fitness_6}\n")
                    archivo.write(f"pdb_select_6: {pdb_select_6}\n")
                    archivo.write(f"poblacion_6: {poblacion_6}\n")
                    archivo.write(f"best_fitness_energ_6: {best_fitness_energ_6}\n") 
                    archivo.write(f"prob_pob_6: {prob_pob_6}\n") 
                if 7 in islas:
                    archivo.write(f"best_execution_7: {best_execution_7}\n")
                    archivo.write(f"best_execution_dist_7: {best_execution_dist_7}\n")
                    archivo.write(f"best_fitness_7: {best_fitness_7}\n")
                    archivo.write(f"pdb_select_7: {pdb_select_7}\n")
                    archivo.write(f"poblacion_7: {poblacion_7}\n")
                    archivo.write(f"best_fitness_energ_7: {best_fitness_energ_7}\n") 
                    archivo.write(f"prob_pob_7: {prob_pob_7}\n") 
                if 8 in islas:
                    archivo.write(f"best_execution_8: {best_execution_8}\n")
                    archivo.write(f"best_execution_dist_8: {best_execution_dist_8}\n")
                    archivo.write(f"best_fitness_8: {best_fitness_8}\n")
                    archivo.write(f"pdb_select_8: {pdb_select_8}\n")
                    archivo.write(f"poblacion_8: {poblacion_8}\n")
                    archivo.write(f"best_fitness_energ_8: {best_fitness_energ_8}\n") 
                    archivo.write(f"prob_pob_8: {prob_pob_8}\n") 
                if 9 in islas:
                    archivo.write(f"best_execution_9: {best_execution_9}\n")
                    archivo.write(f"best_execution_dist_9: {best_execution_dist_9}\n")
                    archivo.write(f"best_fitness_9: {best_fitness_9}\n")
                    archivo.write(f"pdb_select_9: {pdb_select_9}\n")
                    archivo.write(f"poblacion_9: {poblacion_9}\n")
                    archivo.write(f"best_fitness_energ_9: {best_fitness_energ_9}\n") 
                    archivo.write(f"prob_pob_9: {prob_pob_9}\n") 
                if 10 in islas:
                    archivo.write(f"best_execution_10: {best_execution_10}\n")
                    archivo.write(f"best_execution_dist_10: {best_execution_dist_10}\n")
                    archivo.write(f"best_fitness_10: {best_fitness_10}\n")
                    archivo.write(f"pdb_select_10: {pdb_select_10}\n")
                    archivo.write(f"poblacion_10: {poblacion_10}\n")
                    archivo.write(f"best_fitness_energ_10: {best_fitness_energ_10}\n") 
                    archivo.write(f"prob_pob_10: {prob_pob_10}\n") 
                if 11 in islas:
                    archivo.write(f"best_execution_11: {best_execution_11}\n")
                    archivo.write(f"best_execution_dist_11: {best_execution_dist_11}\n")
                    archivo.write(f"best_fitness_11: {best_fitness_11}\n")
                    archivo.write(f"pdb_select_11: {pdb_select_11}\n")
                    archivo.write(f"poblacion_11: {poblacion_11}\n")
                    archivo.write(f"best_fitness_energ_11: {best_fitness_energ_11}\n") 
                    archivo.write(f"prob_pob_11: {prob_pob_11}\n") 
                if 12 in islas:
                    archivo.write(f"best_execution_12: {best_execution_12}\n")
                    archivo.write(f"best_execution_dist_12: {best_execution_dist_12}\n")
                    archivo.write(f"best_fitness_12: {best_fitness_12}\n")
                    archivo.write(f"pdb_select_12: {pdb_select_12}\n")
                    archivo.write(f"poblacion_12: {poblacion_12}\n")
                    archivo.write(f"best_fitness_energ_12: {best_fitness_energ_12}\n") 
                    archivo.write(f"prob_pob_12: {prob_pob_12}\n") 
                if 13 in islas:
                    archivo.write(f"best_execution_13: {best_execution_13}\n")
                    archivo.write(f"best_execution_dist_13: {best_execution_dist_13}\n")
                    archivo.write(f"best_fitness_13: {best_fitness_13}\n")
                    archivo.write(f"pdb_select_13: {pdb_select_13}\n")
                    archivo.write(f"poblacion_13: {poblacion_13}\n")
                    archivo.write(f"best_fitness_energ_13: {best_fitness_energ_13}\n") 
                    archivo.write(f"prob_pob_13: {prob_pob_13}\n") 
                if 14 in islas:
                    archivo.write(f"best_execution_14: {best_execution_14}\n")
                    archivo.write(f"best_execution_dist_14: {best_execution_dist_14}\n")
                    archivo.write(f"best_fitness_14: {best_fitness_14}\n")
                    archivo.write(f"pdb_select_14: {pdb_select_14}\n")
                    archivo.write(f"poblacion_14: {poblacion_14}\n")
                    archivo.write(f"best_fitness_energ_14: {best_fitness_energ_14}\n") 
                    archivo.write(f"prob_pob_14: {prob_pob_14}\n") 
                if 15 in islas:
                    archivo.write(f"best_execution_15: {best_execution_15}\n")
                    archivo.write(f"best_execution_dist_15: {best_execution_dist_15}\n")
                    archivo.write(f"best_fitness_15: {best_fitness_15}\n")
                    archivo.write(f"pdb_select_15: {pdb_select_15}\n")
                    archivo.write(f"poblacion_15: {poblacion_15}\n")
                    archivo.write(f"best_fitness_energ_15: {best_fitness_energ_15}\n") 
                    archivo.write(f"prob_pob_15: {prob_pob_15}\n") 
                if 16 in islas:
                    archivo.write(f"best_execution_16: {best_execution_16}\n")
                    archivo.write(f"best_execution_dist_16: {best_execution_dist_16}\n")
                    archivo.write(f"best_fitness_16: {best_fitness_16}\n")
                    archivo.write(f"pdb_select_16: {pdb_select_16}\n")
                    archivo.write(f"poblacion_16: {poblacion_16}\n")
                    archivo.write(f"best_fitness_energ_16: {best_fitness_energ_16}\n") 
                    archivo.write(f"prob_pob_16: {prob_pob_16}\n") 
                if 17 in islas:
                    archivo.write(f"best_execution_17: {best_execution_17}\n")
                    archivo.write(f"best_execution_dist_17: {best_execution_dist_17}\n")
                    archivo.write(f"best_fitness_17: {best_fitness_17}\n")
                    archivo.write(f"pdb_select_17: {pdb_select_17}\n")
                    archivo.write(f"poblacion_17: {poblacion_17}\n")
                    archivo.write(f"best_fitness_energ_17: {best_fitness_energ_17}\n") 
                    archivo.write(f"prob_pob_17: {prob_pob_17}\n") 
                if 18 in islas:
                    archivo.write(f"best_execution_18: {best_execution_18}\n")
                    archivo.write(f"best_execution_dist_18: {best_execution_dist_18}\n")
                    archivo.write(f"best_fitness_18: {best_fitness_18}\n")
                    archivo.write(f"pdb_select_18: {pdb_select_18}\n")
                    archivo.write(f"poblacion_18: {poblacion_18}\n")
                    archivo.write(f"best_fitness_energ_18: {best_fitness_energ_18}\n") 
                    archivo.write(f"prob_pob_18: {prob_pob_18}\n") 
                if 19 in islas:
                    archivo.write(f"best_execution_19: {best_execution_19}\n")
                    archivo.write(f"best_execution_dist_19: {best_execution_dist_19}\n")
                    archivo.write(f"best_fitness_19: {best_fitness_19}\n")
                    archivo.write(f"pdb_select_19: {pdb_select_19}\n")
                    archivo.write(f"poblacion_19: {poblacion_19}\n")
                    archivo.write(f"best_fitness_energ_19: {best_fitness_energ_19}\n") 
                    archivo.write(f"prob_pob_19: {prob_pob_19}\n") 
                if 20 in islas:
                    archivo.write(f"best_execution_20: {best_execution_20}\n")
                    archivo.write(f"best_execution_dist_20: {best_execution_dist_20}\n")
                    archivo.write(f"best_fitness_20: {best_fitness_20}\n")
                    archivo.write(f"pdb_select_20: {pdb_select_20}\n")
                    archivo.write(f"poblacion_20: {poblacion_20}\n")
                    archivo.write(f"best_fitness_energ_20: {best_fitness_energ_20}\n") 
                    archivo.write(f"prob_pob_20: {prob_pob_20}\n") 
                    
                    
                    
        "las isals coperan entre si con probabiliudad prob_cop, se seleccionan de cada isla n_colab y si cada elemento es mejor que el peor de la otra isla entonces se transfiera la informacion de una isla a otra"
        for ijikk, cop in enumerate(islas):
            for cop1 in islas[ijikk+1:]:  
                if random.choices([1, 0], weights=[prob_copera, 1-prob_copera], k=1)==[1]:
                    print('****************************************************************************copera************************************************')
                    nombre_variable = 'best_execution_' + str(cop)
                    best_execution_ = globals()[nombre_variable]
                    nombre_variable = 'best_execution_dist_' + str(cop)
                    best_execution_dist_ = globals()[nombre_variable]
                    nombre_variable = 'best_fitness_' + str(cop)
                    best_fitness_ = globals()[nombre_variable]
                    nombre_variable = 'pdb_select_' + str(cop)
                    pdb_select_ = globals()[nombre_variable]
                    nombre_variable = 'best_fitness_energ_' + str(cop)
                    best_fitness_energ_ = globals()[nombre_variable]
                    
                    nombre_variable = 'best_execution_' + str(cop1)
                    best_execution__ = globals()[nombre_variable]
                    nombre_variable = 'best_execution_dist_' + str(cop1)
                    best_execution_dist__ = globals()[nombre_variable]
                    nombre_variable = 'best_fitness_' + str(cop1)
                    best_fitness__ = globals()[nombre_variable]
                    nombre_variable = 'pdb_select_' + str(cop1)
                    pdb_select__ = globals()[nombre_variable]
                    nombre_variable = 'best_fitness_energ_' + str(cop1)
                    best_fitness_energ__ = globals()[nombre_variable]
        
                    total_fitness = sum(best_fitness_)
                    probabilidades = [iji / total_fitness for iji in best_fitness_]
                    muestra = np.random.choice(list(range(len(best_fitness_))), size=n_colab, p=probabilidades, replace=False)
                    muestra_enteros1 = [int(i) for i in muestra]
                    
                    for ittp in muestra_enteros1:
                        if best_fitness__[len(best_fitness__)-1]<best_fitness_[ittp]:
                            best_execution__[len(best_fitness__)-1]=copy.deepcopy(best_execution_[ittp])
                            best_execution_dist__[len(best_fitness__)-1]=copy.deepcopy(best_execution_dist_[ittp])
                            best_fitness__[len(best_fitness__)-1]=copy.deepcopy(best_fitness_[ittp])
                            pdb_select__[len(best_fitness__)-1]=copy.deepcopy(pdb_select_[ittp])
                            best_fitness_energ__[len(best_fitness__)-1]=copy.deepcopy(best_fitness_energ_[ittp])
                            best_fitness__, best_execution__, pdb_select__, best_fitness_energ__,best_execution_dist__=ordena_mejores_energia(best_fitness__, best_execution__, pdb_select__, best_fitness_energ__,best_execution_dist__)
                    
                    total_fitness = sum(best_fitness__)
                    probabilidades = [iji / total_fitness for iji in best_fitness__]
                    muestra = np.random.choice(list(range(len(best_fitness__))), size=n_colab, p=probabilidades, replace=False)
                    muestra_enteros2 = [int(i) for i in muestra]
                    
                    for ittp in muestra_enteros2:
                        if best_fitness_[len(best_fitness_)-1]<best_fitness__[ittp]:
                            best_execution_[len(best_fitness_)-1]=copy.deepcopy(best_execution__[ittp])
                            best_execution_dist_[len(best_fitness_)-1]=copy.deepcopy(best_execution_dist__[ittp])
                            best_fitness_[len(best_fitness_)-1]=copy.deepcopy(best_fitness__[ittp])
                            pdb_select_[len(best_fitness_)-1]=copy.deepcopy(pdb_select__[ittp])
                            best_fitness_energ_[len(best_fitness_)-1]=copy.deepcopy(best_fitness_energ__[ittp])
                            best_fitness_, best_execution_, pdb_select_, best_fitness_energ_,best_execution_dist_=ordena_mejores_energia(best_fitness_, best_execution_, pdb_select_, best_fitness_energ_,best_execution_dist_)  
        for ijikk, cop in enumerate(islas):
            nombre_variable = 'best_execution_' + str(cop)
            best_execution_ = globals()[nombre_variable]
            nombre_variable = 'best_execution_dist_' + str(cop)
            best_execution_dist_ = globals()[nombre_variable]
            nombre_variable = 'best_fitness_' + str(cop)
            best_fitness_ = globals()[nombre_variable]
            nombre_variable = 'pdb_select_' + str(cop)
            pdb_select_ = globals()[nombre_variable]
            nombre_variable = 'best_fitness_energ_' + str(cop)
            best_fitness_energ_ = globals()[nombre_variable]
            cambio_00=0
            for iippkk in range(len(best_fitness_)-1): 
                if best_fitness_[iippkk]==100:
                    best_fitness_[iippkk]=0
                if best_fitness_[iippkk]==best_fitness_[iippkk+1] and best_execution_[iippkk]==best_execution_[iippkk+1]:
                    best_fitness_[iippkk+1]=0
                    cambio_00=1
            if cambio_00==1:
                best_fitness_, best_execution_, pdb_select_, best_fitness_energ_,best_execution_dist_=ordena_mejores_energia(best_fitness_, best_execution_, pdb_select_, best_fitness_energ_,best_execution_dist_)         
    return poblacion_1, best_execution_1,best_fitness_1,pdb_select_1,best_fitness_energ_1            
