import numpy as np
import re
from ..descproteins import GAAC,EGAAC,CKSAAGP,GDPC,GTPC,BLOSUM62,CTDC,CTDT,CTDD
from Bio.SVDSuperimposer import SVDSuperimposer
from ..EDA_tools.EDAtools import similitud_MC,entropia_descrip_val,entropia_descrip_prob
from transformers import EsmTokenizer, EsmModel
import torch


"Mapas de contacto"
def mapa_contacto(BB):
    BB_temp=BB[1::4]
    MC=[[0 for _ in range(len(BB_temp))]for _ in range(len(BB_temp)-1)]
    for i in range(len(BB_temp)-1):
        temporal= np.linalg.norm(BB_temp[(i+1):] - BB_temp[i], axis=1)
        MC[i][(i+1):]=temporal
    return MC


def descriptores_seleccionados(secuencia,modelo):
    fastas = [['nombre',secuencia,'Chain','Chain']]
    userDefinedOrder = 'ACDEFGHIKLMNPQRSTVWY'
    userDefinedOrder = re.sub('[^ACDEFGHIKLMNPQRSTVWY]', '', userDefinedOrder)
    if len(userDefinedOrder) != 20:
        userDefinedOrder = 'ACDEFGHIKLMNPQRSTVWY'
    myAAorder = {
        'alphabetically': 'ACDEFGHIKLMNPQRSTVWY',
        'polarity': 'DENKRQHSGTAPYVMCWIFL',
        'sideChainVolume': 'GASDPCTNEVHQILMKRFYW',
        'userDefined': userDefinedOrder
    }
    myOrder = 'ACDEFGHIKLMNPQRSTVWY'
    kw = {'path': 'nada', 'order': myOrder, 'type': 'Protein'}
    cmd = modelo + '.' + modelo + '(fastas, **kw)'
    encodings = eval(cmd)
    return encodings[1][2:]


"descritores ESM2"
def ESM2_desc(secuencia,tokenizer,model_ESM2):
	inputs = tokenizer(secuencia, return_tensors="pt", padding=True, truncation=True)
	outputs = model_ESM2(**inputs)
	last_hidden_states = outputs.last_hidden_state
	x = last_hidden_states.detach()
	descriptores_ESM2= np.array([])
	for i in range(1,(len(x[0])-1)):
		descriptores_ESM2=np.concatenate((descriptores_ESM2, x[0][i].numpy()))
	return descriptores_ESM2
		


"determina los descriptores para una secuencia de aminoacidos segun el metodo seleccionado"
def descriptores(secuencia,tokenizer,model_ESM2):
    choices=['GAAC', 'EGAAC', 'CKSAAGP','GDPC','GTPC','BLOSUM62',
                  'CTDC', 'CTDT', 'CTDD']
    descriptores_salida = [None for _ in range(len(choices))] 
    iii=0
    for meot in choices:
        descriptores_salida[iii]=descriptores_seleccionados(secuencia,meot)
        iii+=1
    descriptores_salida.append(ESM2_desc(secuencia,tokenizer,model_ESM2))
    return descriptores_salida


"determina las metricas utilizadas en el fitness"
"se incluyen los descriptores fisico quimico"
def fitness_gdt_rmsd_mc_fisquim(x,y,MC_BB,corte,descriptor_ref,descriptor_temp):
    x=np.array(x)
    y=np.array(y)
    sup = SVDSuperimposer()
    sup.set(x, y)
    sup.run()
    rms = sup.get_rms()
    y_on_x = sup.get_transformed()
    distancias = np.linalg.norm(x - y_on_x, axis=1)
    distancias_sal=list(distancias[1::4])
    distancias1=distancias
    gdt=0
    for i in range(len(corte)):
        gdt = gdt+np.count_nonzero(distancias1 <= corte[i])
    gdt=gdt/(len(corte)*len(distancias1))
    temoral_MC=mapa_contacto(y)
    MC_similitud=similitud_MC(MC_BB, temoral_MC)
    divKl=[]
    posi_observaciones=[1,2,5,6,7,8,9]
    posi_prob=[0,3,4]
    for lugar in posi_observaciones:
        divKl.append(entropia_descrip_val(descriptor_ref, descriptor_temp, lugar))
    for lugar in posi_prob:
        divKl.append(entropia_descrip_prob(descriptor_ref, descriptor_temp, lugar))
    divKl = [x for x in divKl if x != float('inf')]
    if divKl==[]:divKl=2.5
    return rms, gdt, MC_similitud, divKl,distancias_sal


"determina las metricas utilizadas en el fitness"
"no se incluyen los descriptores fisico quimico"
def fitness_gdt_rmsd_mc(x,y,MC_BB,corte):
    x=np.array(x)
    y=np.array(y)
    sup = SVDSuperimposer()
    sup.set(x, y)
    sup.run()
    rms = sup.get_rms()
    y_on_x = sup.get_transformed()
    distancias = np.linalg.norm(x - y_on_x, axis=1)
    distancias_sal=list(distancias[1::4])
    distancias1=distancias
    gdt=0
    for i in range(len(corte)):
        gdt = gdt+np.count_nonzero(distancias1 <= corte[i])
    gdt=gdt/(len(corte)*len(distancias1))
    temoral_MC=mapa_contacto(y)
    MC_similitud=similitud_MC(MC_BB, temoral_MC)
    return rms, gdt, MC_similitud,distancias_sal

"agrega las metricas cuando se tienen descriptores"
def agrega_rmsd_gdt_E_MC_divKl(MC_similitud,energia_desing,rms,gdt,divKl,a,b):
    temporal_fitnes=(1/(1+rms))+gdt+(1/(1+MC_similitud))
    #return temporal_fitnes+ (temporal_fitnes/3)/(1+np.exp((energia_desing+a)/b))+1/(1+np.mean(divKl))
    return temporal_fitnes+1/(1+np.mean(divKl))

"agrega las metricas cuando no se tienen descriptores"
def agrega_rmsd_gdt_E_MC(MC_similitud,energia_desing,rms,gdt,a,b):
    temporal_fitnes=(1/(1+rms))+gdt+(1/(1+MC_similitud))
    return temporal_fitnes+ (temporal_fitnes/3)/(1+np.exp((energia_desing+a)/b))
