from .Algorithm_evolutionary.algorithm_evolutionary import EDA_tres_capas,EDA_isla
import os
import pyrosetta

#ruta_carpeta_bd_selected = os.path.join(directorio_trabajo, 'BD_selected')
#print(archivos_en_bd_selected[0:15])
#archivos_en_bd_selected = archivos_en_bd_selected[2:]
#print(archivos_en_bd_selected[0:15])

def run(max_generations = 1000, population_size = 5, sample_size = 3):
    pyrosetta.init()
    directorio_trabajo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_carpeta_bd_selected = os.path.join(directorio_trabajo, 'target_pdbs')
    archivos_en_bd_selected = os.listdir(ruta_carpeta_bd_selected)
    for i in archivos_en_bd_selected:
        pdb_file_path = os.path.join(ruta_carpeta_bd_selected,i)

        "se determinaran los descriptores fisicoquimicos 1 si si 0 si no"
        desc_si_no=1

        "ESMfold se ejecuta de manera local"
        local=0

        "Cuando se esta trabajando en colab"
        nueva_ruta='algo'

        "1 si es continuacion de una ejecucion anterior, 0 otro caso"
        # if i=='2wqg.pdb':
        #     continua=1
        # else:
        #     continua=0
        continua = 0

        "ultimo dato guardado de las ejecuciones anteriores, rellenar si continua=1"
        numnum=2460
        "valores de a y b en la funcion objetivo para dilatar la energia"
        a=30
        b=30

        "cantidad de individuos por los que estara formada la poblacion"
        n_pop=population_size
        "ejecuciones de cada individuo por etapa"
        n_rep=sample_size
        "Cantidad de ejecuciones a determinar el fitnes"
        n_fit=3
        "cantidad a actualizar del propio"
        act_prop=2
        "cantidad a actualizar del global"
        act_glob=1
        "cantidad de mejores elementos guardados (las ejecuciones de menor fitness)"
        n_best=80
        "posicion en el arreglo donde se encuetra el elmento de menor fitness"
        pos_min=n_best-1
        "Numero de generaciones"
        n_generaciones=max_generations
        "cantidad de elementod de la poblacion que se reiniciaran"
        n_reinicia=8
        "numero de generaciones para reiniciarce"
        #n_genra_reinicio=50
        n_genra_reinicio=20

        "Corte para el GDT"
        corte=[1,2,4,8]

        "Variables y nombres"
        aminoacidos_nombre = ['Valina','Leucina','Isoleucina','Metionina','Fenilalanina','Lisina','Arginina','Histidina','Ácido aspártico','Ácido glutámico','Asparagina','Glutamina','Tirosina','Triptófano','Serina','Treonina','Prolina','Alanina','Glicina','Cisteína']
        amino = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

        "Islas a utilizar en el algoritmo"
        islas=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        #islas=[15,16,17,18,19,20]

        "con que probabilidad va ha haber coperacion entre las islas"
        prob_copera=0.10

        "cantidad de elementos a ser seleccionados para colaborar"
        n_colab=1
        #EDA_tres_capas(pdb_file_path,desc_si_no,continua,numnum,n_pop,n_rep,n_fit,act_prop,act_glob,n_best,pos_min,n_generaciones,n_reinicia,n_genra_reinicio,corte,amino,a,b,local,nueva_ruta)
        #poblacion, best_execution,best_fitness,pdb_select,best_fitness_energ=Algoritmo_evolutivo_energia_MC_descriptor(continua,n,n_pop,n_rep,n_fit,act_prop,act_glob,amino,num_amino,n_best,n_generaciones,pos_min,BB,w1,w2,corte,tokenizer,model,nrep,mat_frec,nombre_sin_extension,n_func,MC_BB,ruta_archivo,numnum,secuencia_ref)
        "cantidad de elementos utilizados para actualizar la red"
        tot_act_red=1500

        "detener cuando se tenga un gdt deceado"
        #max_gdt=0.90
        max_gdt=0.9999
        EDA_isla(pdb_file_path,desc_si_no,continua,numnum,n_pop,n_rep,n_fit,act_prop,act_glob,n_best,pos_min,n_generaciones,
                    n_reinicia,n_genra_reinicio,corte,amino,a,b,local,nueva_ruta, islas,prob_copera,n_colab,tot_act_red,max_gdt)
