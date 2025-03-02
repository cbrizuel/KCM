from .Algorithm_evolutionary.algorithm_evolutionary import EDA_tres_capas,EDA_isla
import os
import pyrosetta


#archivos_en_bd_selected = archivos_en_bd_selected[0:1]
#archivos_en_bd_selected = archivos_en_bd_selected[700:]

def run(max_generations = 1000, 
        population_size = 5, 
        sample_size = 5,
        input_folder = 'benchmark_pdbs'):
    pyrosetta.init()
    import os
    directorio_trabajo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_carpeta_bd_selected = os.path.join(directorio_trabajo, input_folder)
    archivos_en_bd_selected = os.listdir(ruta_carpeta_bd_selected)
    print(archivos_en_bd_selected[0:15])
    for ii in archivos_en_bd_selected:
        # Borra todas las variables del espacio de nombres global
        for var in list(globals().keys()):
            if var not in ('__name__', '__doc__', '__loader__', '__package__', '__spec__','__file__','ii','pyrosetta','max_generations','population_size','sample_size'):
                del globals()[var]
        from .Algorithm_evolutionary.algorithm_evolutionary import EDA_tres_capas,EDA_isla
        import os
        directorio_trabajo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ruta_carpeta_bd_selected = os.path.join(directorio_trabajo, input_folder)
        archivos_en_bd_selected = os.listdir(ruta_carpeta_bd_selected)        
        pdb_file_path = os.path.join(ruta_carpeta_bd_selected,ii)

        "se determinaran los descriptores fisicoquimicos 1 si si 0 si no"
        desc_si_no=1

        "ESMfold se ejecuta de manera local"
        local=0

        "Cuando se esta trabajando en colab"
        nueva_ruta='algo'

        "1 si es continuacion de una ejecucion anterior, 0 otro caso"
        # if ii=='7spo.pdb':
        #     continua=1
        # else:
        #     continua=0
        continua = 0

        "ultimo dato guardado de las ejecuciones anteriores, rellenar si continua=1"
        numnum=260
        "valores de a y b en la funcion objetivo para dilatar la energia"
        a=100
        b=100

        "cantida1 de individuos por los que estara formada la poblacion"
        n_pop=population_size
        "ejecuciones de cada individuo por etapa"
        n_rep=sample_size
        "Cantidad de ejecuciones a determinar el fitnes"
        n_fit=5
        "cantidad a actualizar del propio"
        act_prop=2
        "cantidad a actualizar del global"
        act_glob=3
        "cantidad de mejores elementos guardados (las ejecuciones de menor fitness)"
        n_best=150
        "posicion en el arreglo donde se encuetra el elmento de menor fitness"
        pos_min=n_best-1
        "Numero de generaciones"
        n_generaciones=max_generations
        "cantidad de elementod de la poblacion que se reiniciaran"
        n_reinicia=10
        "numero de generaciones para reiniciarce"
        n_genra_reinicio=10

        "Corte para el GDT"
        corte=[1,2,4,8]

        "Variables y nombres"
        aminoacidos_nombre = ['Valina','Leucina','Isoleucina','Metionina','Fenilalanina','Lisina','Arginina','Histidina','Ácido aspártico','Ácido glutámico','Asparagina','Glutamina','Tirosina','Triptófano','Serina','Treonina','Prolina','Alanina','Glicina','Cisteína']
        amino = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

        "Islas a utilizar en el algoritmo"
        islas=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
        #islas=[21]
        #islas=[5,6,7,8,11,12,15,16,18,20]

        "con que probabilidad va ha haber coperacion entre las islas"
        prob_copera=0.10

        "cantidad de elementos a ser seleccionados para colaborar"
        n_colab=1
        #EDA_tres_capas(pdb_file_path,desc_si_no,continua,numnum,n_pop,n_rep,n_fit,act_prop,act_glob,n_best,pos_min,n_generaciones,n_reinicia,n_genra_reinicio,corte,amino,a,b,local,nueva_ruta)
        #poblacion, best_execution,best_fitness,pdb_select,best_fitness_energ=Algoritmo_evolutivo_energia_MC_descriptor(continua,n,n_pop,n_rep,n_fit,act_prop,act_glob,amino,num_amino,n_best,n_generaciones,pos_min,BB,w1,w2,corte,tokenizer,model,nrep,mat_frec,nombre_sin_extension,n_func,MC_BB,ruta_archivo,numnum,secuencia_ref)
        "cantidad de elementos utilizados para actualizar la red"
        tot_act_red=2000

        "detener cuando se tenga un gdt deceado"
        max_gdt=0.99
        EDA_isla(pdb_file_path,desc_si_no,continua,numnum,n_pop,n_rep,n_fit,act_prop,act_glob,n_best,pos_min,n_generaciones,
                    n_reinicia,n_genra_reinicio,corte,amino,a,b,local,nueva_ruta, islas,prob_copera,n_colab,tot_act_red,max_gdt)
