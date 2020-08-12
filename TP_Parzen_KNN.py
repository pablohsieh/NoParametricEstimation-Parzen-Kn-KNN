# Teoria de deteccion y estimacion - FIUBA 
# 1er cuatrimestre de 2020

# Autor: PABLO HSIEH 
# Padron: 97363

# Estimacion no parametrica - PARZEN, KNN


import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#import scipy.integrate as integrate
#from scipy.integrate import quad
#import math


# Gaussiana ------------------------------------------------------------------
def gaussiana(x,mu,sigma): 
#Es la pdf gaussiana
#Para el caso del error se usa mu=0, sigma=1
    return (np.exp( (-(x-mu)**2)/(2*sigma**2)))*(1/(np.sqrt(2*np.pi)*sigma))

                
# Estimaciones ---------------------------------------------------------------
def ventana(x,h):
# Ventana rectangular que cumple pdf
# Se puede verificar la integral con: quad(ventana,-np.inf,np.inf,args=(h))
    if np.abs(x) <= np.abs(h/2):
        return 1/h
    else:
        return 0

def window(x):
# Ventana rectangular de altura 1 y ancho 1
    if np.abs(x) <= np.abs(1/2):
        return 1
    else:
        return 0
    
def window_gauss(x,h,desvio):
# Ventana gaussiana de media 0 y varianza sigma
# La media es 0 porque si no lo fuera entonces la ventana estaria en cualquier lugar
# Una gaussiana en 3sigma cae a casi 0, se podria usar 4sigma tambien  
# Necesito que la gaussiana este dentro de la ventana de long h, por eso
# 4*sigma = h/2 => sigma = h/8  
    #sigma = h/6
    sigma = h/(2*desvio)    
    if np.abs(x) <= np.abs(1/2):
        return gaussiana(x,0,sigma)
    else:
        return 0

def select_window(gauss,x,h,desvio):
    if gauss == 1:
        return window_gauss(x,h,desvio)
    else:
        return window(x)


def estimacion_parzen(muestras,soporte,h,gauss,desvio):
# pdf estimada en 1d
    N = len(muestras)
    p_hat = np.zeros(len(soporte))
    for x in range(len(soporte)):
        for x_i in muestras:
            #p_hat[x] = p_hat[x] + window((soporte[x]-x_i)/h)/h
            p_hat[x] = p_hat[x] + select_window( gauss, (soporte[x]-x_i)/h , h, desvio)/h
        p_hat[x] = p_hat[x]/N
    return p_hat

    
def vol_k_vecinos(muestras,k,x):
# Obtengo los k vecinos mas cercanos a x
    distancia = abs(muestras - x) # obtengo las distancias entre las muestras y el valor donde estoy
    distancia_ordenada = sorted(distancia) # ordeno las distancias de menor a mayor
    v = distancia_ordenada[k-1] # el k vecino mas cercano esta a distancia v de x
    return v*2
    
def estimacion_kn(muestras,soporte,k):
    N = len(muestras)
    p_hat = np.zeros(len(soporte))
    if k == 1:
        for x in range(len(soporte)):
            p_hat[x] = k/(2*abs(x-vol_k_vecinos(muestras,k,soporte[x])))
        return p_hat
    else:
        for x in range(len(soporte)):
            p_hat[x] = k/(N*vol_k_vecinos(muestras,k,soporte[x]))
        return p_hat


# Funcion que devuelve la imagen en x ---------------------------------------
def f(dominio,imagen,x):
    paso = dominio[1]-dominio[0]
    delta = paso/2
    for i in range(len(dominio)):
        if (x >= dominio[i]-delta and x < dominio[i]+delta):
            index = i
            return imagen[index]




# KNN Clasificacion ---------------------------------------------------------

def agregar_dist_clase(muestra,clase,x):
# Obtengo las distancias de cada punto a la referencia x
# Devuelvo una matriz Nx2: col1=distancias, col2=clase(1 o 0)
    N = len(muestra)
    if clase == 1:
        clase = np.ones(N) 
    else:
        clase = np.zeros(N)
        
    distancia = abs(muestra-x)
    dist_ordenada = sorted(distancia)
    
    aux = np.column_stack((dist_ordenada,clase))
    
    return aux


def es_clase_a_KNN(muestra_a,muestra_b,x,k):
# obtengo array de distancias ordenadas por cercania a x y clase(0 o 1)
    dist_x_a = agregar_dist_clase(muestra_a,0,x)
    dist_x_b = agregar_dist_clase(muestra_b,1,x)
# obtengo un unico array con distancias y clases
    z = np.vstack((dist_x_a, dist_x_b))
# ordeno el array por cercania a punto x
    dist_x = np.array(sorted(z,key=lambda x: x[0]))
# me quedo con la columna de clase
    clases = dist_x[:,1]
#    return clases
# cuento la cantidad de k vecinos de clase = 1, es decir clase b
    suma = sum(clases[0:k])
    if suma >= np.floor(k/2): # si tengo muchos 1 es porque es de clase b
        return 1 # no es de clase a, devuelvo un 1
    else: # suma < k/2, o sea que es de clase = 0, es decir clase a
        return 0 # es de clase a y devuelvo 0
    
    
def sep_clases_KNN(muestra_a,muestra_b,array_muestra,k):
# Devuelvo dos arrays, uno con las muestras de la clase a y otro de la b por regla KNN
    array_clase_a = np.zeros(0)
    array_clase_b = np.zeros(0)
#    print(array_muestra[0])
    for i in range(len(array_muestra)):
        aux = es_clase_a_KNN(muestra_a,muestra_b,array_muestra[i],k)
        if aux == 0: #array_muestra[i] es de clase a
            array_clase_a = np.append(array_clase_a,array_muestra[i])
        else:
            array_clase_b = np.append(array_clase_b,array_muestra[i])
    return array_clase_a,array_clase_b


def sep_clases_agregar_col_KNN(muestra_a,muestra_b,array_muestra,k):
# Devuelvo array_muestra con una columna extra con la clasificacion obtenida
    array_clasif = np.zeros(0)
    for i in range(len(array_muestra)):
       # print(array_muestra)
        clase = es_clase_a_KNN(muestra_a,muestra_b,array_muestra[i][0],k)
        #print(clase)
        array_clasif = np.append(array_clasif,clase)

    aux = np.column_stack((array_muestra,array_clasif))

    return aux


# CLASIFICADOR DICOTOMICO ----------------------------------------------------
# Con las distribuciones estimadas y la probabilidad a priori se arma un 
#clasificador bayesiano dicotomico.
def es_clase_a(p_priori_a,p_estimada_a,p_priori_b,p_estimada_b,muestra,soporte):
    p_a = p_priori_a*f(soporte,p_estimada_a,muestra)
    p_b = p_priori_b*f(soporte,p_estimada_b,muestra)
    #g = p_a - p_b
    if p_a > p_b: # es de clase a
        return 0
    else: #es de clase b
        return 1
        
def sep_clases(p_priori_a,p_estimada_a,p_priori_b,p_estimada_b,array_muestra,soporte):
# Devuelvo dos arrays, uno con las muestras de la clase a y otro de la b
    array_clase_a = np.zeros(0)
    array_clase_b = np.zeros(0)
    for i in range(len(array_muestra)):
        aux = es_clase_a(p_priori_a,p_estimada_a,p_priori_b,p_estimada_b,array_muestra[i],soporte)
        if aux == 0: #array_muestra[i] es de clase a
            array_clase_a = np.append(array_clase_a,array_muestra[i])
        else:
            array_clase_b = np.append(array_clase_b,array_muestra[i])
    return array_clase_a,array_clase_b

def sep_clases_agregar_col(p_priori_a,p_estimada_a,p_priori_b,p_estimada_b,array_muestra,soporte):
# Devuelvo array_muestra con una columna extra con la clasificacion obtenida
    array_clasif = np.zeros(0)
    for i in range(len(array_muestra)):
        clase = es_clase_a(p_priori_a,p_estimada_a,p_priori_b,p_estimada_b,array_muestra[i][0],soporte)
        array_clasif = np.append(array_clasif,clase)

    aux = np.column_stack((array_muestra,array_clasif))

    return aux
        

def agregar_clase(array,clase):
# Agrego una columna extra con la clase de la muestra
    N = len(array)
    if clase == 1:
        clase = np.ones(N) 
    elif clase == 0:
        clase = np.zeros(N)   
    aux = np.column_stack((array,clase))   
    return aux



def graf_histograma(muestras_1,label_1,muestras_2,label_2,n,mu_1,mu_2):
    fig = plt.figure()
    plt.hist(muestras_1,bins='auto',alpha = 0.65,label='F1:'+label_1)
    plt.hist(muestras_2,bins='auto',alpha = 0.65,label='F2:'+label_2)
    #plt.axvline(x=mu_1, color='k', linestyle='dashed',linewidth=0.5)
    #plt.axvline(x=mu_2, color='k', linestyle='dashed',linewidth=0.5)
    plt.xlabel('Soporte')
    plt.ylabel('Cantidad de ocurrencias')
    plt.title('Histograma de '+str(n)+' muestras para F1 y F2.')
    plt.legend()
    plt.show
    return fig

def graf(soporte,p_F1,p_F2,p_teo_F1,p_teo_F2,label,h_k):
    fig = plt.figure()
    plt.plot(soporte,p_F1,color='b',linestyle='dashed',linewidth=1.2,label='F1 estimada')
    plt.plot(soporte,p_F2,color='r',linestyle='dashed',linewidth=1.2,label='F2 estimada')
    plt.plot(soporte,p_teo_F1,color='b',linestyle='solid',linewidth=1,label='F1 teórica')
    plt.plot(soporte,p_teo_F2,color='r',linestyle='solid',linewidth=1,label='F2 teórica')
    plt.xlabel('Soporte')
    plt.ylabel('pdf estimada')
    plt.title('Comparación entre distribución teórica y estimada.\n'+label+str(h_k))
    plt.grid()
    plt.legend()
    plt.show()
    return fig


def graf_puntos_clasif(real_F1,real_F2,clas1_F1,clas2_F1,clas1_F2,clas2_F2,label_title): 
    fig = plt.figure()
    plt.plot(clas1_F1,len(clas1_F1)*[0],'bx',linewidth=1,label='clase F1')
    plt.plot(clas2_F1,len(clas2_F1)*[0],'r.',linewidth=1,label='clase F2')
    plt.plot(real_F1,len(real_F1)*[2],'bx',linewidth=1)#,label='Real F1')
    plt.plot(clas1_F2,len(clas1_F2)*[1],'bx',linewidth=1)#,label='clase F1')
    plt.plot(clas2_F2,len(clas2_F2)*[1],'r.',linewidth=1)#,label='clase F2')
    plt.plot(real_F2,len(real_F2)*[2],'r.',linewidth=1)#,label='Real F2')
    
    # Recuadro del muestras
    naranja = '#f5be58'
    plt.axhspan(1.9, 2.1, alpha=0.35, color = naranja,label = 'Muestras a clasificar')
    plt.axhspan(-0.1, 0.1, alpha=0.15, color = 'b' ,label = 'Clasificación de las provenientes de F1')
    plt.axhspan(0.9, 1.1, alpha=0.15, color = 'r' ,label = 'Clasificación de las provenientes de F2')
    
    plt.title(label_title)
    plt.grid()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=2)
    plt.show()
    return fig


def cant_miss(muestra):
    cant = 0
    for i in range(len(muestra)):
        if muestra[i][2] != muestra[i][1]:
            cant = cant + 1
    return cant

def hacer_clasificacion(imprimir, papw1, pF1, papw2, pF2, sample_F1test, sample_F2test, soport, h_k, label_estimacion,label_impresion):
# ---- Clasificacion
# Utilizo las muestras de prueba para clasificar con las densidades estimadas
#    n_test = len(sample_F1test) + len(sample_F2test)    

    test_class_1_F1, test_class_2_F1 = sep_clases(papw1, pF1, papw2, pF2, sample_F1test[:,0],soport)
    test_class_1_F2, test_class_2_F2 = sep_clases(papw1, pF1, papw2, pF2, sample_F2test[:,0],soport)

    muestras_1 = sep_clases_agregar_col(papw1, pF1, papw2, pF2, sample_F1test,soport)
    muestras_2 = sep_clases_agregar_col(papw1, pF1, papw2, pF2, sample_F2test,soport)
    
    cant_miss_F1 = cant_miss(muestras_1)
    cant_miss_F2 = cant_miss(muestras_2)
        
    err_clasif_F1 = cant_miss_F1 / len(muestras_1) 
    err_clasif_F2 = cant_miss_F2 / len(muestras_2) 
    
    err_clasif_F1 = round(err_clasif_F1, 4)
    err_clasif_F2 = round(err_clasif_F2, 4)
    
    label_titulo = str('Clasificación de ')+str(n_test)+str(' muestras usando las distribuciones estimadas.\n')+label_estimacion+str(h_k)+str('.\nError de clasificación: Clase F1=')+str(err_clasif_F1)+str(', Clase F2=')+str(err_clasif_F2)

    fig_graf = graf_puntos_clasif(sample_F1test[:,0],sample_F2test[:,0],test_class_1_F1,test_class_2_F1,test_class_1_F2,test_class_2_F2,label_titulo)
    if imprimir == 1:
        output_filename = 'fig_d-Muestras-Clasif-' + label_impresion + '.png'
        fig_graf.savefig(output_filename,bbox_inches='tight')
 

##############################################################################   
######### MAIN ###############################################################   
##############################################################################  


pap_w1 = 0.4
pap_w2 = 0.6

F1_mu = 1
F1_sigma = 4
F2_mu = 4
F2_sigma = 4

n=10**4
n_test=10**2
#n=100
#n_test=10

imprimir = 0

label_F1 = str('Normal(1,4)')
label_F2 = str('Normal(4,4)')
label_parzen = str('Estimación utilizando ventanas de Parzen')
#label_window_gauss = str(', Gaussiana(0,h/4) h=')
label_window_rect = str(', Rectangular h=')
label_kn = str('Estimación utilizando Kn vecinos más cercanos, k= ')
#x=np.zeros(n)

# Generacion de n muestras normales de media mu y varianza sigma
#sample = np.random.normal(mu,sigma,n)
sample_F1 = np.random.normal(F1_mu,F1_sigma,n)
sample_F2 = np.random.normal(F2_mu,F2_sigma,n)

# Generacion de muestras a clasificar
#sample_F1_test = np.random.normal(F1_mu,F1_sigma,n_test)
#sample_F2_test = np.random.normal(F2_mu,F2_sigma,n_test)

sample_F1_test = np.random.normal(F1_mu,F1_sigma,40)
sample_F2_test = np.random.normal(F2_mu,F2_sigma,60)

sample_F1_test = agregar_clase(sample_F1_test,0)
sample_F2_test = agregar_clase(sample_F2_test,1)


fig_graf = graf_histograma(sample_F1, label_F1, sample_F2, label_F2,n,F1_mu,F2_mu)
if imprimir == 1:
    fig_graf.savefig('fig_a-Histograma.png',bbox_inches='tight')


# El soporte va a ser 0 para valores menores a (muestra_min - h/2) y mayores a
# (muestra_min + h/2)
# Para el ejercicio son dos gaussianas de varianza 4 y media 1 y 4, no deberia
# tener nada fuera de un soporte (-15;20)
soporte_minimo = -15
soporte_maximo = 20
#soporte_minimo =  np.floor(min(sample_F1)-h/2)
#soporte_maximo = np.ceil(max(sample_F2)+h/2)
soporte = np.arange(soporte_minimo,soporte_maximo,step=0.1)


h = np.array([115/np.sqrt(n), #h=1.15
              100/np.sqrt(n), #h=1
              80/np.sqrt(n),  #h=0.8
              50/np.sqrt(n),  #h=0.5
              25/np.sqrt(n)]) #h=0.25
    
    
for i in range(len(h)):
# PARZEN ---- Estimacion con ventana GAUSSIANA ------ PARZEN ------- PARZEN --

##### sigma = h/4
    label_window_gauss = str(', Gaussiana(0,h/4) h=')
    label_imprimir = 'Parzen_winGauss(sigma=h4)_h=' + str(h[i]) 
        
    p_F1 = estimacion_parzen(sample_F1,soporte,h[i],1,2)
    p_F2 = estimacion_parzen(sample_F2,soporte,h[i],1,2)

    fig_graf = graf(soporte,p_F1,p_F2,gaussiana(soporte,F1_mu,F1_sigma),gaussiana(soporte,F2_mu,F2_sigma),label_parzen+label_window_gauss,h[i])
    if imprimir == 1:
        output_filename = 'fig_b-' + label_imprimir + '.png'
        fig_graf.savefig(output_filename,bbox_inches='tight')
        
# ---- Clasificacion
    label_estimacion = label_parzen+label_window_gauss
    hacer_clasificacion(imprimir,pap_w1, p_F1, pap_w2, p_F2, sample_F1_test, sample_F2_test, soporte, h[i], label_estimacion, label_imprimir)


##### sigma = h/2
    label_window_gauss = str(', Gaussiana(0,h/2) h=')
    label_imprimir = 'Parzen_winGauss(sigma=h2)_h=' + str(h[i]) 
        
    p_F1 = estimacion_parzen(sample_F1,soporte,h[i],1,1)
    p_F2 = estimacion_parzen(sample_F2,soporte,h[i],1,1)

    fig_graf = graf(soporte,p_F1,p_F2,gaussiana(soporte,F1_mu,F1_sigma),gaussiana(soporte,F2_mu,F2_sigma),label_parzen+label_window_gauss,h[i])
    if imprimir == 1:
        output_filename = 'fig_b-' + label_imprimir + '.png'
        fig_graf.savefig(output_filename,bbox_inches='tight')

# ---- Clasificacion
# Utilizo las muestras de prueba para clasificar con las densidades estimadas
    label_estimacion = label_parzen+label_window_gauss
    hacer_clasificacion(imprimir,pap_w1, p_F1, pap_w2, p_F2, sample_F1_test, sample_F2_test, soporte, h[i], label_estimacion, label_imprimir)


##### sigma = h/8
    label_window_gauss = str(', Gaussiana(0,h/8) h=')
    label_imprimir = 'Parzen_winGauss(sigma=h8)_h=' + str(h[i])  
    
    p_F1 = estimacion_parzen(sample_F1,soporte,h[i],1,4)
    p_F2 = estimacion_parzen(sample_F2,soporte,h[i],1,4)

    fig_graf = graf(soporte,p_F1,p_F2,gaussiana(soporte,F1_mu,F1_sigma),gaussiana(soporte,F2_mu,F2_sigma),label_parzen+label_window_gauss,h[i])
    if imprimir == 1:
        output_filename = 'fig_b-' + label_imprimir + '.png'
        fig_graf.savefig(output_filename,bbox_inches='tight')

# ---- Clasificacion
    label_estimacion = label_parzen+label_window_gauss
    hacer_clasificacion(imprimir,pap_w1, p_F1, pap_w2, p_F2, sample_F1_test, sample_F2_test, soporte, h[i], label_estimacion, label_imprimir)



# PARZEN ---- Estimacion con ventana RECTANGULAR ------- PARZEN ------- PARZEN
    p_F1 = estimacion_parzen(sample_F1,soporte,h[i],0,2) #el 2 no importa aca
    p_F2 = estimacion_parzen(sample_F2,soporte,h[i],0,2)

    fig_graf = graf(soporte,p_F1,p_F2,gaussiana(soporte,F1_mu,F1_sigma),gaussiana(soporte,F2_mu,F2_sigma),label_parzen+label_window_rect,h[i])
    if imprimir == 1:
        output_filename = 'fig_b-Parzen_winRect_h=' + str(h[i]) + '.png'
        fig_graf.savefig(output_filename,bbox_inches='tight')

# ---- Clasificacion
# Utilizo las muestras de prueba para clasificar con las densidades estimadas
    label_estimacion = label_parzen+label_window_rect
    label_imprimir = 'Parzen_winRect_h=' + str(h[i])
    hacer_clasificacion(imprimir,pap_w1, p_F1, pap_w2, p_F2, sample_F1_test, sample_F2_test, soporte, h[i], label_estimacion, label_imprimir)



# Kn ---- Estimacion con k vecinos ---- Kn ---- Kn ---- Kn ---- Kn ---- Kn ---

k=np.array([1,10,50,100])

for i in range(len(k)):
    p_F1 = estimacion_kn(sample_F1,soporte,k[i])
    p_F2 = estimacion_kn(sample_F2,soporte,k[i])
    fig_graf = graf(soporte,p_F1,p_F2,gaussiana(soporte,F1_mu,F1_sigma),gaussiana(soporte,F2_mu,F2_sigma),label_kn,k[i])
    if imprimir == 1:
        output_filename = 'fig_c-KnVecinos_k=' + str(k[i]) + '.png'
        fig_graf.savefig(output_filename,bbox_inches='tight')
    
    # ---- Clasificacion
    # Utilizo las muestras de prueba para clasificar con las densidades estimadas  
    label_imprimir = 'Kn_vecinos_k='+ str(k[i])
    hacer_clasificacion(imprimir,pap_w1, p_F1, pap_w2, p_F2, sample_F1_test, sample_F2_test, soporte, h[i], label_kn, label_imprimir)



# ---- Clasificacion KNN ----
k=np.array([1,11,51])

for i in range(len(k)):
    # ---- Clasificacion KNN
    # Utilizo las muestras de prueba para clasificar con las densidades estimadas
    test_clase_1_F1, test_clase_2_F1 = sep_clases_KNN(sample_F1,sample_F2,sample_F1_test[:,0],k[i])
    test_clase_1_F2, test_clase_2_F2 = sep_clases_KNN(sample_F1,sample_F2,sample_F2_test[:,0],k[i])

    muestra_1_clasif = sep_clases_agregar_col_KNN(sample_F1,sample_F2,sample_F1_test,k[i])
    muestra_2_clasif = sep_clases_agregar_col_KNN(sample_F1,sample_F2,sample_F2_test,k[i])
    
    cant_miss_F1 = cant_miss(muestra_1_clasif)
    cant_miss_F2 = cant_miss(muestra_2_clasif)
    
    error_clasif_F1  = cant_miss_F1 / len(muestra_1_clasif)
    error_clasif_F2  = cant_miss_F2 / len(muestra_2_clasif)

    error_clasif_F1 = round(error_clasif_F1, 4)
    error_clasif_F2 = round(error_clasif_F2, 4)

    label_title = str('Clasificación de ')+str(n_test)+str(' muestras usando KNN, k=')+str(k[i])+str('.\nError de clasificación: Clase F1=')+str(error_clasif_F1)+str(', Clase F2=')+str(error_clasif_F2)
    fig_graf = graf_puntos_clasif(sample_F1_test[:,0],sample_F2_test[:,0],test_clase_1_F1,test_clase_2_F1,test_clase_1_F2,test_clase_2_F2,label_title)
    if imprimir == 1:
        output_filename = 'fig_e-Muestras-Clasif-KNN_k=' + str(k[i]) + '.png'
        fig_graf.savefig(output_filename,bbox_inches='tight')
        
        


