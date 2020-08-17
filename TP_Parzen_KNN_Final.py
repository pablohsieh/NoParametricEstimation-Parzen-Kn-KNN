# Teoria de deteccion y estimacion - FIUBA 
# 1er cuatrimestre de 2020

# Autor: PABLO HSIEH 
# Padron: 97363

# Estimacion no parametrica - PARZEN, KN, Regla de desicion KNN


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Gaussiana ------------------------------------------------------------------
def gaussiana(x,mu,sigma): 
#Es la pdf gaussiana
#Para el caso del error se usa mu=0, sigma=1
    return (np.exp( (-(x-mu)**2)/(2*sigma**2)))*(1/(np.sqrt(2*np.pi)*sigma))

def p_error(lim_inf,lim_sup,mu,sigma): # Probabilidad de error integrando pdf gaussiana
# el lim inferior es la dist mahalanobis/2
    return quad(gaussiana,lim_inf,lim_sup,args=(mu,sigma))
                
# Estimacion ------ PARZEN ---------------------------------------------------
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
# La media es 0 porque se centra la ventana sobre la muestra de entrenamiento
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
# funcion para seleccionar la ventana a utilizar: gaussiana o rectangular
    if gauss == 1:
        return window_gauss(x,h,desvio)
    else:
        return window(x)

def estimacion_parzen(muestras,soporte,h,gauss,desvio):
# pdf estimada en 1d por ventana de parzen
    N = len(muestras)
    p_hat = np.zeros(len(soporte))
    for x in range(len(soporte)):
        for x_i in muestras:
            #p_hat[x] = p_hat[x] + window((soporte[x]-x_i)/h)/h
            p_hat[x] = p_hat[x] + select_window( gauss, (soporte[x]-x_i)/h , 
                                                h, desvio)/h
        p_hat[x] = p_hat[x]/N
    return p_hat

# Estimacion ------ Kn vecinos -----------------------------------------------
def vol_k_vecinos(muestras,k,x):
# Obtengo los k vecinos mas cercanos a x
    distancia = abs(muestras - x) # obtengo las distancias entre las muestras 
                                  # y el valor donde estoy
    distancia_ordenada = sorted(distancia) #distancias de menor a mayor
    v = distancia_ordenada[k-1] #k vecino mas cercano esta a distancia v de x
    return v*2
    
def estimacion_kn(muestras,soporte,k):
# se realiza la estimacion por kn vecinos
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
#se modela que la imagen vale lo mismo desde dominio-delta hasta dominio+delta
    paso = dominio[1]-dominio[0]
    delta = paso/2
    for i in range(len(dominio)):
        if (x >= dominio[i]-delta and x < dominio[i]+delta):
            index = i
            return imagen[index]


# KNN Clasificacion ---------------------------------------------------------
def agregar_dist_clase(muestra,clase,x):
#Obtengo las distancias ordenadas por cercania de cada punto a la referencia x
# Devuelvo una matriz Nx2: col1=distancias, col2=clase(1 o 0)
    N = len(muestra)
    if clase == 1:
        clase = np.ones(N) 
    else:
        clase = np.zeros(N)
# se obtiene la distancia en modulo de cada muestra de entrenamiento hasta la
# x a clasificar
    distancia = abs(muestra-x)
# se ordena a las muestras de entrenamiento por cercania a la x a clasificar
    dist_ordenada = sorted(distancia)
# se agrega una columna con la clase de las muestras de entrenamiento
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
# cuento la cantidad de k vecinos de clase = 1, es decir clase b
    suma = sum(clases[0:k])
    if suma >= np.floor(k/2): # si tengo muchos 1 es porque es de clase b
        return 1 # no es de clase a, devuelvo un 1
    else: # suma < k/2, o sea que es de clase = 0, es decir clase a
        return 0 # es de clase a y devuelvo 0
    
def sep_clases_KNN(muestra_a,muestra_b,array_muestra,k):
# Devuelvo dos arrays, uno con las muestras de la clase a 
# y otro de las de clase b por regla KNN
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
        clase = es_clase_a_KNN(muestra_a,muestra_b,array_muestra[i][0],k)
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
        
def sep_clases(p_priori_a,p_estimada_a,p_priori_b,p_estimada_b,
               array_muestra,soporte):
# Devuelvo dos arrays, uno con las muestras de la clase a y otro de la b
    array_clase_a = np.zeros(0)
    array_clase_b = np.zeros(0)
    for i in range(len(array_muestra)):
        aux = es_clase_a(p_priori_a,p_estimada_a,p_priori_b,p_estimada_b,
                         array_muestra[i],soporte)
        if aux == 0: #array_muestra[i] es de clase a
            array_clase_a = np.append(array_clase_a,array_muestra[i])
        else:
            array_clase_b = np.append(array_clase_b,array_muestra[i])
    return array_clase_a,array_clase_b

def sep_clases_agregar_col(p_priori_a,p_estimada_a,p_priori_b,p_estimada_b,
                           array_muestra,soporte):
# Devuelvo array_muestra con una columna extra con la clasificacion obtenida
    array_clasif = np.zeros(0)
    for i in range(len(array_muestra)):
        clase = es_clase_a(p_priori_a,p_estimada_a,p_priori_b,p_estimada_b,
                           array_muestra[i][0],soporte)
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


def cant_miss(muestra):
# Se obtiene la cantidad de etiquetas diferentes, 
# es decir la cantidad de muestras clasificadas erroneamente
    cant = 0
    for i in range(len(muestra)):
        if muestra[i][2] != muestra[i][1]:
            cant = cant + 1
    return cant

def hacer_clasificacion(imprimir, papw1, pF1, papw2, pF2, 
                        sample_F1test, sample_F2test, soport, h_k, 
                        label_estimacion,label_impresion,bound):
# ---- Clasificacion
# Utilizo las muestras de prueba para clasificar con las densidades estimadas 
# Se separan las muestras clasificandolas
    test_class_1_F1, test_class_2_F1 = sep_clases(papw1, pF1, papw2, pF2, 
                                                  sample_F1test[:,0],soport)
    test_class_1_F2, test_class_2_F2 = sep_clases(papw1, pF1, papw2, pF2, 
                                                  sample_F2test[:,0],soport)
# Se clasifican las muestras poniendole una etiqueta de clasificacion
    muestras_1 = sep_clases_agregar_col(papw1, pF1, papw2, pF2, 
                                        sample_F1test,soport)
    muestras_2 = sep_clases_agregar_col(papw1, pF1, papw2, pF2, 
                                        sample_F2test,soport)
# Se calcula la cantidad de muestras que fueron mal clasificadas
    cant_miss_F1 = cant_miss(muestras_1)
    cant_miss_F2 = cant_miss(muestras_2)
# Se obtiene erl error de clasificacion
    err_clasif_F1 = cant_miss_F1 / len(muestras_1) 
    err_clasif_F2 = cant_miss_F2 / len(muestras_2)
# Se redondea el error de clasif con 4 cifras significativas
    err_clasif_F1 = round(err_clasif_F1, 4) 
    err_clasif_F2 = round(err_clasif_F2, 4)
    
    label_titulo = str('Clasificación de ')+str(n_test)+str(' muestras usando las distribuciones estimadas.\n')+label_estimacion+str(h_k)+str('.\nError de clasificación: Clase F1=')+str(err_clasif_F1)+str(', Clase F2=')+str(err_clasif_F2)
    
    fig_graf = graf_puntos_clasif(sample_F1test[:,0],sample_F2test[:,0],
                                  test_class_1_F1,test_class_2_F1,
                                  test_class_1_F2,test_class_2_F2,
                                  label_titulo,bound)
    if imprimir == 1:
        output_filename = 'fig_d-Muestras-Clasif-' + label_impresion + '.png'
        fig_graf.savefig(output_filename,bbox_inches='tight')


# Graficos -------------------------------------------------------------------
def graf_gaussianas(soporte,p_teo_F1,p_teo_F2,bound):
# Se grafica la distribucion real y la estimada
# p_teo_F1 y p_teo_F2 son las distribuciones reales teoricas
    fig = plt.figure()
    plt.plot(soporte,p_teo_F1,color='b',linestyle='solid',linewidth=1,
             label='F1 teórica: N(1,4)')
    plt.plot(soporte,p_teo_F2,color='r',linestyle='solid',linewidth=1,
             label='F2 teórica: N(4,4)')
    plt.axvline(x=bound, color='k', linestyle='dashed',linewidth=1.2,
                label = 'Región de desición, x_B='+str(round(bound,6)))
    plt.xlabel('Soporte')
    plt.ylabel('Distribución')
    plt.title('Distribuciones teóricas y región de desición.')
    plt.grid()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=2)
    plt.show()
    return fig

def graf_histograma(muestras_1,label_1,muestras_2,label_2,n,mu_1,mu_2):
# Se grafica el histograma standard de las muestras generaddas
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
# Se grafica la distribucion real y la estimada
# p_teo_F1 y p_teo_F2 son las distribuciones reales teoricas
# p_F1 y p_F2 son las distribuciones estimadas
# h_k es el valor de longitud de ventana h o cant de vecinos kn usados
    fig = plt.figure()
    plt.plot(soporte,p_F1,color='b',linestyle='dashed',linewidth=1.2,
             label='F1 estimada')
    plt.plot(soporte,p_F2,color='r',linestyle='dashed',linewidth=1.2,
             label='F2 estimada')
    plt.plot(soporte,p_teo_F1,color='b',linestyle='solid',linewidth=1,
             label='F1 teórica')
    plt.plot(soporte,p_teo_F2,color='r',linestyle='solid',linewidth=1,
             label='F2 teórica')
    plt.xlabel('Soporte')
    plt.ylabel('pdf estimada')
    plt.title('Comparación entre distribución teórica y estimada.\n'
              +label+str(h_k))
    plt.grid()
    plt.legend()
    plt.show()
    return fig

def graf_puntos_clasif(real_F1,real_F2,clas1_F1,clas2_F1,clas1_F2,clas2_F2,
                       label_title,bound): 
# Se grafican las muestras a clasificar y clasificadas
# real_F1 y real_F2 son las muestras a clasificar provenientes de F1 y F2
# Clas1_F1 son las muestras de F1 clasificadas como F1
# Clas2_F1 son las muestras de F1 clasificadas como F2
# Clas1_F2 son las muestras de F2 clasificadas como F1
# Clas2_F2 son las muestras de F2 clasificadas como F2
    fig = plt.figure()
    plt.plot(clas1_F1,len(clas1_F1)*[0],'bx',linewidth=1,label='clase F1') 
    plt.plot(clas2_F1,len(clas2_F1)*[0],'r.',linewidth=1,label='clase F2')
    plt.plot(real_F1,len(real_F1)*[2],'bx',linewidth=1) 
    plt.plot(clas1_F2,len(clas1_F2)*[1],'bx',linewidth=1) 
    plt.plot(clas2_F2,len(clas2_F2)*[1],'r.',linewidth=1) 
    plt.plot(real_F2,len(real_F2)*[2],'r.',linewidth=1) 
    plt.axvline(x=bound, color='k', linestyle='dashed',linewidth=1.2,
                label = 'Región de desición, x_B='+str(round(bound,6)))
# Recuadro del muestras
    naranja = '#f5be58'
    plt.axhspan(1.9, 2.1, alpha=0.35, color = naranja,
                label = 'Muestras a clasificar')
    plt.axhspan(-0.1, 0.1, alpha=0.15, color = 'b' ,
                label = 'Clasificación de las provenientes de F1')
    plt.axhspan(0.9, 1.1, alpha=0.15, color = 'r' ,
                label = 'Clasificación de las provenientes de F2')
    plt.title(label_title)
    plt.grid()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=2)
    plt.show()
    return fig

def graficar_ventanas(h):
# Grafico las ventanas de parzen gaussiana con desvio h,h/2,h/4,h/8 
# y rectuangular de long h          
    soporte = np.arange(-h/2,h/2,step=0.01)
    y1 = gaussiana(soporte,0,h/1)
    y2 = gaussiana(soporte,0,h/2)
    y6 = gaussiana(soporte,0,h/6)
    y4 = gaussiana(soporte,0,h/4)
    y8 = gaussiana(soporte,0,h/8)
    
    fig = plt.figure()
    plt.plot(soporte, y1, linewidth=1.2,label='N(0,h^2)')
    plt.plot(soporte, y2, linewidth=1.2,label='N(0,(h/2)^2)')
    plt.plot(soporte, y4, linewidth=1.2,label='N(0,(h/4)^2)')
    plt.plot(soporte, y6, linewidth=1.2,label='N(0,(h/6)^2)')    
    plt.plot(soporte, y8, linewidth=1.2,label='N(0,(h/8)^2)')
    plt.plot((-h/2,h/2), (1,1), 'k-',linewidth=1.2, label='Rectangular h')
    plt.plot((-h/2,-h/2), (0,1), 'k--',linewidth=1.2)
    plt.plot((h/2,h/2), (0,1), 'k--',linewidth=1.2)
    plt.xlabel('Soporte')
    plt.ylabel('Amplitud de ventana')
    plt.title('Comparación entre ventanas utilizadas, h=' + str(h))
    plt.grid()
    plt.legend()
    plt.show()
    return fig


##############################################################################   
######### MAIN ###############################################################   
##############################################################################  


pap_w1 = 0.4 #probabilidad a priori de la clase 1
pap_w2 = 0.6 #probabilidad a priori de la clase 2

F1_mu = 1 #Media de la distribucion 1 
F1_sigma = 4 #Varianza de la distribucion 1
F2_mu = 4 #Media de la distribucion 2
F2_sigma = 4 #Varianza de la distribucion 2

# El soporte va a ser 0 para valores menores a (muestra_min - h/2) y mayores a
# (muestra_min + h/2)
# Para el ejercicio son dos gaussianas de varianza 4 y media 1 y 4, no deberia
# tener nada fuera de un soporte (-15;20)
soporte_minimo = -15
soporte_maximo = 20
#soporte_minimo =  np.floor(min(sample_F1)-h/2)
#soporte_maximo = np.ceil(max(sample_F2)+h/2)
soporte = np.arange(soporte_minimo,soporte_maximo,step=0.1)

n=10**4  #cantidad de muestras de entrenamiento
n_test=10**2  #cantidad de muestras a clasificar

# Para imprimir o exportar las figuras obtenidas: imprimir = 1, sino 0
imprimir = 1

# Grafico las ventanas dependiendo del desvio, h=1 para posterior analisis
#fig_graf = graficar_ventanas(1)
#if imprimir == 1:
#    fig_graf.savefig('fig_0-compar_ventanas.png',bbox_inches='tight')

# Etiquetas utilizadas para imprimir o titulos
label_F1 = str('Normal(1,4)')
label_F2 = str('Normal(4,4)')
label_parzen = str('Estimación utilizando ventanas de Parzen')
label_window_rect = str(', Rectangular h=')
label_kn = str('Estimación utilizando Kn vecinos más cercanos, k= ')

# Generacion de n muestras normales de media mu y varianza sigma
#sample = np.random.normal(mu,sigma,n)
sample_F1 = np.random.normal(F1_mu,F1_sigma,n)
sample_F2 = np.random.normal(F2_mu,F2_sigma,n)

# Generacion de muestras a clasificar
sample_F1_test = np.random.normal(F1_mu,F1_sigma,40) #genero con P(w1)=0.4
sample_F2_test = np.random.normal(F2_mu,F2_sigma,60) #genero con P(w2)=0.6

sample_F1_test = agregar_clase(sample_F1_test,0) #La clase w1 -> columna de 0
sample_F2_test = agregar_clase(sample_F2_test,1) #La clase w2 -> columna de 1

bound = (np.log(pap_w2)-np.log(pap_w1)+1/8+2)*(4/5)
#bound = 2.024372
fig_graf = graf_gaussianas(soporte,gaussiana(soporte,F1_mu,F1_sigma),
                           gaussiana(soporte,F2_mu,F2_sigma),bound)
#if imprimir == 1:
#    fig_graf.savefig('fig_0-gaussianas_boundary.png',bbox_inches='tight')

#obtencion del error teorico
error_x_dadoF1 = p_error(-np.inf,bound,F1_mu,np.sqrt(F1_sigma))
error_F1 = error_x_dadoF1[0] * pap_w1
print(round(error_F1,6)) #Esto es el error de clasificar la muestra como w2 cuando era w1
error_x_dadoF2 = p_error(bound,np.inf,F2_mu,np.sqrt(F2_sigma))
error_F2 = error_x_dadoF2[0] * pap_w2
print(round(error_F2,6)) #Error de clasificar como w1 cuando era w2

fig_graf = graf_histograma(sample_F1,label_F1,sample_F2,label_F2,n,F1_mu,F2_mu)
if imprimir == 1:
    fig_graf.savefig('fig_a-Histograma.png',bbox_inches='tight')


## Se analizo la utilizacion de varios h, se opto por utilizar h entre 0.8 y 1
#h = np.array([115/np.sqrt(n),100/np.sqrt(n),80/np.sqrt(n),
#              50/np.sqrt(n),25/np.sqrt(n)]) #h=1.15,1,0.8,0.5,0.25
#h = np.array([95/np.sqrt(n), 90/np.sqrt(n), 85/np.sqrt(n)]) #h=0.95,0.9,0.85   
#h = np.array([120/np.sqrt(n), 110/np.sqrt(n)]) #h=1.2,1.1   
h = np.array([110/np.sqrt(n)]) 
   
for i in range(len(h)):
# PARZEN ---- Estimacion con ventana GAUSSIANA ------ PARZEN ------- PARZEN --

##### sigma = h/6
    label_window_gauss = str(', Gaussiana(0,(h/6)^2) h=')
    label_imprimir = 'Parzen_winGauss(sigma=h6)_h=' + str(h[i]) 
        
    p_F1 = estimacion_parzen(sample_F1,soporte,h[i],1,3)
    p_F2 = estimacion_parzen(sample_F2,soporte,h[i],1,3)

    fig_graf = graf(soporte,p_F1,p_F2,gaussiana(soporte,F1_mu,F1_sigma),
                    gaussiana(soporte,F2_mu,F2_sigma),
                    label_parzen+label_window_gauss,h[i])
    if imprimir == 1:
        output_filename = 'fig_b-' + label_imprimir + '.png'
        fig_graf.savefig(output_filename,bbox_inches='tight')
        
# ---- Clasificacion
    label_estimacion = label_parzen+label_window_gauss
    hacer_clasificacion(imprimir,pap_w1, p_F1, pap_w2, p_F2, 
                        sample_F1_test, sample_F2_test, soporte, h[i], 
                        label_estimacion, label_imprimir,bound)


## Se analizaron los resultados y se vio que lo mejor es sigma = h/6
## Entonces todo lo que sigue va comentado

##### sigma = h/2
#    label_window_gauss = str(', Gaussiana(0,(h/2)^2) h=')
#    label_imprimir = 'Parzen_winGauss(sigma=h2)_h=' + str(h[i]) 
        
#    p_F1 = estimacion_parzen(sample_F1,soporte,h[i],1,1)
#    p_F2 = estimacion_parzen(sample_F2,soporte,h[i],1,1)

#    fig_graf = graf(soporte,p_F1,p_F2,gaussiana(soporte,F1_mu,F1_sigma),gaussiana(soporte,F2_mu,F2_sigma),label_parzen+label_window_gauss,h[i])
#    if imprimir == 1:
#        output_filename = 'fig_b-' + label_imprimir + '.png'
#        fig_graf.savefig(output_filename,bbox_inches='tight')

# ---- Clasificacion
# Utilizo las muestras de prueba para clasificar con las densidades estimadas
#    label_estimacion = label_parzen+label_window_gauss
#    hacer_clasificacion(imprimir,pap_w1, p_F1, pap_w2, p_F2, sample_F1_test, sample_F2_test, soporte, h[i], label_estimacion, label_imprimir,bound)

##### sigma = h/4
#    label_window_gauss = str(', Gaussiana(0,(h/4)^2) h=')
#    label_imprimir = 'Parzen_winGauss(sigma=h4)_h=' + str(h[i]) 
        
#    p_F1 = estimacion_parzen(sample_F1,soporte,h[i],1,2)
#    p_F2 = estimacion_parzen(sample_F2,soporte,h[i],1,2)

#    fig_graf = graf(soporte,p_F1,p_F2,gaussiana(soporte,F1_mu,F1_sigma),gaussiana(soporte,F2_mu,F2_sigma),label_parzen+label_window_gauss,h[i])
#    if imprimir == 1:
#        output_filename = 'fig_b-' + label_imprimir + '.png'
#        fig_graf.savefig(output_filename,bbox_inches='tight')
        
# ---- Clasificacion
#    label_estimacion = label_parzen+label_window_gauss
#    hacer_clasificacion(imprimir,pap_w1, p_F1, pap_w2, p_F2, sample_F1_test, sample_F2_test, soporte, h[i], label_estimacion, label_imprimir, bound)


##### sigma = h/8
#    label_window_gauss = str(', Gaussiana(0,(h/8)^2) h=')
#    label_imprimir = 'Parzen_winGauss(sigma=h8)_h=' + str(h[i])  
    
#    p_F1 = estimacion_parzen(sample_F1,soporte,h[i],1,4)
#    p_F2 = estimacion_parzen(sample_F2,soporte,h[i],1,4)

#    fig_graf = graf(soporte,p_F1,p_F2,gaussiana(soporte,F1_mu,F1_sigma),gaussiana(soporte,F2_mu,F2_sigma),label_parzen+label_window_gauss,h[i])
#    if imprimir == 1:
#        output_filename = 'fig_b-' + label_imprimir + '.png'
#        fig_graf.savefig(output_filename,bbox_inches='tight')

# ---- Clasificacion
#    label_estimacion = label_parzen+label_window_gauss
#    hacer_clasificacion(imprimir,pap_w1, p_F1, pap_w2, p_F2, sample_F1_test, sample_F2_test, soporte, h[i], label_estimacion, label_imprimir, bound)



# PARZEN ---- Estimacion con ventana RECTANGULAR ------- PARZEN ------- PARZEN
    p_F1 = estimacion_parzen(sample_F1,soporte,h[i],0,2) #el 2 no importa aca
    p_F2 = estimacion_parzen(sample_F2,soporte,h[i],0,2)

    fig_graf = graf(soporte,p_F1,p_F2,gaussiana(soporte,F1_mu,F1_sigma),
                    gaussiana(soporte,F2_mu,F2_sigma),
                    label_parzen+label_window_rect,h[i])
    if imprimir == 1:
        output_filename = 'fig_b-Parzen_winRect_h=' + str(h[i]) + '.png'
        fig_graf.savefig(output_filename,bbox_inches='tight')

# ---- Clasificacion
# Utilizo las muestras de prueba para clasificar con las densidades estimadas
    label_estimacion = label_parzen+label_window_rect
    label_imprimir = 'Parzen_winRect_h=' + str(h[i])
    hacer_clasificacion(imprimir, pap_w1, p_F1, pap_w2, p_F2,
                        sample_F1_test, sample_F2_test, soporte, h[i],
                        label_estimacion, label_imprimir, bound)


# Kn ---- Estimacion con k vecinos ---- Kn ---- Kn ---- Kn ---- Kn ---- Kn ---


k=np.array([1,10,50,100])

for i in range(len(k)):
    p_F1 = estimacion_kn(sample_F1,soporte,k[i])
    p_F2 = estimacion_kn(sample_F2,soporte,k[i])
    fig_graf = graf(soporte,p_F1,p_F2,gaussiana(soporte,F1_mu,F1_sigma),
                    gaussiana(soporte,F2_mu,F2_sigma),label_kn,k[i])
    if imprimir == 1:
        output_filename = 'fig_c-KnVecinos_k=' + str(k[i]) + '.png'
        fig_graf.savefig(output_filename,bbox_inches='tight')
    
# ---- Clasificacion
# Utilizo las muestras de prueba para clasificar con las densidades estimadas  
    label_imprimir = 'Kn_vecinos_k='+ str(k[i])
    hacer_clasificacion(imprimir,pap_w1, p_F1, pap_w2, p_F2, 
                        sample_F1_test, sample_F2_test, soporte, k[i], 
                        label_kn, label_imprimir, bound)



# ---- Clasificacion KNN ----
k=np.array([1,11,51])

for i in range(len(k)):
# ---- Clasificacion KNN
# Utilizo las muestras de prueba para clasificar con las densidades estimadas
    test_clase_1_F1, test_clase_2_F1 = sep_clases_KNN(sample_F1,sample_F2,
                                                      sample_F1_test[:,0],k[i])
    test_clase_1_F2, test_clase_2_F2 = sep_clases_KNN(sample_F1,sample_F2,
                                                      sample_F2_test[:,0],k[i])

    muestra_1_clasif = sep_clases_agregar_col_KNN(sample_F1,sample_F2,
                                                  sample_F1_test,k[i])
    muestra_2_clasif = sep_clases_agregar_col_KNN(sample_F1,sample_F2,
                                                  sample_F2_test,k[i])
    
    cant_miss_F1 = cant_miss(muestra_1_clasif)
    cant_miss_F2 = cant_miss(muestra_2_clasif)
    
    error_clasif_F1  = cant_miss_F1 / len(muestra_1_clasif)
    error_clasif_F2  = cant_miss_F2 / len(muestra_2_clasif)

    error_clasif_F1 = round(error_clasif_F1, 4)
    error_clasif_F2 = round(error_clasif_F2, 4)

    label_title = str('Clasificación de ')+str(n_test)+str(' muestras usando KNN, k=')+str(k[i])+str('.\nError de clasificación: Clase F1=')+str(error_clasif_F1)+str(', Clase F2=')+str(error_clasif_F2)
    fig_graf = graf_puntos_clasif(sample_F1_test[:,0],sample_F2_test[:,0],
                                  test_clase_1_F1,test_clase_2_F1,
                                  test_clase_1_F2,test_clase_2_F2,label_title,
                                  bound)
    if imprimir == 1:
        output_filename = 'fig_e-Muestras-Clasif-KNN_k=' + str(k[i]) + '.png'
        fig_graf.savefig(output_filename,bbox_inches='tight')
        
        


