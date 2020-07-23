# Teoria de deteccion y estimacion - FIUBA 
# 1er cuatrimestre de 2020

# Autor: PABLO HSIEH 
# Padron: 97363

# Estimacion no parametrica - PARZEN, KNN


import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#import scipy.integrate as integrate
from scipy.integrate import quad
#import math


# Gaussiana ------------------------------------------------------------------
def gaussiana(x,mu,sigma): 
#Es la pdf gaussiana
#Para el caso del error se usa mu=0, sigma=1
    return (np.exp( (-(x-mu)**2)/(2*sigma**2)))*(1/(np.sqrt(2*np.pi)*sigma))

#def muestras_gauss(n,m,s):
#    muestra = np.zeros(n)
#    for i in range(n):
#        muestra[i] = np.random.normal(mu,sigma,1)
#    return muestra

# Error ----------------------------------------------------------------------
def mahalanobis_distance(d,x,y,cov_xy): #distancia de mahalanobis entre x e y
    a = x - y
    if d ==1: #caso escalar
        b = 1/cov_xy
    else: #caso de dimension > 1
        b = np.linalg.inv(cov_xy)
    r = np.dot(a,np.dot(a,b))
    return np.sqrt(r)

def p_error(lim_inf): # Probabilidad de error integrando pdf gaussiana
# el lim inferior es la dist mahalanobis/2
    return quad(gaussiana,lim_inf,np.inf,args=(0,1))

            

        
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
    
def window_gauss(x,h):
# Ventana gaussiana de media 0 y varianza sigma
# La media es 0 porque si no lo fuera entonces la ventana estaria en cualquier lugar
# Una gaussiana en 3sigma cae a casi 0, se podria usar 4sigma tambien  
# Necesito que la gaussiana este dentro de la ventana de long h, por eso
# 4*sigma = h/2 => sigma = h/8  
    #sigma = h/6
    sigma = h/8    
    if np.abs(x) <= np.abs(1/2):
        return gaussiana(x,0,sigma)
    else:
        return 0

def select_window(gauss,x,h):
    if gauss == 1:
        return window_gauss(x,h)
    else:
        return window(x)


def estimacion_parzen(muestras,soporte,h,gauss):
# pdf estimada en 1d
    N = len(muestras)
    p_hat = np.zeros(len(soporte))
    for x in range(len(soporte)):
        for x_i in muestras:
            #p_hat[x] = p_hat[x] + window((soporte[x]-x_i)/h)/h
            p_hat[x] = p_hat[x] + select_window( gauss, (soporte[x]-x_i)/h , h)/h
        p_hat[x] = p_hat[x]/N
    return p_hat

#def parzen_estimate(muestras,soporte,h):
 #   N = len(muestras)
  #  estimate = []
   # for x in soporte:
    #    sum = 0
     #   for x_i in muestras:
      #      sum = sum + window((x-x_i)/h)/h
       # estimate.append(sum/N)
    #return estimate
    
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

# CLASIFICADOR DICOTOMICO ----------------------------------------------------
# Con las distribuciones estimadas y la probabilidad a priori se arma un 
#clasificador bayesiano dicotomico.
def es_clase_a(p_priori_a,p_estimada_a,p_priori_b,p_estimada_b,muestra,soporte):
    p_a = p_priori_a*f(soporte,p_estimada_a,muestra)
    p_b = p_priori_b*f(soporte,p_estimada_b,muestra)
    #g = p_a - p_b
    if p_a > p_b: # es de clase a
        return 1
    else: #es de clase b
        return 0
        
def sep_clases(p_priori_a,p_estimada_a,p_priori_b,p_estimada_b,array_muestra,soporte):
# Devuelvo dos arrays, uno con las muestras de la clase a y otro de la b
    array_clase_a = np.zeros(0)
    array_clase_b = np.zeros(0)
#    print(array_muestra[0])
    for i in range(len(array_muestra)):
        aux = es_clase_a(p_priori_a,p_estimada_a,p_priori_b,p_estimada_b,array_muestra[i],soporte) == 1
        if aux == 1: #array_muestra[i] es de clase a
            array_clase_a = np.append(array_clase_a,array_muestra[i])
        else:
            array_clase_b = np.append(array_clase_b,array_muestra[i])
    return array_clase_a,array_clase_b
        

# Realizacion de graficos

#def graficar(x_p_hat,p_hat,x_p_teo,p_teo):
#    plt.plot(x_p_hat,p_hat,color='b',linestyle='dashed',linewidth=1.25,label='Parzen')
    #plt.bar(x_p_hat,p_hat,color='b',align='center',alpha = 0.25 ,label='Parzen barras')
#    plt.plot(x_p_teo,p_teo,color='r',linestyle='solid',linewidth=1.25,label='Teórica')
#    plt.xlabel('soporte')
#    plt.ylabel('pdf estimada')
#    plt.xticks(x) 
#    plt.grid()
#    plt.legend()
#    plt.show()
    
 
#def graf_test(x,y):
#    plt.plot(x,y,color='b',linestyle='dashed',linewidth=1)
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.xticks(x) 
#    plt.grid()
#    plt.legend()
#    plt.show()

#def graf_test_2(x,y,z):
#    plt.plot(x,y,color='b',linestyle='dashed',linewidth=1,label='1er arg')
#    plt.plot(x,z,color='r',linestyle='solid',linewidth=1,label='2do arg')
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.xticks(x) 
#    plt.grid()
#    plt.legend()
#    plt.show()

def graf_histograma(muestras_1,label_1,muestras_2,label_2,n,mu_1,mu_2):
    fig = plt.figure()
    plt.hist(muestras_1,bins='auto',alpha = 0.65,label='F1:'+label_1)
    plt.hist(muestras_2,bins='auto',alpha = 0.65,label='F2:'+label_2)
    plt.axvline(x=mu_1, color='k', linestyle='dashed',linewidth=0.5)
    plt.axvline(x=mu_2, color='k', linestyle='dashed',linewidth=0.5)
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

#def graf_muestras_clasif(soporte,p_F1,p_F2,muestras_F1,muestras_F2):#,label,h_k):
# muestras_F1 y muestras_F2 son las muestras clasificadas en clase F1 y F2
#    fig = plt.figure()
#    plt.plot(soporte,p_F1,color='b',linestyle='dashed',linewidth=1.25,label='F1 estimada')
#    plt.plot(soporte,p_F2,color='r',linestyle='dashed',linewidth=1.25,label='F2 estimada')
#    plt.plot(muestras_F1,len(muestras_F1)*[0],'bx',linewidth=1,label='clase F1')
#    plt.plot(muestras_F2,len(muestras_F2)*[0],'r.',linewidth=1,label='clase F2')
#    plt.xlabel('Soporte')
#    plt.ylabel('pdf estimada')
#    plt.title('Distribuciones estimadas y muestras clasificadas.\n')#+label+str(h_k))
#    plt.grid()
#    plt.legend()
#    plt.show()
#    return fig

def graf_puntos_clasif(real_F1,real_F2,clas1_F1,clas2_F1,clas1_F2,clas2_F2,n_test,error_F1,error_F2,label_estimacion,h_k):
    fig = plt.figure()
    plt.plot(test_clase_1_F1,len(test_clase_1_F1)*[0],'bx',linewidth=1,label='clase F1')
    plt.plot(test_clase_2_F1,len(test_clase_2_F1)*[0],'r.',linewidth=1,label='clase F2')
    plt.plot(sample_F1_test,len(sample_F1_test)*[1],'bx',linewidth=1)#,label='Real F1')
    plt.plot(test_clase_1_F2,len(test_clase_1_F2)*[3],'bx',linewidth=1)#,label='clase F1')
    plt.plot(test_clase_2_F2,len(test_clase_2_F2)*[3],'r.',linewidth=1)#,label='clase F2')
    plt.plot(sample_F2_test,len(sample_F2_test)*[4],'r.',linewidth=1)#,label='Real F2')
    plt.title('Clasificación de '+str(n_test)+' muestras usando las distribuciones estimadas.\n'+label_estimacion+str(h_k)+'.\nError de clasificación: Clase F1='+str(error_clasif_F1)+', Clase F2='+str(error_clasif_F2) )#+label+str(h_k))
    plt.grid()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=4)
    plt.show()
    return fig


######### MAIN #######


pap_w1 = 0.4
pap_w2 = 0.6

F1_mu = 1
F1_sigma = 4
F2_mu = 4
F2_sigma = 4

#n=10**4
n=100
n_test=100

label_F1 = str('Normal(1,4)')
label_F2 = str('Normal(4,4)')
label_parzen = str('Estimación utilizando ventanas de Parzen')
label_window_gauss = str(', Gaussiana(0,h/8) h=')
label_window_rect = str(', Rectangular h=')
label_kn = str('Estimación utilizando Kn vecinos más cercanos, k= ')
#x=np.zeros(n)

# Generacion de n muestras normales de media mu y varianza sigma
#sample = np.random.normal(mu,sigma,n)
sample_F1 = np.random.normal(F1_mu,F1_sigma,n)
sample_F2 = np.random.normal(F2_mu,F2_sigma,n)

# Generacion de muestras a clasificar
sample_F1_test = np.random.normal(F1_mu,F1_sigma,n_test)
sample_F2_test = np.random.normal(F2_mu,F2_sigma,n_test)

#s_F1 = sample_F1 * pap_w1 
#s_F2 = sample_F2 * pap_w2 
#px = p_x_w1 * pap_w1 + p_x_w2 * pap_w2 
#fig_graf = graf_histograma(s_F1, label_F1, s_F2, label_F2,n)


fig_graf = graf_histograma(sample_F1, label_F1, sample_F2, label_F2,n,F1_mu,F2_mu)
#fig_graf.savefig('fig_a-Histograma.png',bbox_inches='tight')

#h = math.ceil(np.sqrt(len(n)))
h = 1

# El soporte va a ser 0 para valores menores a (muestra_min - h/2) y mayores a
# (muestra_min + h/2)
# Para el ejercicio son dos gaussianas de varianza 4 y media 1 y 4, no deberia
# tener nada fuera de un soporte (-15;20)
soporte_minimo =  np.floor(min(sample_F1)-h/2)
soporte_maximo = np.ceil(max(sample_F2)+h/2)
soporte = np.arange(soporte_minimo,soporte_maximo,step=0.1) 
# con step=0.1 tarda casi 8 minutos

# PARZEN ---- Estimacion con ventana GAUSSIANA ------ PARZEN ------- PARZEN --
p_F1 = estimacion_parzen(sample_F1,soporte,h,1)
p_F2 = estimacion_parzen(sample_F2,soporte,h,1)


fig_graf = graf(soporte,p_F1,p_F2,gaussiana(soporte,F1_mu,F1_sigma),gaussiana(soporte,F2_mu,F2_sigma),label_parzen+label_window_gauss,h)
output_filename = 'fig_b-Parzen_winGauss_h=' + str(h) + '.png'
#fig_graf.savefig(output_filename,bbox_inches='tight')


#fig_graf = graf_muestras_clasif(soporte,p_F1,p_F2,sample_F1_test,sample_F2_test)
#output_filename = 'fig_d-Clasif-Parzen-winGauss_h=' + str(h) + '.png'
#fig_graf.savefig(output_filename,bbox_inches='tight')

# ---- Clasificacion
# Utilizo las muestras de prueba para clasificar con las densidades estimadas
test_clase_1_F1, test_clase_2_F1 = sep_clases(pap_w1, p_F1, pap_w2, p_F2, sample_F1_test,soporte)
test_clase_1_F2, test_clase_2_F2 = sep_clases(pap_w1, p_F1, pap_w2, p_F2, sample_F2_test,soporte)

error_clasif_F1 = len(test_clase_2_F1) / n_test
error_clasif_F2 = len(test_clase_1_F2) / n_test


fig_graf = graf_puntos_clasif(sample_F1_test,sample_F2_test,test_clase_1_F1,test_clase_2_F1,test_clase_1_F2,test_clase_2_F2,n_test,error_clasif_F1,error_clasif_F2,label_parzen+label_window_gauss,h)
output_filename = 'fig_d-Muestras-Clasif-Parzen-winGauss_h=' + str(h) + '.png'
#fig_graf.savefig(output_filename,bbox_inches='tight')

#fig = plt.figure()
#plt.plot(test_clase_1_F1,len(test_clase_1_F1)*[0],'bx',linewidth=1,label='clase F1')
#plt.plot(test_clase_2_F1,len(test_clase_2_F1)*[0],'r.',linewidth=1,label='clase F2')
#plt.plot(sample_F1_test,len(sample_F1_test)*[1],'bx',linewidth=1)#,label='Real F1')
#plt.plot(test_clase_1_F2,len(test_clase_1_F2)*[3],'bx',linewidth=1)#,label='clase F1')
#plt.plot(test_clase_2_F2,len(test_clase_2_F2)*[3],'r.',linewidth=1)#,label='clase F2')
#plt.plot(sample_F2_test,len(sample_F2_test)*[4],'r.',linewidth=1)#,label='Real F2')
#plt.title('Clasificación de '+str(n_test)+' muestras. Error de clasificación:\nClase F1='+str(error_clasif_F1)+', Clase F2='+str(error_clasif_F2) )#+label+str(h_k))
#plt.grid()
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=4)
#plt.show()


# PARZEN ---- Estimacion con ventana RECTANGULAR ------- PARZEN ------- PARZEN
p_F1 = estimacion_parzen(sample_F1,soporte,h,0)
p_F2 = estimacion_parzen(sample_F2,soporte,h,0)

fig_graf = graf(soporte,p_F1,p_F2,gaussiana(soporte,F1_mu,F1_sigma),gaussiana(soporte,F2_mu,F2_sigma),label_parzen+label_window_rect,h)
output_filename = 'fig_b-Parzen_winRect_h=' + str(h) + '.png'
#fig_graf.savefig(output_filename,bbox_inches='tight')

# ---- Clasificacion
# Utilizo las muestras de prueba para clasificar con las densidades estimadas
test_clase_1_F1, test_clase_2_F1 = sep_clases(pap_w1, p_F1, pap_w2, p_F2, sample_F1_test,soporte)
test_clase_1_F2, test_clase_2_F2 = sep_clases(pap_w1, p_F1, pap_w2, p_F2, sample_F2_test,soporte)
error_clasif_F1 = len(test_clase_2_F1) / n_test
error_clasif_F2 = len(test_clase_1_F2) / n_test


fig_graf = graf_puntos_clasif(sample_F1_test,sample_F2_test,test_clase_1_F1,test_clase_2_F1,test_clase_1_F2,test_clase_2_F2,n_test,error_clasif_F1,error_clasif_F2,label_parzen+label_window_rect,h)
output_filename = 'fig_d-Muestras-Clasif-Parzen-winRect_h=' + str(h) + '.png'
#fig_graf.savefig(output_filename,bbox_inches='tight')




# Kn ---- Estimacion con k vecinos ---- Kn ---- Kn ---- Kn ---- Kn ---- Kn ---

k=np.array([1,10,50,100])

for i in range(len(k)):
    p_F1 = estimacion_kn(sample_F1,soporte,k[i])
    p_F2 = estimacion_kn(sample_F2,soporte,k[i])
    fig_graf = graf(soporte,p_F1,p_F2,gaussiana(soporte,F1_mu,F1_sigma),gaussiana(soporte,F2_mu,F2_sigma),label_kn,k[i])
    output_filename = 'fig_c-KnVecinos_k=' + str(k[i]) + '.png'
#    fig_graf.savefig(output_filename,bbox_inches='tight')
    
    # ---- Clasificacion
    # Utilizo las muestras de prueba para clasificar con las densidades estimadas
    test_clase_1_F1, test_clase_2_F1 = sep_clases(pap_w1, p_F1, pap_w2, p_F2, sample_F1_test,soporte)
    test_clase_1_F2, test_clase_2_F2 = sep_clases(pap_w1, p_F1, pap_w2, p_F2, sample_F2_test,soporte)
    error_clasif_F1 = len(test_clase_2_F1) / n_test
    error_clasif_F2 = len(test_clase_1_F2) / n_test

    fig_graf = graf_puntos_clasif(sample_F1_test,sample_F2_test,test_clase_1_F1,test_clase_2_F1,test_clase_1_F2,test_clase_2_F2,n_test,error_clasif_F1,error_clasif_F2,label_kn,k[i])
    output_filename = 'fig_d-Muestras-Clasif-Kn_vecinos_k=' + str(k[i]) + '.png'
#    fig_graf.savefig(output_filename,bbox_inches='tight')




#fig_1=plt.figure(1)
#plt.plot(soporte,p_F1,color='b',linestyle='dashed',linewidth=1.25,label='Parzen F1')
#plt.plot(soporte,p_F2,color='r',linestyle='dashed',linewidth=1.1,label='Parzen F2')
#plt.plot(soporte,gaussiana(soporte,F1_mu,F1_sigma),color='b',linestyle='solid',linewidth=1.25,label='Teórica F1')
#plt.plot(soporte,gaussiana(soporte,F2_mu,F2_sigma),color='r',linestyle='solid',linewidth=1.1,label='Teórica F2')
#plt.xlabel('soporte')
#plt.ylabel('pdf estimada')
#plt.title('Comparación entre distribución teórica y estimada')
#plt.grid()
#plt.legend()
#plt.show()
#fig_1.savefig(out_filename_parzen,bbox_inches='tight')



#p = estimacion_parzen(sample,soporte,h)
#p_cristina = parzen_estimate(sample,soporte,h)


#graf_test(soporte,p_F1)
#graf_test_2(soporte,p,p_cristina)
#graficar(soporte,p,soporte,gaussiana(soporte,mu,sigma))


#########

#h = math.ceil(np.sqrt(len(y)))

#z = gaussiana(x,mu,sigma)

#graf_test(x,z)
#graf_test(x,gaussiana(x,mu,sigma))

#p = estimacion(y,x,h)
#graficar(x, p, x, z)

#graf_test(x,p)




# Ejemplo ej clase Jonas
#D = np.array([2,3,4,8,10,11,12])

#h = math.ceil(np.sqrt(len(D)))
#x = np.arange(0,14, step=1)
#p_estimada = estimacion_parzen(D,x,h)
#p_cris = parzen_estimate(D,x,h)

#fig_1 = plt.figure(1)
#plt.plot(x,p_estimada,color='darkorange',linestyle='dashed',linewidth=1,label='PARZEN')
#plt.bar(x, p_estimada,color='tab:orange',alpha = 0.4,align='center')
#plt.xlabel('x')
#plt.ylabel('p estimada')
#plt.xticks(x) 
#plt.grid()
#plt.legend()
#plt.show()

#fig_2 = plt.figure(2)
#plt.plot(x,p_cris,color='darkorange',linestyle='dashed',linewidth=1,label='PARZEN CRIS')
#plt.bar(x, p_cris,color='tab:orange',alpha = 0.4,align='center')
#plt.xlabel('x')
#plt.ylabel('p estimada')
#plt.xticks(x) 
#plt.grid()
#plt.legend()
#plt.show()

#f = open("TP_resultados.txt", "w") # Voy a imprimirlo en un archivo        