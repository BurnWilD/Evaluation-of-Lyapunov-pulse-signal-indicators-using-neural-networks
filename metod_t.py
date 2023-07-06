#Методы нахождения временной задержки и размерности вложения
import numpy as np
import math

def autocorrelation(x):#автокорреляционная функция
    N=len(x)# количество элементов временного ряда
    auto_corr=np.zeros(N)#Создаем массив длины N, заполненный нулями
    mean=np.mean(x)#среднее значение ряда
    var=np.var(x)#Дисперсия ряда
    for lag in range(1,N): #тут тау
        acov=0.0
        count=0
        for i in range(1,N-lag): # тут t
            acov+=(x[i]-mean)*(x[i+lag]-mean)
            count+=1
        acov/=N #Вычисляем несмещенную автоковариацию
        auto_corr[lag]=acov/var # Нормализиуем на дисперсию
        if(auto_corr[lag]==0):
            return lag
        elif (auto_corr[lag]<0):
            return lag-1
        else: pass
    return lag-1

def MU_IN(x):#Метод взаимной информации
    N=len(x)#Длина временного ряда
    L=math.floor(math.log2(N))+1# Вычислили L по формуле Старка   
    c=math.ceil((max(x)-min(x))/L)#размах интервала
    segments=[i for i in range(math.floor(min(x)-1),math.floor(max(x)),c) ]
    segments.append(max(x))
    segments = tuple(zip(segments[:-1], segments[1:]))   
    Smax=-1000000
    A=np.zeros(L)
    for i in range(len(x)): A[next((idx for idx, (sec, fir) in enumerate(segments) if sec < x[i] <= fir), None)]+=1
    A/=N
    for tay in range(1,N):
        B=np.zeros(L)
        AB=np.zeros((L,L))
        c=0
        for i in range (len(x)-tay):#вычисляем под статические данные   
            res = next((idx for idx, (sec, fir) in enumerate(segments) if sec < x[i+tay] <= fir), None)
            B[res]+=1
            AB[res][next((idx for idx, (sec, fir) in enumerate(segments) if sec < x[i] <= fir), None)]+=1  
        B/=(N)
        AB/=(N-tay)
        for i in range(L): 
            for j in range(L):
                c-=0 if AB[i][j]/(A[i]*B[j])==0 else AB[i][j]*math.log2(AB[i][j]/(A[i]*B[j]))
        if(c<=Smax): return tay-2 
        else: Smax=c
    return tay-2

def fnn(x):
    P=1000000
    Rt=2
    def scallar(point1, point2):#функция вычисления расстояния между двумя точками
        return sum([(point1[i] - point2[i])**2 for i in range(len(point1))])**0.5
    
    def neighbour(i,point1,m):
        dmin=1000000000000
        for j in range(N-m):
            if i!=j:
                point2=x[j:j+m]
                distance=scallar(point1, point2)
                if distance<dmin:
                    dmin=distance      
        return dmin
    
    m=0
    N=len(x)
    d=np.zeros(20)
    j=0
    while ((P/N)>0.001):
        P=0
        m+=1
        for i in range(N-m-1):
            dis1=neighbour(i,x[i:i+m],m)#находим первого соседа для размерности m
            dis2=neighbour(i,x[i:i+m+1],m+1)
            Ri=0 if dis1==0 else dis2/dis1         
            if Ri>Rt: P+=1
        print(m, P/N)
        d[j]=P/N
        j+=1
    return m,d
