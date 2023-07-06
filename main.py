import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
from keras.layers import Dense
from keras.models import Sequential
import keras
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
scaler=MinMaxScaler(feature_range=(0, 1))
np.set_printoptions(precision=15, floatmode='fixed')#делаем 10 знаков после запятой
from dynamic_systems import *
from metod_t import *

xs=real_time_series()
dataset=xs.reshape((-1,1))#превратили вектор в вектор векторов
dataset = scaler.fit_transform(dataset)#нормализовали данные

train_size = int(len(dataset) * 0.70)
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:] #Разбили датасет на тренировочную и тестовую выборку

tay=MU_IN(xs)
input_nodes = 11#количество входных узлов
hidden_nodes = 7#количество скрытых узлов 
output_nodes = 1#количество выходных узлов
num_epochs=600#количество эпох
d = 1e-6#Очень маленькое значение
def split_data(x,tay,input_nodes):#Формирование данных
    NN=len(x)-(((input_nodes)*tay))
    a=np.zeros((NN,input_nodes))
    b=[]
    for i in range(NN):
        for j in range(input_nodes):
            a[i][j]=x[i+tay*j]
        b.append(x[i+tay*input_nodes])
    return a,np.array(b)
X_train,y_train=split_data(train,tay,input_nodes)
X_test,y_test=split_data(test,tay,input_nodes)


#Определение и обучение нейронной сети
model =Sequential([
    Dense(hidden_nodes,input_shape=(input_nodes,),activation='tanh'),
    Dense(output_nodes,activation='linear')#выходной узел
])

optimizer = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer,loss='mse',metrics=['mae'])
model.fit(X_train,y_train,epochs=num_epochs)

def pred_in_steps(x):#Предсказывание на n шагов
    x=x.reshape(1,(len(x)))
    b=[]
    for i in range(60):
        a=model.predict(x)
        b.append(a)
        x=np.roll(x,-1)        
        x[0][-1]=a
    return b

a1=pred_in_steps(X_test[len(X_test)-1])
X_test2=np.copy(X_test)
X_test2[:,0]=X_test2[:,0]+d#Добавляем смещение
a2=pred_in_steps(X_test2[len(X_test)-1])

def lin_reg(d):#Вычисляем наклон прямой регрессии и строим график
    plt.xlabel('n')
    plt.ylabel('Ln(d)')
    plt.plot(d,label='d')
    ter=np.array([int(i) for i in range(len(d))]).reshape((-1,1))
    model2 = LinearRegression().fit(ter, d)
    plt.plot(ter,model2.predict(ter),label='LinearRegression')
    s='Наклон прямой равен:'+str(model2.coef_)
    plt.title(s)
    print('Наклон прямой равен:', model2.coef_)
    plt.legend()
    plt.grid()
    plt.show()  
    return 
def calcD(pred_a,pred_b):#Вычисляем разность между двумя траекториями
    d=[]
    for i in range(len(pred_a)):
        if (pred_b[i]-pred_a[i])!=0:#исключить из выборки 0 значения
            wer=math.log(np.abs(pred_b[i]-pred_a[i]))
            if(wer<0):
                d.append(wer)
    return d

d=calcD(a1,a2)
lin_reg(d)
