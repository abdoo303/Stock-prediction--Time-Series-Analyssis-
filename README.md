# 

this is a model using time series analysis to predict trends of of falls or rises of stock markets shares; 
## Importation


```bash
import math
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
import pandas_datareader as web
import pandas as pd 
import numpy  as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dense
print("done")
```


```python
df.shape
```
## Plotting the data of close vs time
```python
plt.figure(figsize=(16,8))
plt.title("close price history")
plt.plot(df['Close'])
plt.ylabel("Close in $",fontsize=18)
plt.xlabel("Date",fontsize=18)
plt.show()
```
## Fetching the data form yahoo finance

```python
close= df['Close'].values
close
```
## set the size of train data to 80% of the total size
```python
len_of_train= math.ceil(.8*len(close))
len_of_train
```


```python
#initializing the scaler and transforming the close column
scaler= MinMaxScaler(feature_range=(0,1))
close=close.reshape(-1,1)
scaled_close= scaler.fit_transform(close)
```

## Preparing data to be fed into the model
```python
train_data= scaled_close[0:len_of_train,:]

x_train=[]
y_train=[]
for i in range(60,len_of_train):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])

x_train,y_train= np.array(x_train),np.array(y_train)

```


```python
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape
```
## the model uses 2 LSTM layers and 2 Dense layers
```python

model= Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(50))
model.add(Dense(25))
model.add(Dense(1))

```
## Fitting the model
```python
model.compile(optimizer='adam',loss="mean_squared_error")
history= model.fit(x_train,y_train,batch_size=1,epochs=1)
```
## Preparing validation data to be tested
```python


y_test=close[len_of_train:,:]
x_test=[]
test_data=scaled_close[len_of_train-60:,:]

for i in range(60,test_data.shape[0]):
    x_test.append(test_data[i-60:i,0])
x_test,y_test=np.array(x_test),np.array(y_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
x_test.shape
```
##  Predicition!
```python
y_pred= model.predict(x_test)
y_pred=scaler.inverse_transform(y_pred)
```
## Root Mean Squared Error
```python
rmse= np.sqrt(np.mean((y_pred-y_test)**2))
rmse
```
```python
train=df[:len_of_train]
valid=df[len_of_train:]
valid["Prediction"]=y_pred
```
## plotting the train data and the actual data of validaions and prediction. 
```python
plt.figure(figsize=(16,8))
plt.title("VAlidddddation")
plt.xlabel("Date")
plt.ylabel("Real Close and Predictoin")
plt.plot(train['Close'])
plt.plot(valid[["Close","Prediction"]])
plt.legend(["Real","Val","Prediction"],loc="lower right")
plt.show()
```
[the source vid](https://www.youtube.com/watch?v=QIUxPv5PJOY&list=LL&index=3)

## License
[GitHub.Abdelrahim](https://github.com/abdoo303)