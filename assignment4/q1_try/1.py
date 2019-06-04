#%% [markdown]
# ## Import Libraries
#%%
from keras import layers, Input, models
import numpy as np
import pickle
import sklearn

#%% [markdown]
# ## Generate Data
#%%
seq_range = 10
sine = np.array([np.sin(i) for i in np.arange(0,50,0.001)])
X1 = []
Y1 = []
for i in range(5000):
    r = np.random.randint(0, high=4990)
    X1.append((sine[r:r+10]))
    Y1.append(sine[r+10])

    r = np.random.randint(0, high=4)
    for j in range(r):
        X1[-1][9-j] = 0.0 # Padding ka c*udap



X1, Y1 = np.array(X1), np.array(Y1)
# X1, Y1 = sklearn.utils.shuffle(X1,Y1)
X1, Y1 = X1.reshape(5000,-1,1), Y1.reshape(-1,1)
print(X1.shape, Y1.shape)


#%%[markdown]
# ## Construct Model
#%%


m_input = Input(batch_shape=(None,10,1), name='line')
m = layers.SimpleRNN(10, return_sequences=True)(m_input)
m = layers.SimpleRNN(5, return_sequences=False)(m)
m = layers.Dense(1)(m)
# m = layers.LSTM((1), return_sequences=True)(m)
m = models.Model(m_input,m)
m.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
m.summary()




#%% [markdown]
# ## Train Model
#%%

m.fit(X1,Y1, epochs=30, batch_size=5)

#%% [markdown]
# ## Test
#%%

x = np.array([np.sin(i) for i in np.arange(4,5,0.001)])[:10].reshape(1,-1,1)

y = m.predict(x)
print(y)

#%% [markdown]
# ## Plots
#%%
import matplotlib.pyplot as plt


#%%
