import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets._samples_generator import make_blobs #to generate data clusters

X, Y = make_blobs(n_samples=100, centers=2, cluster_std=0.6, random_state=0)

c1 = np.array([1, 2])   #create centers
c2 = np.array([2.5, 2])

fig,axs = plt.subplots(2,2) #create subplots
axs = axs.reshape(-1)

for j in range(4):
    class1 = np.zeros((1,2)) #create class to store the data points
    class2 = np.zeros((1,2))
    
    for z in range(X.shape[0]):
        d1 = np.linalg.norm(c1 - X[z, :], ord=2) #compute distances
        d2 = np.linalg.norm(c2 - X[z, :], ord=2)
        if d1<d2:
            class1 = np.append(class1, X[z,:].reshape((1,2)),axis=0) #reshape to ensure that we append a point
        else:
            class2 = np.append(class2, X[z,:].reshape((1,2)),axis=0)
            
    class1 = np.delete(class1, (0), axis=0) # remove zeros used to initialize
    class2 = np.delete(class2, (0), axis=0)
    
    #plots
    axs[j].plot(class1[:,0],class1[:,1],'ro',ms=5)
    axs[j].plot(class2[:,0],class2[:,1],'bo',ms=5)
    axs[j].plot(c1[0],c1[1],'ko',ms=10)
    axs[j].plot(c2[0],c2[1],'ko',ms=10) 
    
    c1 = np.array([np.mean(class1[:,0]),np.mean(class1[:,1])]) #recalculate centers
    c2 = np.array([np.mean(class2[:,0]),np.mean(class2[:,1])]) 
    
plt.show()