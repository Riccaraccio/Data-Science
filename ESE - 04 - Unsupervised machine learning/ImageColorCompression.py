from matplotlib.image import imread #to read the image
import numpy as np 
import matplotlib.pyplot as plt 

image = imread("StillLife.jpg") #import image

data = image.reshape(image.shape[0]*image.shape[1], image.shape[2]) #reshape

def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data
    
    # choose a random subset
    rng = np.random.RandomState(0) #for everyone to get the same results
    i = rng.permutation(data.shape[0])[:N] #extract N random points
    colors = colors[i]/255; #normalize color value to be in range (0,1)
    
    R, G, B = data[i].T #extract R,G,B value for each pixel
    
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(R, G, c=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 255), ylim=(0, 255))

    ax[1].scatter(R, B, c=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 255), ylim=(0, 255))

    fig.suptitle(title, size=20)
    plt.show()

plot_pixels(data, title='Input color space: 16 million possible colors')

from sklearn.cluster import MiniBatchKMeans #faster for very large datasets
kmeans = MiniBatchKMeans(16) #number of clusers
kmeans.fit(data) #fit data

# kmeans.predict(data) assigns each data point to a cluster, return Index of the cluster each sample belongs to
# kmeans.cluster_centers_ gives the centroid of each cluster.
# kmeans.cluster_centers_[kmeans.predict(data)] effectively retrieves the centroid corresponding to the cluster each data point is assigned to
new_colors = kmeans.cluster_centers_[kmeans.predict(data)] 

new_colors = new_colors.astype(int) #convert values of the centers to int, (0,255) no floats
plot_pixels(data, colors=new_colors, title="Reduced color space: 16 colors")

image_recolored = new_colors.reshape(image.shape)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[1].imshow(image_recolored)
ax[1].set_title('16-color Image')
plt.show()
