import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from skimage import io, color
from sklearn.preprocessing import scale



filename = '/Users/zhujunli/Desktop/data/ortraining-images/LGG_small/0/slice_01_02.jpg'


image = cv2.imread(filename)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
vectorized = image.reshape(-1,3)
vectorized = np.float32(vectorized)
kmeans = KMeans(n_clusters=20, random_state=0).fit(vectorized)
res = kmeans.cluster_centers_[kmeans.labels_.flatten()] 

segmented_image = res.reshape((image.shape))
label = kmeans.labels_.reshape((image.shape[0],image.shape[1]))

output = segmented_image.astype(np.uint8)
print('res:', res.shape)
print('segment:', segmented_image.shape)


plt.imshow(output)


columns = 4
rows = 5
fig=plt.figure(figsize=(8, 8))
for i in range(1, columns*rows +1):
    component=np.zeros(image.shape,np.uint8)
    component[label==i]=image[label==i]    
    fig.add_subplot(rows, columns, i)
    plt.title('label:' + str(i))
    plt.imshow(component)
plt.show()


