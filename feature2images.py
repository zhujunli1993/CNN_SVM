import numpy as np

from PIL import Image

data_conv3 = np.genfromtxt('/Users/zhujunli/Desktop/features/feature_MaxPool_2.csv', delimiter=',')
data_conv3 = data_conv3 / data_conv3.max()
data_conv3 = data_conv3.reshape(32, 32* 32* 128)

data_conv2 = np.genfromtxt('/Users/zhujunli/Desktop/features/feature_MaxPool_1.csv', delimiter=',')
data_conv2 = data_conv2 / data_conv2.max()
data_conv2 = data_conv2.reshape(32, 64* 64* 64)

data_conv1 = np.genfromtxt('/Users/zhujunli/Desktop/features/feature_MaxPool_0.csv', delimiter=',')
data_conv1 = data_conv1 / data_conv1.max()
data_conv1 = data_conv1.reshape(32, 128* 128* 32)
data_fc = np.genfromtxt('/Users/zhujunli/Desktop/features/feature_Relu_3.csv', delimiter=',')
data_fc = data_fc / data_fc.max()
data_fc = data_fc.reshape(32, 1024)
for i in range(0,31):
    feature_map = data_conv3[i].reshape(32,32,128)
    img = Image.fromarray(feature_map[:,:,0], 'L')
    img.save('/Users/zhujunli/Desktop/featuresmap3/image_'+str(i)+'.png',format='PNG')   

for i in range(0,31):
    feature_map = data_conv2[i].reshape(64,64,64)
    img = Image.fromarray(feature_map[:,:,0], 'L')
    img.save('/Users/zhujunli/Desktop/featuresmap2/image_'+str(i)+'.png',format='PNG')   


for i in range(0,31):
    feature_map = data_conv1[i].reshape(128,128,32)
    img = Image.fromarray(feature_map[:,:,0], 'L')
    img.save('/Users/zhujunli/Desktop/featuresmap1/image_'+str(i)+'.png',format='PNG') 


feature_map = data_fc.reshape(32,1024)
img = Image.fromarray(feature_map, 'L')
img.save('/Users/zhujunli/Desktop/featuresmap0/image_'+str(i)+'.png',format='PNG') 