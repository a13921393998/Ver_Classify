import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import string
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import PIL
# from captcha.image import ImageCaptcha
#引入相关库


characters = string.digits + string.ascii_uppercase
model = load_model('aaaaaaa.h5')
height = 80
width = 170
batch_size = 1
n_class = 36

def decoder(y,rando_str):
    y = np.argmax(np.array(y), axis=2)[:,0]
    print('最大值下标', y)
    return ''.join([characters[x] for x in y])


x = np.zeros((batch_size, height, width, 3), dtype=np.float)
im = Image.open('your_file1.png')
out = im.resize((170, 80))
x[0] = np.array(out).astype('float32')/255.0
result = model.predict(x)
print('预测样本', decoder(y=result, rando_str=characters))



plt.imshow(im)
plt.show()
















