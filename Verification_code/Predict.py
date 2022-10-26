import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import string, random
from captcha.image import ImageCaptcha
import numpy as np
from keras.layers import Input, MaxPooling2D, Dense, BatchNormalization, Flatten, Conv2D
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from PIL import Image
#引入相关库


#将概率最大的四个字符转换为字符串
def decoder(y, random_str):
    y = np.argmax(np.array(y),axis=2)[:, 0]
    print('最大值下标',y)
    return ''.join([random_str[x] for x in y])


def randomCode():
    # 随机生成验证码
    raw = string.digits + string.ascii_uppercase
    code = ''.join(random.sample(raw,4))
    return code, raw


def GEN(height=80,width=170,batch_size=1,n_class=36):
    # 样本生成器
    print('样本生成')
    x = np.zeros((batch_size,height,width,3),dtype=np.float)
    #生成随机验证码样本的储存器
    y = [np.zeros((batch_size,n_class),dtype=np.uint8) for i in range(4)]
    #随机验证码样本的标注储存器


    #做验证码生成器
    generator = ImageCaptcha(height=height,width=width)
    #随机生成验证码
    while 1:
        #将随机生成的验证码放入验证码样本储存器中
        for i in range(batch_size):
            random_str, raw = randomCode()
            genCode = generator.generate_image(random_str)
            x[i] = np.array(genCode).astype('float32')/255.0
            #0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
            for j,ch in enumerate(random_str):
                #j为位置，ch为字母数字共36个
                #将样本标注放入样本标注储存器中
                y[j][i:] = 0
                #将样本标注储存器归零，防止数据出错
                y[j][i,random_str.find(ch)] = 1
                #在样本标注储存器中字母数字所在位置由0改成1进行标注
        yield x, y, raw, genCode, random_str


model = load_model('aaaaaaa.h5')
# 调用模型
x, y, raw, genCode, random_str = next(GEN())
# 调用验证码图片的生成器
result = model.predict(x)
# 进行预测
print('预测样本', decoder(y=result, random_str= raw))
# 输出预测结果


plt.imshow(genCode)
plt.title('Truth value:->%s'%random_str, fontsize=30)
plt.show()