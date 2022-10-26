import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import string, random
from captcha.image import ImageCaptcha
import numpy as np
from keras.layers import Input, MaxPooling2D, Dense, BatchNormalization, Flatten, Conv2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from PIL import Image


# 引入相关库


def randomCode():
    # 随机生成验证码
    raw = string.digits + string.ascii_uppercase
    # 将0-9，A-Z的数字放入raw变量中，用于随机生成验证码
    random_str = ''.join(random.sample(raw, 4))
    # 随机生成验证码
    return random_str, raw


def GEN(height=80, width=170, batch_size=32, n_class=36):
    # 样本生成器
    print('样本生成')
    x = np.zeros((batch_size, height, width, 3), dtype=np.float)
    # 生成随机验证码样本的储存器
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(4)]
    # 随机验证码样本的标注储存器
    generator = ImageCaptcha(height=height, width=width)
    # 做验证码生成器
    # 随机生成验证码
    while 1:
        # 将随机生成的验证码放入验证码样本储存器中
        for i in range(batch_size):
            random_str, raw = randomCode()
            x[i] = np.array(generator.generate_image(random_str)).astype('float32') / 255.0
            # 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
            for j, ch in enumerate(random_str):
                # j为位置，ch为字母数字共36个
                # 将样本标注放入样本标注储存器中
                y[j][i, :] = 0
                # 将样本标注储存器归零，防止数据出错
                y[j][i, raw.find(ch)] = 1
                # 在样本标注储存器中字母数字所在位置由0改成1进行标注
        yield x, y


# 网络结构
# 训练使用的模型结构
def trainBreakModel():
    h, w, nclass = 80, 170, 36
    # 定义图片高度和宽度
    input_tensor = Input(shape=(h, w, 3))
    # 定义照片信息输入的tensor
    x = input_tensor
    # 防止数据覆盖，所以另x = input_tensor
    for i in range(4):
        x = Conv2D(filters=32 * 2 ** i, kernel_size=(3, 3), activation='relu')(x)
        # filters卷积核的作用对图像进行加权平均，取出其特征
        # filters是卷积核的数目（即输出的维度），kernel_size是过滤器大小，activation = ‘relu’是激活函数
        # 常用的激活函数RelU函数，tanh函数，Sigmoid函数
        # relu激活函数的优点，有效降低幂函数运算，提高运算效率，有效降低梯度消失的问题
        # relu函数表达式为y=max(0,x)
        # Conv2D是二维卷积，这里输入的为二维图片，所以使用Conv2D
        x = Conv2D(filters=32 * 2 ** i, kernel_size=(3, 3), activation='relu')(x)
        x = BatchNormalization(axis=3)(x)
        # normalization 归一化 ：将数据转换到 [0, 1] 之间
        # axis的值取决于按照input的哪一个维度进行，这里的input_tensor为shape=(h,w,3)，所以参数为3
        x = MaxPooling2D((2, 2))(x)
        # 2D输入的最大池化层
        # 将输入的图像划分为若干个矩形区域，对每个子区域输出最大值。
        # 池化操作后的结果相比其输入缩小了。池化层的引入是仿照人的视觉系统对视觉输入对象进行降维和抽象。
        # 特征不变形：池化操作是模型更加关注是否存在某些特征而不是特征具体的位置。
        # 特征降维：池化相当于在空间范围内做了维度约减，从而使模型可以抽取更加广范围的特征。同时减小了下一层的输入大小，进而减少计算量和参数个数。
        # 在一定程度上防止过拟合，更方便优化。
    x = Flatten()(x)
    # Flatten层用来将输入“压平”，把多维的输入一维化，用于从卷积层到全连接层的过渡。Flatten不影响batch的大小。
    x = [Dense(nclass, activation='softmax', name='D%d' % (n + 1))(x) for n in range(4)]
    # 激活函数softmax用于多分类问题
    # 全连接层Dense即全连接层，逻辑上等价于这样一个函数：
    # 权重W为m*n的矩阵.输入x为n维向量.激活函数Activation.偏置bias.输出向量out为m维向量.out=Activation(Wx+bias),即一个线性变化加一个非线性变化产生输出。
    model = Model(inputs=input_tensor, outputs=x)
    # 建立模型
    return model


# plot_model(trainBreakModel(),to_file='bbbbtrain.png',show_shapes=True)
# 打印模型


# 训练主程序
def train():
    model = trainBreakModel()
    # 从函数中获取建立好的模型

    check_point = ModelCheckpoint(
        filepath='aaaaaaa.h5',
        # 模型存放的路径
        save_best_only=True
        # save_best_only当设置为True时，监测值有改进时才会保存当前的模型,既保证储存的是当前建立的最好模型
    )

    model.compile(
        loss='categorical_crossentropy',
        # 目标函数，或称损失函数，是网络中的性能函数，也是编译一个模型必须的两个参数之一。
        # categorical_crossentropy
        # 多分类的对数损失函数，与softmax分类器相对应的损失函数，理同上。
        # 此损失函数与上一类同属对数损失函数，sigmoid和softmax的区别主要是，sigmoid用于二分类，softmax用于多分类。
        # softmax计算损失函数详情请见图
        optimizer='adadelta',
        # optimizer优化器
        # adadelta优化函数详情请见图
        metrics=['accuracy']
        # metrics: 列表，包含评估模型在训练和测试时的性能的指标
    )

    model.fit_generator(
        GEN(),
        # generator：生成器函数，生成器的输出应该为：
        # 一个形如（inputs，targets）的tuple。这里使用的是这个
        # 一个形如（inputs, targets,sample_weight）的tuple。
        # 所有的返回值都应该包含相同数目的样本。生成器将无限在数据集上循环。
        # 每个epoch以经过模型的样本数达到samples_per_epoch时，记一个epoch结束
        epochs=10,
        # epochs：整数，数据迭代的轮数，既训练10轮
        steps_per_epoch=50,
        # steps_per_epoch：整数，当生成器返回steps_per_epoch次数据时计一个epoch结束，执行下一个epoch，这里既一轮训练调用50次生成器GEN()
        validation_data=GEN(),
        # validation_data：具有以下三种形式之一
        # 生成验证集的生成器,这里使用的是这个
        # 一个形如（inputs,targets）的tuple
        # 一个形如（inputs,targets，sample_weights）的tuple
        validation_steps=10,
        # validation_steps: 当validation_data为生成器时，本参数指定验证集的生成器返回次数
        # 设置验证多少次数据后取平均值作为此epoch训练后的效果
        callbacks=[check_point, TensorBoard(log_dir='logs', histogram_freq=1)]
        # callbacks第一个参数filepath，这里使用check_point参数，其作用是将每个epoch的训练结果保存到模型中，优化模型
        # TensorBoard是Tensorflow库的可视化工具用于查看模型的生成情况和loss，acc的变换情况
        # log_dir='logs'是存放可视化数据的路径
        # histogram_freq默认为0，不打开可视化程序面板。1打开loss，acc两个可视化面板
        # 在Terminal中输入tensorboard --logdir=./logs查看可视化
    )


if __name__ == "__main__":
    # 本文件调用train()禁止外部文件调用train()这个函数
    train()
