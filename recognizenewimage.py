from PIL import Image,ImageFilter
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

myGraph=tf.Graph()
with myGraph.as_default():
    #定义输入样本
    #下载样本集
    xs=tf.placeholder(tf.float32,[784])#输入样本xs，其形式是n个图片，每个图片的格式是28*28的向量,并引入tf
    ys=tf.placeholder(tf.float32,[10])#输入label
    #转换样本的形式以便于进行卷积
    x_input=tf.reshape(xs,[-1,28,28,1])#转换形式是n个纵深的图片，图片是尺寸是28*28*1
    ##添加dropout
    keep_prob=tf.placeholder(tf.float32)

    '''对输入的图像进行第一次卷积'''
    #定义卷积函数的参数
    #定义权重weight
    w_conv1=tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))#权重是可训练参数，使用Variable方法定义，
    ###并利用截断正态分布赋初值，采用5*5大小的窗口，将输入厚度为1的图片映射成输出为32层的feature map
    #定义偏差bias
    tf.summary.histogram('w1',w_conv1)
    b_conv1=tf.Variable(tf.constant(0.1,shape=[32]))#由于tf矩阵运算中的加法具有broadcaster属性，只需要声明输出行数
    #进行卷积运算
    tf.summary.histogram('b1',b_conv1)
    conv1=tf.nn.relu(tf.nn.conv2d(x_input,w_conv1,strides=[1,1,1,1],padding='SAME')+b_conv1)#2d卷积方法，输入x_input，w_conv1，
    #######步长2*2，填充至一样大小，此时shape为[-1,28,28,32]
    tf.summary.histogram('conv1',conv1)

    '''对feature map进行第一次池化'''
    pool1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')#最大池化，ksize经验赋值，shape([-1,14,14,32])

    '''进行第二次卷积'''
    w_conv2=tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
    b_conv2=tf.Variable(tf.constant(0.1,shape=[64]))
    conv2=tf.nn.relu(tf.nn.conv2d(pool1,w_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)#shape=[-1,14,14,64]

    '''进行第二次池化'''
    pool2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')#shape=[-1,7,7,64]

    '''连接全连接层1'''
    #定义全连接层的参数
    #定义权重weight,目的是将每个图片的3维tensor特征变为1维数组特征列表，故先对池化后的feature map进行reshape
    pool2_f=tf.reshape(pool2,[-1,7*7*64])
    w_fullc1=tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
    b_fullc1=tf.Variable(tf.constant(0.1,shape=[1024]))
    fullc1=tf.nn.relu(tf.matmul(pool2_f,w_fullc1)+b_fullc1)#矩阵运算并激活,shape=[-1,7*7*1024]
    #使用dropout
    fullc1_dropout=tf.nn.dropout(fullc1,keep_prob)

    '''连接全连接层2分类'''
    w_fullc2=tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
    b_fullc2=tf.Variable(tf.constant(0.1,shape=[10]))
    prediction=tf.nn.softmax(tf.matmul(fullc1,w_fullc2)+b_fullc2)#矩阵运算并激活,shape=[-1,10]

'''读取checkpoint'''
with tf.Session(graph=myGraph) as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    saver=tf.train.Saver()
    saver.restore(sess,'my_net/3.ckpt')
    '''导入新图片数据'''
    im=Image.open('image/2.jpg').convert('L')
    width=float(im.size[0])
    height=float(im.size[1])
    newImage=Image.new('L',(28,28),(255))
    nwidth=int(round(20/height*width))
    img=im.resize((nwidth,20),Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    wleft=int(round((20-nwidth)/2))
    newImage.paste(img,(wleft,4))
    array_image=list(newImage.getdata())
    array_image_1=sess.run(tf.reshape(array_image,(28,28)))
    turn_array_to_image=Image.fromarray(array_image_1)
    plt.imshow(turn_array_to_image)
    plt.show()
    pre_value=tf.argmax(prediction,1)
    print(sess.run(prediction,feed_dict={xs:array_image}))
    print(sess.run(pre_value,feed_dict={xs:array_image}))
    













