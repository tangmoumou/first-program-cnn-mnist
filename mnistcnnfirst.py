import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

#定义输入样本
#下载样本集
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)
xs=tf.placeholder(tf.float32,[None,784])#输入样本xs，其形式是n个图片，每个图片的格式是28*28的向量,并引入tf
ys=tf.placeholder(tf.float32,[None,10])#输入label
#转换样本的形式以便于进行卷积
x_input=tf.reshape(xs,[-1,28,28,1])#转换形式是n个纵深的图片，图片是尺寸是28*28*1
##添加dropout
keep_prob=tf.placeholder(tf.float32)

'''展示第五张图片'''
#定义展示图片
plt.imshow(mnist.train.images[4].reshape((28,28)),cmap='gray_r')
#定义标题、坐标轴等
plt.title('%i'%np.argmax(mnist.train.labels[4]))
plt.show()

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

##'''输出预测结果'''
###打开会话
##sess=tf.Session()
###全局变量初始化
##init=tf.global_variables_initializer()
##sess.run(init)
###提共样本
##xs=mnist.index(1,3)
###进行预测
##print (sess.run(prediction))

'''定义损失函数'''
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))#预测值是softmax模式，采用交叉熵作为损失函数
'''查看cross_entropy'''
tf.summary.scalar('cross_entropy',cross_entropy)

'''定义训练方法'''
train_step=tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)#采用Adam优化算法

'''保存训练'''
saver=tf.train.Saver()

'''打开会话'''
sess=tf.Session()

'''合并summary添加graph'''
merged=tf.summary.merge_all()
writer=tf.summary.FileWriter('logs/',sess.graph)

'''全局变量初始化'''
init=tf.global_variables_initializer()
sess.run(init)

'''保存到指定路径的指定文件'''
save_path=saver.save(sess,'my_net/3.ckpt')
print('Save to path:',save_path)

'''开始训练'''
#定义训练次数
for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.6})
    '''输出每次训练之后的精度'''
    if i%50==0:
        #获取预测值
        rs=sess.run(merged,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.6})
        writer.add_summary(rs,i)
        y_pre=sess.run(prediction,feed_dict={xs:mnist.test.images[:1000]})
        ys_v=mnist.test.labels[:1000]
        #比较预测与真实值是否一致转换成布尔矩阵
        pre_correct=tf.equal(tf.argmax(y_pre,1),tf.argmax(ys_v,1))
        #将布尔矩阵映射成0，1，并取平均
        accuracy=tf.reduce_mean(tf.cast(pre_correct,tf.float32))
        #运行得出结果
        result=sess.run(accuracy)
        print (result)


'''添加dropout避免过拟合，增加plot展示，增加tensorboard可视化'''
#添加dropout
#添加keep_prob,对output前一个函数进行dropout处理

#添加plot展示
##'''展示第五张图片'''
###定义展示图片
##plt.imshow(mnist.train.images[4].reshape((28,28)),cmap='gray_r')
###定义标题、坐标轴等
##plt.title('%i'%np.argmax(mnist.train.labels[4]))
##plt.show()

#添加tensorboard可视化
#添加graph
##writer=tf.summary.FileWriter("logs/")
#添加histograms
#tf.summary.histogram()
#添加scalars
#tf.summary.scalar()
#合并并添加

'''保存训练结果，并运用于实际'''
#添加saver
##saver=tf.train.Saver()
##save_path=saver.save(sess,'my_net/save_net.ckpt')
##print ("保存到：",save_path)
