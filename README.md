# first-program-cnn-mnist
第一次用cnn编译一个mnist图像识别，采用的是最简单的tensortflow框架的卷积神经网络方式
include use the pillow library to deal with the image simply（convert mode，resize，paste，create a new image and reshape to be an array for it can fit to the input of the cnn）
create two conv layer two pool layer and two full connect layer
compute cross function by cross_entropy
train the net throught ADAM method
and add tensorboard to the net 
save data by Saver
of course there are save.restore() to restore the data
