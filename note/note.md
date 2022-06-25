# python 中的[:-1]和[::-1]的具体使用
```python
a='python'
b=a[::-1] # 倒着一个一个取，最后的结果是把字符串进行翻转
print(b) #nohtyp
c=a[::-2] # 倒着隔一个取一个，倒数第一个最先取
print(c) #nhy
#从后往前数的话，最后一个位置为-1，倒数第二个位置为-2
d=a[:-1] #从位置0到位置-1之前的数
print(d) #pytho
e=a[:-2] #从位置0到位置-2之前的数
print(e) #pyth
```
# 图片的读入 
通常是有两种读入方式，分别是用PIL中的Image读入和用openCV读入。PIL(Python Imaging Library)是Python中最基础的图像处理库，OpenCV是一个很强大的图像处理库，适用面更广。两种读入方式是有区别的，主要有以下几个区别

图片格式不同，Image读入的是“RGB”，Opencv读入的是“BGR”。
读入图片的尺寸不同，Image读入的是 w h，Opencv读入的是h w c。其中w是宽，h是高，c是通道数。
Image读入是Image类无法直接显示其像素点的值（可以转换成numpy显示），Opencv读入的直接是numpy的格式。可以直接显示其像素值。

# 使用Image.open读出图像，加convert('RGB')的作用
读出来的图像是RGBA四通道的，A通道为透明通道，该通道值对深度学习模型训练来说暂时用不到，因此使用convert(‘RGB’)进行通道转换

# a[:,:,0]
```python
import numpy as np
a = [[[1,2,3],[4,5,6]]]
a = np.array(a)
b = a[:,:,0]
print(b) # [[1 4]]
# 把最后一维数组中，所有下标为0的数取出来
```
# fire.Fire()
```python
import fire
if __name__ == '__main__':
    fire.Fire()
# 该方法可以直接在命令行运行文件
# 例如：当前文件名为test.py
# 命令为：python test.py 方法名 必填参数值 可变参数 关键字参数
```
# 类名和self
在没有__init__方法的里面，类型调用和self调用是不同的，类名调用全局的值都是默认值，而self调用的才是当前值

# a[:,0]和a[:,0::4]的区别
```python
import numpy as np
a = np.array([[1,2,3,4],[5,6,7,8]])

b = a[:,0]
print(b)
# 输出为[1,5]
c = a[:,0::4]
print(c)
# 输出为[ [1]
#        [5]
#             ]
```

