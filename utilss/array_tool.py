"""
tools to convert specified type
"""
import torch as t
import numpy as np

# detach()
# .detach()就是返回一个新的tensor，并且这个tensor是从当前的计算图中分离出来的。但是返回的tensor和原来的tensor是共享内存空间的。
# 举个例子来说明一下detach有什么用。 如果A网络的输出被喂给B网络作为输入， 如果我们希望在梯度反传的时候只更新B中参数的值，而不更新A中的参数值，这时候就可以使用detach()
# .cpu()
# 将数据的处理设备从其他设备（如.cuda()拿到cpu上），不会改变变量类型，转换后仍然是Tensor变量。
# .numpy()
# Tensor.numpy()将Tensor转化为ndarray，这里的Tensor可以是标量或者向量（与item()不同）转换前后的dtype不会改变
# .item()
# 将一个Tensor变量转换为python标量（int float等）常用于用于深度学习训练时，将loss值转换为标量并加，以及进行分类任务，计算准确值值时需要
# .data()
# Tensor.data和Tensor.detach()一样， 都会返回一个新的Tensor， 这个Tensor和原来的Tensor共享内存空间，一个改变，另一个也会随着改变，且都会设置新的Tensor的requires_grad属性为False。
# 这两个方法只取出原来Tensor的tensor数据， 丢弃了grad、grad_fn等额外的信息。
# tensor.data是不安全的, 因为 x.data 不能被 autograd 追踪求微分

def tonumpy(data):
    # 该函数的功能是把data转化成numpy类型的数据
    # isinstance 用来判断一个对象的变量类型
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, t.Tensor):
        return data.detach().cpu().numpy()


def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = t.from_numpy(data)
    # 为什么已经是tensor类型了，还要进行data.detach()
    # 因为这里不需要更新data之前模型的参数  而上面一开始就是numpy类型，显然不能进行反向传播
    if isinstance(data, t.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor


def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, t.Tensor):
        return data.item()