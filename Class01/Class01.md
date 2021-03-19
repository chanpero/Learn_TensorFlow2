# 笔记1
## 1.张量Tensor
| 维数 | 名字 | 例子 |
| --- | --- | --- |
| 0-D | 标量 scalar | s=1024 |
| 1-D | 向量 vector | v=[1, 2, 3] |
| 2-D | 矩阵 matrix | m=[[1,2,3],[4,5,6],[7,8,9]] |
| n-D | 张量 tensor | t=[[[...]]] |
张量可以表示0阶到n阶数组
*****
## 2. 常用函数
```python
a = tf.constant([1,5],dtype=tf.int64)
b = tf.convert_to_tensor( a, dtype=tf.int64 )

a = tf.zeros([2, 3])
b = tf.ones(4)
c = tf.fill([2, 2], 9)
```

```python
# 生成正态分布的随机数，默认均值为0，标准差为1
d = tf.random.normal ([2, 2], mean=0.5, stddev=1)

# 生成截断式正态分布的随机数,保证范围在（μ-2σ，μ+2σ）内
e = tf.random.truncated_normal ([2, 2], mean=0.5, stddev=1)

# 生成均匀分布随机数,范围 [minval, maxva)
f = tf.random.uniform([2, 2], minval=0, maxval=1)

# 强制转换格式
tf.cast (x1, tf.int32)
# 计算张量维度上元素的最小值、最大值、均值、和
tf.reduce_min(x1)
tf.reduce_max(x1)
tf.reduce_mean(x1)
tf.reduce_sum(x1, axis=0)
```

对应元素四则运算：</br>
+ tf.add(a,b)
+ tf.subtract(a,b)
+ tf.multiply(a,b)
+ tf.divide(b,a)

+ tf.square(a)
+ tf.pow(a, 3)
+ tf.sqrt(a)
+ tf.matmul(a, b) - 矩阵乘法

```python
# tf.Variable()将变量标记为“可训练”，被标记的变量会在反向传播中记录梯度信息。神经网络训练中，常用该函数标记待训练参数。
w1 = tf.Variable(tf.random.normal([4, 3], mean=0, stddev=1))

# 切分传入张量的第一维度，生成输入特征/标签对，构建数据集
tf.data.Dataset.from_tensor_slices((input_features, labels))

# with结构记录计算过程，gradient求出张量的梯度
with tf.GradientTape() as tape:
    w = tf.Variable(tf.constant(3.0))
    loss = tf.pow(w,2) #loss=w2 loss’=2w
grad = tape.gradient(loss,w)
print(grad)

# 转换独热编码
tf.one_hot(labels, depth=classes)

tf.nn.softmax(y)

# w -= a
w.assign_sub(a)

# 返回张量沿指定维度最大值的索引
tf.argmax(x, axis)
```

## 3.神经网络实现鸢尾花分类
+ 准备数据
    + 数据集读入 - load_data
    + 数据集乱序 - shuffle
    + 生成训练集和测试集（即 x_train / y_train）
    + 配成对(输入特征，标签)，每次读入一小撮（batch）- from_tensor_slices
+ 搭建网络
    + 定义神经网路中所有可训练参数 - tf.Variable()
+ 参数优化
    + 嵌套循环迭代，with结构更新参数，显示当前loss - with GradientTape()
+ 测试效果
    + 计算当前参数前向传播后的准确率，显示当前acc
+ acc / loss可视化

