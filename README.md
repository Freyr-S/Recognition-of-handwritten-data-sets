# Recognition-of-handwritten-data-sets
## 1.定义一个类neuralnetwork
###  该类包括初始化函数、训练函数、查询函数
--------
初始化  
```
 def __init__(self, inputcodes, hiddencodes, outputcodes, learningrate):
        self.incodes = inputcodes
        self.hicodes = hiddencodes
        self.outcodes = outputcodes
        self.lr = learningrate
        self.wih = numpy.random.normal(0.0, pow(self.hicodes, -0.5), (self.hicodes,self.incodes))
        self.who = numpy.random.normal(0.0, pow(self.outcodes, -0.5), (self.outcodes,self.hicodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
```
inputcodes为输入层节点数，hiddencodes为隐藏层节点数，outputcodes为输出层节点数，learningrate为学习率，wih为输入层到隐藏层的权重矩阵，who为隐藏层到输出层的权重矩阵，activation_function为激活函数

----
训练train
```
    def train(self, input_list, target_list) :
        inputs = numpy.array(input_list, ndmin = 2).T
        target = numpy.array(target_list, ndmin = 2).T
        hidden_input = numpy.dot(self.wih, inputs)
        hidden_output = self.activation_function(hidden_input)
        final_input = numpy.dot(self.who, hidden_output)
        final_output = self.activation_function(final_input)
        output_error = target - final_output
        hidden_error = numpy.dot(self.who.T, output_error)
        self.who += self.lr * numpy.dot((output_error * final_output * (1 - final_output)), numpy.transpose(hidden_output))
        self.wih += self.lr * numpy.dot((hidden_error * hidden_output * (1 - hidden_output)),
                                        numpy.transpose(inputs))
        pass
 ```
基于所计算输出与目标输出之间的误差，不断改进权重，训练网络
 
 ----
 查询query
 ```
     def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T
        hidden_input = numpy.dot(self.wih, inputs)
        hidden_output = self.activation_function(hidden_input)
        final_input = numpy.dot(self.who, hidden_output)
        final_output = self.activation_function(final_input)

        return final_output
  ```
  计算输出  
  ## 2.训练网络
  ```
  traindata_file = open("C:/Users/86189/.kaggle/train.csv", 'r')
traindata_list = traindata_file.readlines()
traindata_file.close()
```
读取训练数据集并用列表记录
```
    for record in traindata_list[1:]:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:])/ 255.0 * 0.99) + 0.01
        target = numpy.zeros(output_nodes) + 0.01
        target[int(all_values[0])] = 0.99
        n.train(inputs, target)
        pass
 ```
 因为要跳过列表traindata_list第一项的字符串，所以要遍历列表traindata_list除第一项外的其他项 `for record in traindata_list[1:]`
 同样要跳过列表all_values第一项的标签，所以为`all_values[1:]`
 ```
 testdata_file = open("C:/Users/86189/.kaggle/test.csv",'r')
testdata_list = testdata_file.readlines()
testdata_file.close()
```
读取测试数据集并用列表记录
```
list = []
id = 1

for record in testdata_list[1:]:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    lable = numpy.argmax(outputs)
    list.append([id,lable])
    df = pd.DataFrame(list, columns=['ImageId', 'Label'])
    id += 1
    pass
```
计算结果。同样要跳过列表testdata_list的第一项 `testdata_list[1:]`，但因为此时列表all_values没有标签，所以不用跳过第一项 `numpy.asfarray(all_values)`。
将结果按照规定的要求先存入列表 ` list.append([id,lable])`，再用list的数据创建dataframe`df = pd.DataFrame(list, columns=['ImageId', 'Label'])`
最后保存为csv文件`df.to_csv('submission.csv', index=False)`

----
将结果拿到kaggle上比赛，识别率为0.97100

----
完整代码：
```
import numpy
import scipy.special
import pandas as pd

class neuralnetwork :

    def __init__(self, inputcodes, hiddencodes, outputcodes, learningrate):
        self.incodes = inputcodes
        self.hicodes = hiddencodes
        self.outcodes = outputcodes
        self.lr = learningrate
        self.wih = numpy.random.normal(0.0, pow(self.hicodes, -0.5), (self.hicodes,self.incodes))
        self.who = numpy.random.normal(0.0, pow(self.outcodes, -0.5), (self.outcodes,self.hicodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, input_list, target_list) :
        inputs = numpy.array(input_list, ndmin = 2).T
        target = numpy.array(target_list, ndmin = 2).T
        hidden_input = numpy.dot(self.wih, inputs)
        hidden_output = self.activation_function(hidden_input)
        final_input = numpy.dot(self.who, hidden_output)
        final_output = self.activation_function(final_input)
        output_error = target - final_output
        hidden_error = numpy.dot(self.who.T, output_error)
        self.who += self.lr * numpy.dot((output_error * final_output * (1 - final_output)), numpy.transpose(hidden_output))
        self.wih += self.lr * numpy.dot((hidden_error * hidden_output * (1 - hidden_output)),
                                        numpy.transpose(inputs))
        pass

    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T
        hidden_input = numpy.dot(self.wih, inputs)
        hidden_output = self.activation_function(hidden_input)
        final_input = numpy.dot(self.who, hidden_output)
        final_output = self.activation_function(final_input)

        return final_output

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1

n = neuralnetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

traindata_file = open("C:/Users/86189/.kaggle/train.csv", 'r')
traindata_list = traindata_file.readlines()
traindata_file.close()

num = 5

for i in range(num):
    for record in traindata_list[1:]:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:])/ 255.0 * 0.99) + 0.01
        target = numpy.zeros(output_nodes) + 0.01
        target[int(all_values[0])] = 0.99
        n.train(inputs, target)
        pass
    pass

testdata_file = open("C:/Users/86189/.kaggle/test.csv",'r')
testdata_list = testdata_file.readlines()
testdata_file.close()

list = []
id = 1

for record in testdata_list[1:]:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    lable = numpy.argmax(outputs)
    list.append([id,lable])
    df = pd.DataFrame(list, columns=['ImageId', 'Label'])
    id += 1
    pass

df.to_csv('submission.csv', index=False)
```
