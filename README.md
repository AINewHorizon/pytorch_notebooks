(关注公众号：AI新视野，发送‘资料’二字，免费获取50G人工智能视频教程！)
---
# 深度学习框架PyTorch 1.2 入门教程

代码地址：https://github.com/AINewHorizon/pytorch_notebooks

这篇教程将为你全面的介绍使用PyTorch训练神经网络的基本知识。

![](https://tuchuang-1259787532.cos.ap-beijing.myqcloud.com/image/1567390342391922.png)

本文会介绍如何使用PyTorch构建一个神经网络模型。比如，会堆叠少量的层，构建一个图像分类器模型，然后评估这个模型。

这次的教程会比较短，并且尽可能地避免使用"术语"和太难懂的代码。就是说，这可能是你能用PyTorch构建出的最基础的神经网络模型。实际上，这次要讲的非常基础，非常适合PyTorch和机器学习的初学者。


## 起步

在开始上手写代码之前，你需要先安装最新版本的 PyTorch。

```bash
pip3 install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
```

现在需要导入一些模块，这些模块将有助于获得必要的函数，以帮助建立的神经网络模型。主要导入的模块有 torch 和 torchvision。它们包含了让你入门 PyTorch 所需的大部分功能。但是，由于这是一个机器学习教程，还需要 torch.nn、torch.nn.functional 和 torchvision.transforms，它们都包含实用函数以构建的模型。可能不会使用下面列出的所有模块，但它们是你开始机器学习项目之前需要导入的典型模块。

```python
## The usual imports    
import torch    
import torch.nn as nn    
import torch.nn.functional as F    
import torchvision    
import torchvision.transforms as transforms    
## for printing image    
import matplotlib.pyplot as plt    
import numpy as np
```

使用下面的命令检查 PyTorch 版本，以确保你正在使用的是本教程使用的正确版本。在本教程中，使用 PyTorch 1.2。

```python
print(torch.__version__)
```


## 数据加载

开始一个机器学习的工程，首先需要加载数据。这里使用 [MNIST](http://yann.lecun.com/exdb/mnist/ "MNIST")数据集。这个数据集可以看做是机器学习的入门数据集。

该数据集中包含一系列的手写数字图像，图像大小为28*28。下面会对数据进行介绍，加载数据的时候采用批量大小为32，如下图所示。

![](https://tuchuang-1259787532.cos.ap-beijing.myqcloud.com/image/1567391575130544.png)

下面是加载数据的完整步骤：

-   利用transform模块加载数据，并将数据转换为tensor，tensor是一种储存数据结构一种有效的方式。
-   采用Dataloader类构建非常方便的数据加载器，这有助于将数据批量输送到神经网络模型中，从现在开始将接触到batch的概念，现在先将它看做是数据的子集。
-   如上所述，还将通过在数据加载器中设置批处理参数来创建批量数据，在这里将其设置为32，如果你设置成64也可以。
  
```python
## parameter denoting the batch size    
BATCH_SIZE = 32    
## transformations    
transform = transforms.Compose(    
[transforms.ToTensor()])    
## download and load training dataset    
trainset = torchvision.datasets.MNIST(root='./data', train=True,    
                                        download=True, transform=transform)    
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,    
                                        shuffle=True, num_workers=2)    
## download and load testing dataset    
testset = torchvision.datasets.MNIST(root='./data', train=False,    
                                        download=True, transform=transform)    
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,    
                                        shuffle=False, num_workers=2)
```

下面来仔细看看trainset和testset包含的内容：

```python
print(trainset)
print(testset)
## output
Dataset MNIST     
Number of datapoints: 60000     
Root location: ./data     
Split: Train     
StandardTransform 
Transform: Compose(ToTensor()) 
Dataset MNIST     
Number of datapoints: 10000     
Root location: ./data     
Split: Test    
StandardTransform Transform: Compose(ToTensor())
```
下面详细介绍几个参数：

-   BATCH_SIZE是模型中使用的batch的大小。
-   transform是用于对数据进行转换的模块。下面我将展示一个示例，以确切地演示它是如何为其使用的 training set 和 testset 提供更多信息的，testset 包含实际的 dataset对象。  注意，设置train=True对应的是训练数据，train=False对应的是测试数据。训练和测试数据的比例是85%/15%,即训练数据60000，测试数据10000个。

-   trainloader储存着数据加载器的实例，可以对数据进行打乱和构建批处理。

再看一看transforms.Compose(...) 函数和它的功能。随便生成一张图像，看看它是怎么使用的。

```python
image = transforms.ToPILImage(mode='L')(torch.randn(1, 96, 96))
```

显示生成的图像

plt.imshow(image)

输出结果

![](https://tuchuang-1259787532.cos.ap-beijing.myqcloud.com/image/1567392711919199.png)

现在有了一张图像，对它应用一些虚拟变换。对图像进行45度旋转，以下的转换过程将解决这个问题：

```python
## dummy transformation    
dummy_transform = transforms.Compose(    
    [transforms.RandomRotation(45)])    

dummy_result = dummy_transform(image)    
plt.imshow(dummy_result)
```

输出结果

![](https://tuchuang-1259787532.cos.ap-beijing.myqcloud.com/image/1567392773936172.png)

你可以使用transforms.Compose(...)对图像进行多种变换。Pytorch内置了多种变换函数，事实上你也可以自己写转换函数。看看另外一个变换的效果：旋转+水平翻转

```python
## dummy transform     
dummy2_transform = transforms.Compose(    
    [transforms.RandomRotation(45), transforms.RandomVerticalFlip()])    

dummy2_result = dummy2_transform(image)    
plt.imshow(dummy2_result)
```

输出结果

![](https://tuchuang-1259787532.cos.ap-beijing.myqcloud.com/image/1567392837243178.png)  

是不是看起来很酷，你可以尝试其他的转换方法。关于进一步研究的数据的主题，让接下来仔细看看的图像数据集。


## 探索数据

作为一名从业者和研究人员，我总是花费一些时间和精力来探索和理解我的数据集。这是一个有趣的并且很有意义的做法，以确保训练模型之前一切井然。

让检查训练和测试数据集包含的内容。我将使用 matplotlib 库从数据集打印出一些图像。使用一点numpy代码，我可以将图像转换为正确的格式来打印出来。下面我打印出一整批 32 张图像：

```python
## functions to show an image    
def imshow(img):    
    #img = img / 2 + 0.5     # unnormalize    
    npimg = img.numpy()    
    plt.imshow(np.transpose(npimg, (1, 2, 0)))    

## get some random training images    
dataiter = iter(trainloader)    
images, labels = dataiter.next()    
## show images    
imshow(torchvision.utils.make_grid(images))
```

输出结果：

![](https://tuchuang-1259787532.cos.ap-beijing.myqcloud.com/image/1567392938151316.png)  

打印batches的维度信息：

```python
for images, labels in trainloader:    
    print("Image batch dimensions:", images.shape)    
    print("Image label dimensions:", labels.shape)    
    break
```

结果

```python
Image batch dimensions: torch.Size([32, 1, 28, 28]) 
Image label dimensions: torch.Size([32])
```


## 构建模型

现在可以构建用于执行图像分类任务的卷积神经网络模型了。为了简化，的将堆叠使用一个dense层，一个dropout层和一个output层来训练模型。

关于模型的讨论：

首先，以下结构涉及名为MyModel的类，是用于在PyTorch中构建神经网络模型的标准代码：

```python
## the model    
class MyModel(nn.Module):    
    def __init__(self):    
        super(MyModel, self).__init__()    
        self.d1 = nn.Linear(28 * 28, 128)    
        self.dropout = nn.Dropout(p=0.2)    
        self.d2 = nn.Linear(128, 10)    
    def forward(self, x):    
        x = x.flatten(start_dim = 1)    
        x = self.d1(x)    
        x = F.relu(x)    
        x = self.dropout(x)    
        logits = self.d2(x)    
        out = F.softmax(logits, dim=1)    
        return out
```

-   各层定义在def _init()函数里。super(...).__init__()  是将所有的东西组合在一块。模型中堆叠 了一个隐藏层 (self.d1），其后跟着一个dropout层 (self.dropout)，然后是分类的输出层(self.d2)。
-   nn.Linear(...)定义 了dense层，其中的输入和输出维度。这里的输入和输出分别与输入的特征和输出的特征所分别对应。
-   nn.Dropout（）用于定义Dropout层，Dropout层是在深度学习中用于防止过拟合的方法。 这意味着Dropout在模型训练过程中扮演着一个正则化的功能。使用这个方法主要是为 了的模型在其他数据集上也能表现良好。Dropout随机的将神经网络中的一些单元置为0，在构建的模型中将Dropout的参数设置为0.2.了解更多关于Dropout的信息，请阅读关于Dropout的[说明文档](https://pytorch.org/docs/stable/nn.html#dropout "说明文档")。
-   模型的入口也就是数据输入到神经网络模型的位置放在了forward（）函数之下。通常也会添加其他变换函数，用于训练过程中对图像进行变换。
-   在forward函数中，对输入的数据进行一系列的计算。1）将图像拉平，从2D的图像（28*28）转化为1D（1*784）；2）将1D的图像按照批次输入到第一个隐含层；3）隐含层的输出采用非线性激活函数Relu。了解Relu函数的功能并不是很重要，但他的作用确实异常的显著，使得在大数据集上训练更快更有效；4）正如上面解释的那样，dropout层可以帮助解决模型在训练数据上过拟合的问题；5）紧接着将dropout的结果输入到分类层d2;6)最后的结果输入到softmax函数中，将概率分布归一化，从而帮助计算分类的准确率；7）这就是最后的输出结果。

直观地说，下面就是刚刚构建的模型图。但是隐藏层比图中显示的大很多，但由于空间限制，该图应被视为实际模型的近似表示。

![](https://tuchuang-1259787532.cos.ap-beijing.myqcloud.com/image/1567395105127276.png)

正如我在前面的教程中所做的那样，我总是鼓励用一个批处理来测试模型，以确保输出的维度符合的预期。请注意，是怎样迭代数据加载器，它可以方便地存储图像和标签对。out 包含模型的输出，这是应用了softmax层的logits来帮助预测。

```python
## test the model with 1 batch    
model = MyModel()    
for images, labels in trainloader:    
    print("batch size:", images.shape)    
    out = model(images)    
    print(out.shape)    
    break
```

输出结果：

```python
batch size: torch.Size([32, 1, 28, 28]) 
torch.Size([32, 10])
```

可以清楚地看到，返回批次中有10个输出值与批次中的每个图像相关联；这些值将被用于检查模型的性能。


## 训练模型

在准备好训练模型之前，需要设置一个损失函数、一个优化器和一个效用函数来计算模型的准确性：

```python
learning_rate = 0.001    
num_epochs = 5    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
model = MyModel()    
model = model.to(device)    
criterion = nn.CrossEntropyLoss()    
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```

-   learning_rate 学习率，用于优化模型的权重，可以看做是模型的一个参数。

-   num_epochs训练步骤数目，为了让训练时间不太长，设置epoch为5.

-   device训练模型的硬件设备，如果有GPU的话，采用GPU训练，否则，默认采用CPU训练。

-   model构建的模型实例。

-   model.to(divice)设置模型在什么硬件设备上训练。 

-   criterion用于计算模型损失的度量标准，通过前向传播和反向传播优化权重。

-   optimizer优化工具，在反向传播中调整权重，注意，它需要一个学习率和模型参数，这些是优化器的一部分。稍后会详细介绍。

效用函数将在下面进行定义，它有助于计算模型的准确率。了解它是如何计算的在目前来看并不是很重要，只需要了解它是通过比较模型的输出结果（预测）和实际目标值（数据集的标签）来计算预测的准确率。

```python
## utility function to compute accuracy    
def get_accuracy(output, target, batch_size):    
    ''' Obtain accuracy for training round '''    
    corrects = (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()    
    accuracy = 100.0 * corrects/batch_size    
    return accuracy.item()
```


### 开始训练

现在可以训练模型了，代码如下：

```python
## train the model    
for epoch in range(num_epochs):    
    train_running_loss = 0.0    
    train_acc = 0.0    

    ## commence training    
    model = model.train()   

    ## training step    
    for i, (images, labels) in enumerate(trainloader):    

        images = images.to(device)    
        labels = labels.to(device)    

        ## forward + backprop + loss    
        predictions = model(images)    
        loss = criterion(predictions, labels)    
        optimizer.zero_grad()    
        loss.backward()    

        ## update model params    
        optimizer.step()    

        train_running_loss += loss.detach().item()    
        train_acc += get_accuracy(predictions, labels, BATCH_SIZE)  

    model.eval()    
    print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \    
    %(epoch, train_running_loss / i, train_acc/i))
```

-   训练模型的第一步是定义训练循环

```python
for epoch in range(num_epochs):
    ...
```

-   定义了两个变量training_running_loss和train_acc，帮助在不同批次训练时监视训练精度和损失。

-   model.train()设置模型的模式，准备训练。

-   注意，是在dataloader上迭代数据，这很方便地将的数据和标签一一对应。

-   第二个for循环，指的是在每一步训练过程中，迭代batch中全部的数据。

-   往模型中传入数据通过model(image),输出结果代表模型的预测结果。

-   预测结果和实际类别标签进行对应和比较，从而计算训练损失。

-   在更新权重之前，需要做以下工作：1）利用optimizer重置梯度变量（optimizer.zero_grad() ）；2）这一步是安全的，并不会重写模型计算的梯度，模型上一次计算的梯度暂时存储在缓存里，可以通过loss.backward() 进行重访；3）loss.backward()计算模型损失各参数对应的梯度；4）optimizer.step()确保模型参数更新；5）最终获得损失和精度，通过这两个指标可以告诉模型训练的情况。

输出结果如下：

```
Epoch: 0 | Loss: 1.6167 | Train Accuracy: 86.02 
Epoch: 1 | Loss: 1.5299 | Train Accuracy: 93.26 
Epoch: 2 | Loss: 1.5143 | Train Accuracy: 94.69 
Epoch: 3 | Loss: 1.5059 | Train Accuracy: 95.46 
Epoch: 4 | Loss: 1.5003 | Train Accuracy: 95.98
```
训练完毕，可以清楚地看到损失值一直在下降，精度一直在上升，说明的模型是有效的，可以用于图像的分类任务。

可以通过对测试数据计算精度，来验证的模型是否在分类任务表现良好。通过下面的代码，你可以看到，在MINIST分类任务上，的模型表现的很好。
```python
test_acc = 0.0    
for i, (images, labels) in enumerate(testloader, 0):    
    images = images.to(device)    
    labels = labels.to(device)    
    outputs = model(images)    
    test_acc += get_accuracy(outputs, labels, BATCH_SIZE)    

print('Test Accuracy: %.2f'%( test_acc/i))
```
输出测试精度：
```
Test Accuracy: 96.32
```


## 结语

恭喜你已完成本教程👏。这是一个全面的教程，目的是对使用神经网络和PyTorch进行基本的图像分类做一个非常基本的介绍。\
本教程深受此TensorFlow教程的启发（<https://www.tensorflow.org/beta/tutorials/quickstart/beginner>）。 感谢相应参考文献的作者所做的宝贵工作。


🌎[ GitHub Repo](https://github.com/AINewHorizon/pytorch_notebooks " GitHub Repo") （Github 仓库）

---
![AIcode.jpg](https://tuchuang-1259787532.cos.ap-beijing.myqcloud.com/image/undefinedAIcode.jpg)
<span style="display:block;text-align:center;color:orangered;">请长按或扫描关注本公众号</span>
![](https://tuchuang-1259787532.cos.ap-beijing.myqcloud.com/image/20190502-092925-73ef.gif)
