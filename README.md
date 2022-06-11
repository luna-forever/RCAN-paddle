# 论文复现：[Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://paperswithcode.com/paper/image-super-resolution-using-very-deep)

*****

* RCAN
  * [一、简介](#一简介)
  
  * [二、复现结果](#二复现结果)
  
  * [三、个人看法](#三个人看法)
  
    

# **一、简介**

***

基于paddlepaddle框架复现Residual Channel Attention Networks(RCAN).

#### **论文**

Zhang, Yulun, et al. "Image super-resolution using very deep residual channel attention networks." *Proceedings of the European conference on computer vision (ECCV)*. 2018.

#### **参考项目**

https://github.com/yulunzhang/RCAN

#### **项目aistudio地址**

https://aistudio.baidu.com/aistudio/projectdetail/4203398
版本为ver1.0，运行`main.ipynb`即可开始训练。

# 二、复现结果

#### **指标（在set14上测试）**

|        模型        | PSNR  |  SSIM  |
| :----------------: | :---: | :----: |
|        论文        | 28.98 | 0.7910 |
|     Paddle训练     | 28.99 | 0.7913 |
| 预训练模型权重转换 | 28.98 | 0.7910 |

![image-202206111708uku](.\fig\image-20220611170828001.png)

![image-20220611172737254](.\fig\image-20220611172737254.png)

#### **部分结果展示**

### Zebra

原图![原图](.\fig\image-20220611173134818.png)ESDR![image-20220611173244056](.\fig\image-20220611173244056.png)

RCAN![image-20220611173324596](.\fig\image-20220611173324596.png)

### Flower

原图![image-20220611173448803](.\fig\image-20220611173448803.png)ESDR![image-20220611173510165](.\fig\image-20220611173510165.png)

RCAN![image-20220611173538691](.\fig\image-20220611173538691.png)

### Bridge

原图![image-20220611173643430](.\fig\image-20220611173643430.png)ESDR![image-20220611173701424](.\fig\image-20220611173701424.png)

RCAN![image-20220611173725449](.\fig\image-20220611173725449.png)



# 三、个人看法

1.论文中作者为了跟前人的文章对比，损失函数和前人一致采用了L1loss。可以尝试其他的损失函数，看看哪个效果好，如L2、感知和对抗损失等。

2.CA中平均池化可以换成更复杂的聚合功能，实现更多样化，更精确的结果。

3.卷积层的输出无法利用局部感受野外的上下文信息，可以替换成Transformer，对图片全局进行建模（替代平均池化的作用），这方面的论文现在有SwinIR，ELAN等。
