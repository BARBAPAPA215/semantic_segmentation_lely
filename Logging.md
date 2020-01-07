
# 3/12/2019 

周计划 找 paper：
* 注意引用率 
* 2019+2018
* 开源，有github
* 一两周找 然后精读
* 范围：bmvc， Iccv， cvpr， eccv…

新方法和deeplab v3+  
先用已有的数据集  
下周三 lely开会 11/12  
19/12 来lely  
12，1月 两周假期 

# 13/12/2019

-准备讨论 （仅考虑有github的论文，以及 2018，2019 发表的）

*  DeepLabv3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation 2018  
在PASCAL VOC 2012数据集上的准确度排名第一  
PASCAL VOC 2012 排名前几名的都是2018年的论文，SANet（2019）排名第六  
该论文在Cityscapes 排名第九  

*  Cityscapes 的排名第一为HRNetV2 + OCR (w/ ASP) : Object-Contextual Representations for Semantic Segmentation  
该论文提出好多种方法，在Cityscapes上排名占据1.3.6

*  Cityscapes数据集上排名第十的论文Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation  
仅比DeepLab v3+ 低1.2%  
但是是更轻量级的模型

*  轻量级网络, 权衡速度和准确性
*  实时语义分割， 最佳的是DFANet，旷视的论文，2019年发表的好像还没有开源但是有非官方的code，相比于先前的 BiSeNet 和 ICNet 等，在相近精度的条件下 DFANet 可以极大地减少运算量。而和 ENet 等小计算量方法相比，DFANet的精度有巨大提升。在计算量受限的前提下DFANet是能在高分辨率图片上达到准确度媲美主流“大模型”的轻量化网络。

-Feedback
* deeplabv3+ 以及另外三个左右的paper精读

* ps: 其他的数据集 比如coco，
* 注意加星量
* cityspace的数据集变化较少，不是最佳选择
* 不考虑实时的，表现不好
* 下周三 周四来lely

# Gpu
ssh -X deep@172.26.41.164 显示图片  
ssh deep@172.26.41.164  
密码： learning  
screen -S yuanhao

# 7/1/2020  

-讨论文章
* 1.Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation （60,254）

* 2.Context Encoding for Semantic Segmentation （1164）  
https://github.com/zhanghang1989/PyTorch-Encoding  
ADE20K val:6  
PASCAL Context : 10  
Document 写的很详细  
测试pre-trained model 还在运行没有报错

* 3.Object-Contextual Representations for Semantic Segmentation （669）  
https://github.com/PkuRainBow/OCNet.pytorch
ADE20K: 2  
Cityscapes test: 1  
COCO-Stuff test: 1  
PASCAL Context : 1  
ADE20K val: 2 

* 4.Dual Attention Network for Scene Segmentation（1,244）
https://github.com/junfu1115/DANet  
COCO-Stuff test: 2  
Cityscapes test: 13  
PASCAL Context : 8

* 5.HRNetV2(HRNetV2-W48)： Deep High-Resolution Representation Learning for Visual Recognition （700）    
https://github.com/CSAILVision/semantic-segmentation-pytorch  
http://scenesegmentation.csail.mit.edu/  
PASCAL Context: # 3
运行报错

-数据集
* PASCAL-Context Dataset：(少)  
https://cs.stanford.edu/~roozbeh/pascal-context/

* ADE20K: 多, 稀疏  
是用来做 scene parsing 的一个非常大的数据集合，包含 150 中物体类型，由 MIT CSAIL 研究组维护。数据集在 http://groups.csail.mit.edu/vision/datasets/ADE20K/

-运行

* 5.HRNetV2(HRNetV2-W48)： Deep High-Resolution Representation Learning for Visual Recognition 
现在的问题:  
train and test：要求 update gpu driver 或者 改 pytorch 的 version（ compiled with current gpu driver）

        pytorch conda install pytorch torchvision cuda90 -c pytorch  
	问题2 modulenotfounderror，例如 tensorboardx: (用pip3)  
	问题3 8 errors detected in the compilation of “/tmp/tmpxft_00001c20_00000000-6_inplace_abn_cuda.cpp1.ii” 
（修复过pip之后没有重新运行）

    (改参数，4个gpu  
    不能在terminal 里面使用conda。。重装了anaconda  
    Python 2.7？）
* 2.Context Encoding for Semantic Segmentation
        pip install torch-encoding  
	Screw up pip  
	(Modulenotfounderror: pip._vendor.retrying （unable install packages)  
	试Pip3: exception  
	request.exceptions.HTTPError: 404 client error 但是网页存在 https://pypi.org/simple/torch-encoding/ ）  
	解决： 重装 pip

        which pip
	/usr/local/bin/pip

		/usr/local/bin/pip uninstall pip
		apt-get remove pythonl-pip
		apt-get install python-pip

    Test pre-trained model 目前没有报错, evaluation还在跑  
    没有yaml文件 调参？

ps: MAX_ITER一开始调参时可以设小一点，能够短时间内看效果

-问题：
* 训练用public的数据集，测试用自己的数据集？
* 用pre-trained 的model 测试我们的数据集？

-Feedback
* 注意Licence
* 注意运行时间
* visualize 结果
* train自己的model（pretrain的model表现异常好）
