# 【深度学习模型训练】使用MGN DG-Net算法进行行人重识别

关于本实验的参考资料如下：

MGN论文：https://arxiv.org/abs/1804.01438v1

MGN github：https://github.com/seathiefwang/MGN-pytorch

DG-Net论文：https://arxiv.org/abs/1904.07223

DG-Net github：https://github.com/NVlabs/DG-Net#news

本实验使用以上两种算法为研究基础，在复现论文结果的基础上进行部分改进。原理部分请查阅论文，这里仅对实验平台的搭建及过程进行讲解。由于疫情期间，刚开始研究的时候没有连上学校实验室的服务器，所以使用了kaggle notebook平台，kaggle notebook平台可以完成代码的修改及中小模型的训练。在无法使用服务训练时，是调整代码、训练中小模型的一个较好替代。以下简单介绍下kaggle的深度学习平台及简单的使用方法。



## kaggle notebook

kaggle notebook是kaggle为深度学习研究人员打造的深度学习平台，拥有相对完善的预装环境，以上两种模型在pytorch环境下进行训练，均可在kaggle notebook上直接跑，基本不需要安装其他关联包。

### 平台限制

https://www.kaggle.com/docs/kernels

每个Notebook编辑会话都具有以下资源：

- **单会话9小时执行时间，一周30小时的GPU总使用时间**

- 5 GB自动保存的磁盘空间（/ kaggle /工作中）

- 16 GB的临时暂存磁盘空间（/ kaggle /工作区之外）

  CPU规格

  - 4个CPU核心
  - 16 GB的RAM

  GPU规格

  - 2个CPU核心
  - 13 GB的RAM

根据平台限制需要合理规划处理器的使用和代码的运行时间。

### 预装操作

如果需要预装其他相关包，可按照如下命令进行。对于Python，可以在前面加！来运行任意的shell命令到代码单元。

```python
#使用pip安装新软件包
!pip install my-new-package
#升级或降级现有软件包
!pip install my-existing-package==X.Y.Z
```
### 使用步骤

1. 在官网https://www.kaggle.com/ 注册一个kaggle账号，并进入个人空间

2. 在个人空间左侧栏里选择notebooks，进入后选择 new notebook选项，随后设置notebook语言和形式。我这里选择的是Python和Notebook，随后进入编辑页面

3. kaggle notebook使用方法类似于jupter notebook，需要上传自己的数据集和训练的Python代码，点击+Add data即可。在代码单元使用shell命令运行，生成的结果可在output下查看和下载。(注意，上传文件需要科学上网,并且较大的输出文件难以下载)

4. 若想使用GPU训练，账号需要绑定手机号，完成手机验证后，在settings-accelerator下可选用。TPU貌似需要美国的账号才可使用。



## 服务器平台搭建

由以上介绍，**kaggle使用GPU的时间有限，一周只有30小时的总使用时间并且单个会话限制9小时**。所以对于GAN等生成模型或者较大的网络而言（50层级以上），训练时间难以满足。使用服务器搭建平台训练网络无疑的更优。


### 服务器连接步骤

以下为首次使用服务器的小伙伴进行介绍。

1. 首先需要联系服务器管理员为你创建一个用户，设置用户密码。普通用户创建后，会在服务器的/home/下创建相应的文件夹(用户目录），每个普通用户在自己的用户目录下搭建自己的环境

2. 注意服务器的Nvidia驱动版本， CUDA版本，避免tensorflow pytorch对版本的不支持问题

3. 通过SSH连接服务器（一般使用putty进行命令行操作，使用winscp进行py文件和数据集传输）

   - putty

     创建会话：首先输入正确的IP地址，同时可以将地址进行存储方便以后访问。

     正确连接后，跳出服务器命令行，输入用户名和密码，正确连接    

   - winscp

     与putty的连接方法类似，创建会话后输入正确的用户名和密码即可。

### 环境搭建

1. 登录完成后，需要切换到自己的用户文件夹下进行配置。

      ```python
      cd /home/XXX
      ```

2. 推荐使用**Anaconda**为多环境配置，设置多个虚拟环境运行，减少版本冲突的影响。

3. 创建新的虚拟环境并激活，这里使用Anaconda3，python3.6的环境

      ```shell
      conda create -n XXX python=3.6
      conda activate XXX
      ```

4. 增加新的安装包，可以使用pip conda两种方法

      ```shell
      pip install numpy
      conda install numpy
      ```

5. 安装pytorch和tensorflow

      ```shell
      conda install pytorch==1.0.0 torchvision==0.2.1
      conda install tensorflow-gpu=1.1
      ```

6. 使用GPU进行深度学习时，为了避免与他人同时使用一个GPU导致训练效率过低，可时刻查看显卡使用情况

      ```shell
      watch -n 1 nvidia-smi
      #若默认显卡被占用，修改代码使用其余显卡
      config = tf.ConfigProto(allow_soft_placement=True)
      config.gpu_options.allow_growth = True
      os.environ["CUDA_VISIBLE_DEVICES"] = "0"
      ```

7. 若挂载的VPN时常断连导致训练中断，可在指令前加nohup，后加&来使用后台进行训练

      ```shell
      nohup  训练指令 > train.log & 
      ```



## 使用MGN进行行人重识别

训练及测试均使用Market-1501，这里我在原网络的损失函数和架构上做了一点小变动，各个文件的主要改动如下：

pytorch4      ——原始网络  CrossEntropy+Triplet损失 & 1-2-3支路架构

pytorch11     ——损失改进  CrossEntropy+Triplet+CenterLoss损失 & 1-2-3支路架构

pytorch25     ——损失改进  CrossEntropy+MSMLoss损失 & 1-2-3支路架构       

pytorch29     ——架构改进  CrossEntropy+Triplet损失 & 1-2-4支路架构

pytorch30     ——架构改进  CrossEntropy+Triplet损失 & 1-2-4支路架构

MGN-pytorch-master   ——原始网络详细注释



### 训练

```python
#pytorch4、pytorch29、pytorch30指令
python3 /home/youwenjing/pytorch4/MGN-pytorch-master/main.py --reset --datadir /home/youwenjing/Market-1501/ --batchid 8 --batchtest 16 --test_every 50 --epochs 400 --decay_type step_320_380 --loss 1*CrossEntropy+2*Triplet --margin 1.2 --random_erasing --save adam_0 --nGPU 1  --lr 2e-4 --optimizer ADAM  --re_rank --amsgrad
#pytorch11指令
python3 /home/youwenjing/pytorch11/MGN-pytorch-master/main.py --reset --datadir /home/youwenjing/Market-1501/ --batchid 8 --batchtest 16 --test_every 50 --epochs 400 --decay_type step_320_380 --loss 1*CrossEntropy+2*Triplet+0.005*CenterLoss --margin 1.2 --random_erasing --save adam_1 --nGPU 1  --lr 2e-4 --optimizer ADAM --re_rank --amsgrad
#pytorch25指令
python3 /home/youwenjing/pytorch25/MGN-pytorch-master/main.py --reset --datadir /home/youwenjing/Market-1501/  --batchid 8 --batchtest 16 --test_every 50 --epochs 400 --decay_type step_320_380 --loss 1*CrossEntropy+2*MSMLLoss --margin 1.2 --random_erasing --save adam_2 --nGPU 1  --lr 2e-4 --optimizer ADAM --re_rank --re_rank --amsgrad
```

`--datadir` 数据集存储位置

`--batchid` 每个batch行人数目

`--test_every` 每几个batch测试一次模型

`--epochs`总共的训练周期数

`--save`存储训练损失函数曲线、验证曲线、模型的文件夹名称

`--random_erasing`使用随机擦除

`--re_rank`**使用重排序手段**

### 测试

```shell
python3 /home/youwenjing/pytorch4/MGN-pytorch-master/main.py --datadir /home/youwenjing/Market-1501/ --margin 1.2 --save sgd_0 --cpu --test_only --resume 0 --pre_train /home/youwenjing/experiment/adam_0/model/model_best.pt
```

`--datadir`测试数据集位置

`--save`测试结果存储文件夹名

`--test_only` 表示仅做测试

`--pre_train` 测试的模型存储位置



### 结果展示

| 网络损失               | mAP    | rank-1 | rank-3 | rank-5 | rank-10 |
| ---------------------- | ------ | ------ | ------ | ------ | ------- |
| Softmax&trihard        | 0.8775 | 0.9466 | 0.9721 | 0.9801 | 0.9905  |
| Softmax&msml           | 0.8777 | 0.9415 | 0.9733 | 0.9804 | 0.9869  |
| Softmax&trihard&center | 0.8785 | 0.9489 | 0.9733 | 0.9810 | 0.9890  |

| 网络结构    | mAP    | rank-1 | rank-3 | rank-5 | rank-10 |
| ----------- | ------ | ------ | ------ | ------ | ------- |
| resnet1-3-4 | 0.8711 | 0.9480 | 0.9733 | 0.9804 | 0.9890  |
| resnet1-2-3 | 0.8775 | 0.9466 | 0.9721 | 0.9801 | 0.9905  |
| resnet1-2-4 | 0.8726 | 0.9457 | 0.9685 | 0.9765 | 0.9875  |




## 使用DG-Net进行行人重识别

由于原作者的github说明文档十分清晰，以下仅简要介绍训练测试步骤，详细步骤及代码请转到https://github.com/NVlabs/DG-Net#news 进行下载。代码的详细注释可查看 DG-Net-master，注释参考https://blog.csdn.net/weixin_43013761/article/details/102364512 。

1、下载代码到本地

2、下载数据集Market-1501，准备数据集（相同id放在同一目录下），注意修改 dataset path

```python
python prepare-market.py          # for Market-1501
```

3、使用预训练的教师模型，放置./models

4、训练DG-Net,检查 `configs/latest.yaml`.改变 data_root field 并开始训练

```python
python3 /home/youwenjing/DG-Net-master/train.py --config /home/youwenjing/DG-Net-master/configs/latest.yaml
```

5、图片输出和模型文件存储在 `outputs/latest`.检查loss日志

```python
 tensorboard --logdir logs/latest
```

### 测试

1、生成图片质量评估

阅读  `./visual_tools/README.md` 进行感官和客观的评估

`./visual_tools/test_folder.py` 生成图片并进行客观评估 注意修改[SSIM](https://github.com/layumi/PerceptualSimilarity) and [FID](https://github.com/layumi/TTUR)  data path .

```python
#感官结果
(pytorch1.0)
python /home/youwenjing/DG-Net-master/show_swap.py
(pytorch1.0)
python /home/youwenjing/DG-Net-master/show_rainbow.py  
#客观结果
#使用pytorch1.0环境，使用网络生成大量图片测试SSIM FID指数 --name为使用的模型文件夹名称 --which_epoch 为模型训练次数
(pytorch1.0)
python /home/youwenjing/DG-Net-master/test_folder.py --name E0.5new_reid0.5_w30000 --which_epoch 100000
#使用tensorflow1.1环境，测试FID指数 
(tensorflow1.1)
python /home/youwenjing/TTUR/TTUR-master/fid.py /home/youwenjing/kaggle/train_all /home/youwenjing/kaggle/off-gan_id1 --gpu 0
#使用pytorch1.0环境，测试SSIM指数 
(pytorch1.0)
python /home/youwenjing/PerceptualSimilarity-master/compute_market.py --dir /home/youwenjing/kaggle/off-gan_id1
```

2、模型精确度测试

```
(pytorch1.0)python /home/youwenjing/DG-Net-master/test_2label.py --name E0.5new_reid0.5_w30000  --which_epoch 100000
```

3、验证结构空间与外观空间

```
(pytorch1.0)
python /home/youwenjing/DG-Net-master/show_smooth.py
(pytorch1.0)
python /home/youwenjing/DG-Net-master/show_smooth_structure.py
```

