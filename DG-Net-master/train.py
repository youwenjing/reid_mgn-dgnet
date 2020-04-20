"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_loss, get_config, write_2images, Timer
import argparse
from trainer import DGNet_Trainer
import torch.backends.cudnn as cudnn
import torch
import numpy.random as random
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/latest.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--name', type=str, default='latest_ablation', help="outputs path")
parser.add_argument("--resume", action="store_true")  # 是否为继续训练，如果之前中断过，继续训练设置为store_false，重0开始训练设置为store_true
parser.add_argument('--trainer', type=str, default='DGNet', help="DGNet")  # 选择训练的的网络
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
opts = parser.parse_args()
# 处理多GPU情况
str_ids = opts.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gpu_ids.append(int(str_id))
num_gpu = len(gpu_ids)

cudnn.benchmark = True

# Load experiment setting
# 如果是在上次训练模型的中断的基础上进行训练，则获得上次训练模型的配置文件
if opts.resume:
    config = get_config('./outputs/'+opts.name+'/config.yaml')
# 否则就使用config指定的yaml配置文件
else:
    config = get_config(opts.config)
max_iter = config['max_iter'] # 设置迭代次数
display_size = config['display_size']  # outputs/latest/images目录下图片中，每行人的数目
config['vgg_model_path'] = opts.output_path  # 设置输出结果路径

# Setup model and data loader
if opts.trainer == 'DGNet':
    trainer = DGNet_Trainer(config, gpu_ids)
    trainer.cuda()

random.seed(7) # 固定随机数结果
# 把图片分成a,b，然后a,b图像进行合成
train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
train_a_rand = random.permutation(train_loader_a.dataset.img_num)[0:display_size] #随机抽取图片
train_b_rand = random.permutation(train_loader_b.dataset.img_num)[0:display_size] 
test_a_rand = random.permutation(test_loader_a.dataset.img_num)[0:display_size] 
test_b_rand = random.permutation(test_loader_b.dataset.img_num)[0:display_size] 
# 把图片拼接起来，方便统一显示
train_display_images_a = torch.stack([train_loader_a.dataset[i][0] for i in train_a_rand]).cuda()
train_display_images_ap = torch.stack([train_loader_a.dataset[i][2] for i in train_a_rand]).cuda()
train_display_images_b = torch.stack([train_loader_b.dataset[i][0] for i in train_b_rand]).cuda()
train_display_images_bp = torch.stack([train_loader_b.dataset[i][2] for i in train_b_rand]).cuda()
test_display_images_a = torch.stack([test_loader_a.dataset[i][0] for i in test_a_rand]).cuda()
test_display_images_ap = torch.stack([test_loader_a.dataset[i][2] for i in test_a_rand]).cuda()
test_display_images_b = torch.stack([test_loader_b.dataset[i][0] for i in test_b_rand]).cuda()
test_display_images_bp = torch.stack([test_loader_b.dataset[i][2] for i in test_b_rand]).cuda()

# Setup logger and output folders
# 设置输出目录和打印日志
# 如果不是继续训练,也就是重新训练，拷贝一些文件到outputs/latest下面，其目的是为了，保留了训练文件和网络结构
if not opts.resume:
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name)) # 在训练过程中实时地观察loss/accuracy曲线
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copyfile(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder
    shutil.copyfile('trainer.py', os.path.join(output_directory, 'trainer.py')) # copy file to output folder
    shutil.copyfile('reIDmodel.py', os.path.join(output_directory, 'reIDmodel.py')) # copy file to output folder
    shutil.copyfile('networks.py', os.path.join(output_directory, 'networks.py')) # copy file to output folder
else:
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", opts.name))
    output_directory = os.path.join(opts.output_path + "/outputs", opts.name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
# Start training 设置迭代次数和周期
# 如果是继续训练，使用预训练模型，先加载模型
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
config['epoch_iteration'] = round( train_loader_a.dataset.img_num  / config['batch_size'] )
print('Every epoch need %d iterations'%config['epoch_iteration'])
nepoch = 0 
    
print('Note that dataloader may hang with too much nworkers.')

if num_gpu>1:  # 设置多GPU情况
    print('Now you are using %d gpus.'%num_gpu)
    trainer.dis_a = torch.nn.DataParallel(trainer.dis_a, gpu_ids)
    trainer.dis_b = trainer.dis_a
    trainer = torch.nn.DataParallel(trainer, gpu_ids)

while True:
 # 循环获得训练数据,a,b
    for it, ((images_a,labels_a, pos_a),  (images_b, labels_b, pos_b)) in enumerate(zip(train_loader_a, train_loader_b)):
        if num_gpu>1:
            trainer.module.update_learning_rate()
        else:
            trainer.update_learning_rate()
        #  images_a与pos_a同一ID不一样的图片 images_b 与pos_b同一ID不一样的图片 labels_a, labels_b分别表示ID类别
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()
        pos_a, pos_b = pos_a.cuda().detach(), pos_b.cuda().detach()
        labels_a, labels_b = labels_a.cuda().detach(), labels_b.cuda().detach()

        with Timer("Elapsed time in update: %f"):
            # Main training code
            # 进行前向传播
            x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p = \
                                                                                 trainer.forward(images_a, images_b, pos_a, pos_b)
            # 进行反向传播 
            if num_gpu>1:
                trainer.module.dis_update(x_ab.clone(), x_ba.clone(), images_a, images_b, config, num_gpu)
                trainer.module.gen_update(x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p, images_a, images_b, pos_a, pos_b, labels_a, labels_b, config, iterations, num_gpu)
            else: 
                trainer.dis_update(x_ab.clone(), x_ba.clone(), images_a, images_b, config, num_gpu=1)
                trainer.gen_update(x_ab, x_ba, s_a, s_b, f_a, f_b, p_a, p_b, pp_a, pp_b, x_a_recon, x_b_recon, x_a_recon_p, x_b_recon_p, images_a, images_b, pos_a, pos_b, labels_a, labels_b, config, iterations, num_gpu=1)

            torch.cuda.synchronize()

        # Dump training stats in log file
        # 打印训练日志
        if (iterations + 1) % config['log_iter'] == 0:
            print("\033[1m Epoch: %02d Iteration: %08d/%08d \033[0m" % (nepoch, iterations + 1, max_iter), end=" ")
            if num_gpu==1:
                write_loss(iterations, trainer, train_writer)
            else:
                write_loss(iterations, trainer.module, train_writer)

        # Write images
        # 达到迭代次数，进行图片保存
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                if num_gpu>1:
                    test_image_outputs = trainer.module.sample(test_display_images_a, test_display_images_b)
                else:
                    test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
            write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
            del test_image_outputs
        # 图片显示
        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                if num_gpu>1:
                    image_outputs = trainer.module.sample(train_display_images_a, train_display_images_b)
                else:
                    image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            del image_outputs
        # Save network weights
        # 模型保存
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            if num_gpu>1:
                trainer.module.save(checkpoint_directory, iterations)
            else:
                trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')

    # Save network weights by epoch number
    nepoch = nepoch+1
    if(nepoch + 1) % 10 == 0:
        if num_gpu>1:
            trainer.module.save(checkpoint_directory, iterations)
        else:
            trainer.save(checkpoint_directory, iterations)

