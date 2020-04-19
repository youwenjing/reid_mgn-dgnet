from importlib import import_module
from torchvision import transforms
from random_erasing import RandomErasing
from data.sampler import RandomSampler
from torch.utils.data import dataloader

class Data:
    def __init__(self, args):
       # 重置图像分辨率参数 依概率p水平翻转 转化为tensor 对数据按通道进行标准化
        train_list = [
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        #随机擦除图像增强
        if args.random_erasing:
            train_list.append(RandomErasing(probability=args.probability, mean=[0.0, 0.0, 0.0]))

        train_transform = transforms.Compose(train_list)
       # 一起组成几个变换
        test_transform = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    # 判断是否仅仅测试
        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())
            self.trainset = getattr(module_train, args.data_train)(args, train_transform, 'train')
    # 从数据集中加载 batch_size个随机选取的图像数据 
            self.train_loader = dataloader.DataLoader(self.trainset,
                            sampler=RandomSampler(self.trainset,args.batchid,batch_image=args.batchimage),
                            #shuffle=True,
                            batch_size=args.batchid * args.batchimage,
                            num_workers=args.nThread)
        else:
            self.train_loader = None
        
        if args.data_test in ['Market1501']:
            module = import_module('data.' + args.data_train.lower())
            self.testset = getattr(module, args.data_test)(args, test_transform, 'test')
            self.queryset = getattr(module, args.data_test)(args, test_transform, 'query')

        else:
            raise Exception()

        self.test_loader = dataloader.DataLoader(self.testset, batch_size=args.batchtest, num_workers=args.nThread)
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=args.batchtest, num_workers=args.nThread)
        
