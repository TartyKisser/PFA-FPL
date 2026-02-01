import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import numpy as np
from utils.utils import set_seed, base_path, get_dataloader, Logger
from methods.model import Model 
import argparse


parser = argparse.ArgumentParser() 

parser.add_argument('--dataset_name', type=str, default='VG',
                    choices=['COCO2014', 'CUB', 'NUS', 'VG'])
parser.add_argument('--n_way', type=int, default=10)
parser.add_argument('--n_shot', type=int, default=1)
parser.add_argument('--max_epoch', type=int, default=500)
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=0)

# 返回一个初始化的原型网络
def get_model():
    model = Model(device=device,
                n_way=n_way,
                n_shot=n_shot,
                n_query=n_query,
                verbose=True)
    return model


def _train():
    print('Start Training!')
    model = get_model()
    best_mAP = 0
    for epoch in range(max_epoch):
        avg_loss = model.train_loop(train_loader, train_dict)
        print('epoch %d training done | Loss: %f' % (epoch, avg_loss))
        if epoch > -1:
            result = model.test_loop(val_loader, val_dict)
            mAP, mAP_std = result['mAP'], result['mAP-std']
            print(f'{mAP}±{mAP_std}')
            print('Epoch %d validation done | MAP: %.4f±%.4f' % (epoch, mAP, mAP_std))
            if mAP > best_mAP:
                best_mAP = mAP
                model.save(train_dir, epoch='best_mAP')
            if epoch == max_epoch - 1:
                model.save(train_dir, epoch=epoch)
        if (epoch + 1) <= 200:
            if (epoch + 1) % 100 == 0:
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5
        else:
            if (epoch + 1) % 50 == 0:
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5


def _test():
    print('Start Testing!')
    model = get_model()
    model.load(train_dir, epoch='best_mAP')
    result = model.test_loop(test_loader, test_dict)
    mAP, mAP_std = result['mAP'], result['mAP-std']
    print(f'{mAP}±{mAP_std}')


if __name__ == '__main__':
    args = parser.parse_args()
    dataset_name = args.dataset_name
    device = args.device
    seed = args.seed
    n_way = args.n_way
    max_epoch = args.max_epoch
    n_shot = args.n_shot
    num_workers = args.num_workers

    image_size = 224
    set_seed(seed)
    if dataset_name == 'COCO':
        n_way = np.minimum(n_way, 16)
    else:
        n_way = np.minimum(n_way, 20)
    n_query = n_way // 2

    with open('data/COCO2014.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    train_dict = data['train']
    val_dict = data['test']
    test_dict = data['test']

    train_dir = os.path.join(base_path, 'save', dataset_name,
                             f'{max_epoch}_{n_way}_{n_shot}_{n_query}_{seed}',
                             'train')
    log_dir = os.path.join(base_path, 'save', dataset_name,
                           f'{max_epoch}_{n_way}_{n_shot}_{n_query}_{seed}',
                           'log')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_loader = get_dataloader(dataset_name=dataset_name, phase='train', n_way=n_way, n_shot=n_shot,
                                  n_query=n_query, transform=True, num_iter=200,
                                  num_workers=num_workers, image_size=image_size)
    val_loader = get_dataloader(dataset_name=dataset_name, phase='test', n_way=n_way, n_shot=n_shot,
                                n_query=n_query, transform=False, num_iter=100, num_workers=num_workers,
                                image_size=image_size)
    test_loader = get_dataloader(dataset_name=dataset_name, phase='test', n_way=n_way, n_shot=n_shot,
                                 n_query=n_query, transform=False, num_iter=1000,
                                 num_workers=num_workers, image_size=image_size)

    print = Logger(f'{log_dir}/log.txt').logger.warning
    print(
        f'{dataset_name}, max_epoch: {max_epoch}, n_way: {n_way}, n_shot: {n_shot}, n_query: {n_query}, seed: {seed}')

    if not os.path.exists(os.path.join(train_dir, f'{max_epoch - 1}.tar')):
        _train()
    _test()


# python run.py --dataset_name COCO2014 --n_way 10 --n_shot 1 --max_epoch 500 --num_workers 8 --device cuda:0 --seed 0