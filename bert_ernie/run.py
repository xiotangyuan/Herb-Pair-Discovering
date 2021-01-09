# coding: UTF-8
import time
import torch
import numpy as np
# from train_eval import beipoyuanli
# from train_eval import train,init_network
from importlib import import_module
from train_eval import *
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--model', default='bert',type=str, required=True, help='choose a model: Bert, ERNIE')
parser.add_argument('--model', default='bert',type=str, help='choose a model: Bert, ERNIE')
args = parser.parse_args()
class shishuwunai():
    def wunaijiade(self):
        dataset = 'THUCNews'  # 数据集
        model_name = args.model  # bert
        x = import_module('models.' + model_name)
        config = x.Config(dataset)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样

        start_time = time.time()
        print("Loading data...")
        test_data = build_dataset(config)
        # test_data = build_dataset(config)
        test_iter = build_iterator(test_data, config)
        # print("test_data",test_data)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)

        # train
        model = x.Model(config)  # .to(config.device)
        # model = x.Model(config)

        # beipoyuanli.train(config, model, train_iter, dev_iter, test_iter)
        aa = train(config, model,  test_iter)
        # train(config, model, test_iter)
        return aa

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

