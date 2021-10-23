import argparse
import dataloader
import TTSR
import loss
import trainer
import torch
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

args={
      
### log setting
 'save_dir': 'test_result',
 'reset': True,
 'log_file_name': 'TTSR.log',
 'logger_name': 'TTSR',
 
### device setting
 'cpu': False,
 'num_gpu': 1,
 
### dataset setting
 'dataset': 'CUFED',
 'dataset_dir': '../Data/CUFED',
 
### dataloader setting
 'num_workers': 1,
 
### model setting
 'num_res_blocks': '16+16+8+4',
 'n_feats': 64,
 'res_scale': 1.0,
 
### loss setting
 'GAN_type': 'WGAN_GP',
 'GAN_k': 2,
 'tpl_use_S': False,
 'tpl_type': 'l2',
 'rec_w': 1.0,
 'per_w': 1,
 'tpl_w': 1,
 'adv_w': 0,
 
### optimizer setting
 'beta1': 0.9,
 'beta2': 0.999,
 'eps': 1e-08,
 'lr_rate': 0.0001,
 'lr_rate_dis': 0.0001,
 'lr_rate_lte': 1e-05,
 'decay': 999999,
 'gamma': 0.5,
 
### training setting
 'batch_size': 1,
 'train_crop_size': 40,
 'num_init_epochs': 0,
 'num_epochs': 5,
 'print_every': 1,
 'save_every': 2,
 'val_every': 999999,
 
### evaluate / test / finetune setting
 'eval': False,
 'eval_save_results': False,
 'model_path': 'model_save/TTSR.pt',
 'test': True,
 'lr_path': './test_input/lr/3.png',
 'ref_path': './test_input/ref/3.png'}

args = argparse.Namespace(**args)

_dataloader = dataloader.get_dataloader(args) if (1) else None
device = torch.device('cpu' if args.cpu else 'cuda')



_model = TTSR.TTSR(args).to(device)
_loss_all = loss.get_loss_dict(args)
t = trainer.Trainer(args, _dataloader, _model, _loss_all)
#t.load('model_save/TTSR.pt')

if (1):
    t.load(model_path=args.model_path)
    t.test()
if (0):
    t.load(model_path=args.model_path)
    t.evaluate()
if(0):
    for epoch in range(1, args.num_init_epochs+1):
        t.train(current_epoch=epoch, is_init=True)
    for epoch in range(1, args.num_epochs+1):
        t.train(current_epoch=epoch, is_init=False)
        if (epoch % args.val_every == 0):
            t.evaluate(current_epoch=epoch)
