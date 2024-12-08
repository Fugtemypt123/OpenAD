import os
from os.path import join as opj
import torch
import torch.nn as nn
import numpy as np
from gorilla.config import Config
from utils import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", help="train config file path")
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Number of gpus to use"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="The checkpoint to be resume"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.work_dir != None:
        cfg.work_dir = args.work_dir
    if args.gpu != None:
        cfg.training_cfg.gpu = args.gpu
        
    log_dir = './log/openad_pn2/OPENAD_PN2_FULL_SHAPE_Release'
    logger = IOStream(opj(log_dir, 'run.log'))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu
    num_gpu = len(cfg.training_cfg.gpu.split(','))
    logger.cprint('Use %d GPUs: %s' % (num_gpu, cfg.training_cfg.gpu))
    if cfg.get('seed') != None:
        set_random_seed(cfg.seed)
        logger.cprint('Set seed to %d' % cfg.seed)
    model = build_model(cfg).cuda() 

    if args.checkpoint != None:
        print("Loading checkpoint....")
        _, exten = os.path.splitext(args.checkpoint)
        if exten == '.t7':
            checkpoint  = torch.load(args.checkpoint)
            pretrained_state_dict = checkpoint
            # print(pretrained_state_dict[0])
            # raise ValueError('With Great Power Comes Great Responsibility!')
            model_dict = model.state_dict()
            partial_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}
            model_dict.update(partial_state_dict)
            model.load_state_dict(model_dict)
            print("Successfully loaded partial checkpoint!")
        elif exten == '.pth':
            check = torch.load(args.checkpoint)
            model.load_state_dict(check['model_state_dict'])
    else:
        print("Training from scratch!")

    dataset_dict = build_dataset(cfg)
    loader_dict = build_loader(cfg, dataset_dict)
    train_loss = build_loss_clpp()
    optim_dict = build_optimizer(cfg, model)

    training = dict(
        model=model,
        dataset_dict=dataset_dict,
        loader_dict=loader_dict,
        loss=train_loss,
        optim_dict=optim_dict,
        logger=logger
    )

    task_trainer = TrainerCLPP(cfg, training)
    task_trainer.run()
