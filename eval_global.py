import os
import argparse
from gorilla.config import Config
from os.path import join as opj
from utils import *
from utils.eval import eval_global_util
import torch
from torch.utils.data.dataloader import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Test model on unseen affordances")
    parser.add_argument("--config", help="config file path")
    parser.add_argument("--checkpoint", help="the dir to saved model")
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Number of gpus to use"
    )
    parser.add_argument(
        "--test_num",
        type=int,
        default=50,
        help="The number of tests"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)

    from dataset.AffordanceNet import TestDataset
    test_dataset = TestDataset(cfg.data.data_root, test_num=args.test_num)

    logger = IOStream(opj(cfg.work_dir, 'result_' + cfg.model.type + '.log'))
    if cfg.get('seed', None) != None:
        set_random_seed(cfg.seed)
        logger.cprint('Set seed to %d' % cfg.seed)
        
    if args.gpu != None:
        cfg.training_cfg.gpu = args.gpu

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu

    model = build_model(cfg).cuda()

    if args.checkpoint == None:
        print("Please specify the path to the saved model")
        exit()
    else:
        print("Loading model....")
        _, exten = os.path.splitext(args.checkpoint)
        if exten == '.t7':
            model.load_state_dict(torch.load(args.checkpoint))
        elif exten == '.pth':
            check = torch.load(args.checkpoint)
            model.load_state_dict(check['model_state_dict'])

    # TODO: hard code the batchsize to be 1
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    val_affordance = cfg.training_cfg.val_affordance
  
    eval_global_util(model, test_loader)