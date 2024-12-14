import os
import argparse
from gorilla.config import Config
from os.path import join as opj
from utils import *
import torch
from models.openad_pn2 import OpenAD_PN2

def parse_args():
    parser = argparse.ArgumentParser(description="Test model on unseen affordances")
    parser.add_argument("--OpenAD_config", help="config file path")
    parser.add_argument("--OpenAD_checkpoint", help="the dir to saved OpenAD model")
    parser.add_argument("--CLPP_checkpoint", help="the dir to saved CLPP model")
    parser.add_argument("--CLPP_config", help="config file path")
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Number of gpus to use"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.OpenAD_config)
    CLPP_config = Config.fromfile(args.CLPP_config)

    logger = IOStream(opj(cfg.work_dir, 'result_' + cfg.model.type + '.log'))
    if cfg.get('seed', None) != None:
        set_random_seed(cfg.seed)
        logger.cprint('Set seed to %d' % cfg.seed)
        
    if args.gpu != None:
        cfg.training_cfg.gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training_cfg.gpu

    OpenAD_model = build_model(cfg).cuda()
    CLPP_model = build_model(CLPP_config).cuda()

    if args.CLPP_checkpoint == None or args.OpenAD_checkpoint == None:
        print("Please specify the path to the saved model")
        exit()
    else:
        print("Loading model....")
        # Load OpenAD
        _, exten = os.path.splitext(args.OpenAD_checkpoint)
        print("exten: ", exten)
        if exten == '.t7':
            OpenAD_model.load_state_dict(torch.load(args.OpenAD_checkpoint))
        elif exten == '.pth':
            check = torch.load(args.OpenAD_checkpoint)
            OpenAD_model.load_state_dict(check['model_state_dict'])
        else:
            print("Invalid file format")
            exit()
        # Load CLPP
        _, exten = os.path.splitext(args.CLPP_checkpoint)
        if exten == '.t7':
            CLPP_model.load_state_dict(torch.load(args.CLPP_checkpoint))
        elif exten == '.pth':
            check = torch.load(args.CLPP_checkpoint)
            CLPP_model.load_state_dict(check['model_state_dict'])
        else:
            print("Invalid file format")
            exit()

    dataset_dict = build_dataset(cfg)
    loader_dict = build_loader(cfg, dataset_dict)

    OpenAD_layers = (OpenAD_model.fp3, OpenAD_model.fp2, OpenAD_model.fp1, OpenAD_model.bn1, OpenAD_model.conv1)
    CLPP_layers = (CLPP_model.sa1, CLPP_model.sa2, CLPP_model.sa3)
    model = OpenAD_PN2.from_checkpoint(CLPP_layers, OpenAD_layers)

    val_loader = loader_dict.get("val_loader", None)
    val_affordance = cfg.training_cfg.val_affordance
    mIoU = evaluation(logger, model, val_loader, val_affordance)