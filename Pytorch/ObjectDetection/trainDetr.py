import os
import wandb
import argparse
import shutil
import torch
from torch.cuda.amp import autocast, GradScaler
torch.backends.cuda.cufft_plan_cache.set_max_split_size(1500)

mydir = os.getcwd() # would be the MAIN folder
mydir_tmp = mydir + "/detr/" # add the testA folder name
mydir_new = os.chdir(mydir_tmp) # change the current working directory
mydir = os.getcwd() # set the main directory again, now it calls testA

from main import main



splitsFolder = "C:/Users/MICHE/Documents/Datasets/datasetSplits/"
fold_type = ["reg", "rand"]
k = 4
for fold in fold_type:
    for i in range(k):
        datasetFolder = splitsFolder + fold + "_" + str(i) + "/"
        
        # Init wandb
        run = wandb.init(sync_tensorboard=True,
                        settings=wandb.Settings(start_method="thread", console="off"), 
                        project = "detrFineTune")
        wandb.run.name = "detrFineTune_" + fold + "_" + str(i)
        
        
        pathDataset = "../cocoDatasetTrial/"
        pathOutput = "../output/"
        pretrainedFile = "C:/Users/MICHE/Desktop/Master/MTP/torch/detr-r50-e632da11.pth"
        backbone = "resnet50"
        batchSize = 2
        epochs = 50
        
        # Create folder of outputs
        if not os.path.exists(pathOutput):
            os.makedirs(pathOutput)
        
        def get_args_parser():
            parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
            parser.add_argument('--lr', default=1e-4, type=float)
            parser.add_argument('--lr_backbone', default=1e-5, type=float)
            parser.add_argument('--batch_size', default=batchSize, type=int)
            parser.add_argument('--weight_decay', default=1e-4, type=float)
            parser.add_argument('--epochs', default=epochs, type=int)
            parser.add_argument('--lr_drop', default=200, type=int)
            parser.add_argument('--clip_max_norm', default=0.1, type=float,
                                help='gradient clipping max norm')

            # Model parameters
            parser.add_argument('--frozen_weights', type=str, default=None,
                                help="Path to the pretrained model. If set, only the mask head will be trained")
            # * Backbone
            parser.add_argument('--backbone', default=backbone, type=str,
                                help="Name of the convolutional backbone to use")
            parser.add_argument('--dilation', action='store_true',
                                help="If true, we replace stride with dilation in the last convolutional block (DC5)")
            parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                                help="Type of positional embedding to use on top of the image features")

            # * Transformer
            parser.add_argument('--enc_layers', default=6, type=int,
                                help="Number of encoding layers in the transformer")
            parser.add_argument('--dec_layers', default=6, type=int,
                                help="Number of decoding layers in the transformer")
            parser.add_argument('--dim_feedforward', default=2048, type=int,
                                help="Intermediate size of the feedforward layers in the transformer blocks")
            parser.add_argument('--hidden_dim', default=256, type=int,
                                help="Size of the embeddings (dimension of the transformer)")
            parser.add_argument('--dropout', default=0.1, type=float,
                                help="Dropout applied in the transformer")
            parser.add_argument('--nheads', default=8, type=int,
                                help="Number of attention heads inside the transformer's attentions")
            parser.add_argument('--num_queries', default=100, type=int,
                                help="Number of query slots")
            parser.add_argument('--pre_norm', action='store_true')

            # * Segmentation
            parser.add_argument('--masks', action='store_true',
                                help="Train segmentation head if the flag is provided")

            # Loss
            parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                                help="Disables auxiliary decoding losses (loss at each layer)")
            # * Matcher
            parser.add_argument('--set_cost_class', default=1, type=float,
                                help="Class coefficient in the matching cost")
            parser.add_argument('--set_cost_bbox', default=5, type=float,
                                help="L1 box coefficient in the matching cost")
            parser.add_argument('--set_cost_giou', default=2, type=float,
                                help="giou box coefficient in the matching cost")
            # * Loss coefficients
            parser.add_argument('--mask_loss_coef', default=1, type=float)
            parser.add_argument('--dice_loss_coef', default=1, type=float)
            parser.add_argument('--bbox_loss_coef', default=5, type=float)
            parser.add_argument('--giou_loss_coef', default=2, type=float)
            parser.add_argument('--eos_coef', default=0.1, type=float,
                                help="Relative classification weight of the no-object class")

            # dataset parameters
            parser.add_argument('--dataset_file', default='coco')
            parser.add_argument('--coco_path', default=datasetFolder, type=str)
            parser.add_argument('--coco_panoptic_path', type=str)
            parser.add_argument('--remove_difficult', action='store_true')

            parser.add_argument('--output_dir', default=pathOutput,
                                help='path where to save, empty for no saving')
            parser.add_argument('--device', default='cuda',
                                help='device to use for training / testing')
            parser.add_argument('--seed', default=42, type=int)
            parser.add_argument('--resume', default=pretrainedFile, help='resume from checkpoint')
            parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                                help='start epoch')
            parser.add_argument('--eval', action='store_true')
            parser.add_argument('--num_workers', default=0, type=int)

            # distributed training parameters
            parser.add_argument('--world_size', default=1, type=int,
                                help='number of distributed processes')
            parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
            return parser
        
        parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
        args = parser.parse_args()
        main(args)
        
        wandb.finish()
        
        # Copy results
        dest = splitsFolder + fold + str(i) + "_output/"
        
        # Create folder of results
        if not os.path.exists(dest):
            os.makedirs(dest)
            
        source = pathOutput
        shutil.copytree(source, dest, dirs_exist_ok=True)