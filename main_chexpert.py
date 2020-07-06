import argparse
import torch
import util
import pandas as pd
import numpy as np
import sklearn.metrics as sk_metrics
from constants import *
from logger import Logger
from torchvision import transforms
from dataset.chexpert import get_dataloader 
from networks.densenet import CheXpert


def train(args):

    # get dataloader
    train_transform = transforms.Compose([
            transforms.Resize((args.resize_shape, args.resize_shape)),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(transform_args.rotate),
            transforms.RandomCrop((args.crop_shape, args.crop_shape)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)
    ])
    train_dataset_args = {
        'data_path': CHEXPERT_TRAIN_CSV, 
        'img_type': args.img_type,
        'data_transform': train_transform,
    }
    valid_dataset_args = {
        'data_path': CHEXPERT_VALID_CSV, 
        'img_type': args.img_type,
        'data_transform': train_transform,
    }
    dataloader_args = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers
    }
    train_loader = get_dataloader(dataloader_args, train_dataset_args)
    valid_loader = get_dataloader(dataloader_args, valid_dataset_args)

    # put model on device
    model = CheXpert(model_name="densenet121", num_classes=14)
    model = model.to(args.device)
    model.train()

    # optimizer 
    optimizer = util.set_optimizer(opt=args, model=model)

    # loss function 
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")

    # define logger
    logger = Logger(log_dir=args.log_dir, metrics_name=args.eval_metrics, args=args)

    global_step = 0

    # iterate over epoch
    for epoch in args.num_epoch:

        # training loop
        for inputs, targets in train_loader:

            # validation loop 
            if global_step % args.iters_per_eval:
                model.eval()
                probs = []
                gt = []
                with torch.no_grad():

                    for val_inputs, val_targets in valid_loader:

                        batch_logits = model(val_inputs.to(args.device))
                        batch_probs = torch.sigmoid(batch_logits)

                        probs.append(batch_probs.cpu())
                        gt.append(val_targets.cpu())

                metrics = util.evaluate(probs, gt, args.threshold)

                # log into tensorboard 
                avg_metric = {}
                for m in ["auroc", "auprc", "accuracy", "precision", "recall"]:
                    metrics_list = []
                    for pathology in CHEXPERT_COMPETITION_TASKS:
                        metrics_list.append(metrics[pathology][m])
                    
                    avg = sum(metrics_list) / len(metrics_list)
                    avg_metric[f"val/{m}"] = avg
                logger.log_dict(avg_metric, global_step, "val")

                # save to log file
                log_dict = {}
                for pathology, pathology_metrics in metrics.items():
                    for metric, value in pathology_metrics.items():
                        log_dict[f"{pathology}_{metric}"] = [value]
                logger.log_iteration(log_dict, global_step, "val")

                # log image
                logger.log_image(val_inputs, global_step)

                # save checkpoint
                logger.save_checkpoint(model, metrics, global_step)

            model.train()
            with torch.set_grad_enabled(True):

                # Run the minibatch through the model.
                logits = model(inputs.to(args.device))

                # Compute the minibatch loss.
                loss = loss_fn(logits, targets.to(args.device))

                # Perform a backward pass.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            global_step += 1 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_type', type=str, default='Frontal', choices=['All','Frontal'])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--iters_per_eval', type=int, default=100)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--num_epoch', type=str, default='10')
    parser.add_argument('--resize_shape', type=int, default=320)
    parser.add_argument('--crop_shape', type=int, default=320)
    parser.add_argument('--optimizer', type=str, default="adam", choices=["sdg", "adam"])
    parser.add_argument('--lr', type=float, default=1e-3)
    # TODO: default momentum, dampening, weight decay
    parser.add_argument('--momentum', type=float, default=0.1)
    parser.add_argument('--sgd_dampening', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.001)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--log_dir', type=str, default="./logs")
    parser.add_argument('--eval_metrics', type=str, default="auroc", choices=["auroc", "auprc", "accuracy", "precision", "recall"])

    args = parser.parse_args()

    # set gpu and device
    args.gpu_ids = [int(ids) for ids in args.gpu_ids.split(",")]
    if len(args.gpu_ids) > 0 and torch.cuda.is_available():
        # Set default GPU for `tensor.to('cuda')`
        torch.cuda.set_device(args.gpu_ids[0])
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    train(args)