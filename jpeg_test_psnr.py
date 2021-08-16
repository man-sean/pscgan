import argparse
import utils.argparse_utils as argparse_utils

import os
from pytorch_lightning import seed_everything
import utils.utils as utils
from datasets.factory import factory as ds_fac
from tqdm import tqdm
from pytorch_lightning.metrics import PSNR

if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser = argparse_utils.add_base_args(parent=parent_parser, test_flag=False)
    parent_parser = argparse_utils.add_test_args(parent_parser)
    parent_parser = argparse_utils.add_optional_args(parent_parser)
    args = parent_parser.parse_args()

    config, experiment_dir = argparse_utils.override_config(parent_parser, args, test_flag=False)
    device_ids = list(range(args.n_gpus))
    num_gpus = len(device_ids)
    accelerator = config['accelerator']
    if accelerator != 'ddp':
        device_ids = device_ids[0]
    seed_everything(0)  # Required for Distributed Data Parallel

    datamodule = ds_fac[config['dataset_cfg']['type']](config['dataset_cfg'], num_gpus)
    datamodule.setup()

    # Create logger
    if args.out_dir is None:
        results_path = os.path.join(experiment_dir, 'results')
    else:
        results_path = args.out_dir
    utils.mkdir(results_path)

    psnr = PSNR(1)

    pbar = tqdm(total=len(datamodule.test_dataset), desc='Computing PSNR')

    # read and save few images
    for idx in range(len(datamodule.test_dataset)):
        example = datamodule.test_dataset[idx]
        x, y = example['real'], example['noisy']
        psnr.update(x, y)
        pbar.update()

    print(f'PSNR: {psnr.compute()}')
