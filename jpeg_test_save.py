import argparse
import utils.argparse_utils as argparse_utils

import os
from pytorch_lightning import seed_everything
import utils.utils as utils
from datasets.factory import factory as ds_fac
from torchvision.utils import save_image

if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser = argparse_utils.add_base_args(parent=parent_parser, test_flag=False)
    parent_parser = argparse_utils.add_train_args(parent_parser)
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

    # read and save few images
    for idx in range(10):
        print(f'reading clean and compressed image {idx}...', end=' ')
        example = datamodule.training_dataset[idx]
        print(f'saving images')
        save_image(example['real'], os.path.join(results_path, f'{idx}_real.png'))
        save_image(example['noisy'], os.path.join(results_path, f'{idx}_compressed.png'))
