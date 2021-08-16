import torch
import torchjpeg.codec
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.lightning import CenterCropMinAxis, DatasetBase, torchvision_classes


class JPEGCompress():
    def __init__(self, qf: int = 100):
        self.qf = qf
        print(f'[JPEGCompress] initiated with qf: {self.qf}')

    def __call__(self, img):
        img = self.pad(img)
        dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.quantize_at_quality(img, self.qf)
        compressed = torchjpeg.codec.reconstruct_full_image(Y_coefficients, quantization, CbCr_coefficients, dimensions)
        return compressed

    def pad(self, img, macroblock_size=2 * 8):
        shape = torch.Tensor(list(img.shape))
        padding = (torch.ceil(shape / macroblock_size) * macroblock_size - shape).long()
        return torch.nn.functional.pad(img.unsqueeze(0), [0, padding[2], 0, padding[1]], 'replicate').squeeze(0)


class JPEGDatamoduleLightning(pl.LightningDataModule):
    def __init__(self, config, num_gpus):
        super().__init__()
        self.training_set_path = config['training_set_path'] if 'training_set_path' in config else None
        self.val_set_path = self.training_set_path
        self.test_set_path = config['test_set_path']
        self.jpeg_qf = config['jpeg_qf']
        self.train_batch_size = config['train_batch_size']
        self.num_workers = config['num_workers']
        self.val_batch_size = config['test_batch_size']
        self.test_batch_size = config['test_batch_size']
        self.accelerator = config['accelerator']
        self.torchvision_class = torchvision_classes[config['torchvision_class']]
        self.len_multiplier = config['len_multiplier'] if 'len_multiplier' in config else 1
        self.num_gpus = num_gpus
        self.training_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        pass

    @property
    def train_transforms(self):
        return transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])

    @property
    def val_transforms(self):
        return transforms.Compose([transforms.ToTensor()])

    @property
    def test_transforms(self):
        return transforms.Compose([transforms.ToTensor()])

    def setup(self, stage=None):
        compress_transform = JPEGCompress(self.jpeg_qf)
        if self.training_set_path is not None:
            self.training_dataset = DatasetBase(dataset_path=self.training_set_path,
                                                target_transform=self.train_transforms,
                                                input_transform=compress_transform,
                                                torchvision_class=self.torchvision_class,
                                                len_multiplier=self.len_multiplier)
        if self.val_set_path is not None:
            self.val_dataset = DatasetBase(dataset_path=self.val_set_path,
                                           target_transform=self.test_transforms,
                                           input_transform=compress_transform,
                                           torchvision_class=self.torchvision_class,
                                           len_multiplier=1)
        if self.test_set_path is not None:
            self.test_dataset = DatasetBase(dataset_path=self.test_set_path,
                                            target_transform=self.test_transforms,
                                            input_transform=compress_transform,
                                            torchvision_class=self.torchvision_class,
                                            len_multiplier=1)

    def train_dataloader(self):
        if self.training_dataset is not None:
            return DataLoader(self.training_dataset,
                              batch_size=self.train_batch_size // self.num_gpus
                              if self.accelerator == 'ddp' else self.train_batch_size,
                              shuffle=True,
                              num_workers=self.num_workers,
                              drop_last=True,
                              pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.val_batch_size // self.num_gpus
                          if self.accelerator == 'ddp' else self.val_batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=True,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.test_batch_size // self.num_gpus
                          if self.accelerator == 'ddp' else self.test_batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=True,
                          pin_memory=True)

    def test_dataloader_custom_batch_size(self, batch_size):
        return DataLoader(self.test_batch_size // self.num_gpus
                          if self.accelerator == 'ddp' else self.test_batch_size,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=True,
                          pin_memory=True)


class CenterCropResizeDataset(JPEGDatamoduleLightning):
    def __init__(self, config, num_gpus):
        super().__init__(config, num_gpus)

    @property
    def train_transforms(self):
        return transforms.Compose([CenterCropMinAxis(-1),
                                   transforms.Resize(128),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor()])

    @property
    def test_transforms(self):
        return transforms.Compose([CenterCropMinAxis(-1),
                                   transforms.Resize(128),
                                   transforms.ToTensor()])

    @property
    def val_transforms(self):
        return transforms.Compose([CenterCropMinAxis(-1),
                                   transforms.Resize(128),
                                   transforms.ToTensor()])
