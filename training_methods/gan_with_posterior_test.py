import os.path

import torch
from torch.nn.functional import interpolate
from torchjpeg.dct import images_to_batch, batch_to_images, Stats, to_rgb
from torchjpeg.quantization import quantize_multichannel, dequantize_multichannel, ijg, dequantize

from pytorch_lightning.metrics import MeanSquaredError as MSE
from pytorch_lightning.metrics import PSNR
from torchvision.utils import save_image

from training_methods.gan import GAN
from utils.utils import mkdir


class GANWithPosteriorTest(GAN):
    def on_test_epoch_start(self):
        self.degradation_mse = MSE().to(self.device)
        self.degradation_psnr = PSNR(1).to(self.device)
        self.qf = float(self.config['jpeg_qf'])

    def compress_batch(self, batch):
        dct = images_to_batch(batch)
        mat_y = ijg.get_coefficients_for_qualities(torch.full((1,), self.qf).to(batch.device), 'luma')
        mat_c = ijg.get_coefficients_for_qualities(torch.full((1,), self.qf).to(batch.device), 'chroma')
        mat = torch.cat([mat_y, mat_c], dim=1)
        y_coeffs, cb_coeffs, cr_coeffs = quantize_multichannel(dct, mat)
        y_coeffs = dequantize(y_coeffs, mat[:, 0:1, :, :])
        cb_coeffs = dequantize(cb_coeffs, mat[:, 1:2, :, :])
        cr_coeffs = dequantize(cr_coeffs, mat[:, 1:2, :, :])

        y = batch_to_images(y_coeffs, channel='y')
        cb = batch_to_images(cb_coeffs, channel='cb')
        cr = batch_to_images(cr_coeffs, channel='cr')

        cb = interpolate(cb, y[0,0].shape, mode="nearest")
        cr = interpolate(cr, y[0,0].shape, mode="nearest")

        spatial = torch.cat([y, cb, cr], dim=1)
        spatial = to_rgb(spatial * 255)  # noqa
        spatial = spatial.clamp(0, 255)
        spatial = spatial / 255

        return spatial

    def save_batch(self, y, y_hat, idx):
        save_image(torch.cat((y, y_hat), dim=-1), os.path.join(self.test_path, f"recompress_{idx}.png"))

    def test_step(self, batch, batch_idx):
        x, y = self.batch_postprocess(batch)
        with torch.no_grad():
            # decompress y
            x_hat = self(y, noise_stds=1)
            # recompress y
            y_hat = self.compress_batch(x_hat)
            # update metrics
            self.degradation_mse.update(y_hat, y)
            self.degradation_psnr.update(y_hat, y)
            # save random batch
            if batch_idx in self.test_cfg['save_batch']:
                mkdir(self.test_path, remove=False)
                self.save_batch(y, y_hat, batch_idx)

    def test_epoch_end(self, outputs):
        self.log("Test set recompression MSE", self.degradation_mse.compute(), prog_bar=True, logger=True)
        self.log("Test set recompression PSNR", self.degradation_psnr.compute(), prog_bar=True, logger=True)
