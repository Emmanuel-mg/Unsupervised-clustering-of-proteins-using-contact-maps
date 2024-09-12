import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from torch.utils.data import TensorDataset, DataLoader

# Setting the seed
L.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

cmaps = os.listdir('cmaps')
maps = []
for map in cmaps:
    max_size = 408
    im_reshape = np.zeros((max_size, max_size))
    im = np.asarray(Image.open(os.path.join('cmaps', map)))
    im_correct = im[im.any(axis=1)][:, im.any(axis=0)]
    n, p = im_correct.shape
    # The data is rescalde between 0 and 1
    im_rescale = - (im_correct / 255) + 1
    # Center and reshape the data
    im_rescale = np.pad(im_rescale, [((max_size - n) // 2, (max_size - n) - (max_size - n) // 2), ((max_size - p) // 2, (max_size - p) - (max_size - p) // 2)])
    maps.append(im_rescale[None,:,:].astype(np.float32))

data_tensors = torch.stack([torch.tensor(arr, dtype=torch.float32) for arr in maps])
train_dataset = TensorDataset(data_tensors[8:], data_tensors[8:])
train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True, num_workers=4)
test_dataset = TensorDataset(data_tensors[:8], data_tensors[:8])
test_dataloader = DataLoader(test_dataset, batch_size = 8, num_workers=4)

# Create the encoder decoder
class AutoEncoder(L.LightningModule):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        super().__init__()
        c_hid = base_channel_size
        self.encoder = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 408x408 => 204x204
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1, stride=2), # 408x408 => 102x102
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 102x102 => 53x53
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2), # 102x102 => 27x27
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 25x25 => 12x12
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 169 * c_hid, latent_dim),
        )
        self.linear_decoder = nn.Sequential(
            nn.Linear(latent_dim, 2 * 169 * c_hid),
            act_fn()
        )
        self.decoder = nn.Sequential( 
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 12x12 => 25x25
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=0, padding=1, stride=2),  # 25x25 => 51x51
            act_fn(),
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 51x51 => 102x102
            act_fn(),
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 102x102 => 204x204
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),  # 204x204 => 408x408
            nn.Sigmoid()
        )
    def forward(self, x):
        x_ae = self.encoder(x)
        x_rec = self.linear_decoder(x_ae).reshape(x_ae.shape[0], -1, 13, 13)
        return self.decoder(x_rec)
    
    def get_reconstruction_loss(self, x):
        x_pred = self.forward(x)
        loss = torch.nn.functional.mse_loss(x, x_pred, reduction = "none")
        return loss.mean()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch):
        x, _ = batch
        loss = self.get_reconstruction_loss(x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, _ = batch
        loss = self.get_reconstruction_loss(x)
        self.log("val_loss", loss)
        return loss

class GenerateCallback(Callback):
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True)
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)
        
ae = AutoEncoder(num_input_channels=1, base_channel_size=24, latent_dim=1000)

trainer = L.Trainer(      
    default_root_dir='logs',
    log_every_n_steps = 1,
    accelerator="auto",
    devices=1,
    max_epochs=100,
    callbacks=[
    ModelCheckpoint(save_weights_only=True),
    GenerateCallback(data_tensors[:8], every_n_epochs=1),
    LearningRateMonitor("epoch")
    ])
trainer.fit(ae, train_dataloader, test_dataloader)