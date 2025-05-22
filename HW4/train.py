import os, argparse, numpy as np
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from utils.dataset_utils import RainSnowDataset
from utils.schedulers import LinearWarmupCosineAnnealingLR
from net.model import PromptIR

class RainSnowModule(pl.LightningModule):
    def __init__(self, lr=2.2e-4):
        super().__init__()
        self.net     = PromptIR(decoder=True)
        self.loss_fn = nn.MSELoss()
        self.lr      = lr

        self.epoch_loss_sum = 0.0 ###
        self.epoch_batch_count = 0 ###

    def forward(self, x, pidx):
        return self.net(x, pidx)

    def training_step(self, batch, batch_idx):
        deg, cln, pidx = batch
        pred = self(deg, pidx)
        loss = self.loss_fn(pred, cln)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        self.epoch_loss_sum += loss.item() ###
        self.epoch_batch_count += 1 ###
        return loss
        

    def on_train_epoch_start(self):
        # reset at the start of each epoch
        self.epoch_loss_sum = 0.0
        self.epoch_batch_count = 0

    def on_train_epoch_end(self):
        # compute average and append to file
        avg = self.epoch_loss_sum / max(1, self.epoch_batch_count)
        epoch = self.current_epoch + 1
        with open("loss.txt", "a") as f:
            f.write(f"Epoch {epoch:03d}: {avg:.6f}\n")

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=opt,warmup_epochs=15,max_epochs=150)
        return [opt],[scheduler]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs',     type=int, default=100)
    args = p.parse_args()

    transform = transforms.Compose([transforms.ToTensor()])

    ds = RainSnowDataset(args.data_root, transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    model     = RainSnowModule()
    logger    = TensorBoardLogger('logs/', name='RainSnow')
    ckpt_cb   = ModelCheckpoint(dirpath='ckpts/',
                                filename='{epoch}',
                                save_top_k=-1,              
                                every_n_epochs=1,
                                save_last=False
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu', devices=1,
        logger=logger, callbacks=[ckpt_cb]
    )
    trainer.fit(model, train_dataloaders=loader)

    # Save final weights
    #torch.save(model.net.state_dict(), 'best_allinone_IR.pth')

if __name__=='__main__':
    main()
