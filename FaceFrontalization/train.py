import warnings
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from boarder import test_fn
from config import *

warnings.filterwarnings('ignore')
from utils import load_checkpoint, save_checkpoint
from dataset import MapDataset
from models import UNET, Discriminator
import torch.optim as optim


def train_fn(
        disc, gen, train_loader, board_loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, step
):
    loop = tqdm(train_loader)
    for idx, (x, y) in enumerate(loop):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # Print losses occasionally and print to tensorboard
        if idx % 10 == 0:
            # print(
            #    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {idx}/{len(loader)} \
            #       Loss D: {D_loss:.4f}, loss G: {G_loss:.4f}"
            # )

            test_fn(gen, board_loader, step)

            step += 1
        loop.set_postfix(
            D_real=torch.sigmoid(D_real).mean().item(),
            D_fake=torch.sigmoid(D_fake).mean().item(),
        )
        loop.update()
    return step


def main():
    disc = Discriminator(in_channels=3).to(DEVICE)
    # gen = Generator(in_channels=3, features=64).to(DEVICE)
    # gen = SegNet(in_channels=3, n_classes=3).to(DEVICE)
    gen = UNET(in_channels=3, out_channels=3).to(DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999), )
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    dataset = MapDataset(root_dir=ALL_DIR)

    val_size = 16
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,  # dataloader suggestion
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    if LOAD_MODEL:
        load_checkpoint(
            weights_paths + weight_ver + '_' + CHECKPOINT_GEN, gen, opt_gen, LEARNING_RATE,
        )
        load_checkpoint(
            weights_paths + weight_ver + '_' + CHECKPOINT_DISC, disc, opt_disc, LEARNING_RATE,
        )
    step = 0
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch}")
        step = train_fn(
            disc, gen, train_loader, val_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler, step)

        if SAVE_MODEL and epoch % 3 == 0:
            save_checkpoint(gen, opt_gen, filename=weights_paths + str(epoch) + '_' + CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=weights_paths + str(epoch) + '_' + CHECKPOINT_DISC)


if __name__ == "__main__":
    main()
