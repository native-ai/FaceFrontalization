import torch
import torchvision
from config import writer_fake,writer_real

def test_fn(
        gen, board_loader, step, DEVICE
):
    with torch.no_grad():
        image, target = next(iter(board_loader))
        fake = gen(image.to(DEVICE))
        # take out (up to) 16 examples
        img_grid_real = torchvision.utils.make_grid(target, normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)

        writer_real.add_image("Real", img_grid_real, global_step=step)
        writer_fake.add_image("Fake", img_grid_fake, global_step=step)
