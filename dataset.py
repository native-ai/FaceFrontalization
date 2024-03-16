import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class MapDataset(Dataset):
    def __init__(self, root_dir,angle=[15,30,45,60,75,90]):
        self.root_dir = root_dir
        self.angle=angle
        self.list_files=[]
        self.prepare()
    def __len__(self):
        return len(self.list_files)
    def prepare(self,):
        for a in self.angle:
            for lst in os.listdir(self.root_dir+str(a)):
                if not lst.endswith("test.png"):
                    self.list_files.append(str(a)+"/"+lst)
                
    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        input_image = np.array(Image.open(img_path[:-4]+"_test.png"))
        target_image = np.array(Image.open(img_path))

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image


if __name__ == "__main__":
    root_dir=r"data/TP-GAN-MultiPIE-test-Setting2/"

    dataset = MapDataset(root_dir=root_dir)
    loader = DataLoader(dataset, batch_size=16)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        exit(1)