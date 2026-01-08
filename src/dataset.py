import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class XRayDataLoaders:
    def __init__(self, data_dir, image_size=512, batch_size=32, num_workers=2):
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def _build_transforms(self):
        train_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomEqualize(),
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.25]),
        ])

        eval_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.25]),
        ])
        return train_transforms, eval_transforms

    def get_dataloaders(self):
        train_trans, eval_trans = self._build_transforms()

        # Assume estrutura: data_dir/train, data_dir/val, data_dir/test
        train_ds = ImageFolder(os.path.join(self.data_dir, 'train'), transform=train_trans)
        val_ds = ImageFolder(os.path.join(self.data_dir, 'val'), transform=eval_trans)
        test_ds = ImageFolder(os.path.join(self.data_dir, 'test'), transform=eval_trans)

        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, 
                              num_workers=self.num_workers, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, 
                            num_workers=self.num_workers, pin_memory=True)
        test_dl = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, 
                             num_workers=self.num_workers, pin_memory=True)

        return train_dl, val_dl, test_dl
    