from Utils import show_images
from Models import ResUNet
from Dataset import CustomCityscapes

from torchvision import transforms
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse


parser = argparse.ArgumentParser(description='Cityscapes segmentation using a residual U-Net')
parser.add_argument('data_dir', type=str)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--verbose', type=int, default=0)
args = parser.parse_args()

DATASET_DIR = args.data_dir
BATCH_SIZE = args.batch_size
DEVICE = args.device
LR = args.lr
VERBOSE = args.verbose
EPOCHS = args.epochs

if __name__ == '__main__':
    ### Load dataset
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])

    train_dataset = CustomCityscapes(root=DATASET_DIR, split='train', mode='fine', target_type='color', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)

    val_dataset = CustomCityscapes(root=DATASET_DIR, split='val', mode='fine', target_type='color', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)

    # Definition of model, loss and optimizer
    resunet = ResUNet(3, 3).to(DEVICE)
    resunet_optim = torch.optim.Adam(resunet.parameters(), lr=LR)

    criterion = nn.MSELoss()
    display_step = 100

    step = 0
    for epoch in range(EPOCHS):
        for images, masks in tqdm(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            resunet_optim.zero_grad()
            pred = resunet(images)
            resunet_loss = criterion(pred, masks)
            resunet_loss.backward()
            resunet_optim.step()

            if step % display_step == 0 and step > 0:
                print("Epoch {} Step {} Loss {}".format(epoch, step, resunet_loss))
                if VERBOSE > 0:
                    show_images(images)
                    show_images(masks)
                    show_images(pred)
            step += 1

