
import os
import glob
import time
import argparse
import random
import scipy.io
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from model import CrowdStampedeYantra as CSRNet

CROP_SIZE = 512

class QNRFDataset(Dataset):
    def __init__(self, root_dir, split='Train', transform=None, crop_size=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.crop_size = crop_size

        self.split_dir = os.path.join(root_dir, split)
        self.img_paths = sorted(glob.glob(os.path.join(self.split_dir, 'img_*.jpg')))

        self.valid_indices = []
        for idx, img_path in enumerate(self.img_paths):
            gt_path = img_path.replace('.jpg', '_ann.mat')
            if os.path.exists(gt_path):
                self.valid_indices.append(idx)
            else:
                print(f"Warning: GT not found for {img_path}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        img_path = self.img_paths[real_idx]
        gt_path = img_path.replace('.jpg', '_ann.mat')

        image = Image.open(img_path).convert('RGB')

        points = scipy.io.loadmat(gt_path)['annPoints']

        if self.crop_size:
            w, h = image.size
            if w < self.crop_size or h < self.crop_size:
                scale = max(self.crop_size/w, self.crop_size/h)
                new_w, new_h = int(w*scale)+1, int(h*scale)+1
                image = image.resize((new_w, new_h))
                points = points * scale
                w, h = new_w, new_h

            dx = random.randint(0, w - self.crop_size)
            dy = random.randint(0, h - self.crop_size)

            image = image.crop((dx, dy, dx+self.crop_size, dy+self.crop_size))

            mask = (points[:, 0] >= dx) & (points[:, 0] < dx+self.crop_size) &                   (points[:, 1] >= dy) & (points[:, 1] < dy+self.crop_size)
            points = points[mask]
            points = points - [dx, dy]

        target = np.zeros((self.crop_size, self.crop_size), dtype=np.float32) if self.crop_size else np.zeros((image.size[1], image.size[0]), dtype=np.float32)

        downsample_ratio = 16
        t_w, t_h = (self.crop_size // downsample_ratio, self.crop_size // downsample_ratio) if self.crop_size else (image.size[0]//downsample_ratio, image.size[1]//downsample_ratio)
        target = np.zeros((t_h, t_w), dtype=np.float32)

        for pt in points:
            x, y = int(pt[0] // downsample_ratio), int(pt[1] // downsample_ratio)
            if 0 <= y < t_h and 0 <= x < t_w:
                target[y, x] = 1.0

        from scipy.ndimage import gaussian_filter
        target = gaussian_filter(target, sigma=1)

        if points.shape[0] > 0:
            t_sum = target.sum()
            if t_sum > 0:
                target = target * (points.shape[0] / t_sum)

        if self.transform:
            image = self.transform(image)

        target = torch.from_numpy(target).unsqueeze(0)

        return image, target

class CustomTransform:
    def __call__(self, img):
        img = 255.0 * F.to_tensor(img)
        img[0,:,:] = img[0,:,:] - 92.8207477031
        img[1,:,:] = img[1,:,:] - 95.2757037428
        img[2,:,:] = img[2,:,:] - 104.877445883
        return img

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_ds = QNRFDataset(
        args.data,
        split='Train',
        transform=CustomTransform(),
        crop_size=CROP_SIZE
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = CSRNet(load_weights=False).to(device)

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
        else:
            print(f"Checkpoint {args.resume} not found.")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(size_average=False).to(device)

    best_mae = 1e9

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        for i, (img, target) in enumerate(train_loader):
            img = img.to(device)
            target = target.to(device)

            output = model(img)

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} Mean Loss: {epoch_loss / len(train_loader):.4f}")

        save_path = f"checkpoint_epoch_{epoch+1}.pth"
        torch.save({'state_dict': model.state_dict()}, save_path)
        print(f"Saved checkpoint: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to dataset (e.g., data/UCF-QNRF_ECCV18)')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')

    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
