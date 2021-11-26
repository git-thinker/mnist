import torch
import torchvision
import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset


class PostIDLocator():
    """
    This is used to locate postid in a full picture
    """
    def __init__(self, threshold=0.45, shrink=0):
        self.threshold = threshold
        self.shrink = shrink

    def locate(self, im):
        """
        Function takes raw cv2.imread() and binar-ified
        (H, W)
        """

        # Focus on left-upper half
        im = im[:im.shape[0] // 2, :im.shape[1] // 2, :]

        # Cast into gray
        im = im.mean(axis=2) / 255.0
        
        # Cast into Black and White (Black as Background)
        im = cv2.threshold(im, self.threshold, 1, cv2.THRESH_BINARY)[1]

        # Right Up edge Detection using Boolean Approach

        im_right = np.roll(im, shift=(1,), axis=(1,))
        im_up  = np.roll(im, shift=(-1,), axis=(0,))

        im_right = np.logical_or(im, im_right)
        im_up = np.logical_or(im, im_up)

        im_right = np.logical_xor(im, im_right)
        im_up = np.logical_xor(im, im_up)

        im = np.logical_or(im_right, im_up)
        
        idxW = np.sort(im.sum(axis=0).argsort()[-12:])
        idxH = np.sort(im.sum(axis=1).argsort()[-2:])

        return [
            (
                idxW[i]+self.shrink, 
                idxH[0]+self.shrink, 
                idxW[i+1]-self.shrink, 
                idxH[1]-self.shrink
            ) for i in range(0, len(idxW), 2)]

class Binarify(torch.nn.Module):
    def __init__(self, threshold=0.5):
        super(Binarify, self).__init__()
        self.threshold = threshold
    
    def forward(self, img):
        return Image.fromarray(1-cv2.threshold(np.asarray(img)/255.0, self.threshold, 1, cv2.THRESH_BINARY)[1])
        
class EnvelopDataset(Dataset):
    def __init__(self, dir:str, threshold=0.45, shrink=0):
        self.dir = dir
        self.file_list = os.listdir(dir)
        self.locator = PostIDLocator(threshold=threshold, shrink=shrink)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            Binarify(threshold=0.5),
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = os.path.join(self.dir, self.file_list[idx])
        im = Image.open(path)
        points = self.locator.locate(cv2.imread(path))
        return torch.stack([self.transform(im.crop(item)) for item in points]), path

class EnvelopDevDataset(Dataset):
    def __init__(self, dir:str, threshold=0.45, shrink=0):
        self.dir = dir
        self.file_list = os.listdir(dir)
        self.locator = PostIDLocator(threshold=threshold, shrink=shrink)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=1),
            Binarify(threshold=0.5),
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = os.path.join(self.dir, self.file_list[idx])
        im = Image.open(path)
        points = self.locator.locate(cv2.imread(path))
        labels = torch.tensor([(int(self.file_list[idx][:-4])-1+i)%10 for i in range(6)])
        ims = [self.transform(im.crop(item)) for item in points] 

        for i in range(len(ims)):
            nonzero_idx_1 = torch.nonzero(ims[i].squeeze().sum(dim=-2)>1)
            shift_1 = 16 - (nonzero_idx_1.max() + nonzero_idx_1.min()).item() // 2
            nonzero_idx_2 = torch.nonzero(ims[i].squeeze().sum(dim=-1)>1)
            shift_2 = 16 - (nonzero_idx_2.max() + nonzero_idx_2.min()).item() // 2
            ims[i] = torch.roll(input=ims[i], shifts=(shift_2, shift_1), dims=(-2, -1))

        return torch.stack(ims), labels, path


class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.Sequence = torch.nn.Sequential(
            # (1, 32, 32)
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),

            torch.nn.BatchNorm2d(num_features=32),

            # (32, 32, 32)
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # (32, 16, 16)
            torch.nn.ReLU(),

            # (32, 16, 16)
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            

            torch.nn.BatchNorm2d(num_features=64),

            # (64, 16, 16)
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # (64, 8, 8)
            torch.nn.ReLU(),

            # (64, 8, 8)
            torch.nn.Flatten(start_dim=1),

            # (64 * 8 * 8 = 4096)
            torch.nn.Linear(in_features=4096, out_features=1024),

            # (1024, )
            torch.nn.ReLU(),

            # (1024, )
            torch.nn.Linear(in_features=1024, out_features=128),

            torch.nn.ReLU(),

            # (128, )
            torch.nn.Linear(in_features=128, out_features=10),

            # (10, )
            torch.nn.Softmax(),
        )
        

    def forward(self, x):
        return self.Sequence(x)


class Transpose(torch.nn.Module):
    def __init__(self, dim0:int, dim1:int) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x:torch.Tensor):
        return x.transpose(dim0=self.dim0, dim1=self.dim1)

class AttentionNet(torch.nn.Module):
    def __init__(self):
        super(AttentionNet, self).__init__()
        self.sequence = torch.nn.Sequential(*[
            # ViT-like Embedding
            # (1, 32, 32)
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=4), 
            # (32, 8, 8)
            torch.nn.Flatten(start_dim=-2, end_dim=-1),
            # (32, 64)
            Transpose(-1, -2),
            # (64, 32)
            torch.nn.TransformerEncoder(
                encoder_layer=torch.nn.TransformerEncoderLayer(
                    d_model=32, 
                    nhead=4
                ), 
                num_layers=12, 
                norm=torch.nn.LayerNorm(normalized_shape=32)
            ),
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(in_features=32*64, out_features=256),
            torch.nn.GELU(),
            torch.nn.Linear(in_features=256, out_features=10),
            torch.nn.Softmax(dim=-1),
        ])
        

    def forward(self, x):
        return self.sequence(x)

class VitNet(torch.nn.Module):
    def __init__(self):
        super(VitNet, self).__init__()
        self.cls_token = torch.nn.Parameter(torch.rand(1, 1, 32))
        self.positional_embedding = torch.nn.Parameter(torch.rand(1, 65, 32))
        self.Embedding = torch.nn.Sequential(*[
            # ViT-like Embedding
            # (1, 32, 32)
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=4), 
            # (32, 8, 8)
            torch.nn.Flatten(start_dim=-2, end_dim=-1),
            # (32, 64)
            Transpose(-1, -2),
            # (64, 32)
        ])

        self.TransformerEncoder = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=32, 
                nhead=4,
                norm_first=True,
                batch_first=True
            ),
            num_layers=4, 
            norm=torch.nn.LayerNorm(normalized_shape=32),
        )
        self.Output = torch.nn.Sequential(*[
            torch.nn.Flatten(start_dim=1, end_dim=-1),
            torch.nn.Linear(in_features=32, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=10),
            torch.nn.Softmax(dim=-1),
        ])
        

    def forward(self, x:torch.Tensor):
        batch_num = x.shape[0]
        # (1, 32, 32)
        x = self.Embedding(x)
        # (64, 32)
        cls_token = self.cls_token.expand(batch_num, 1, 32)
        x = torch.cat((cls_token, x), dim=1)

        # (65, 32)
        x += self.positional_embedding

        # (65, 32)
        x = self.TransformerEncoder(x)

        # (65, 32)
        cls_token_final = x[:, 0, :]

        x = self.Output(cls_token_final)
        return x