"""MNIST and CIFAR-10 dataloaders."""
import torch
from torchvision import datasets, transforms


def get_datasets(name: str, root: str = "./data"):
    if name == "mnist":
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST(root, train=True, download=True, transform=tf)
        test_ds = datasets.MNIST(root, train=False, download=True, transform=tf)
        info = dict(in_channels=1, image_size=28, num_classes=10)
    elif name == "cifar10":
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_tf = transforms.Compose([transforms.ToTensor(), normalize])
        train_ds = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
        test_ds = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
        info = dict(in_channels=3, image_size=32, num_classes=10)
    else:
        raise ValueError(name)
    return train_ds, test_ds, info


def make_loaders(train_ds, test_ds, batch_size: int, num_workers: int = 0):
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader
