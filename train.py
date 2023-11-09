import torch
import numpy as np
from scipy.stats import norm
from math import prod
#import torchaudio as ta
import torchvision as tv
import lightning as L

torch.set_float32_matmul_precision("medium")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, other_dataset, transform=None):
        self.items = tuple((transform(x), y) if transform else (x, y) for x, y in other_dataset)
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]


# Define experimental optimizer
class ConvolutionalOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr):
        params = tuple(params)
        super().__init__(params, defaults=dict(lr=lr))

        self.lr = lr
        self.base_optimizer = torch.optim.Adam(params, lr=lr)
        self.param_groups = self.base_optimizer.param_groups

        kernel_range = np.arange(-1, 1, 0.1)
        pdf = norm.pdf(kernel_range, 0, 1)
        self.bellcurve_1d = torch.tensor(pdf, dtype=torch.float32, device=device, requires_grad=False)
        self.bellcurve_2d = torch.outer(self.bellcurve_1d, self.bellcurve_1d)
        self.bellcurve_1d = self.bellcurve_1d.unsqueeze(0).unsqueeze(0)
        self.bellcurve_2d = self.bellcurve_2d.unsqueeze(0).unsqueeze(0)

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def step(self, closure=None):
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    elif len(p.grad.shape) == 1: # Linear Layer updates
                        inp1 = p.grad.unsqueeze(0).unsqueeze(0)
                        new_grad = torch.nn.functional.conv1d(inp1, self.bellcurve_1d, padding="same")
                        p.grad = new_grad.squeeze(0).squeeze(0)
                    elif len(p.grad.shape) == 2: # FC layer updates
                        inp = p.grad.unsqueeze(0).unsqueeze(0)
                        new_grad = torch.nn.functional.conv2d(inp, self.bellcurve_2d, padding="same")
                        p.grad = new_grad.squeeze(0).squeeze(0)
                    elif len(p.grad.shape) == 4: # Convolutional layer updates
                        pass
                    else:
                        raise NotImplementedError(
                            f"Only 1D and 4D parameter updates are supported. Got parameter tensor of shape {p.grad.shape}."
                        )
        self.base_optimizer.step(closure=closure)


# Wrap model in LightningModule
class Model(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return ConvolutionalOptimizer(self.model.parameters(), lr=1e-3)


# hparams
batch_size = True # How much data? Yes.
ds_num_workers = 16
devices = 8
dataset = tv.datasets.CIFAR10
dataset_name = dataset.__name__.upper()
dataset_path = f"/tmp/{dataset_name}"

# Load Dataset
cifar_train = dataset(root=dataset_path, train=True, download=True, transform=tv.transforms.ToTensor())
cifar_test = dataset(root=dataset_path, train=False, download=True, transform=tv.transforms.ToTensor())
train_dataset = InMemoryDataset(cifar_train) if batch_size is True else cifar_train
test_dataset = InMemoryDataset(cifar_test) if batch_size is True else cifar_test
loader_args = lambda ds: dict(num_workers=ds_num_workers, batch_size=(len(ds) if batch_size is True else batch_size))
train_loader = torch.utils.data.DataLoader(train_dataset, **loader_args(train_dataset))
test_loader = torch.utils.data.DataLoader(test_dataset, **loader_args(test_dataset))

# Load model
model_name = "resnet18"
resnet_model = torch.hub.load("pytorch/vision:v0.6.0", model_name, verbose=False, weights="ResNet18_Weights.DEFAULT")
total_params = sum(prod(l.shape) for l in resnet_model.parameters())
trainable_params = sum(prod(l.shape) for l in resnet_model.parameters() if l.requires_grad)
print(f"Loaded model: {model_name}")
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
# print(f"Parameters:")
# [print(f"{n:<28} {p.shape}") for n, p in resnet_model.named_parameters()]

# Train model
model = Model(resnet_model)
trainer = L.Trainer(max_epochs=100, log_every_n_steps=1, devices=devices)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
