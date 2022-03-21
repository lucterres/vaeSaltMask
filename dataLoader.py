import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms

TRAIN_MASK_DIR = './train/'

transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
X_train = torchvision.datasets.ImageFolder(root=TRAIN_MASK_DIR, transform=transform)

#print(f"Shape of training data: {X_train.shape}")
print(f"Data type: {type(X_train)}")




