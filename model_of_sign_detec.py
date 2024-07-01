import torch
import os
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# The path of the dataset
train_path = r'C:\Users\Mo Khaled\PycharmProjects\OpenCv\Sign-Language-Digits-Dataset/train'
valid_path = r'C:\Users\Mo Khaled\PycharmProjects\OpenCv\Sign-Language-Digits-Dataset/valid'
test_path = r'C:\Users\Mo Khaled\PycharmProjects\OpenCv\Sign-Language-Digits-Dataset/test'

# Loading the images
def load_images_from_folder(folder, image_size=(100, 100)):
    images = []
    labels = []
    label_dict = {folder_name: idx for idx, folder_name in enumerate(os.listdir(folder))}
    for folder_name, label in label_dict.items():
        folder_path = os.path.join(folder, folder_name)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, image_size)  # Resize to specified size
                images.append(img)
                labels.append(label)
            else:
                print(f"Warning: Failed to read image {img_path}")
    return np.array(images), np.array(labels)

train_images, train_labels = load_images_from_folder(train_path)
test_images, test_labels = load_images_from_folder(test_path)

# Normalize images and convert to torch tensors
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# Reshape images to match input size (N, C, H, W)
train_images = train_images.reshape(-1, 1, 100, 100)
test_images = test_images.reshape(-1, 1, 100, 100)

# Convert to torch tensors
train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_images), torch.tensor(train_labels, dtype=torch.long))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_images), torch.tensor(test_labels, dtype=torch.long))

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(28224, 64)  # Adjust input size according to the final feature map size
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No softmax here, as CrossEntropyLoss applies it
        return x

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

    print(f'[{epoch + 1}] loss: {running_loss / n_total_steps:.3f}')

print('Finished Training')
PATH = 'cnn.pth'
torch.save(model.state_dict(), PATH)
'''
# Retrieve a batch of test data
examples = iter(test_loader)
example_data, example_targets = next(examples)

# Plot some example images
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(example_data[i][0], cmap='gray')
plt.show()
'''