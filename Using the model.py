import torch
import cv2
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


# Load the model architecture
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


# Initialize the model and load the saved weights
model = ConvNet()
model.load_state_dict(torch.load('cnn.pth'))
model.eval()  # Set the model to evaluation mode


# Preprocess the image
def preprocess_image(image_path, image_size=(100, 100)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    img = cv2.resize(img, image_size)
    img = img.astype(np.float32) / 255.0
    img = img.reshape(-1, 1, 100, 100)  # Reshape to match input size (N, C, H, W)
    img = torch.tensor(img)
    return img


# Predict function
def predict(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)

    # Make prediction
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        return predicted.item()


# Example usage
image_path = r'C:\Users\Mo Khaled\PycharmProjects\OpenCv\Sign-Language-Digits-Dataset\test\0\IMG_4203.JPG'
predicted_label = predict(image_path)
print(f'Predicted Label: {predicted_label}')
