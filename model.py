
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

img_size = 28

# Image recognition model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*4*4, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 1)  # Only one output for regression

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn1(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.bn2(self.conv4(x)))
        x = self.pool(x)
        x = x.view(-1, 64*4*4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

net = Net()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# Load pre-trained parameters
model_path = "model_cnn.pth"
try:
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
except Exception as e:
    print(f"Error loading the model due to {e}")

def predict(img):
    # Preprocessing for the model input
    img = img.convert("L")  # Convert to grayscale
    img = img.resize((img_size, img_size))  # Resize the image
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    img = transform(img)
    x = img.reshape(1, 1, img_size, img_size).to(device)

    # Prediction
    with torch.no_grad():
        y = net(x)

    # Return the predicted score
    return y.item()
