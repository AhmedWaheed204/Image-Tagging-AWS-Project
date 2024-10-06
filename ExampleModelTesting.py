import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## Defining the Architecture of saved Model
class MultiLabelClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelClassifier, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        return x  # Return logits directly
num_classes = 80
model = MultiLabelClassifier(num_classes).to(device)


# Load the model state dict to the CPU
model.load_state_dict(torch.load('/kaggle/input/aws_model/pytorch/default/1/multi_label_classifier.pth', map_location=torch.device('cpu')))

##Function for Testing
def test_single_image(model, image_path, categories_file, transform, device):
    # Load categories
    categories = pd.read_csv(categories_file, header=None)
    category_names = categories.iloc[:, 0].tolist()  # Assuming the first column contains category names

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    # Set model to evaluation mode
    model.eval()

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()  # Apply sigmoid to outputs

    # Get top 5 predictions
    top5_indices = probabilities.argsort()[-5:][::-1]
    top5_probabilities = probabilities[top5_indices]
    top5_categories = [category_names[i] for i in top5_indices]

    # Print results
    print(f"Top 5 predictions for image: {image_path}")
    for category, prob in zip(top5_categories, top5_probabilities):
        print(f"{category}: {prob:.4f}")


#Testing
image_path = '/kaggle/input/coco-dataset-for-multi-label-image-classification/imgs/imgs/test/000000000265.jpg'
test_single_image(model, 
                 image_path,
                 '/kaggle/input/coco-dataset-for-multi-label-image-classification/labels/labels/categories.csv',
                 transform,
                 device='cpu'
                 )
