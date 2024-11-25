import os
import torch
from torch import nn
from torchvision import transforms, datasets
from torchvision.models import resnet18

# Define the RCNN model
class RCNN(nn.Module):
    def __init__(self, num_classes):
        super(RCNN, self).__init__()
        self.cnn = resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        self.rnn = nn.LSTM(input_size=512, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.permute(0, 2, 3, 1).contiguous()
        cnn_features = cnn_features.view(batch_size, -1, 512)
        rnn_out, _ = self.rnn(cnn_features)
        rnn_out = rnn_out[:, -1, :]
        out = self.fc(rnn_out)
        return out

# Load the trained model and class mapping
def load_model(model_path, class_mapping):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(class_mapping)
    model = RCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

# Transform for the test images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# Predict function
def predict_images_in_folder(folder_path, model, device, transform, class_mapping):
    results = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            image_path = os.path.join(root, file)
            image = datasets.folder.default_loader(image_path)
            image = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)
            predicted_label = list(class_mapping.keys())[list(class_mapping.values()).index(predicted.item())]
            results.append((image_path, predicted_label))
    return results

if __name__ == "__main__":
    # Model and folder paths
    model_path = "rcnn_plant_disease.pth"  # Path to the trained model
    test_folder = "predict/"  # Path to the folder with test images

    # Class mapping (update this with the actual mapping used during training)
    class_mapping = {
        "Pepper__bell___Bacterial_spot": 0,
        "Pepper__bell___healthy": 1,
        "Potato___Early_blight": 2,
        "Potato___healthy": 3,
        "Potato___Late_blight": 4,
        "Tomato__Target_Spot": 5,
        "Tomato__Tomato_mosaic_virus": 6,
        "Tomato__Tomato_YellowLeaf__Curl_Virus": 7,
        "Tomato_Bacterial_spot": 8,
        "Tomato_Early_blight": 9,
        "Tomato_healthy": 10,
        "Tomato_Late_blight": 11,
        "Tomato_Leaf_Mold": 12,
        "Tomato_Septoria_leaf_spot": 13,
        "Tomato_Spider_mites_Two_spotted_spider_mite": 14,
    }

    # Load model
    model, device = load_model(model_path, class_mapping)

    # Predict images in the test folder
    predictions = predict_images_in_folder(test_folder, model, device, transform, class_mapping)

    # Display predictions
    print("\nPredictions:")
    for image_path, predicted_label in predictions:
        print(f"Image: {image_path}, Predicted Label: {predicted_label}")
