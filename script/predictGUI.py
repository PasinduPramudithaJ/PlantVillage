import sys
import torch
from torch import nn
from torchvision import transforms, datasets
from torchvision.models import resnet18
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QStackedWidget
)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt

# RCNN Model Definition
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

# Load Model and Class Mapping
def load_model(model_path, class_mapping):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(class_mapping)
    model = RCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

# Image Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# Predict Function
def predict_image(image_path, model, device, transform, class_mapping):
    image = datasets.folder.default_loader(image_path)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    predicted_label = list(class_mapping.keys())[list(class_mapping.values()).index(predicted.item())]
    return predicted_label

# Prescriptions for each disease
prescriptions = {
    "Pepper__bell___Bacterial_spot": "Use copper-based fungicides and rotate crops.",
    "Pepper__bell___healthy": "No treatment required. Maintain healthy practices.",
    "Potato___Early_blight": "Apply fungicides like mancozeb or chlorothalonil.",
    "Potato___healthy": "No action needed. Maintain healthy practices.",
    "Potato___Late_blight": "Use fungicides such as metalaxyl or chlorothalonil.",
    "Tomato__Target_Spot": "Use fungicides and ensure proper air circulation.",
    "Tomato__Tomato_mosaic_virus": "Remove infected plants and use virus-free seeds.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whiteflies and use resistant plant varieties.",
    "Tomato_Bacterial_spot": "Use copper-based sprays and resistant varieties.",
    "Tomato_Early_blight": "Apply mancozeb or chlorothalonil fungicides.",
    "Tomato_healthy": "No treatment required. Maintain good crop health.",
    "Tomato_Late_blight": "Apply chlorothalonil or mancozeb fungicides.",
    "Tomato_Leaf_Mold": "Use fungicides and avoid overhead watering.",
    "Tomato_Septoria_leaf_spot": "Remove infected leaves and apply fungicides.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Use miticides and introduce natural predators.",
}

# Main GUI
class PlantDiseaseApp(QMainWindow):
    def __init__(self, model, device, transform, class_mapping):
        super().__init__()
        self.model = model
        self.device = device
        self.transform = transform
        self.class_mapping = class_mapping

        self.setWindowTitle("Plant Disease Detector")
        self.setGeometry(300, 200, 800, 600)
        self.setWindowIcon(QIcon("icon/icon.png"))

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.home_page()
        self.detection_page()

        self.stacked_widget.setCurrentIndex(0)

    def home_page(self):
        self.home_widget = QWidget()
        layout = QVBoxLayout(self.home_widget)

        title_label = QLabel("🌿 Plant Disease Detector 🌿")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 28px; font-weight: bold; color: green;")
        layout.addWidget(title_label)

        start_button = QPushButton("Start Diagnosis")
        start_button.setStyleSheet("font-size: 20px; background-color: #4CAF50; color: white; padding: 15px;")
        start_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        layout.addWidget(start_button)

        self.stacked_widget.addWidget(self.home_widget)

    def detection_page(self):
        self.main_widget = QWidget()
        layout = QVBoxLayout(self.main_widget)

        self.image_label = QLabel("Upload an Image")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("font-size: 18px; color: gray; border: 1px solid #ddd;")
        layout.addWidget(self.image_label)

        self.result_label = QLabel("Prediction: ")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        layout.addWidget(self.result_label)

        self.prescription_label = QLabel("Prescription: ")
        self.prescription_label.setAlignment(Qt.AlignCenter)
        self.prescription_label.setWordWrap(True)
        self.prescription_label.setStyleSheet("font-size: 16px; color: #555;")
        layout.addWidget(self.prescription_label)

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.setStyleSheet("font-size: 20px; background-color: #4CAF50; color: white;width: 200px; padding: 15px;")
        self.upload_button.clicked.connect(self.upload_image)
        layout.addWidget(self.upload_button)

        back_button = QPushButton("Back to Home")
        back_button.setStyleSheet("font-size: 18px; background-color: #f44336; color: white;width: 200px; padding: 15px;")
        back_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        layout.addWidget(back_button)

        self.stacked_widget.addWidget(self.main_widget)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png)")
        if file_path:
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            self.image_label.setText("")

            predicted_label = predict_image(file_path, self.model, self.device, self.transform, self.class_mapping)
            prescription = prescriptions.get(predicted_label, "No prescription available.")

            self.result_label.setText(f"Prediction: {predicted_label}")
            self.prescription_label.setText(f"Prescription: {prescription}")

if __name__ == "__main__":
    model_path = "rcnn_plant_disease.pth"
    class_mapping = { "Pepper__bell___Bacterial_spot": 0,
        "Pepper__bell___healthy": 1,
        "Potato___Early_blight": 2,
        "Potato___healthy": 4,
        "Potato___Late_blight": 3,
        "Tomato__Target_Spot": 11,
        "Tomato__Tomato_mosaic_virus": 13,
        "Tomato__Tomato_YellowLeaf__Curl_Virus": 12,
        "Tomato_Bacterial_spot": 5,
        "Tomato_Early_blight": 6,
        "Tomato_healthy": 14,
        "Tomato_Late_blight": 7,
        "Tomato_Leaf_Mold": 8,
        "Tomato_Septoria_leaf_spot": 9,
        "Tomato_Spider_mites_Two_spotted_spider_mite": 10,}
    model, device = load_model(model_path, class_mapping)

    app = QApplication(sys.argv)
    window = PlantDiseaseApp(model, device, transform, class_mapping)
    window.show()
    sys.exit(app.exec_())