import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.models import resnet50
from PIL import Image
import torchvision.datasets as datasets



# Definir transformaciones para preprocesamiento de imágenes
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Descargar y cargar el conjunto de datos de entrenamiento
train_dataset = datasets.ImageFolder('C:/Users/MICHE/Documents/Datasets/MIT_large_train/train', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Descargar y cargar el conjunto de datos de prueba
test_dataset = datasets.ImageFolder('C:/Users/MICHE/Documents/Datasets/MIT_large_train/test', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
# Cargar la imagen de entrada
input_image = Image.open("C:/Users/MICHE/Documents/Datasets/MIT_large_train/train/tallbuilding/a212018.jpg")
input_tensor = transform(input_image)
input_batch = input_tensor.unsqueeze(0)

# Verificar si hay soporte de GPU y cargar el modelo en el dispositivo adecuado
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo previamente guardado
model = resnet50(pretrained=False)
num_classes = 8
model.fc = nn.Linear(2048, num_classes)
model.load_state_dict(torch.load("C:/Users/MICHE/Desktop/Master/MTP/torch/model_1.pth"))
model.to(device)
model.eval()

# Realizar inferencia en la imagen de entrada
with torch.no_grad():
    input_batch = input_batch.to(device)
    output = model(input_batch)

# Obtener las predicciones y las probabilidades
probabilities = torch.softmax(output, dim=1)[0]
predicted_class_index = torch.argmax(probabilities).item()
predicted_class = train_dataset.classes[predicted_class_index]

print(f"Predicción: {predicted_class}")
print(f"Probabilidades: {probabilities}")
