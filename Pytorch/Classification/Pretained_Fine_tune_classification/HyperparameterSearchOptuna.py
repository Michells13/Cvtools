# -*- coding: utf-8 -*-
"""
Hyperparameter search after a fine tune training using optuna 

Created on Mon May 22 11:05:39 2023

@author: MICHELL 
"""
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet50




def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    
    
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
    
    
    # Definir el modelo ResNet pre-entrenado
    model = resnet50(pretrained=True)
    num_classes = 8
    
    # Congelar los pesos de las capas convolucionales existentes
    for param in model.parameters():
        param.requires_grad = False
    
    # Reemplazar la capa lineal final para la clasificación de 1000 clases por una nueva capa lineal para 8 clases
    model.fc = nn.Linear(2048, num_classes)
    
    # Definir la función de pérdida y el optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)

    
    # Mover el modelo a GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Entrenamiento del modelo
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        best_val_acc = 0.
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
    
            optimizer.zero_grad()
    
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
    
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            correct_predictions += (predicted == labels).sum().item()
    
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / len(train_loader.dataset)
        if epoch_accuracy >best_val_acc: 
            best_val_acc = epoch_accuracy
            #checkpoint = {'state_dict': model.state_dict()}
            torch.save(model.state_dict(), "model_1.pth")
    
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")
    
    # Evaluación del modelo en el conjunto de datos de prueba
    model.eval()
    test_loss = 0.0
    test_correct_predictions = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
    
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
    
            test_loss += loss.item()
            test_correct_predictions += (predicted == labels).sum().item()
    
    test_loss /= len(test_loader)
    test_accuracy = test_correct_predictions / len(test_loader.dataset)
    print('loss validation :',test_loss)
    print('accuracy validation :',test_accuracy)
    return test_accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
best_trial = study.best_trial
best_learning_rate = best_trial.params['lr']