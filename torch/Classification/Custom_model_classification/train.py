import wandb
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


configs = dict(
    dataset = 'MIT_small_train_1',
    n_class = 8,
    image_width = 256,
    image_height = 256,
    batch_size = 32,
    model_name = 'VGG_SP_NOFC_keras',
    epochs = 100,
    learning_rate = 0.01,
    optimizer = 'nadam',
    loss_fn = 'categorical_crossentropy',
    metrics = ['accuracy'],
    weight_init = "glorot_normal",
    activation = "relu",
    regularizer = "l2",
    reg_coef = 0.01,
    # Data augmentation
    width_shift_range = 0,
    height_shift_range = 0,
    horizontal_flip = False,
    vertical_flip = False,
    rotation_range = 0,
    brightness_range = [0.8, 1.2],
    zoom_range = 0.15,
    shear_range = 0

)


class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
        # Initialize weights using Glorot Normalization
        nn.init.xavier_normal_(self.depthwise.weight)
        nn.init.xavier_normal_(self.pointwise.weight)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class VGG_SP_NOFC(nn.Module):
    def __init__(self, config):
        super(VGG_SP_NOFC, self).__init__()
        # Separable conv 1
        self.sep1 = SeparableConv(3, 16)
        # Batch norm 1
        self.batch1 = nn.BatchNorm2d(16)
        
        # Separable conv 2
        self.sep2 = SeparableConv(16, 32)
        # Batch norm 2
        self.batch2 = nn.BatchNorm2d(32)
        
        # Separable conv 3
        self.sep3 = SeparableConv(32, 64)
        # Batch norm 3
        self.batch3 = nn.BatchNorm2d(64)
        
        # Separable conv 4
        self.sep4 = SeparableConv(64, 128)
        # Batch norm 4
        self.batch4 = nn.BatchNorm2d(128)
        
        # Separable conv 5
        self.sep5 = SeparableConv(128, 256)
        # Batch norm 5
        self.batch5 = nn.BatchNorm2d(256)
        
        # FC
        self.fc = nn.Linear(256, config.n_class)
        nn.init.xavier_normal_(self.fc.weight)
        
        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Global average pooling
        self.gAvPool = nn.AvgPool2d(kernel_size=16, stride=1)
        
        # Activations
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)
        
        

    def forward(self, x):
        # Sep 1
        x = self.pool(self.batch1(self.relu(self.sep1(x))))
        # Sep 2
        x = self.pool(self.batch2(self.relu(self.sep2(x))))
        # Sep 3
        x = self.pool(self.batch3(self.relu(self.sep3(x))))
        # Sep 4
        x = self.pool(self.batch4(self.relu(self.sep4(x))))
        # Sep 5
        x = self.batch5(self.relu(self.sep5(x)))
        # Global pooling
        x = self.gAvPool(x)
        x = torch.squeeze(x)
        # FC
        x = self.fc(x)
        
        # If it is in eval mode
        #if not self.training:
            # Softmax
        #    x = self.softmax(x)
        
        return x


train_data_dir='C:/Users/MICHE/Documents/Datasets/MIT_large_train/train'
test_data_dir = "C:/Users/MICHE/Documents/Datasets/MIT_large_train/test"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def model_pipeline(hyperparameters):

    # access all HPs through wandb.config, so logging matches execution!
    config = wandb.config
    
    # make the model, data, and optimization problem
    model, train_loader, val_loader, criterion, optimizer = make(config)
    total_params = sum(
    	param.numel() for param in model.parameters()
    )
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Param. num: ", total_params)
    print("Trainable Param. num: ", train_params)
    model = model.to(device)
    
    # and use them to train the model
    train(model, train_loader, val_loader, criterion, optimizer, config)
    
    return model


def make(config):
    # No data augmentation
    transformNDA = transforms.Compose([
        transforms.Resize((configs["image_height"], configs["image_width"])),
        transforms.ToTensor(),
        #lambda x: x/255.
    ])
    # Data augmentation
    transformDA = transforms.Compose([
        transforms.Resize((configs["image_height"], configs["image_width"])),
        transforms.ToTensor(),
        #lambda x: x/255.,
        transforms.ColorJitter(brightness=config.brightness_range),
        transforms.RandomResizedCrop((configs["image_height"], configs["image_width"]), [1.-config.zoom_range, 1.+config.zoom_range], [1,1])
        
    ])
    
    # Make the data
    train_dataset = ImageFolder(train_data_dir, transform=transformDA)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_dataset = ImageFolder(test_data_dir, transform=transformNDA)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Make the model)
    model = VGG_SP_NOFC(config = config)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.NAdam(model.parameters(), lr=config.learning_rate)#, weight_decay=config.reg_coef)
    
    return model, train_loader, val_loader, criterion, optimizer


def train(model, train_loader, val_loader, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    
    # Run training and track with wandb
    best_val_acc = 0.
    for epoch in range(config.epochs):
        loss = 0.
        total = 0.
        correct = 0.
        for images, labels in train_loader:

            lossC, totalC, correctC = train_batch(images, labels, model, optimizer, criterion, config)
            loss += lossC
            total += totalC
            correct += correctC

        loss /= len(train_loader)
        acc = correct / total
        # Report metrics every epoch
        train_log(loss, acc, epoch)
        val_acc = val(epoch, model, val_loader, criterion, best_val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    print("Best val acc.: ", best_val_acc)

def train_batch(images, labels, model, optimizer, criterion, config):
    model.train()
    images, labels = images.to(device), labels.to(device)
    
    total, correct = 0, 0
    # Forward pass 
    outputs = model(images)
    loss = criterion(outputs, labels) + config.reg_coef * model.fc.weight.norm(2)
    
    # Backward pass 
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()
    
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    return loss, total, correct

def train_log(loss, acc, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss})
    print(f"Train loss after epoch {str(epoch)}: {loss:.3f}")
    wandb.log({"epoch": epoch, "accuracy": acc})
    print(f"Train acc after epoch {str(epoch)}: {acc:.3f}")
    

    
def val(epoch, model, val_loader, criterion, best_acc):
    model.eval()

    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        loss = 0.
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        loss /= len(val_loader)
        print(f"Accuracy of the model on the {total} " +
              f"val images: {correct / total:%}")
        
        wandb.log({"epoch": epoch, "val_accuracy": correct / total})
        print(f"Val acc after epoch {str(epoch)}: {(correct / total):.3f}")
        wandb.log({"epoch": epoch, "val_loss": loss})
        print(f"Val loss after epoch {str(epoch)}: {loss:.3f}")

    acc = correct / total
    
    if acc > best_acc: 
        checkpoint = {'state_dict': model.state_dict()}
        torch.save(checkpoint, 'model.pth')
        torch.onnx.export(model, images, "model.onnx")
        
        # Save the model in the exchangeable ONNX format
        #wandb.save("model.onnx")
    
    return acc

run = wandb.init(project='VGG_SP_NOFC', config=configs, job_type='train')
wandb.run.name = "PYTORCH_train"

# Build, train and analyze the model with the pipeline
model = model_pipeline(configs)

wandb.finish()