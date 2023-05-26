import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn as nn



'''
This code runs an inference of a simple classifier model
 built from the other .py file in the same folder, at the 
 end it prints the name of the classes of each sample , 
 in order to make it work you have to modify the paths of 
 the model and the test images.
'''

labels = ["Opencountry", "coast", "forest", "highway", "inside_city", "mountain", "street", "tallbuilding"]
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
        self.fc = nn.Linear(256, config["n_class"])
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
configs = dict(
    dataset = 'MIT_small_train_1',
    n_class = 8,
    image_width = 256,
    image_height = 256,
    batch_size = 32,
    model_name = 'VGG_SP_NOFC_keras',
    epochs = 5,
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
# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create an instance of the ImageFolder dataset for your new data
test_data_dir = "C:/Users/MICHE/Documents/Datasets/MIT_large_train/test"
test_dataset = ImageFolder(test_data_dir, transform=transform)

# Create a data loader for the test dataset
batch_size = 32
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the saved model checkpoint
checkpoint_path = "C:/Users/MICHE/Desktop/Master/MTP/torch/model.pth"
checkpoint = torch.load(checkpoint_path)

# Create an instance of the model using the model architecture class
model = VGG_SP_NOFC(config=configs)

# Load the model's state dictionary from the checkpoint
model.load_state_dict(checkpoint['state_dict'])

# Set the model to evaluation mode
model.eval()

# Move the model to the appropriate device (CPU or GPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Perform inference on the test dataset
predictions = []
saved=[]
for images, _ in test_loader:
    images = images.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    predictions.extend(predicted.tolist())
 

# Print the predicted labels
print(predictions)
print(len(predictions))
for i in range(1,300):
    print(labels[predictions[i]])