import torch
from torchvision import datasets, transforms
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import torch.optim as optim
import time
import torch.utils.data
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR, StepLR, CosineAnnealingLR
import torch
from torch.optim import AdamW
import matplotlib.pyplot as plt
from models import *
import pandas as pd
from models import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
import time



# Function to load a batch file and return a dictionary
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#Load dataset, combine batches, split, and visualize images
def load_and_prepare_data():
    data_batches, label_batches = [], []
    for i in range(1, 6):
        batch = unpickle(f'data/KaggleData/cifar-10-python/cifar-10-batches-py/data_batch_{i}')
        data_batches.append(batch[b'data'])
        label_batches.append(batch[b'labels'])
    X, y = np.concatenate(data_batches), np.concatenate(label_batches)
    
    # Split into training and validation sets (80-20 split)
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = load_and_prepare_data()

#Define transformations
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transformations = {
    'train': transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(), 
        transforms.RandomApply([transforms.RandomErasing()], p=0.5),
        transforms.Normalize(*stats)]),
        
    'valid': transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(*stats)]),
        
    #Normalization only 
    'normalization_only': transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(*stats)])
}
#Adjusting the CIFAR10Dataset class initialization to accept 'data_mode'
class CIFAR10Dataset(Dataset):
    def __init__(self, data, labels, transform=None, data_mode='default'):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.data_mode = data_mode
        if data_mode == 'train_Enhanced':
            self.data = np.concatenate((self.data, self.data), axis=0)
            self.labels = np.concatenate((self.labels, self.labels), axis=0)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.data_mode == 'train_Enhanced' and idx >= len(self.labels) // 2:
            transform = transformations['train']
        else:
            transform = self.transform
        image = self.data[idx % len(self.labels)].reshape(3, 32, 32).transpose(1, 2, 0)
        image = Image.fromarray(image.astype('uint8'), 'RGB')
        image = transform(image)
        return image, self.labels[idx % len(self.labels)]



def main():
 

    #Create datasets and DataLoader instances
    datasets = {
        'train_Enhanced': CIFAR10Dataset(X_train, y_train, transform=transformations['normalization_only'], data_mode='train_Enhanced'),
        'valid': CIFAR10Dataset(X_val, y_val, transform=transformations['valid'])
    }

    # Update loaders for each dataset
    loaders = {
        'train_Enhanced': DataLoader(datasets['train_Enhanced'], batch_size=64, shuffle=True, num_workers=2, pin_memory=True),
        'valid': DataLoader(datasets['valid'], batch_size=64, shuffle=False, num_workers=2,pin_memory=True)
    }

    print('Amount of Train Data batches (Enhanced):', len(loaders['train_Enhanced']))
    print('Amount of Valid Data batches:', len(loaders['valid']))

    print('Amount of training images (Enhanced):', len(datasets['train_Enhanced']))
    print('Amount of Validation images:', len(X_val))


    #---------Define Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(0))

    #--------------Training
    def train_and_evaluate_model(model, model_name, loaders, device, num_epochs=1):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.00001)
        scheduler = ExponentialLR(optimizer, gamma=0.98)

        train_loss_history, valid_loss_history, valid_accuracy_history, train_accuracy_history = [], [], [], []
        best_accuracy = 0.0
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            model.train()
            running_loss, total_train_loss, total_correct_train, num_batches, num_train_examples = 0.0, 0.0, 0, 0, 0
            for i, (inputs, labels) in enumerate(loaders["train_Enhanced"]):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device,non_blocking=True)
                labels = labels.long()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),0.1)
                optimizer.step()

                running_loss += loss.item()
                total_train_loss += loss.item() * inputs.size(0)
                num_batches += 1

                _, predicted = torch.max(outputs.data, 1)
                total_correct_train += (predicted == labels).sum().item()
                num_train_examples += labels.size(0)

                if (i + 1) % 100 == 0 or i == len(loaders["train_Enhanced"]) - 1:
                    print(f"{model_name}, train_Enhanced - Epoch: {epoch} [{(i + 1) * len(inputs)}/{len(loaders['train_Enhanced'].dataset)} "
                        f"({100. * (i + 1) / len(loaders['train_Enhanced']):.0f}%)], Weight Decay: 0.0001,  Loss: {running_loss / num_batches:.4f}")

            avg_train_loss = total_train_loss / num_train_examples 
            train_loss_history.append(avg_train_loss)

            train_accuracy = 100. * total_correct_train / num_train_examples
            train_accuracy_history.append(train_accuracy)

            model.eval()
            total_valid_loss, total_correct_valid, num_valid_batches = 0.0, 0, 0
            for inputs, labels in loaders['valid']:
                inputs, labels = inputs.to(device,non_blocking=True), labels.to(device,non_blocking=True)
                labels = labels.long()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_valid_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_correct_valid += (predicted == labels).sum().item()
                num_valid_batches += 1

            avg_valid_loss = total_valid_loss / len(loaders['valid'].dataset) 
            valid_loss_history.append(avg_valid_loss)

            valid_accuracy = 100. * total_correct_valid / len(loaders['valid'].dataset)
            valid_accuracy_history.append(valid_accuracy)

            print(f"End of Epoch: {epoch}, Avg. Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Avg. Valid Loss: {avg_valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.2f}%")

            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                model_path = f'{model_name}_Final_best_modelNoDropouts.pth'
                torch.save(model.state_dict(), model_path)
                print(f"New best model saved: {model_path}")
                torch.cuda.empty_cache()
            scheduler.step()

        execution_time = time.time() - start_time
        print(f"Training completed. Total execution time: {execution_time:.2f} seconds")

        return train_loss_history, valid_loss_history, valid_accuracy_history, train_accuracy_history
    #-----------------Training call


    num_epochs = 200  # --------------number of Epochs

    all_metrics_final = {
        'train_losses': [],
        'valid_losses': [],
        'valid_accuracies': [],
        'train_accuracies': []
    }

    # #Define Model
    # def ResNet3_with_dropout_30DR():
    #     return ResNet3Dropouts(BasicBlockDropouts, [4, 4, 3], dropout_rate=0)

    # modelResnet3_443_30DR = ResNet3_with_dropout_30DR()
    # total_paramsResnet3_443_30DR = sum(p.numel() for p in modelResnet3_443_30DR.parameters())
    # print(f"Total parameters modelResnet3_443_30DR: {total_paramsResnet3_443_30DR}")

    def Resnet3_443Exp():
        return ResNet3(BasicBlock, [4,4,3])
    model = Resnet3_443Exp().to(device)
    total_paramsResnet3_443 = sum(p.numel() for p in model.parameters())
    print(f"Total parameters modelResnet3_443: {total_paramsResnet3_443}")
    # Re-initialize the model for each weight decay to ensure training starts fresh

    metrics = train_and_evaluate_model(model, "Resnet3_443Exp", loaders, device, num_epochs)





#------------- test dataset prediction
    model_path='Resnet3_443Exp_Final_best_modelNoDropouts.pth'

    # Load best Model
    # model_path = 'modelResnet3_443_30DR_Final_best_model.pth'
    #Load Testing data
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    test_data_dict = unpickle('./data/cifar_test_nolabels.pkl')
    test_images = test_data_dict[b'data']
    test_images = test_images.reshape(-1, 3, 32, 32)

    #Load Best Model, Normalize test data, do an inference and create submission csv file
    test_images = torch.from_numpy(test_images).float()
    if test_images.max() > 1.0:
        test_images /= 255.0
    test_images = test_images.permute(0, 1, 2, 3)
    normalize = transforms.Normalize(*stats)
    normalized_images = torch.stack([normalize(img) for img in test_images])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    normalized_images = normalized_images.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for img in normalized_images:
            img = img.unsqueeze(0).to(device) 
            output = model(img)
            pred = output.argmax(dim=1)
            predictions.append(pred.item())
    print(len(predictions))
    submission_df = pd.DataFrame({
        'ID': list(range(len(predictions))),
        'Labels': predictions
    })
    submission_csv_path = 'submission.csv'
    submission_df.to_csv(submission_csv_path, index=False)
    print(submission_csv_path)



#-------------- plot results
    # Unpack and store the returned metrics
    all_metrics_final['train_losses'], \
    all_metrics_final['valid_losses'], \
    all_metrics_final['valid_accuracies'], \
    all_metrics_final['train_accuracies'] = metrics


    plt.figure(figsize=(20, 10))

    # Training Loss
    plt.subplot(2, 2, 1)
    plt.plot(all_metrics_final['train_losses'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Validation Loss
    plt.subplot(2, 2, 2)
    plt.plot(all_metrics_final['valid_losses'], label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Validation Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(all_metrics_final['valid_accuracies'], label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Training Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(all_metrics_final['train_accuracies'], label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    # Save the figure
    plt.savefig('finalModelPerformancePlotnoDropouts.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()  


