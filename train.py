import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models
import matplotlib.pyplot as plt
import time
import os
import json

# Import our dataset loader
from dataset import get_dataloaders

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=10, device='cuda'):
    """
    Trains the PyTorch model and evaluates it after each epoch.
    """
    model = model.to(device)
    
    since = time.time()
    
    best_acc = 0.0
    best_model_wts = model.state_dict()
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            # deep copy the model if it has the best validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

def plot_history(history, save_path="training_history.png"):
    """Plots training and validation accuracy/loss."""
    epochs = range(1, len(history['train_acc']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved training history plot to {save_path}")

def main():
    # 1. Setup Config
    data_dir = r"D:\Arohan_Softwares\Google projects\Leaf_Detection\image data"
    batch_size = 32  # Adjust based on GPU memory
    num_epochs = 5   # Just doing 5 for a quick run, increase to 15-20 for full training
    num_workers = 4  # Set to 0 if encountering Windows multiprocessing issues
    learning_rate = 0.001
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Get DataLoaders
    print("Initializing DataLoaders...")
    dataloaders, class_to_idx = get_dataloaders(data_dir, batch_size=batch_size, num_workers=num_workers)
    num_classes = len(class_to_idx)
    
    # Save the class names to a text file for the Inference script to use
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]
    with open("class_names.txt", 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print("Saved class bindings to class_names.txt")
    
    # 3. Setup Model Architecture
    print("Setting up ResNet-18...")
    # Using a pretrained ResNet18 and replacing the final fully connected layer
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model = model.to(device)
    
    # 4. Define Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Decay LR by a factor of 0.1 every 3 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # 5. Train Model
    print("\nStarting Training...")
    model, history = train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs, device=device)
    
    # 6. Save final Model and History
    torch.save(model.state_dict(), "disease_model.pth")
    print("Saved best model weights to disease_model.pth")
    
    plot_history(history)
    with open("training_history.json", 'w') as f:
        json.dump(history, f)
        
    # 7. Evaluate on Test Set
    if 'test' in dataloaders:
        print("\nEvaluating on Test Set...")
        model.eval()
        running_corrects = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
        test_acc = running_corrects.double() / total
        print(f"Final Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    # Due to Windows multiprocessing quirks with PyTorch, always wrap in main
    main()
