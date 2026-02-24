import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class HierarchicalLeafDataset(Dataset):
    """
    PyTorch Dataset for loading images from a hierarchical directory structure:
    Split (train/test/val) -> Species -> Health Status (Disease) -> Image
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images (e.g., 'image data/train').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        # Parse the directory structure
        self._load_dataset()

    def _load_dataset(self):
        classes = set()
        
        # First pass to find all unique classes (Species___Disease)
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"The directory {self.root_dir} does not exist.")
            
        for species_folder in os.listdir(self.root_dir):
            species_path = os.path.join(self.root_dir, species_folder)
            if not os.path.isdir(species_path):
                continue
                
            for health_status_folder in os.listdir(species_path):
                health_status_path = os.path.join(species_path, health_status_folder)
                if not os.path.isdir(health_status_path):
                    continue
                
                # Create a unique class name: e.g., "Cassava___Healthy"
                class_name = f"{species_folder}___{health_status_folder}"
                classes.add(class_name)
                
        # Create class indexing
        self.classes = sorted(list(classes))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Second pass to load image paths and their corresponding labels
        for species_folder in os.listdir(self.root_dir):
            species_path = os.path.join(self.root_dir, species_folder)
            if not os.path.isdir(species_path):
                continue
                
            for health_status_folder in os.listdir(species_path):
                health_status_path = os.path.join(species_path, health_status_folder)
                if not os.path.isdir(health_status_path):
                    continue
                
                class_name = f"{species_folder}___{health_status_folder}"
                label_idx = self.class_to_idx[class_name]
                
                for image_name in os.listdir(health_status_path):
                    # Filter for image files
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                        image_path = os.path.join(health_status_path, image_name)
                        self.image_paths.append(image_path)
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_paths[idx]
        
        # Load image and convert to RGB (some might be grayscale or RGBA)
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Create a dummy image in case of failure to avoid crashing the loader
            image = Image.new('RGB', (256, 256), color='black')

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(data_dir, batch_size=32, num_workers=4, img_size=(224, 224)):
    """
    Creates and returns PyTorch DataLoaders for train, validation, and test splits.
    
    Args:
        data_dir (str): Base directory containing 'train', 'validation', and 'test' subdirectories.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of subprocesses for data loading.
        img_size (tuple): Target image size (height, width).
    
    Returns:
        dict: A dictionary containing 'train', 'val', and 'test' DataLoaders, and the 'class_to_idx' mapping.
    """
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation and test transforms typically don't include augmentation
    val_test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataloaders = {}
    class_to_idx = None
    
    # Define paths
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'validation')
    test_dir = os.path.join(data_dir, 'test')
    
    # Create datasets and dataloaders
    if os.path.exists(train_dir):
        train_dataset = HierarchicalLeafDataset(root_dir=train_dir, transform=train_transform)
        dataloaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        class_to_idx = train_dataset.class_to_idx
        print(f"Loaded Train Dataset: {len(train_dataset)} images, {len(class_to_idx)} classes.")
        
    if os.path.exists(val_dir):
        val_dataset = HierarchicalLeafDataset(root_dir=val_dir, transform=val_test_transform)
        # Ensure validation set uses the same class_to_idx mapping if you want to be safe, 
        # but the alphabetical sorting in __init__ guarantees consistency if directory structures match.
        dataloaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        print(f"Loaded Validation Dataset: {len(val_dataset)} images.")
        
    if os.path.exists(test_dir):
        test_dataset = HierarchicalLeafDataset(root_dir=test_dir, transform=val_test_transform)
        dataloaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        print(f"Loaded Test Dataset: {len(test_dataset)} images.")

    return dataloaders, class_to_idx

if __name__ == '__main__':
    # Test the dataloaders
    base_dir = r"D:\Arohan_Softwares\Google projects\Leaf_Detection\image data"
    print(f"Testing the dataset loader with directory: {base_dir}")
    
    # We use num_workers=0 for simple testing to avoid multiprocessing issues in some Windows environments
    dataloaders, class_to_idx = get_dataloaders(base_dir, batch_size=8, num_workers=0)
    
    if dataloaders and class_to_idx:
        print("\nClass to Index Mapping (first 5):")
        for i, (k, v) in enumerate(class_to_idx.items()):
            if i < 5:
                print(f"  {k} -> {v}")
            elif i == 5:
                print("  ...")
                
        if 'train' in dataloaders:
            images, labels = next(iter(dataloaders['train']))
            print(f"\nTrain Batch Img Shape: {images.shape}")
            print(f"Train Batch Labels: {labels}")
