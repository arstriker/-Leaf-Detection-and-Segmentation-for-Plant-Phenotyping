import os
import shutil

def create_yolo_dataset(src_dir, dest_dir):
    print(f"Creating YOLO dataset in {dest_dir} from {src_dir}")
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    
    splits = ['train', 'validation', 'test']
    
    for split in splits:
        split_src = os.path.join(src_dir, split)
        split_dest = os.path.join(dest_dir, split if split != 'validation' else 'val') # YOLO typically likes val
        
        if not os.path.exists(split_src):
            continue
            
        os.makedirs(split_dest, exist_ok=True)
        print(f"Processing split: {split}")
        
        for species in os.listdir(split_src):
            species_path = os.path.join(split_src, species)
            if not os.path.isdir(species_path):
                continue
                
            for disease in os.listdir(species_path):
                disease_path = os.path.join(species_path, disease)
                if not os.path.isdir(disease_path):
                    continue
                    
                class_name = f"{species}___{disease}"
                class_dest_path = os.path.join(split_dest, class_name)
                os.makedirs(class_dest_path, exist_ok=True)
                
                for img in os.listdir(disease_path):
                    if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                        img_src_path = os.path.join(disease_path, img)
                        img_dest_path = os.path.join(class_dest_path, img)
                        
                        try:
                            os.link(img_src_path, img_dest_path) # Hardlink
                        except OSError:
                            shutil.copy2(img_src_path, img_dest_path) # Fallback to copy

    print("Success: Dataset created!")

if __name__ == '__main__':
    original_dataset = r"D:\Arohan_Softwares\Google projects\Leaf_Detection\image data"
    yolo_dataset = r"D:\Arohan_Softwares\Google projects\Leaf_Detection\yolo_dataset"
    create_yolo_dataset(original_dataset, yolo_dataset)
