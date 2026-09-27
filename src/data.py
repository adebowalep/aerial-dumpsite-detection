   
import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def preprocess_image(img):
    img = torch.tensor(img).permute(2,0,1)
    return img.to(device).float()

class DumpsiteDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None, common_size=(1024, 1024)):   #(1024, 1024)
        """
        Initialize the DumpsiteDataset.

        Args:
            root_dir (str): Root directory of the dataset.
            transform (callable, optional): A function/transform to apply to the images.
            target_transform (callable, optional): A function/transform to apply to the target data.
            common_size (tuple, optional): Common size for images (width, height).

        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.common_size = common_size

        # List all image files in the root directory
        self.image_files = glob.glob(os.path.join(root_dir, 'images', '*.jpeg'))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image and annotation paths
        img_path, txt_path = self.get_image_and_annotation_paths(idx)
        image = Image.open(img_path).convert("RGB")
        image = np.array(image.resize(self.common_size, resample=Image.BILINEAR))/255.
        
        # Load and process annotations
        boxes, labels = self.load_and_process_annotations(txt_path, image)
        if len(boxes) == 0:
            # If there are no valid bounding boxes, you can choose to skip this sample
            # or return None or a special flag to indicate that it should be excluded.
            return None
        
        target = {
        'boxes': torch.tensor(boxes, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.int64)
        }
        image = preprocess_image(image)
        
        return  image , target

    def get_image_and_annotation_paths(self, idx):
        img_path = self.image_files[idx]

        img_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_name = img_name + '.txt'
        txt_path = os.path.join(self.root_dir, 'annotations', txt_name)

        return img_path, txt_path
    
    def load_and_process_annotations(self, txt_path, image):
        boxes, labels = [], []

        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    label, x_min, y_min, x_max, y_max = map(float, parts)
                    x_min, y_min, x_max, y_max = self.convert_relative_to_absolute_coords(x_min, y_min, x_max, y_max, image)

                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(label)
        
        return boxes, labels    

    def convert_relative_to_absolute_coords(self, x_min, y_min, x_max, y_max, image):
       
        image_width, image_height = self.common_size
        
        #x_min = x_min * image_width
        #y_min = y_min * image_height
        #width = x_max * image_width
        #height = y_max * image_height
        return  x_min, y_min, x_max, y_max
        
    
    def collate_fn(self, batch):
        valid_batch = [item for item in batch if item is not None]
        if not valid_batch:
            return None  # Return None if there are no valid samples

        # Unzip the batch
        images, targets = zip(*valid_batch)
        return images, targets
        #return tuple(zip(*batch))

    