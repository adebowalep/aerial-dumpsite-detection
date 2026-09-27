from tabulate import tabulate
from collections import Counter
import seaborn as sns
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torchvision.ops import nms
import numpy as np
import torch


def decode_output(output,target2label):
    'convert tensors to numpy arrays'
    bbs = output['boxes'].cpu().detach().numpy().astype(np.uint16)
    labels = np.array([target2label[i] for i in output['labels'].cpu().detach().numpy()])
    confs = output['scores'].cpu().detach().numpy()
    ixs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), 0.05)
    bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]

    if len(ixs) == 1:
        bbs, confs, labels = [np.array([tensor]) for tensor in [bbs, confs, labels]]
    return bbs.tolist(), confs.tolist(), labels.tolist()


def generate_label_statistics(train_labels, test_labels):
    all_labels = train_labels + test_labels

    # Calculate the number of occurrences of each class in the combined labels
    class_counts = Counter(all_labels)
    
    # Get a list of unique labels
    unique_labels = sorted(list(set(all_labels)))

    # Prepare the data for tabulation
    table_data = []

    for label in unique_labels:
        train_count = train_labels.count(label)
        test_count = test_labels.count(label)
        total_count = class_counts[label]

        train_percent = (train_count / total_count) * 100
        test_percent = (test_count / total_count) * 100

        table_data.append([label, total_count, train_count, f"{train_percent:.2f}%", test_count, f"{test_percent:.2f}%"])

    # Calculate the total number of images in the train and test sets
    total_train_images = len(train_labels)
    total_test_images = len(test_labels)
    total_images = total_train_images + total_test_images

    # Append a row with totals
    table_data.append(["TOTAL", total_images, total_train_images, "100.00%", total_test_images, "100.00%"])

    # Print the table
    headers = ["Label", "Images", "Train imgs", "Train(%)", "Test imgs", "Test(%)"]
    print(tabulate(table_data, headers, tablefmt="grid"))



def plot_class_distribution(train_labels, test_labels):
    all_labels = train_labels + test_labels

    # Calculate the number of occurrences of each class in the combined labels
    class_counts = Counter(all_labels)

    # Get a list of unique labels
    unique_labels = sorted(list(set(all_labels)))

    # Prepare the data for plotting
    class_names = [label for label, count in class_counts.items()]
    class_counts = [count for label, count in class_counts.items()]

    # Plot the class distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(x=class_counts, y=class_names, hue=class_names, palette="viridis", legend=False)
    plt.title("Class Distribution")
    plt.xlabel("Count")
    plt.ylabel("Class")
    plt.show()

    

def display_images_with_labels_and_bbox(dataset, num_samples=5):
    # Create a DataLoader with a batch size of 1
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Get a batch of samples
    for _ in range(num_samples):
        batch = next(iter(data_loader))
        images, targets = batch

        # Convert tensor to PIL image
        image = F.to_pil_image(images[0].cpu())

        # Create figure and axes with a larger figure size
        fig, ax = plt.subplots(1, figsize=(12, 10))
        ax.imshow(image)

        # Extract bounding boxes and labels
        boxes = targets['boxes'].cpu().numpy()
        labels = targets['labels'].cpu().numpy()

        # Display bounding boxes and labels
        for i in range(len(boxes)):
            box = boxes[i]
            label = labels[i]

            #print(f"Box: {box}, Label: {label}")

            # Check if the box has the expected format
            if len(box) == 1 and len(box[0]) == 4:
                box = box[0]  # Extract the inner array
                # Create a Rectangle patch
                rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                         linewidth=1, edgecolor='r', facecolor='none')

                # Display the rectangle
                ax.add_patch(rect)

                # Set label based on the value
                label_text = 'dumpsite' if int(label) == 1 else 'no_dumpsite'
                plt.text(box[0], box[1], f'Label: {label_text}', color='r')
            else:
                print(f"Skipping invalid bounding box: {box}")

        plt.show()

