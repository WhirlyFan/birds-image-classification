import config
from pathlib import Path
import random
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

config = config.Config()

def display_random_image(image_path: Path, seed=None) -> None:
    if seed:
        random.seed(seed)
    image_path_list = list(image_path.glob('*/*'))
    random_image_path = random.choice(image_path_list)
    image_class = random_image_path.parent.name
    img = Image.open(random_image_path)
    print(f"Random image path: {random_image_path}")
    print(f"Image class: {image_class}")
    print(f"Image height: {img.height}")
    print(f"Image width: {img.width}")
    img_array = np.array(img)
    plt.figure(figsize=(10, 7))
    plt.title(f"Image class: {image_class} | Image shape: {img_array.shape} -> [height, width, color_channels]")
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def plot_transformed_images(image_paths, transform, n=3, seed=None):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths.
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    if seed:
        random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
    plt.show()


def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.savefig(config.ROOT_DIR / 'loss_curves.jpg')
    plt.show()

import torch
from torchvision import transforms
from PIL import Image

def predict_image(image_path, model, device):
    # Define the transform for the input image
    transform = transforms.Compose([
        transforms.Resize((config.INPUT_HEIGHT, config.INPUT_WIDTH)),
        transforms.ToTensor()
    ])

    # Load the image
    image = Image.open(image_path).convert('RGB')

    # Apply the transform to the image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move the image to the appropriate device
    image = image.to(device)

    # Make prediction
    with torch.inference_mode():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()

def find_classes(path: Path) -> Tuple[List[str], Dict[str, int]]:
    """
    Finds the class folders in a given directory and returns the class names and class indices.

    Args:
        path (Path): The path to the directory containing class folders.

    Returns:
        Tuple[List[str], Dict[str, int]]: A tuple containing a list of class names and a dictionary mapping class names to indices.
    """
    # List all subdirectories (assumed to be class folders)
    classes = [d.name for d in path.iterdir() if d.is_dir()]
    # Sort class names to ensure consistent ordering
    classes.sort()
    # Create a dictionary mapping class names to indices
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def random_image(path: Path):
    """
    Returns a random image from a given directory and it's class.

    Args:
        path (Path): The path to the directory containing the image files.

    Returns:
        Tuple: A tuple containing the path to the random image and it's class.
    """
    class_folders = [d for d in path.iterdir() if d.is_dir()]
    random_class = random.choice(class_folders)
    random_image = random.choice(list(random_class.iterdir()))
    return random_image, random_class.name

def display_image(image_path: Path, class_name, predicted_class) -> None:
    """
    Displays an image along with it's class and predicted class.

    Args:
        image_path (Path): The path to the image file.
        class_name (str): The true class name of the
        predicted_class (str): The predicted class name.
    """
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f"True class: {class_name} | Predicted class: {predicted_class}")
    plt.axis('off')
    plt.show()
