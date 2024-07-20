import config
import helpers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.models as models
from torchinfo import summary
from tqdm.auto import tqdm
from timeit import default_timer as timer

config = config.Config()
print(f"Device: {config.DEVICE}")

# Write transform for image
train_transform = transforms.Compose([
    # Flip the images randomly on the horizontal
    transforms.Resize((config.INPUT_HEIGHT, config.INPUT_WIDTH)),
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    transforms.TrivialAugmentWide(num_magnitude_bins=31), # how intense the augmentation should be
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
])

valid_transform = transforms.Compose([
    transforms.Resize((config.INPUT_HEIGHT, config.INPUT_WIDTH)),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((config.INPUT_HEIGHT, config.INPUT_WIDTH)),
    transforms.ToTensor()
])


# Display a random image
# helpers.display_random_image(config.TRAIN_DIR)

# Display random images that have been transformed
image_paths = list(config.TRAIN_DIR.glob(f'*/*{config.IMAGE_TYPE}'))
# helpers.plot_transformed_images(image_paths, train_transform, n=3)

train_data = datasets.ImageFolder(root=config.TRAIN_DIR, transform=train_transform, target_transform=None)
valid_data = datasets.ImageFolder(root=config.VALID_DIR, transform=train_transform)
test_data = datasets.ImageFolder(root=config.TEST_DIR, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False)
test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False)

# img, label = next(iter(train_loader))
# print(len(train_data))
# # Batch size will now be 1, try changing the batch_size parameter above and see what happens
# print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
# print(f"Label shape: {label.shape}")


# Calculate the number of samples to take
num_train_samples = int(len(train_data) * config.SUBSET_FRACTION)

# Create a subset of the dataset
train_subset = Subset(train_data, list(range(num_train_samples)))

# Create DataLoader for the subset
train_subset_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True)

# For validation
valid_subset_size = int(len(valid_data) * config.SUBSET_FRACTION)  # 10% of the validation set
valid_subset = Subset(valid_data, list(range(valid_subset_size)))
valid_subset_loader = DataLoader(valid_subset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False)



class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from:
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*56*56,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        # x = self.conv_block_1(x)
        # print(x.shape)
        # x = self.conv_block_2(x)
        # print(x.shape)
        # x = self.classifier(x)
        # print(x.shape)
        # return x
        return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion

# model_0 = TinyVGG(input_shape=3, # number of color channels (3 for RGB)
#                   hidden_units=10,
#                   output_shape=len(train_data.classes)).to(config.DEVICE)

# # 1. Get a batch of images and labels from the DataLoader
# img_batch, label_batch = next(iter(train_loader))

# # 2. Get a single image from the batch and unsqueeze the image so its shape fits the model
# img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
# print(f"Single image shape: {img_single.shape}\n")

# # 3. Perform a forward pass on a single image
# model_0.eval()
# with torch.inference_mode():
#     pred = model_0(img_single.to(config.DEVICE))

# 4. Print out what's happening and convert model logits -> pred probs -> pred label
# print(f"Output logits:\n{pred}\n")
# print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
# print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
# print(f"Actual label:\n{label_single}")

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(config.DEVICE), y.to(config.DEVICE)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(config.DEVICE), y.to(config.DEVICE)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc



# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):

    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results
model = models.efficientnet_b0(weights = models.EfficientNet_B0_Weights.DEFAULT)
model.to(config.DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

start_time = timer()

model_results = train(model=model,
                        train_dataloader=train_subset_loader,
                        test_dataloader=valid_subset_loader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=config.NUM_EPOCHS)

end_time = timer()

time = end_time - start_time
if time < 60:
    print(f"Execution time: {time:.2f} seconds")
elif time >= 60 and time < 3600:
    print(f"Execution time: {time/60:.2f} minutes")
else:
    print(f"Execution time: {time/3600:.2f} hours")

# Save the model
torch.save(model.state_dict(), config.MODEL_STATE_DICT_PATH)
# Plot the results
helpers.plot_loss_curves(model_results)
