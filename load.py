import config
import torch
from torchvision import models
import helpers

config = config.Config()

# Create a model instance
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Load the state dictionary into the model
model.load_state_dict(torch.load(config.MODEL_STATE_DICT_PATH))

# Move the model to the appropriate device (CPU or GPU)
model = model.to(config.DEVICE)

# Set the model to evaluation mode
model.eval()

# Use the function to predict the class of a new image
test_dir = config.TEST_DIR
classes, class_to_idx = helpers.find_classes(test_dir)
# print(classes)
random_image_path, random_image_class = helpers.random_image(test_dir)
print(random_image_path, random_image_class)
predicted_class_idx = helpers.predict_image(random_image_path, model, config.DEVICE)
helpers.display_image(random_image_path, random_image_class, classes[predicted_class_idx])
