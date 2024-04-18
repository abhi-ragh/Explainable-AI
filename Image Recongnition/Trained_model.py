import torch
import torchvision
from captum.attr import IntegratedGradients
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

test_transforms = transforms.Compose([
    transforms.Resize(256),  # Resize for consistency
    transforms.CenterCrop(224),  # Center crop for evaluation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Define path to your saved model
saved_model_path = "saved_model.pt"

# Load the pre-trained ResNet-18 model without loading pre-trained weights
model = torchvision.models.resnet18(pretrained=False)

# Replace the final fully connected layer with one matching the saved model
num_ftrs = model.fc.in_features  # Number of features from previous layer
model.fc = torch.nn.Linear(num_ftrs, 6)  # Adjust output dimension to 6 for your classes

# Load the state dictionary from the saved model
model.load_state_dict(torch.load(saved_model_path))

# Check if GPU is available and move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


test_image_path = "image_data/seg_pred/3.jpg"
image = Image.open(test_image_path).convert('RGB') 

image_tensor = test_transforms(image)

# Convert to a mini-batch of size 1
image_tensor = image_tensor.unsqueeze(0)

# Move the image tensor to the device
image_tensor = image_tensor.to(device)


ig = IntegratedGradients(model)
# Define target class (adjust for your class labels)
target_class = 5  # Assuming "street" is the 5th class

attributions = ig.attribute(image_tensor, target=target_class)

# Normalize attributions to range [0, 1]
attributions_normalized = attributions / torch.max(attributions.abs())

# Original image processing (assuming you have the original image)
orig_image = np.array(image)  # Assuming image is loaded using PIL
orig_image = orig_image[:, :, ::-1]  # Convert RGB to BGR for matplotlib

# Configure a Matplotlib subplot
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot original image
axs[0].imshow(orig_image)
axs[0].set_title("Original Image")
axs[0].axis('off')

# Average attributions over color channels
attributions_2d = attributions_normalized[0].mean(dim=0)

# Now plot the 2D attributions
heatmap = axs[1].imshow(attributions_2d.detach().cpu().numpy(), cmap='hot', alpha=0.4)

axs[1].set_title("Attributions (Normalized)")
axs[1].axis('off')

# Colorbar for heatmap
fig.colorbar(heatmap, ax=axs[1], label='Attribution Magnitude')

# Get predicted class label
_, predicted_label = torch.max(model(image_tensor).data, 1)

# Print prediction
print(f"Predicted Class: {predicted_label.item()} ({class_names[predicted_label.item()]})")

_, predicted_label = torch.max(model(image_tensor).data, 1)
predicted_class_name = class_names[predicted_label.item()]

# Add text annotation to the original image subplot
axs[0].text(0.05, 0.95, f"Predicted Class: {predicted_class_name}", ha='left', va='top', 
            fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.7))

plt.tight_layout()
plt.show()

