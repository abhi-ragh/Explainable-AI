import torch
import torchvision
from torchvision import transforms
import os
from PIL import Image

# Define device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "image_data"  # Replace with your actual path

train_dir = os.path.join(data_dir, "seg_train")
test_dir = os.path.join(data_dir, "seg_test")

# Training transforms with random augmentations
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Resize with random cropping
    transforms.RandomHorizontalFlip(),   # Random horizontal flip
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Testing transforms for standardization
test_transforms = transforms.Compose([
    transforms.Resize(256),  # Resize for consistency
    transforms.CenterCrop(224),  # Center crop for evaluation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = torchvision.datasets.ImageFolder(test_dir, transform=test_transforms)

batch_size = 32  # Adjust batch size based on your GPU memory
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = torchvision.models.resnet18(pretrained=True)  # Example: ResNet-18

# Freeze pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# Replace last layer
num_ftrs = model.fc.in_features  # Number of features from pre-trained model
model.fc = torch.nn.Linear(num_ftrs, 6)  # 6 output classes for buildings, forest, ...
model.to(device)  # Move model to device (CPU or GPU)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)  # Adjust learning rate (lr)

num_epochs = 40

def train_model(model, criterion, optimizer, num_epochs, train_dataloader, device):
  """Trains the image classification model on the provided data.

  Args:
      model (torch.nn.Module): The image classification model.
      criterion (torch.nn.Module): The loss function.
      optimizer (torch.optim): The optimizer for updating model weights.
      num_epochs (int): The number of training epochs.
      train_dataloader (torch.utils.data.DataLoader): The training data loader.
      device (str): The device to use for training (CPU or GPU).

  Returns:
      None
  """

  print("Starting training...")

  # Track training loss history
  train_loss_history = []

  for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    # Set model to training mode
    model.train()

    total_loss = 0
    for images, labels in train_dataloader:
      # Move data to device (CPU or GPU)
      images, labels = images.to(device), labels.to(device)

      # Forward pass
      outputs = model(images)
      loss = criterion(outputs, labels)

      # Backward pass and parameter update
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_loss += loss.item()  # Track loss for each batch

    # Calculate average loss for the epoch
    average_loss = total_loss / len(train_dataloader)
    train_loss_history.append(average_loss)

    print(f"[Train] Epoch {epoch+1} - Average Loss: {average_loss:.4f}")

  print("Training complete!")

# Train the model
train_model(model, criterion, optimizer, num_epochs, train_dataloader, device)

def predict_image(model, image_path, transform, device):
  """Predicts the class of a single image.

  Args:
      model (torch.nn.Module): The trained image classification model.
      image_path (str): The path to the image file.
      transform (torchvision.transforms): The image transformation used during training.
      device (str): The device to use for prediction (CPU or GPU).

  Returns:
      tuple: A tuple containing the predicted class label (int) and class name (str).
  """

  # Load the image
  image = Image.open(image_path).convert('RGB')  # Ensure RGB format

  # Apply the same transformation used for training
  image = transform(image)

  # Add a batch dimension for the model
  image = image.unsqueeze(0)  # Convert to a mini-batch of size 1

  # Move data to device
  image = image.to(device)

  # Set model to evaluation mode (optional)
  model.eval()

  # Get predictions
  with torch.no_grad():  # Disable gradient calculation for prediction
      outputs = model(image)

  # Get the predicted class index
  _, predicted = torch.max(outputs.data, 1)

  # Get class names from your dataset classes (replace with your actual class names)
  class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

  return predicted.item(), class_names[predicted.item()]

def save_model(model, save_path):
  """Saves the trained model to a file.

  Args:
      model (torch.nn.Module): The trained model to save.
      save_path (str): The path to save the model file.
  """

  # Save model state dictionary
  torch.save(model.state_dict(), save_path)
  print(f"Model saved to: {save_path}")

# Example usage
save_path = "saved_model.pt"  # Replace with your desired save path
save_model(model, save_path)

# Replace with your actual paths and class names
image_path = "image_data/seg_pred/3.jpg"
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']  # Adjust based on your dataset

# Use the same transforms used for training
transform = test_transforms  # Assuming you defined test_transforms earlier

# Make prediction
predicted_label, predicted_class = predict_image(model, image_path, transform, device)

print(f"Predicted Class: {predicted_class} ({predicted_label})")
