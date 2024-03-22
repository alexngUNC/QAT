import torch, torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from main import quantize_model

def train(model: torch.nn.Module):
    #do some training
    return

def test(model_path: str):
    model = torchvision.models.resnet50()
    model = quantize_model(model)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # Modify the last layer for 10 classes
    model.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU!")
    else:
        device = torch.device("cpu")
        print("Using CPU!")
    model.to(device)

    # Example transforms for test data, similar to training data but without augmentation
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Example dataset loading for test data
    test_dataset = torchvision.datasets.CIFAR10(root='test/cifar', train=False, download=True, transform=test_transform)

    # Example data loader for test data
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # do evaluation
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    average_loss = test_loss / len(test_loader)

    print('Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(average_loss, accuracy))

def test_train(freeze=False, epochs=10, model_path=None, save_path='model.pth'):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU!")
    else:
        device = torch.device("cpu")
        print("Using CPU!")

    model = torchvision.models.resnet50(pretrained=True)
    model = quantize_model(model)

    # Freeze all layers except the final fully connected layer
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # Modify the last layer for 10 classes
    if model_path:
        model.load_state_dict(torch.load(model_path))

    # Move model to GPU
    model = model.to(device)

    # Example transforms, you should adjust them according to your dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Example dataset loading, you should replace this with your dataset loading
    train_dataset = torchvision.datasets.CIFAR10(root='train/cifar', train=True, download=False, transform=transform)

    # Example data loader, you should replace this with your data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    for epoch in range(epochs):
      running_loss = 0.0
      correct_predictions = 0
      total_samples = 0

      for i, (inputs, labels) in enumerate(train_loader):
          inputs = inputs.to(device)
          labels = labels.to(device)  # Move data to device

          optimizer.zero_grad()

          outputs = model(inputs)
          loss = criterion(outputs, labels)

          loss.backward()
          optimizer.step()

          running_loss += loss.item()

          # Compute accuracy
          _, predicted = torch.max(outputs, 1)
          correct_predictions += (predicted == labels).sum().item()
          total_samples += labels.size(0)

          if i % 100 == 99:
              print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
              running_loss = 0.0

      # Calculate accuracy after each epoch
      accuracy = correct_predictions / total_samples
      print('Epoch %d Accuracy: %.2f%%' % (epoch + 1, 100 * accuracy))

    print('Finished Training')
    torch.save(model.state_dict(), save_path)
#test_train(freeze=True, epochs=5, save_path='cifar_frozen_quant.pth')
# test_train(freeze=False, epochs=10, model_path='cifar_frozen_quant.pth', save_path='cifar_full_quant.pth')
test("cifar_full_quant.pth")