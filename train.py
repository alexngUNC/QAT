import random
import torch, torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from main import quantize_model

def train(model: torch.nn.Module):
    #do some training
    return

def test(data_path: str, model_path: str, adjust_output = None):
    model = torchvision.models.resnet50(torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model = quantize_model(model)
    if adjust_output:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, adjust_output)
    #model.load_state_dict(torch.load(model_path))
    # model.apply(torch.ao.quantization.disable_observer)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU!")
    else:
        device = torch.device("cpu")
        print("Using CPU!")
    print('Moving model to device...', end="")
    model.to(device)
    print('Done!')

    # Example transforms for test data, similar to training data but without augmentation
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = torchvision.datasets.ImageNet(root=data_path, split='val', transform=test_transform)
    print('Loading data...', end='')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=20)
    print('Done!')

    # do evaluation
    model.eval()  # Set the model to evaluation mode
    print('Evaluating...', end='')
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        correct_val_predictions = 0
        total_val_samples = 0
        for val_inputs, val_labels in test_loader:
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_outputs = model(val_inputs)
            _, val_predicted = torch.max(val_outputs, 1)
            correct_val_predictions += (val_predicted == val_labels).sum().item()
            total_val_samples += val_labels.size(0)
        val_accuracy = correct_val_predictions / total_val_samples
        print('Validation Accuracy: ', 100 * val_accuracy)
    print('Done!')

def test_train(data_path, freeze=False, epochs=10, model_path=None, save_path='model.pth', adjust_output=None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU!")
    else:
        device = torch.device("cpu")
        print("Using CPU!")

    model = torchvision.models.resnet50(torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    #model = torchvision.models.resnet50()
    #model = quantize_model(model)

    # Freeze all layers except the final fully connected layer
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
       
    if adjust_output:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, adjust_output)  # Modify the last layer for 10 classes
    if model_path:
        model.load_state_dict(torch.load(model_path))

    # Move model to GPU
    print('Moving model to GPU...')
    model = model.to(device)

    # Example transforms, you should adjust them according to your dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Example dataset loading, you should replace this with your dataset loading
    train_dataset = torchvision.datasets.ImageNet(root=data_path, split='train', transform=transform)

    # Example data loader, you should replace this with your data loader
    print('Loading data...')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10)
    val_dataset = torchvision.datasets.ImageNet(root=data_path, split='val', transform=transform)
    print('Loading data...', end='')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=10)
    print('Done!')

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    # optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    # Training loop
    print('Training...')
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
        print('Epoch %d Training Accuracy: %.2f%%' % (epoch + 1, 100 * accuracy))

        # Validation
        print('Evaluating...', end='')
        model.apply(torch.ao.quantization.disable_observer)
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            correct_val_predictions = 0
            total_val_samples = 0
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_inputs)
                _, val_predicted = torch.max(val_outputs, 1)
                correct_val_predictions += (val_predicted == val_labels).sum().item()
                total_val_samples += val_labels.size(0)
            val_accuracy = correct_val_predictions / total_val_samples
            print('Epoch %d Validation Accuracy: %.2f%%' % (epoch + 1, 100 * val_accuracy))
        print('Done!')
        model.apply(torch.ao.quantization.enable_observer)
        model.train()
    print('Finished Training')
    torch.save(model.state_dict(), save_path)
#test_train(data_path='../imagenet', freeze=True, epochs=2, save_path='imagenet_frozen_quant.pth')
#test_train(data_path='../imagenet', freeze=False, epochs=15, model_path='imagenet_full_quant.pth', save_path='imagenet_full_quant2.pth')
#test('../imagenet-val', 'imagenet_full_quant.pth')
    
#test_train(data_path='../imagenet', freeze=True, epochs=2, save_path='completely_frozen_resnet.pth')
#test_train(data_path='../imagenet', freeze=False, epochs=10, save_path='resnet_non_quantized_10.pth')
test('../imagenet-val', model_path='')