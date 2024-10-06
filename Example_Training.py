# Define the model (unchanged)
class MultiLabelClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelClassifier, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        return x  # Return logits directly

# Initialize the model
model = MultiLabelClassifier(num_classes).to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Use sigmoid BCE loss
optimizer = torch.optim.Adam(model.resnet.fc.parameters(), lr=0.001)

# Evaluation function
def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0) * num_classes
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# Training loop with validation
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            train_loss += loss.item()

            # Calculate accuracy
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_total += labels.size(0) * num_classes
            train_correct += (predicted == labels).sum().item()

        # Calculate average training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # Evaluate on validation set
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')




        num_epochs = 10
train(model, train_loader, val_loader, criterion, optimizer, num_epochs)


 # Save the trained model
torch.save(model.state_dict(), './multi_label_classifier.pth')  # Specify the save path
