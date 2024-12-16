
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import string
from torch.nn.utils.rnn import pad_sequence

# Preprocess Text Data
def preprocess_text(texts):
    stop_words = set(["a", "and", "the", "of", "is", "in", "to", "it", "on", "for", "with", "this", "that", "an", "by", "as"])
    processed_texts = []
    for text in texts:
        # Convert to lowercase and remove punctuation
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        # Tokenize and remove stopwords
        tokens = [word for word in text.split() if word not in stop_words]
        processed_texts.append(tokens)
    return processed_texts

# Define the IMDb model using MLP
class IMDbModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(IMDbModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # MLP layers
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Embedding lookup
        embedded = self.embedding(x)
        # Mean over words in each review (global average pooling)
        embedded = embedded.mean(dim=1)
        # MLP layers
        x = self.relu(self.fc1(embedded))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Main class to load data, train and evaluate model
class Main:
    def __init__(self, texts, labels, vocab, embedding_dim=100, hidden_dim=256, batch_size=64, epochs=10):
        self.texts = preprocess_text(texts)
        self.labels = labels
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.epochs = epochs

        # Hyperparameters
        self.vocab_size = len(vocab) + 1  # +1 for padding token

        # Train and test split
        self.train_texts, self.test_texts, self.train_labels, self.test_labels = train_test_split(
            self.texts, self.labels, test_size=0.2, random_state=42)

        # Create DataLoader
        self.train_loader, self.test_loader = self.create_data_loaders()

        # Initialize model, loss, and optimizer
        self.model = IMDbModel(self.vocab_size, embedding_dim, hidden_dim, output_dim=1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def text_to_tensor(self, texts):
        """Convert list of tokenized texts to tensor using the vocabulary, with padding."""
        tensor_texts = []
        for text in texts:
            tensor_texts.append(torch.tensor([self.vocab.get(word, self.vocab['<UNK>']) for word in text]))
        # Pad all sequences to the same length
        return pad_sequence(tensor_texts, batch_first=True, padding_value=0)

    def create_data_loaders(self):
        """Create DataLoader for training and testing"""
        train_texts_tensor = self.text_to_tensor(self.train_texts)
        train_labels_tensor = torch.tensor(self.train_labels).float()
        train_dataset = TensorDataset(train_texts_tensor, train_labels_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        test_texts_tensor = self.text_to_tensor(self.test_texts)
        test_labels_tensor = torch.tensor(self.test_labels).float()
        test_dataset = TensorDataset(test_texts_tensor, test_labels_tensor)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def train(self):
        """Train the model"""
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for texts, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(texts)
                loss = self.criterion(outputs.squeeze(), labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                predicted = torch.round(torch.sigmoid(outputs.squeeze()))
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = correct / total
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    def evaluate(self):
        """Evaluate the model on the test set"""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for texts, labels in self.test_loader:
                outputs = self.model(texts)
                predicted = torch.round(torch.sigmoid(outputs.squeeze()))
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")

# Example Usage
if __name__ == "__main__":
    # Example data (replace with actual IMDb reviews and labels)
    texts = ["This movie is great!", "I hated this movie.", "It was okay, not bad.", "Absolutely loved it!", "Worst movie ever."]
    labels = [1, 0, 1, 1, 0]  # 1 = positive, 0 = negative

    # Build vocabulary from training texts
    flat_texts = [word for text in preprocess_text(texts) for word in text]
    vocab = {word: idx+1 for idx, (word, _) in enumerate(Counter(flat_texts).items())}  # 1-based index
    vocab['<UNK>'] = 0  # Add unknown token

    # Create main instance and train
    main = Main(texts, labels, vocab)
    main.train()

    # Evaluate model
    main.evaluate()