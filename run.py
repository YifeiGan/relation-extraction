import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
import sys

#get the file path and read file
train_data_path = sys.argv[1]
test_data_path = sys.argv[2]
output_path = sys.argv[3]
df = pd.read_csv(train_data_path)
df_test = pd.read_csv(test_data_path)

#preprocess the data
df["CORE RELATIONS"] = df["CORE RELATIONS"].fillna('')
df["CORE RELATIONS"] = df["CORE RELATIONS"].apply(lambda i: i.split() if isinstance(i, str) else [])
x = df["UTTERANCES"]
y = df["CORE RELATIONS"]

#vectorize
mlb = MultiLabelBinarizer()
vectorizer = CountVectorizer(
    analyzer = 'char_wb',
    ngram_range = (1, 3),
    max_features=5000,
    min_df = 2
)
x_train = vectorizer.fit_transform(x).toarray()
y_train = mlb.fit_transform(y)

#splitting
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

#vectorize
x_train_vec = torch.FloatTensor(x_train)
x_test_vec = torch.FloatTensor(x_test)
y_train_vec = torch.FloatTensor(y_train)
y_test_vec = torch.FloatTensor(y_test)

#create dataset and dataloader easy for training process
train_dataset = TensorDataset(x_train_vec, y_train_vec)
test_dataset = TensorDataset(x_test_vec, y_test_vec)
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 16)

#build up the model
class RelationExtraction(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 19)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = RelationExtraction(x_train_vec.size(1))
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0027)

#train loop
num_epoch = 100
losses = []
accuracies = []

for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    losses.append(avg_loss)

    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch. no_grad():
        for inputs, labels in train_loader:
            outputs = model(inputs)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions += (predicted == labels).all(dim=1).sum().item()
            total_samples += labels.size(0)
    accuracy = correct_predictions / total_samples
    accuracies.append (accuracy * 100)
    print(f'Epoch {epoch+1}, Loss: {avg_loss}; Training Accuracy: {accuracy * 100:.2f}%')

#load test data and vectorize
x = df_test["UTTERANCES"]
x_test = vectorizer.transform(x).toarray()
x_test_vec = torch.FloatTensor(x_test)

#load model
model.eval
with torch.no_grad():
    outputs = model(x_test_vec)
    predicted = (torch.sigmoid(outputs) > 0.5).float()

core_relations = mlb.inverse_transform(predicted.cpu().numpy())
df_submission = pd.DataFrame({
    'ID' : df_test['ID'],
    'CORE RELATIONS' : ['NONE' if len(relation) == 0 else ' '.join(relation) for relation in core_relations]
})

#save the result to submission.csv
df_submission.to_csv('submission.csv', index=False)
print("submission.csv generated.")