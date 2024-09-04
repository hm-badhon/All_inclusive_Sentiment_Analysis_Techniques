import torch
from torch.cuda.amp import GradScaler, autocast
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import movie_reviews
from tqdm import tqdm

# Download necessary NLTK data
nltk.download("movie_reviews")

# Load the dataset
documents = [
    (" ".join(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

df = pd.DataFrame(documents, columns=["review", "sentiment"])
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'pos' else 0)

X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_data(texts, labels, max_len=256):
    input_ids = []
    attention_masks = []
    
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels.values)
    
    return input_ids, attention_masks, labels

train_inputs, train_masks, train_labels = encode_data(X_train, y_train, max_len=256)
test_inputs, test_masks, test_labels = encode_data(X_test, y_test, max_len=256)

batch_size = 8
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
scaler = GradScaler()

epochs = 2
gradient_accumulation_steps = 4

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    model.train()

    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch_input_ids, batch_attention_mask, batch_labels = tuple(t.to(device) for t in batch)
        
        model.zero_grad()

        with autocast():
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_labels,
            )
            loss = outputs.loss
        
        # Scale the loss and call backward()
        scaler.scale(loss).backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Average training loss: {avg_train_loss}")

model.eval()
correct = 0
total = 0
predictions_list = []
labels_list = []

with torch.no_grad():
    for batch in test_dataloader:
        batch_input_ids, batch_attention_mask, batch_labels = tuple(t.to(device) for t in batch)
        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
        )
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct += (predictions == batch_labels).sum().item()
        total += batch_labels.size(0)
        predictions_list.extend(predictions.cpu().numpy())
        labels_list.extend(batch_labels.cpu().numpy())

accuracy = correct / total
print(f"Validation Accuracy: {accuracy}")

from sklearn.metrics import classification_report
print(f"Classification Report:\n{classification_report(labels_list, predictions_list, target_names=['neg', 'pos'])}")
