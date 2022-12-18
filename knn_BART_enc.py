from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from preprocessing import read_data_split, PropagandaDataset
import json
from transformers import BartTokenizer, BartForSequenceClassification, BartConfig
from torch.utils.data import DataLoader
import torch
from torch import Tensor

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_representations(dataloader, model):
    X = []
    y = []
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['labels'].to(device)
            # Get encoder sentence representation
            encoder_out = model(input_ids=input_ids, attention_mask=attention_mask)
            encoder_last_hidden_state = encoder_out.encoder_last_hidden_state
            eos_mask = input_ids.eq(config.eos_token_id)
            sentnece_representation = encoder_last_hidden_state[eos_mask, :].view(encoder_last_hidden_state.size(0), -1, encoder_last_hidden_state.size(-1))[:, -1, :]
            X.append(sentnece_representation.squeeze().detach().cpu().numpy())
            y.append(label.item())
            torch.cuda.empty_cache()

    return X, y



train_sen, train_labels, train_highlights = read_data_split("./data/protechn_corpus_eval/train")
# print(train_sen, train_labels)
dev_sen, dev_labels, dev_highlights =  read_data_split("./data/protechn_corpus_eval/dev")
test_sen, test_labels, test_highlights =  read_data_split("./data/protechn_corpus_eval/test")

print(len(train_sen))
# train_sen, train_labels = train_sen[:5], train_labels[:5]
# dev_sen, dev_labels = dev_sen[:5], dev_labels[:5]
# test_sen, test_labels = test_sen[:5], test_labels[:5]


## TODO: obtain X_train, X_dev, X_test, y_train, y_dev, y_test form pretrained embeddings
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

train_encodings = tokenizer(train_sen, truncation=True, padding=True)
dev_encodings = tokenizer(dev_sen, truncation=True, padding=True)
test_encodings = tokenizer(test_sen, truncation=True, padding=True)

train_dataset = PropagandaDataset(train_encodings, train_labels)
dev_dataset = PropagandaDataset(dev_encodings, dev_labels)
test_dataset = PropagandaDataset(test_encodings, test_labels)

# model = BartForSequenceClassification.from_pretrained('./saved_models/bart_for_knn')
model = BartForSequenceClassification.from_pretrained('facebook/bart-large')
model.to(device)
model.eval()
config = model.config
# obtain sentence representations
## Pytorch loaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

X_train, y_train =  get_representations(train_loader, model)
X_dev, y_dev =  get_representations(dev_loader, model)
X_test, y_test =  get_representations(test_loader, model)

# print(y_train)
# k_range = [50,100,200,300,500]
k_range = [20]
score_dev = {}
score_test = {}
scores = {}
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_dev = knn.predict(X_dev)
    y_pred_test = knn.predict(X_test)
    # score_dev[k] = metrics.classification_report(y_dev, y_pred_dev, labels=[1,0])
    score_dev = metrics.classification_report(y_dev, y_pred_dev, labels=[1,0])
    score_test = metrics.classification_report(y_test, y_pred_test, labels=[1,0])
    # score_test[k] = metrics.classification_report(y_test, y_pred_test, labels=[1,0])

# scores['dev'] = score_dev
# scores['test'] = score_test
print("DEV Results")
print(score_dev)
print("TEST Results")
print(score_test)

# with open("knn_results.json", "w") as f:
#     json.dump(scores, f)