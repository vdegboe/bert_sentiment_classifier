import torch
import torch.nn as nn
from transformers import BertModel


class MyBertModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_classes=5):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout1 = nn.Dropout(0.3)
        self.hidden = nn.Linear(self.bert.config.hidden_size, 256)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.sortie = nn.Linear(256, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]

        x = self.dropout1(cls_output)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.sortie(x)
        return logits
