import torch
from torch import nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizer, AutoModel, AutoTokenizer
import os

# Models that use mean pooling
POOL_MODELS = {"sentence-transformers/all-MiniLM-L6-v2", "TaylorAI/bge-micro-v2"}

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class LanguageModel(nn.Module):
    def __init__(self, model='distilbert-base-uncased'):
        super(LanguageModel, self).__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
        self.model_name = model
        # Remove the CLIP vision tower
        if "clip" in self.model_name:
            self.model.vision_model = None
        # Freeze the pre-trained parameters (very important)
        for param in self.model.parameters():
            param.requires_grad = False

        # Make sure to set evaluation mode (also important)
        self.model.eval()

    def forward(self, text_batch):
        inputs = self.tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad(): # Ensure no gradients are computed for this forward pass

            if "clip" in self.model_name:
                sentence_embedding = self.model.get_text_features(**inputs)
                return sentence_embedding

            outputs = self.model(**inputs)

        if any(model in self.model_name for model in POOL_MODELS):
            sentence_embeddings = mean_pooling(outputs, inputs['attention_mask'])
            # Normalize embeddings
            sentence_embedding = F.normalize(sentence_embeddings, p=2, dim=1)
        else:
            sentence_embedding = outputs.last_hidden_state[:, 0, :]
        return sentence_embedding
    

class LMHead(nn.Module):
    def __init__(self, embedding_dim=384, hidden_dim=256, num_classes=4):
        super(LMHead, self).__init__()
        
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        #self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        embd = self.fc1(x)
        embd = F.normalize(embd, p=2, dim=1)
        deg_pred = self.fc2(embd)
        return embd, deg_pred