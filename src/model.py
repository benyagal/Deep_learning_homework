"""
PyTorch modell és adatkészlet definíciók.
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

import config

class LegalDataset(Dataset):
    """PyTorch adatkészlet a jogi szövegekhez."""
    def __init__(self, df, tokenizer, max_len, feature_cols, feature_stats=None):
        self.texts = df['paragraph_text'].values
        self.labels = df['label_int'].values
        self.features = df[feature_cols].values if feature_cols else None
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Normalizálás a tanítási adatok statisztikái alapján
        if self.features is not None and feature_stats:
            means = np.array([feature_stats[col][0] for col in feature_cols])
            stds = np.array([feature_stats[col][1] for col in feature_cols])
            # Vágás a kiugró értékek ellen
            self.features = np.clip((self.features - means) / stds, -3.0, 3.0)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        item = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        if self.features is not None:
            item['features'] = torch.tensor(self.features[idx], dtype=torch.float)
        else:
            # Üres tenzort adunk vissza, ha nincsenek extra jellemzők
            item['features'] = torch.empty(0, dtype=torch.float)
        return item

class CoralHead(nn.Module):
    """CORAL kimeneti réteg, amely egy kis MLP-t is tartalmaz a stabilitásért."""
    def __init__(self, hidden_size, num_classes, extra_feat_dim=0, dropout=0.1):
        super().__init__()
        self.use_extra = extra_feat_dim > 0
        in_dim = hidden_size + extra_feat_dim
        
        # Egy egyszerűbb, de hatékonyabb fej architektúra
        self.head_mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(256, num_classes - 1)

    def forward(self, cls_hidden, extra_feats=None):
        if self.use_extra and extra_feats is not None:
            # Ellenőrizzük, hogy az extra_feats nem üres-e
            if extra_feats.shape[1] > 0:
                x = torch.cat([cls_hidden, extra_feats], dim=1)
            else:
                x = cls_hidden
        else:
            x = cls_hidden
        
        x = self.head_mlp(x)
        logits = self.linear(self.dropout(x))
        probs = torch.sigmoid(logits)
        return probs, logits

class CoralModel(nn.Module):
    """A teljes transzformer modell a CORAL fejjel."""
    def __init__(self, model_name, num_classes, extra_feat_dim=0):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name)
        hidden_size = self.base.config.hidden_size
        self.head = CoralHead(hidden_size, num_classes, extra_feat_dim=extra_feat_dim)

    def forward(self, input_ids, attention_mask, extra_feats=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token embedding
        probs, logits = self.head(cls_output, extra_feats)
        return probs, logits

def create_data_loader(df, tokenizer, max_len, batch_size, feature_cols, feature_stats=None, sampler=None):
    """Létrehoz egy DataLoader-t a megadott adatokból."""
    ds = LegalDataset(
        df=df,
        tokenizer=tokenizer,
        max_len=max_len,
        feature_cols=feature_cols,
        feature_stats=feature_stats
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=0, sampler=sampler)
