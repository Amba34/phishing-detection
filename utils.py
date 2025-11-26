# utils.py
import re
import numpy as np
import pandas as pd
from urllib.parse import urlparse

from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch


###############################
# 1. Loading & basic cleaning #
###############################

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect columns: text, label
    df["text"] = df["text"].astype(str).fillna("")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    # Map to binary: spam=1, not spam=0
    df["label_binary"] = df["label"].map({"spam": 1, "not spam": 0})
    return df


################################
# 2. Hand-crafted URL features #
################################

URL_REGEX = re.compile(
    r"(https?://[^\s]+)"
)

def extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return ""

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    texts = df["text"].values

    num_urls = []
    num_digits = []
    text_len = []
    has_ip = []

    for t in texts:
        urls = URL_REGEX.findall(t)
        num_urls.append(len(urls))
        num_digits.append(sum(c.isdigit() for c in t))
        text_len.append(len(t))

        # crude heuristic: URLs with raw IPs
        ip_like = any(re.search(r"\d+\.\d+\.\d+\.\d+", u) for u in urls)
        has_ip.append(int(ip_like))

    df_feat = df.copy()
    df_feat["num_urls"] = num_urls
    df_feat["num_digits"] = num_digits
    df_feat["text_len"] = text_len
    df_feat["has_ip_url"] = has_ip
    return df_feat


#####################
# 3. TF-IDF helper  #
#####################

def build_tfidf_vectorizer(max_features: int = 20000):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english"
    )


########################
# 4. BERT embeddings   #
########################

class BertEmbedder:
    """
    Simple wrapper around a BERT model to get sentence embeddings.
    Uses mean-pooling over last_hidden_state.
    """

    def __init__(self, model_name: str = "bert-base-uncased", device: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )

            enc = {k: v.to(self.device) for k, v in enc.items()}
            outputs = self.model(**enc)
            last_hidden_state = outputs.last_hidden_state  # (B, T, H)
            # mean pooling
            attention_mask = enc["attention_mask"].unsqueeze(-1)
            masked = last_hidden_state * attention_mask
            summed = masked.sum(dim=1)
            counts = attention_mask.sum(dim=1)
            embeddings = (summed / counts).cpu().numpy()
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)
