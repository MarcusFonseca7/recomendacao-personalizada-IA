import warnings
import os
from transformers import logging

# Silencia avisos Python
warnings.filterwarnings("ignore")

# Silencia logs do SentencePiece
os.environ['FLAGS_sentencepiece_warning_level'] = 'FATAL'

# Silencia logs da transformers
logging.set_verbosity_error()

# Bibliotecas principais
import sentencepiece as spm
import re
import torch
import numpy as np
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# 1. Leitura e limpeza dos dados
logs_usuario = pd.read_csv("dadosNike.csv", delimiter=";")

def limpar_texto(texto):
    texto = re.sub(r"[^a-zA-Z0-9 ]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

logs_usuario = logs_usuario.applymap(lambda x: limpar_texto(str(x)))

# Geração de frases descritivas
frases_treinamento = []
for _, row in logs_usuario.iterrows():
    modelo = row.get("produto", "")
    categoria = row.get("categoria", "")
    cor = row.get("cor", "")
    frase = f"Tênis Nike {modelo} {categoria} {cor}"
    frases_treinamento.append(frase)

# Salva frases e frases mascaradas
with open("logs_dataset.txt", "w", encoding="utf-8") as f:
    for frase in frases_treinamento:
        f.write(frase + "\n")
        frase_mask = re.sub(r"Nike \w+", "Nike <mask>", frase)
        f.write(frase_mask + "\n")

# 2. Treinamento do SentencePiece
spm.SentencePieceTrainer.Train("--input=logs_dataset.txt --model_prefix=bpe --vocab_size=300 --model_type=bpe")
sp = spm.SentencePieceProcessor()
sp.Load("bpe.model")

# Tokenização das frases
logs_tokenizados = [sp.encode_as_pieces(frase) for frase in frases_treinamento]

# O BERT4Rec é um modelo de machine learning que aprende históricos de interações com o usuário, utilizando o MLM, que é um sistema de máscara,
# que o sistema mascara algum item e a inteligencia é responsável por prever qual seria esse item. Assim, ele prevê futuras interações que o cliente
# fará e pode ajudar nas recomendações personalizadas.

# 3. Dataset BERT4Rec
class LogDataset(Dataset):
    def __init__(self, tokenized_logs, sp_model, mask_prob=0.15, max_len=200):
        self.tokenized_logs = [sp_model.encode_as_ids(" ".join(log)) for log in tokenized_logs]
        self.mask_prob = mask_prob
        self.sp_model = sp_model
        self.mask_token_id = sp_model.piece_to_id("<mask>")
        self.pad_token_id = sp_model.pad_id() if sp_model.pad_id() > 0 else 0
        self.max_len = max_len

    def __len__(self):
        return len(self.tokenized_logs)

    def __getitem__(self, idx):
        input_ids = self.tokenized_logs[idx]
        labels = np.array(input_ids, dtype=np.int64)
        masked_input = np.array(input_ids, dtype=np.int64)

        for i in range(len(masked_input)):
            if np.random.rand() < self.mask_prob:
                masked_input[i] = self.mask_token_id

        pad_length = self.max_len - len(masked_input)
        attention_mask = np.ones_like(masked_input)
        if pad_length > 0:
            masked_input = np.pad(masked_input, (0, pad_length), constant_values=self.pad_token_id)
            labels = np.pad(labels, (0, pad_length), constant_values=-100)
            attention_mask = np.pad(attention_mask, (0, pad_length), constant_values=0)
        else:
            masked_input = masked_input[:self.max_len]
            labels = labels[:self.max_len]
            attention_mask = attention_mask[:self.max_len]

        return torch.tensor(masked_input), torch.tensor(labels), torch.tensor(attention_mask)

dataset = LogDataset(logs_tokenizados, sp)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 4. Modelo e treinamento
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 5. Treinamento
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids, labels, attention_mask = batch
        input_ids, labels, attention_mask = input_ids.to(device), labels.to(device), attention_mask.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

# 6. Previsão personalizada
def prever_item(masked_sentence, modelos_validos):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    masked_sentence = masked_sentence.replace("<mask>", "[MASK]")

    tokenized_input = tokenizer(masked_sentence, return_tensors="pt")
    input_ids = tokenized_input["input_ids"].to(device)
    attention_mask = tokenized_input["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0]
        mask_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()
        probs = F.softmax(logits[mask_index], dim=-1)

        # Mapeia modelos para IDs de token (mesmo que sejam vários)
        modelo_scores = {}
        for modelo in modelos_validos:
            tokens = tokenizer.tokenize(modelo)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

            # Se algum token não foi reconhecido, ignora
            if all(id != tokenizer.unk_token_id for id in token_ids):
                # Pega a média de probabilidade dos tokens
                media = np.mean([probs[id].item() for id in token_ids])
                modelo_scores[modelo] = media

        top_modelos = sorted(modelo_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    return top_modelos

# 7. Executando a previsão
modelos_validos = sorted(logs_usuario["produto"].unique().tolist())
entrada = "Tênis Nike <mask> preto"
resultado = prever_item(entrada, modelos_validos)

print("Entrada Mascarada:", entrada)
print("Recomendações:")
for modelo, score in resultado:
    print(f"- {modelo}")
