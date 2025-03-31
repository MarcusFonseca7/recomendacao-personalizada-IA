import sentencepiece as spm
import re  # Biblioteca para expressões regulares
import torch
import transformers
import faiss
import numpy as np
from efficientnet_pytorch import EfficientNet
from transformers import BertTokenizer, BertModel, BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# 1. Coleta de Dados (Simulação)
logs_usuario = pd.read_csv("dadosNike.csv", delimiter=";") # Carrega os logs do usuário do arquivo CSV

# Aplicando regex para limpeza dos logs (removendo caracteres especiais e múltiplos espaços)
def limpar_texto(texto):
    texto = re.sub(r"[^a-zA-Z0-9 ]", "", texto)  # Remove caracteres especiais
    texto = re.sub(r"\s+", " ", texto).strip()  # Remove espaços extras
    return texto

logs_usuario = logs_usuario.applymap(lambda x: limpar_texto(str(x)))  # Aplica a função de limpeza a todas as células

# Criando dataset para treinar SentencePiece
with open("logs_dataset.txt", "w", encoding="utf-8") as f:
    for _, row in logs_usuario.iterrows():
        f.write(" ".join(row.values.astype(str)) + "\n")  # Salva os logs processados em um arquivo texto
      
# Treinando SentencePiece
spm.SentencePieceTrainer.Train("--input=logs_dataset.txt --model_prefix=bpe --vocab_size=600 --model_type=bpe")

# 2. Tokenização com SentencePiece-BPE
sp = spm.SentencePieceProcessor()  # Inicializa o processador SentencePiece
sp.Load("bpe.model")  # Carrega o modelo treinado

# Tokeniza os logs convertendo-os em subpalavras
logs_tokenizados = [sp.encode_as_pieces(" ".join(row.values.astype(str))) for _, row in logs_usuario.iterrows()]

# Exibe os logs tokenizados para verificação
print("Logs Tokenizados:", logs_tokenizados)


#

# 3. Preparação dos Dados para o BERT4Rec, modelo de machine learning supervisionado
class LogDataset(Dataset):
    def __init__(self, tokenized_logs, sp_model, mask_prob=0.15, max_len=200):
        self.tokenized_logs = [sp_model.encode_as_ids(" ".join(log)) for log in tokenized_logs]  # Converte para IDs
        self.mask_prob = mask_prob  # Probabilidade de mascarar tokens
        self.sp_model = sp_model
        self.mask_token_id = sp_model.piece_to_id("<mask>")  # ID do token máscara
        self.pad_token_id = sp_model.pad_id() if sp_model.pad_id() > 0 else 0  # Token de padding
        self.max_len = max_len  # Tamanho fixo das sequências

    def __len__(self):
        return len(self.tokenized_logs)

    def __getitem__(self, idx):
        input_ids = self.tokenized_logs[idx]
        labels = np.array(input_ids, dtype=np.int64)
        masked_input = np.array(input_ids, dtype=np.int64)

        # Aplica máscara aleatória nos tokens
        for i in range(len(masked_input)):
            if np.random.rand() < self.mask_prob:
                masked_input[i] = self.mask_token_id  # Substitui pelo token máscara

        # Padding para ajustar ao tamanho fixo
        pad_length = self.max_len - len(masked_input)
        attention_mask = np.ones_like(masked_input)  # Máscara de atenção inicial
        if pad_length > 0:
            masked_input = np.pad(masked_input, (0, pad_length), constant_values=self.pad_token_id)
            labels = np.pad(labels, (0, pad_length), constant_values=-100)  # Ignora padding na loss
            attention_mask = np.pad(attention_mask, (0, pad_length), constant_values=0)  # Zera a máscara nos tokens de padding
        else:
            masked_input = masked_input[:self.max_len]
            labels = labels[:self.max_len]
            attention_mask = attention_mask[:self.max_len]

        return torch.tensor(masked_input), torch.tensor(labels), torch.tensor(attention_mask)

# Criando o dataset e DataLoader
dataset = LogDataset(logs_tokenizados, sp)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 4. Definição do modelo BERT4Rec
model = BertForMaskedLM.from_pretrained("bert-base-uncased")  # Carrega um modelo BERT pré-treinado
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)  # Otimizador AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define se usará GPU ou CPU
model.to(device)  # Move o modelo para o dispositivo

# 5. Treinamento do Modelo
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
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

print("Treinamento do BERT4Rec concluído!")

# 6. Testando uma previsão com o modelo
def prever_item(masked_sentence):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Carrega o tokenizer do BERT
    tokenized_input = tokenizer(masked_sentence.replace("<mask>", "[MASK]"), return_tensors="pt")
    input_ids = tokenized_input["input_ids"].to(device)
    attention_mask = tokenized_input["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0]

        # Obtém os 5 tokens mais prováveis para a máscara
        mask_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()
        probs = F.softmax(logits[mask_index], dim=-1)  # Converte logits para probabilidades
        top_5 = torch.topk(probs, 5)  # Pega os 5 tokens mais prováveis

        top_tokens = [tokenizer.decode([idx]) for idx in top_5.indices]  # Converte IDs em palavras
        top_scores = top_5.values.tolist()

    return list(zip(top_tokens, top_scores))  # Retorna as previsões com suas probabilidades

# Exemplo de uso
mascarado = "Tênis Nike <mask> preto"
resultado = prever_item(mascarado)
print("Entrada Mascarada:", mascarado)
print("Previsão do Modelo:", resultado)

