# Sistema de Recomendações Personalizadas para a Nike

## Visão Geral
Este projeto implementa um sistema de recomendação personalizada para a Nike, focado na recomendação de tênis com base no comportamento do usuário e nas características visuais dos produtos. A abordagem combina aprendizado profundo com técnicas avançadas de recuperação de informações para fornecer sugestões relevantes aos clientes.

## Arquitetura do Sistema
O sistema é composto pelos seguintes componentes principais:

1. **Coleta de Logs:** Captura interações dos usuários na seção de tênis do site da Nike.
2. **Tokenização:** Utiliza SentencePiece-BPE para dividir os logs em tokens, permitindo uma análise eficiente dos padrões de navegação.
3. **Modelo BERT4Rec:**
   - Treinado com Masked Language Model para prever interações futuras.
   - Utiliza Transformers para codificar sequências de interações dos usuários.
   - Prediz produtos de interesse com base em padrões comportamentais.

   ### Próximos passos:
4. **Modelo EfficientNet-B4:**
   - Extrai embeddings visuais das imagens dos produtos.
   - Permite recomendações baseadas na aparência dos itens visitados.
5. **Faiss para Busca Rápida:**
   - Indexa embeddings visuais dos produtos.
   - Realiza buscas eficientes por similaridade para recomendações personalizadas.

**Linguagem utilizada:**
- Python


##Instruções de uso:

1. Copie e cole no terminal da sua IDE: "git clone https://github.com/MarcusFonseca7/recomendacao-personalizada-IA.git"
2. Acesse o diretório, cole: "cd recomendacao-personalizada-IA"
3. Entre no diretório: "code ."
4. Crie o ambiente virtual: "python -m venv venv"
5. Ative ele: "venv\Scripts\activate"
6. Se estiver ativado aparecerá (venv) ao começo do caminho da pasta
7. Instale as bibliotecas: "pip install -r requirements.txt" (pode demorar algum tempo...)
8. Agora só rodar o código e esperar o resultado aparecer no terminal!: "python anuncio.py"
