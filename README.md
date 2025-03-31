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
4. **Modelo EfficientNet-B4:**
   - Extrai embeddings visuais das imagens dos produtos.
   - Permite recomendações baseadas na aparência dos itens visitados.
5. **Faiss para Busca Rápida:**
   - Indexa embeddings visuais dos produtos.
   - Realiza buscas eficientes por similaridade para recomendações personalizadas.

**Linguagem utilizada:**
- Python
