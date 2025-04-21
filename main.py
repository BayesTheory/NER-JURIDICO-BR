import torch
import torch.optim as optim
import os
from preprocessamento import montar_caminhos, checar_arquivos, ler_conll

base_path_lener = './leNER-Br'  # Caminho relativo para a pasta de dados

paths = montar_caminhos(base_path_lener)
checar_arquivos(paths)

dados_treino = ler_conll(paths['file_train_conll'])
dados_dev = ler_conll(paths['file_dev_conll'])

# --- Funções Auxiliares ---
UNK_TOKEN = "<UNK>"
START_TAG = "<START>"
STOP_TAG = "<STOP>"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

def ler_conll(caminho_arquivo):
    sentences = []
    current_sentence_words = []
    current_sentence_tags = []
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    current_sentence_words.append(parts[0])
                    current_sentence_tags.append(parts[-1])
            elif current_sentence_words:
                if len(current_sentence_words) == len(current_sentence_tags):
                    sentences.append((current_sentence_words, current_sentence_tags))
                current_sentence_words = []
                current_sentence_tags = []
        if current_sentence_words and len(current_sentence_words) == len(current_sentence_tags):
            sentences.append((current_sentence_words, current_sentence_tags))
    print(f"Lidas {len(sentences)} sentenças de {caminho_arquivo}")
    return sentences

def prepare_sequence(seq, to_ix, unk_token=None):
    if unk_token:
        idxs = [to_ix.get(w, to_ix[unk_token]) for w in seq]
    else:
        idxs = [to_ix[t] for t in seq]
    return torch.tensor(idxs, dtype=torch.long).to(device)

def construir_vocabularios(training_data):
    word_to_ix = {UNK_TOKEN: 0}
    tag_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            word_lower = word.lower()
            if word_lower not in word_to_ix:
                word_to_ix[word_lower] = len(word_to_ix)
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    tag_to_ix[START_TAG] = len(tag_to_ix)
    tag_to_ix[STOP_TAG] = len(tag_to_ix)
    return word_to_ix, tag_to_ix

def treinar_modelo(model, optimizer, train_data, word_to_ix, tag_to_ix, num_epochs=5, unk_token=UNK_TOKEN):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (sentence, tags) in enumerate(train_data):
            optimizer.zero_grad()
            sentence_in = prepare_sequence([w.lower() for w in sentence], word_to_ix, unk_token)
            targets = prepare_sequence(tags, tag_to_ix)
            if hasattr(model, 'neg_log_likelihood'):
                loss = model.neg_log_likelihood(sentence_in, targets)
            elif hasattr(model, 'compute_loss'):
                loss = model.compute_loss(sentence_in, targets)
            else:
                raise NotImplementedError("O modelo precisa implementar 'neg_log_likelihood' ou 'compute_loss'.")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_data)}], Loss: {loss.item():.4f}')
        avg_loss = total_loss / len(train_data)
        print(f'--- Fim da Epoch [{epoch+1}/{num_epochs}], Loss Média: {avg_loss:.4f} ---')

def avaliar_modelo(model, dev_data, word_to_ix, tag_to_ix, ix_to_tag, unk_token=UNK_TOKEN):
    model.eval()
    with torch.no_grad():
        if dev_data:
            sample_sentence, sample_tags_true = dev_data[0]
            print(f"Sentença de Exemplo: {' '.join(sample_sentence)}")
            print(f"Tags Verdadeiras:   {' '.join(sample_tags_true)}")
            sentence_tensor = prepare_sequence([w.lower() for w in sample_sentence], word_to_ix, unk_token)
            if hasattr(model, 'forward'):
                output = model(sentence_tensor)
                if isinstance(output, tuple) and len(output) == 2:
                    score, predicted_tags_indices = output
                else:
                    predicted_tags_indices = output
                    score = torch.tensor(0.0)
                predicted_tags = [ix_to_tag[ix] for ix in predicted_tags_indices]
                print(f"Tags Preditas:      {predicted_tags}")
                print(f"Score da Predição: {score.item():.4f}")
            else:
                print("O modelo não implementa método 'forward' para predição.")
        else:
            print("Não foi possível carregar dados de desenvolvimento para exemplo de predição.")

# --- Escolha do modelo ---
# Opções: "crf", "bilstm", "bilstm_crf"
MODELO_ESCOLHIDO = "crf"  # Altere para "bilstm" ou "bilstm_crf" conforme desejar

if __name__ == "__main__":
    base_path_lener = './leNER-Br'
    paths = montar_caminhos(base_path_lener)
    file_train_conll = paths['file_train_conll']
    file_dev_conll = paths['file_dev_conll']

    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 5

    print("\n--- Carregando Dados LeNER-Br ---")
    training_data = ler_conll(file_train_conll)
    dev_data = ler_conll(file_dev_conll)
    if training_data is None or dev_data is None:
        raise ValueError("Erro ao carregar os dados de treino ou desenvolvimento.")

    print("\n--- Construindo Vocabulários ---")
    word_to_ix, tag_to_ix = construir_vocabularios(training_data)
    vocab_size = len(word_to_ix)
    tagset_size = len(tag_to_ix)
    ix_to_tag = {v: k for k, v in tag_to_ix.items()}
    print(f"Tamanho do Vocabulário: {vocab_size}")
    print(f"Tamanho do Conjunto de Tags: {tagset_size}")

    print("\n--- Instanciando Modelo e Otimizador ---")
    if MODELO_ESCOLHIDO == "crf":
        from meu_crf import CRFTagger
        model = CRFTagger(
            vocab_size=vocab_size,
            tag_to_ix=tag_to_ix
        ).to(device)
    elif MODELO_ESCOLHIDO == "bilstm":
        from meu_bilstm import BiLSTMTagger
        model = BiLSTMTagger(
            vocab_size=vocab_size,
            tag_to_ix=tag_to_ix,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM
        ).to(device)
    elif MODELO_ESCOLHIDO == "bilstm_crf":
        from meu_bilstm_crf import BiLSTM_CRF
        model = BiLSTM_CRF(
            vocab_size=vocab_size,
            tag_to_ix=tag_to_ix,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM
        ).to(device)
    else:
        raise ValueError("MODELO_ESCOLHIDO deve ser 'crf', 'bilstm' ou 'bilstm_crf'.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print("\n--- Iniciando Treinamento ---")
    treinar_modelo(model, optimizer, training_data, word_to_ix, tag_to_ix, num_epochs=NUM_EPOCHS)

    print("\n--- Exemplo de Predição ---")
    avaliar_modelo(model, dev_data, word_to_ix, tag_to_ix, ix_to_tag)
