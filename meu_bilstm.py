import torch
import torch.nn as nn
import torch.nn.functional as F
# --- Constantes (podem ser definidas globalmente ou passadas) ---
START_TAG = "<START>"
STOP_TAG = "<STOP>"

# --- Funções Auxiliares (podem estar em um arquivo utils) ---
def log_sum_exp(tensor: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """ Calcula log(sum(exp(tensor))) de forma numericamente estável. """
    max_val, _ = torch.max(tensor, dim=dim, keepdim=True)
    # Subtrai max_val antes do exp, soma, tira o log e adiciona max_val de volta.
    # Se max_val for -inf (tensor é todo -inf), o resultado do log(sum(exp)) será -inf.
    # Usar where para evitar log(0) se sum(exp(...)) for zero.
    sum_exp = torch.sum(torch.exp(tensor - max_val), dim=dim, keepdim=True)
    lse = max_val + torch.log(sum_exp.clamp(min=1e-10)) # Clamp para evitar log(0)
    if not keepdim:
        lse = lse.squeeze(dim)
    return lse
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTMTagger, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=False)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.loss_function = nn.NLLLoss()

    def forward(self, sentence):
        # Para predição: retorna as tags preditas (índices)
        tag_scores = self._get_tag_scores(sentence)
        _, predicted_tags = torch.max(tag_scores, dim=1)
        return predicted_tags.tolist()

    def compute_loss(self, sentence, tags):
        # Para treinamento: retorna a loss
        tag_scores = self._get_tag_scores(sentence)
        return self.loss_function(tag_scores, tags)

    def _get_tag_scores(self, sentence):
        # sentence: Tensor (seq_len,)
        embeds = self.word_embeds(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores  # (seq_len, tagset_size)
