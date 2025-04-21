import torch
import torch.nn as nn
from torchcrf import CRF
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
class CRFTagger(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim=None, hidden_dim=None):
        super(CRFTagger, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.num_tags = len(tag_to_ix)
        self.crf = CRF(self.num_tags, batch_first=False)
        # Nenhuma camada de embedding ou LSTM

    def forward(self, sentence):
        # sentence: Tensor (seq_len,)
        # Como não há features, usamos um vetor de zeros como emissões
        seq_len = sentence.size(0)
        emissions = torch.zeros(seq_len, self.num_tags, device=sentence.device)
        # Decodifica a melhor sequência de tags
        tag_seq = self.crf.decode(emissions)
        # torchcrf.decode retorna uma lista de listas (batch), aqui batch=1
        return 0.0, tag_seq[0]

    def compute_loss(self, sentence, tags):
        # sentence: Tensor (seq_len,)
        # tags: Tensor (seq_len,)
        seq_len = sentence.size(0)
        emissions = torch.zeros(seq_len, self.num_tags, device=sentence.device)
        # torchcrf espera shape (seq_len, batch, num_tags), mas usamos batch_first=False
        # Adiciona dimensão de batch (batch=1)
        emissions = emissions.unsqueeze(1)  # (seq_len, 1, num_tags)
        tags = tags.unsqueeze(1)            # (seq_len, 1)
        mask = torch.ones(seq_len, 1, dtype=torch.bool, device=sentence.device)
        # A loss do CRF é negativa da log-likelihood
        nll = -self.crf(emissions, tags, mask=mask, reduction='mean')
        return nll
