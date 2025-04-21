import torch
import torch.nn as nn

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


class CRF(nn.Module):
    """
    Conditional Random Field (CRF) layer.

    Recebe emissões (geralmente de um BiLSTM) e calcula a perda CRF ou
    decodifica a melhor sequência de tags usando Viterbi.

    Args:
        num_tags: Número total de tags (incluindo START e STOP).
        start_tag_ix: Índice da tag START.
        stop_tag_ix: Índice da tag STOP.
        batch_first: Se a primeira dimensão das entradas (emissions) é o batch.
                     Padrão: True.
    """
    def __init__(self, num_tags: int, start_tag_ix: int, stop_tag_ix: int, batch_first: bool = True):
        super().__init__()
        if num_tags <= 0:
            raise ValueError("Invalid number of tags: {}".format(num_tags))
        self.num_tags = num_tags
        self.start_tag_ix = start_tag_ix
        self.stop_tag_ix = stop_tag_ix
        self.batch_first = batch_first

        # Matriz de transições: transitions[i, j] é o score de transitar da tag j para a tag i
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

        # Forçar restrições: nunca transitar PARA start ou DE stop
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Inicializa transições e aplica restrições."""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        # Restrições:
        # - Nenhuma transição *para* START_TAG
        # - Nenhuma transição *de* STOP_TAG
        # Usamos valores muito baixos para simular -infinito
        with torch.no_grad():
            self.transitions.data[:, self.start_tag_ix] = -10000.0
            self.transitions.data[self.stop_tag_ix, :] = -10000.0

    def forward(self, emissions: torch.Tensor, tags: torch.LongTensor, mask: torch.ByteTensor = None) -> torch.Tensor:
        """
        Calcula a perda Negativa Log Likelihood (NLL) do CRF.

        Args:
            emissions (torch.Tensor): Scores de emissão do modelo anterior (e.g., BiLSTM).
                Shape: (batch_size, seq_len, num_tags) se batch_first=True,
                       (seq_len, batch_size, num_tags) caso contrário.
            tags (torch.LongTensor): Sequências de tags verdadeiras (ouro).
                Shape: (batch_size, seq_len) se batch_first=True,
                       (seq_len, batch_size) caso contrário.
            mask (torch.ByteTensor, optional): Máscara binária indicando tokens reais (1) vs padding (0).
                Shape: (batch_size, seq_len) se batch_first=True,
                       (seq_len, batch_size) caso contrário. Default: Assume sem padding.

        Returns:
            torch.Tensor: A perda NLL média sobre o batch (escalar).
                          É igual a: log Z - score(tags_ouro)
        """
        if mask is None:
            # Se máscara não for fornecida, assume que todas as posições são válidas
            mask = torch.ones_like(tags, dtype=torch.uint8, device=emissions.device)

        if not self.batch_first:
            # Converte para batch_first temporariamente para simplificar o código
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # Calcula o log da função de partição (log Z) usando o algoritmo forward
        log_partition_function = self._forward_alg(emissions, mask)

        # Calcula o score da sequência de tags "ouro"
        gold_score = self._score_sequence(emissions, tags, mask)

        # Perda NLL = log Z - score_ouro
        # Retorna a média da perda sobre o batch
        return torch.mean(log_partition_function - gold_score)

    def decode(self, emissions: torch.Tensor, mask: torch.ByteTensor = None) -> list[list[int]]:
        """
        Encontra a melhor sequência de tags usando o algoritmo de Viterbi.

        Args:
            emissions (torch.Tensor): Scores de emissão.
                Shape: (batch_size, seq_len, num_tags) se batch_first=True,
                       (seq_len, batch_size, num_tags) caso contrário.
            mask (torch.ByteTensor, optional): Máscara binária.
                Shape: (batch_size, seq_len) se batch_first=True,
                       (seq_len, batch_size) caso contrário. Default: Assume sem padding.

        Returns:
            list[list[int]]: Uma lista de listas, onde cada lista interna é a
                             sequência de tags predita para um item do batch.
                             Os scores não são retornados aqui, mas podem ser se necessário.
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8, device=emissions.device)

        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # Executa Viterbi para obter as melhores sequências
        _, best_paths = self._viterbi_decode(emissions, mask)
        return best_paths

    def _forward_alg(self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        """ Calcula log(Z), o log da função de partição (normalizador). """
        batch_size, seq_len, num_tags = emissions.shape
        device = emissions.device

        # Inicializa forward_var com -inf, exceto para START_TAG que é 0
        # Shape: (batch_size, num_tags)
        forward_var = torch.full((batch_size, num_tags), -10000.0, device=device)
        forward_var[:, self.start_tag_ix] = 0.0

        # Prepara transições para broadcasting: (1, num_tags, num_tags)
        transitions = self.transitions.unsqueeze(0)

        # Itera pela sequência
        for i in range(seq_len):
            # Emissão do passo atual: (batch_size, 1, num_tags)
            emit_score = emissions[:, i, :].unsqueeze(1)

            # Score do passo anterior + transição + emissão atual
            # forward_var: (batch_size, num_tags) -> (batch_size, num_tags, 1)
            # transitions: (1, num_tags, num_tags)
            # emit_score: (batch_size, 1, num_tags)
            # next_tag_var: (batch_size, num_tags, num_tags)
            next_tag_var = forward_var.unsqueeze(2) + transitions + emit_score

            # Calcula log_sum_exp sobre as tags anteriores para obter o score de chegar em cada tag atual
            # forward_var_t: (batch_size, num_tags)
            forward_var_t = log_sum_exp(next_tag_var, dim=1)

            # Aplica a máscara: se mask[:, i] é 0, mantém o valor antigo de forward_var,
            # senão, atualiza com forward_var_t.
            # mask[:, i].unsqueeze(1): (batch_size, 1)
            mask_i = mask[:, i].unsqueeze(1).type_as(forward_var) # Garante mesmo tipo e device
            forward_var = forward_var_t * mask_i + forward_var * (1 - mask_i)

        # Adiciona a transição final para STOP_TAG
        # forward_var: (batch_size, num_tags)
        # self.transitions[self.stop_tag_ix]: (num_tags,) -> (1, num_tags)
        terminal_var = forward_var + self.transitions[self.stop_tag_ix].unsqueeze(0)

        # Log da função de partição é o log_sum_exp final
        # alpha: (batch_size,)
        alpha = log_sum_exp(terminal_var, dim=1)
        return alpha

    def _score_sequence(self, emissions: torch.Tensor, tags: torch.LongTensor, mask: torch.ByteTensor) -> torch.Tensor:
        """ Calcula o score de uma sequência de tags (não normalizado). """
        batch_size, seq_len, _ = emissions.shape
        device = emissions.device

        # Score inicial: transição de START para a primeira tag real
        # Cria tensor de START tags: (batch_size,)
        start_tags = torch.full((batch_size,), self.start_tag_ix, dtype=torch.long, device=device)
        # Concatena START ao início das tags: (batch_size, seq_len + 1)
        tags_with_start = torch.cat([start_tags.unsqueeze(1), tags], dim=1)

        # Score total começa em zero: (batch_size,)
        score = torch.zeros(batch_size, device=device)

        # Itera pela sequência (incluindo START -> tag[0])
        for i in range(seq_len):
            # Tags atuais e próximas (índices para transições e emissões)
            # current_tags: (batch_size,) tag no passo i (ou START se i=0)
            # next_tags: (batch_size,) tag no passo i+1
            current_tags = tags_with_start[:, i]
            next_tags = tags_with_start[:, i+1]

            # Score de Transição: T[next_tag, current_tag]
            # Shape: (batch_size,)
            transition_scores = self.transitions[next_tags, current_tags]

            # Score de Emissão: E[i, next_tag] (score da tag no passo i)
            # emissions[:, i, :]: (batch_size, num_tags)
            # next_tags: (batch_size,) -> (batch_size, 1)
            # Usa gather para selecionar scores: (batch_size,)
            emission_scores = emissions[:, i, :].gather(dim=1, index=next_tags.unsqueeze(1)).squeeze(1)

            # Adiciona scores ao total, mas APENAS se a posição for válida (mask == 1)
            # mask[:, i]: (batch_size,)
            mask_i = mask[:, i].type_as(score) # Mesmo tipo e device
            score += (transition_scores + emission_scores) * mask_i

        # Adiciona score da transição final para STOP_TAG
        # Pega a última tag real de cada sequência baseado na máscara
        # seq_lengths: (batch_size,) - encontra o índice do último '1' na máscara
        seq_lengths = mask.sum(dim=1)
        # last_tags: (batch_size,) - pega a tag no índice (length - 1)
        last_tags = tags.gather(dim=1, index=(seq_lengths - 1).unsqueeze(1)).squeeze(1)

        # Score de transição da última tag real para STOP
        # Shape: (batch_size,)
        final_transition_score = self.transitions[self.stop_tag_ix, last_tags]

        score += final_transition_score
        return score

    def _viterbi_decode(self, emissions: torch.Tensor, mask: torch.ByteTensor) -> tuple[torch.Tensor, list[list[int]]]:
        """ Decodifica a melhor sequência usando Viterbi. """
        batch_size, seq_len, num_tags = emissions.shape
        device = emissions.device

        # Inicializa viterbi_var com -inf, exceto para START_TAG que é 0
        # Shape: (batch_size, num_tags)
        viterbi_var = torch.full((batch_size, num_tags), -10000.0, device=device)
        viterbi_var[:, self.start_tag_ix] = 0.0

        # Backpointers para reconstruir o caminho: lista de tensores (batch_size, num_tags)
        backpointers = []

        # Prepara transições para broadcasting: (1, num_tags, num_tags)
        transitions = self.transitions.unsqueeze(0)

        # Itera pela sequência
        for i in range(seq_len):
            # Emissão do passo atual: (batch_size, 1, num_tags)
            emit_score = emissions[:, i, :].unsqueeze(1)

            # Score do passo anterior + transição
            # viterbi_var: (batch_size, num_tags) -> (batch_size, num_tags, 1)
            # transitions: (1, num_tags, num_tags)
            # next_tag_var: (batch_size, num_tags, num_tags) - score de chegar na tag j vindo da tag i
            next_tag_var = viterbi_var.unsqueeze(2) + transitions

            # Encontra o melhor score e o índice da tag anterior para cada tag atual
            # best_prev_scores: (batch_size, num_tags) - max score para chegar em cada tag atual
            # best_prev_indices: (batch_size, num_tags) - índice da melhor tag anterior
            best_prev_scores, best_prev_indices = torch.max(next_tag_var, dim=1)

            # Adiciona emissão e armazena backpointer
            # viterbi_var_t: (batch_size, num_tags)
            viterbi_var_t = best_prev_scores + emit_score.squeeze(1)
            backpointers.append(best_prev_indices)

            # Aplica máscara: mantém valor antigo se mask=0
            mask_i = mask[:, i].unsqueeze(1).type_as(viterbi_var)
            viterbi_var = viterbi_var_t * mask_i + viterbi_var * (1 - mask_i) * (i > 0) # Evita zerar o start

        # Adiciona transição final para STOP_TAG
        terminal_var = viterbi_var + self.transitions[self.stop_tag_ix].unsqueeze(0)

        # Encontra a melhor tag final (a que leva para STOP com maior score)
        # best_final_scores: (batch_size,)
        # best_final_tags: (batch_size,) - índice da melhor última tag real
        best_final_scores, best_final_tags = torch.max(terminal_var, dim=1)

        # Backtracking para encontrar os melhores caminhos
        best_paths = []
        for b in range(batch_size):
            # Pega a melhor última tag e o comprimento real da sequência
            best_last_tag = best_final_tags[b].item()
            seq_length = int(mask[b].sum().item())

            # Rastreia de trás para frente usando backpointers
            current_best_tag = best_last_tag
            path = [current_best_tag]
            # Itera de seq_len-1 até 1 (exclusivo)
            for i in range(seq_length - 1, 0, -1):
                 # backpointers[i]: (batch_size, num_tags)
                 # Pega o índice da melhor tag no passo anterior que levou a current_best_tag
                 current_best_tag = backpointers[i][b, current_best_tag].item()
                 path.append(current_best_tag)

            # Inverte o caminho (ele foi construído de trás para frente)
            path.reverse()
            best_paths.append(path)

        return best_final_scores, best_paths
