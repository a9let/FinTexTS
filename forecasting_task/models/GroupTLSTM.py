"""Baytas memory decomposition, KDD 2017 section 3.1.2, on raw MiniLM groups."""

import math
import torch
from torch import nn
from forecasting_task.models.GroupText import TEXT_GROUPS


class TimeAwareCell(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.decomposition = nn.Linear(hidden_size, hidden_size)
        self.cell = nn.LSTMCell(384, hidden_size)

    def adjust(self, memory, delta):
        short = torch.tanh(self.decomposition(memory))
        decay = 1 / torch.log(delta[:, None] + math.e)
        return memory - short + decay * short

    def forward(self, value, hidden, memory, delta, seen):
        adjusted = torch.where(seen[:, None], self.adjust(memory, delta), memory)
        return self.cell(value, (hidden, adjusted))


class GroupTLSTM(nn.Module):
    def __init__(self, hidden_size, window, pred_len):
        super().__init__()
        self.hidden_size, self.window = hidden_size, window
        self.cells = nn.ModuleDict({g: TimeAwareCell(hidden_size) for g in TEXT_GROUPS})
        self.heads = nn.ModuleDict({g: nn.Linear(hidden_size, pred_len) for g in TEXT_GROUPS})

    def forward(self, text_window, calendar_days):
        if text_window.ndim != 4 or text_window.shape[1] < self.window or text_window.shape[2:] != (5, 384):
            raise ValueError('T-LSTM requires separate group vectors within the input window')
        if calendar_days.shape != text_window.shape[:2] or not torch.isfinite(calendar_days).all():
            raise ValueError('T-LSTM requires aligned finite calendar-day ordinals')
        if (calendar_days[:, 1:] <= calendar_days[:, :-1]).any() or (calendar_days != calendar_days.round()).any():
            raise ValueError('Calendar dates must be strictly increasing integer days')
        values = text_window[:, -self.window:].float()
        if not torch.isfinite(values).all():
            raise ValueError('T-LSTM embeddings must be finite')
        days = calendar_days[:, -self.window:]
        present = values.abs().sum(-1) >= 1e-6
        outputs, deltas, states, memories, decays = [], [], [], [], []
        for index, group in enumerate(TEXT_GROUPS):
            hidden = values.new_zeros((len(values), self.hidden_size))
            memory = torch.zeros_like(hidden)
            previous = days.new_zeros(len(values))
            seen = torch.zeros(len(values), dtype=torch.bool, device=values.device)
            dg, hg, cg, ag = [], [], [], []
            for row in range(self.window):
                active = present[:, row, index].nonzero(as_tuple=False).flatten()
                delta_all = days.new_zeros(len(values))
                decay_all = days.new_zeros(len(values))
                if active.numel():
                    delta = torch.where(seen[active], days[active, row] - previous[active], 0.)
                    h, c = self.cells[group](values[active, row, index], hidden[active], memory[active], delta, seen[active])
                    hidden, memory = hidden.index_copy(0, active, h), memory.index_copy(0, active, c)
                    delta_all = delta_all.index_copy(0, active, delta)
                    decay_all = decay_all.index_copy(0, active, torch.where(seen[active], 1 / torch.log(math.e + delta), 1.))
                    previous = previous.index_copy(0, active, days[active, row])
                    seen = seen.index_fill(0, active, True)
                dg.append(delta_all)
                ag.append(decay_all)
                hg.append(hidden)
                cg.append(memory)
            outputs.append(self.heads[group](hidden) * seen[:, None])
            deltas.append(torch.stack(dg, 1))
            decays.append(torch.stack(ag, 1))
            states.append(torch.stack(hg, 1))
            memories.append(torch.stack(cg, 1))
        return torch.stack(outputs, 1), present.any(1), {
            'tlstm_present': present, 'tlstm_delta': torch.stack(deltas, 2), 'tlstm_decay': torch.stack(decays, 2),
            'tlstm_states': torch.stack(states, 2), 'tlstm_memories': torch.stack(memories, 2)}
