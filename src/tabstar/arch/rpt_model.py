from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
from torch import nn, Tensor

from sap_rpt_oss.constants import ModelSize
from sap_rpt_oss.model.torch_model import RPT
from tabstar.arch.config import D_MODEL
from tabstar.arch.interaction import InteractionEncoder
from tabstar.arch.prediction import PredictionHead


class TabStarRPTModel(nn.Module):
    def __init__(
        self,
        rpt_model_size: ModelSize = ModelSize.base,
        d_model: int = D_MODEL,
        rpt_checkpoint_path: Optional[str] = None,
        regression_type: str = "l2",
        classification_type: str = "cross-entropy",
        checkpointing_segments: int = 0,
        tabular_layers: int = 6,
    ):
        super().__init__()
        self.rpt_model_size = rpt_model_size
        self.d_model = d_model
        self.regression_type = regression_type
        self.classification_type = classification_type
        self.checkpointing_segments = checkpointing_segments
        self.tabular_layers = tabular_layers
        self.rpt_checkpoint_path = rpt_checkpoint_path
        self.rpt = RPT(
            rpt_model_size,
            regression_type=regression_type,
            classification_type=classification_type,
            checkpointing_segments=checkpointing_segments,
        )
        if rpt_checkpoint_path:
            self.rpt.load_weights(Path(rpt_checkpoint_path), device=torch.device("cpu"))

        hidden_size = self.rpt.config.hidden_size
        self.proj = nn.Identity() if hidden_size == d_model else nn.Linear(hidden_size, d_model)
        self.tabular_encoder = InteractionEncoder(num_layers=tabular_layers, d_model=d_model)
        self.cls_head = PredictionHead()
        self.reg_head = PredictionHead()

    def forward(self, rpt_data: Dict[str, Tensor | np.ndarray], d_output: int) -> Tensor:
        data = self._prepare_rpt_data(rpt_data)
        input_embeds = self.rpt.embeddings(data, is_regression=True)
        input_embeds[:, -1] = 0
        attn_mask = self.rpt.build_context_attention_mask(data, input_embeds.device)
        attn_mask = attn_mask.type(input_embeds.dtype)
        for layer in self.rpt.in_context_encoder:
            input_embeds = layer(input_embeds, attn_mask)

        encoded = self.proj(input_embeds)
        encoded = self.tabular_encoder(encoded)
        if encoded.shape[1] < d_output + 1:
            raise ValueError(f"Sequence too short for d_output={d_output}: {encoded.shape=}")
        target_tokens = encoded[:, -(d_output + 1):-1]
        if d_output == 1:
            target_scores = self.reg_head(target_tokens)
        else:
            target_scores = self.cls_head(target_tokens)
        target_scores = target_scores.squeeze(dim=-1)
        return target_scores

    def _prepare_rpt_data(self, rpt_data: Dict[str, Tensor | np.ndarray]) -> Dict[str, Tensor]:
        device = next(self.parameters()).device
        prepared: Dict[str, Tensor] = {}
        for key, value in rpt_data.items():
            if isinstance(value, Tensor):
                tensor = value
            else:
                tensor = torch.as_tensor(value)
            prepared[key] = tensor.to(device)
        return prepared

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        config = {
            "rpt_model_size": self.rpt_model_size.name,
            "d_model": self.d_model,
            "regression_type": self.regression_type,
            "classification_type": self.classification_type,
            "checkpointing_segments": self.checkpointing_segments,
            "tabular_layers": self.tabular_layers,
            "rpt_checkpoint_path": self.rpt_checkpoint_path,
        }
        with open(os.path.join(save_directory, "rpt_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=True, indent=2)
