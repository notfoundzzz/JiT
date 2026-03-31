import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2VLModel


class ProjectorAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_mult=2):
        super().__init__()
        hidden_dim = out_dim * hidden_mult
        self.in_proj = nn.Linear(in_dim, out_dim)
        self.in_norm = nn.LayerNorm(out_dim)
        self.adapter = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_norm = nn.LayerNorm(out_dim)

    @property
    def output_dtype(self):
        return self.in_proj.weight.dtype

    def forward(self, x):
        x = self.in_proj(x)
        x = self.in_norm(x)
        x = x + self.adapter(x)
        return self.out_norm(x)


class QwenVLConditionEncoder(nn.Module):
    def __init__(self, model_path, hidden_size, num_tokens, freeze=True):
        super().__init__()
        self.num_tokens = num_tokens
        self.processor = AutoProcessor.from_pretrained(model_path)

        qwen_vl = Qwen2VLModel.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=None,
        )
        self.vision_model = qwen_vl.visual
        vision_dim = self.vision_model.config.hidden_size

        self.projector = ProjectorAdapter(vision_dim, hidden_size)
        self.global_proj = nn.Linear(hidden_size, hidden_size)
        self.token_norm = nn.LayerNorm(hidden_size)
        self.global_norm = nn.LayerNorm(hidden_size)

        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False

    def _to_pil_images(self, cond_img):
        images = ((cond_img + 1.0) / 2.0).clamp(0.0, 1.0)
        images = images.mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(image.astype(np.uint8)) for image in images]

    def _projector_dtype(self):
        return next(self.projector.parameters()).dtype

    def forward(self, cond_img):
        pil_images = self._to_pil_images(cond_img)
        processed = self.processor.image_processor(images=pil_images, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(
            device=cond_img.device,
            dtype=self.vision_model.get_dtype(),
        )
        image_grid_thw = processed["image_grid_thw"].to(cond_img.device)

        vision_outputs = self.vision_model(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.vision_model.spatial_merge_size**2).tolist()
        image_tokens = torch.split(vision_outputs.pooler_output, split_sizes, dim=0)

        pooled_tokens = []
        projector_dtype = self._projector_dtype()
        for tokens in image_tokens:
            tokens = tokens.to(dtype=projector_dtype)
            tokens = self.projector(tokens)
            tokens = F.adaptive_avg_pool1d(tokens.transpose(0, 1).unsqueeze(0), self.num_tokens)
            tokens = tokens.squeeze(0).transpose(0, 1)
            tokens = self.token_norm(tokens)
            pooled_tokens.append(tokens)

        cond_tokens = torch.stack(pooled_tokens, dim=0)
        cond_global = self.global_proj(cond_tokens.mean(dim=1))
        cond_global = self.global_norm(cond_global)
        return cond_global, cond_tokens
