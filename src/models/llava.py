"""LLaVA-1.5-7B backbone: one forward pass -> last-token hidden state + class logits.

Heavy deps (torch, transformers) are imported lazily so this module can be
imported (and registered) on a machine without a GPU / without transformers.
"""

from __future__ import annotations

import numpy as np

from ..registry import register_model, VLMBackbone

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DEFAULT_PROMPT = ("USER: <image>\nWhat is the main object in this image? "
                  "Answer with a single word. ASSISTANT:")


@register_model("llava-1.5-7b")
class LlavaBackbone(VLMBackbone):
    def __init__(self, class_names, dtype="float16", prompt=DEFAULT_PROMPT,
                 batch_size=8, device="cuda"):
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.prompt = prompt
        self.batch_size = batch_size
        self.device = device
        self._torch = torch
        td = {"float16": torch.float16, "bfloat16": torch.bfloat16,
              "float32": torch.float32}[dtype]

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.processor.tokenizer.padding_side = "right"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=td, low_cpu_mem_usage=True).to(device).eval()
        self.feat_dim = self.model.config.text_config.hidden_size

        # class-restricted logits: first sub-token id of each " <class name>"
        tok = self.processor.tokenizer
        self.class_names = list(class_names)
        self.class_token_ids = []
        for name in self.class_names:
            readable = " " + name.replace("_", " ")
            ids = tok(readable, add_special_tokens=False).input_ids
            self.class_token_ids.append(ids[0])
        self.class_token_ids = np.asarray(self.class_token_ids)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def _run_batch(self, images):
        torch = self._torch
        prompts = [self.prompt] * len(images)
        inputs = self.processor(images=images, text=prompts,
                                return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        # last real token index per sample (right padding -> sum(mask)-1)
        last = inputs["attention_mask"].sum(1) - 1              # [B]
        b = torch.arange(len(images), device=self.device)
        hidden = out.hidden_states[-1][b, last]                 # [B, D]
        logits_all = out.logits[b, last]                        # [B, V]
        cls_ids = torch.as_tensor(self.class_token_ids, device=self.device)
        logits = logits_all.index_select(1, cls_ids)           # [B, C]
        return (hidden.float().cpu().numpy(), logits.float().cpu().numpy())

    def extract(self, images) -> dict:
        """images: list[PIL.Image] -> {'hidden':[N,D], 'logits':[N,C]} (numpy)."""
        H, L = [], []
        for i in range(0, len(images), self.batch_size):
            h, l = self._run_batch(images[i:i + self.batch_size])
            H.append(h); L.append(l)
        return {"hidden": np.concatenate(H), "logits": np.concatenate(L)}
