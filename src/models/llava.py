"""LLaVA-1.5-7B backbone: one forward pass -> last-token hidden state.

extract() takes image *paths* and uses a DataLoader with worker processes to
load+preprocess images in parallel, so the GPU stays fed (serial preprocessing
previously left it ~0% utilized). Returns the last-token hidden state; MSP/Energy
logits are produced downstream by a linear probe (see run_all), which is
well-defined -- unlike the near-constant raw class-token logits.

Heavy deps (torch, transformers) are imported lazily so this module can be
imported (and registered) on a machine without a GPU / without transformers.
"""

from __future__ import annotations

import numpy as np

from ..registry import register_model, VLMBackbone

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DEFAULT_PROMPT = ("USER: <image>\nWhat is the main object in this image? "
                  "Answer with a single word. ASSISTANT:")


class _PathDataset:
    """Loads + processes one image per item (runs in DataLoader workers)."""
    def __init__(self, paths, processor, prompt):
        self.paths, self.processor, self.prompt = paths, processor, prompt

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(self.paths[i]).convert("RGB")
        enc = self.processor(images=img, text=self.prompt, return_tensors="pt")
        return {k: v[0] for k, v in enc.items()}   # drop the batch dim


@register_model("llava-1.5-7b")
class LlavaBackbone(VLMBackbone):
    def __init__(self, class_names, dtype="float16", prompt=DEFAULT_PROMPT,
                 batch_size=8, device="cuda", num_workers=8):
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.prompt = prompt
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self._torch = torch
        td = {"float16": torch.float16, "bfloat16": torch.bfloat16,
              "float32": torch.float32}[dtype]

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.processor.tokenizer.padding_side = "right"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=td, low_cpu_mem_usage=True).to(device).eval()
        self.feat_dim = self.model.config.text_config.hidden_size
        self.class_names = list(class_names)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def extract(self, paths) -> dict:
        """paths: list[str] -> {'hidden':[N,D]} (numpy). Parallel preprocessing."""
        import torch
        from torch.utils.data import DataLoader
        # Containers cap ulimit -n at 1024; the default 'file_descriptor' tensor-
        # sharing strategy leaks past that with several workers -> workers die and
        # the loader hangs. 'file_system' shares by name and avoids the FD limit.
        torch.multiprocessing.set_sharing_strategy("file_system")
        ds = _PathDataset(list(paths), self.processor, self.prompt)
        dl = DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers,
                        pin_memory=True, shuffle=False)
        H = []
        for batch in dl:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                out = self.model(**batch, output_hidden_states=True)
            last = batch["attention_mask"].sum(1) - 1                # [B]
            b = torch.arange(last.shape[0], device=self.device)
            hidden = out.hidden_states[-1][b, last]                  # [B, D]
            H.append(hidden.float().cpu().numpy())
        return {"hidden": np.concatenate(H)}
