import torch
import gc
import logging
from typing import Dict
from mohollm.llm.llm import LLMInterface
from transformers import pipeline

logger = logging.getLogger("LOCAL_LLM")


class LOCAL(LLMInterface):
    def __init__(self, model: str, forceCPU: bool = False) -> None:
        super().__init__()
        self.model = model
        self.device = "cpu" if forceCPU or not torch.cuda.is_available() else "cuda"

    def _load_model_to_memory(self):
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            trust_remote_code=True,
            device=self.device,
        )

    def prompt(self, prompt: str, max_number_of_tokens: int = 100, **kwargs) -> Dict:
        message = []
        message.append({"role": "user", "content": prompt})

        generate_kwargs = {
            "do_sample": True,  # Sampling needs to be enabled for temperature to work
            "temperature": 0.7,
            "max_new_tokens": 10000,
        }

        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        response = self.pipe(message, **generate_kwargs)
        content = response[0].get("generated_text")[1].get("content", "")
        return content
