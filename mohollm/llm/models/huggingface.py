import os
import logging
from mohollm.llm.llm import LLMInterface
from huggingface_hub import InferenceClient

logger = logging.getLogger("HUGGINGFACE")


class HUGGINGFACE(LLMInterface):
    def __init__(self, model: str = "", forceCPU: bool = False):
        super().__init__()
        api_key = os.environ.get("HUGGINGFACE_API_KEY")
        if not api_key:
            logger.debug(
                "No API key found. Please set the HUGGINGFACE_API_KEY environment variable"
            )
        self.client = InferenceClient(api_key=api_key)
        self.model = model
        self.rate_limiter = None

    def prompt(self, prompt: str, max_number_of_tokens: str = 1000, **kwargs) -> str:
        self.rate_limiter.add_request(request_text=prompt)
        message = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find information.",
            },
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=message,
            max_tokens=max_number_of_tokens,
            temperature=0.7,
            n=1,
        )
        self.rate_limiter.add_request(request_token_count=response.usage.total_tokens)

        return response.choices[0].message.content
