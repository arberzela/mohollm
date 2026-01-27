import logging
import os
from mohollm.llm.llm import LLMInterface
from openai import OpenAI
from mohollm.utils.rate_limiter import RateLimiter

logger = logging.getLogger("OpenRouter")


class OPENROUTER(LLMInterface):
    def __init__(self, model: str = "", forceCPU: bool = False):
        super().__init__()
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            logger.debug(
                "No API key found. Please set the OPENROUTER_API_KEY environment variable"
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = model
        self.rate_limiter: RateLimiter = None

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
            max_tokens=self.llm_settings.get(
                "max_number_of_tokens", max_number_of_tokens
            ),
            temperature=self.llm_settings.get("temperature", 0.7),
            n=1,
        )
        self.rate_limiter.add_request(request_token_count=response.usage.total_tokens)

        self.update_cost_and_token_usage(response)

        return response.choices[0].message.content
