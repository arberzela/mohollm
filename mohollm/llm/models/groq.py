import logging
import os
from mohollm.llm.llm import LLMInterface
from groq import Groq

logger = logging.getLogger("Groq")


class GROQ(LLMInterface):
    def __init__(self, model: str = "", forceCPU: bool = False):
        super().__init__()
        api_key = os.environ.get("GROQ_AI_API_KEY")
        if not api_key:
            logger.debug(
                "No API key found. Please set the GROQ_AI_API_KEY environment variable"
            )
        self.client = Groq(api_key=api_key)
        self.model = model
        self.rate_limiter = None

    def prompt(self, prompt: str, max_number_of_tokens: str = 100, **kwargs) -> str:
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
            max_tokens=1000,  # max_number_of_tokens,
            temperature=0.7,
            n=1,
        )
        self.rate_limiter.add_request(request_token_count=response.usage.total_tokens)

        return response.choices[0].message.content
