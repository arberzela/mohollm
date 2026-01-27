import logging
from mohollm.llm.llm import LLMInterface
from openai import OpenAI
from mohollm.utils.estimators import estimate_cost

logger = logging.getLogger("DEEPSEEK")


class DEEPSEEK(LLMInterface):
    def __init__(self, model: str = "", forceCPU: bool = False):
        super().__init__()
        self.client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
        self.model = model

    def prompt(self, prompt: str, max_number_of_tokens: str = 1000, **kwargs) -> str:
        message = [
            {
                "role": "system",
                "content": "You are an AI assistant specialized in producing precise, correctly formatted outputs that strictly adhere to all specified constraints and requirements.",
            },
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=message,
            max_tokens=5000,
            temperature=0.6,
            n=1,
        )

        token_usage = response.usage
        cost = estimate_cost(
            {
                "input_cost_per_1000_tokens": self.input_cost_per_1000_tokens,
                "output_cost_per_1000_tokens": self.output_cost_per_1000_tokens,
            },
            token_usage,
        )
        self.statistics.update_cost(cost)
        self.statistics.update_token_usage(token_usage)

        return response.choices[0].message.content
