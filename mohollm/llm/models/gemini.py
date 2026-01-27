import logging
import google.generativeai as genai
from mohollm.utils.estimators import estimate_cost
from mohollm.llm.llm import LLMInterface
import os

logger = logging.getLogger("GEMINI")


class GEMINI(LLMInterface):
    def __init__(self, model: str = "", forceCPU: bool = False):
        super().__init__()

        api_key = os.environ.get("GOOGLE_AI_API_KEY")
        if not api_key:
            logger.debug(
                "No API key found. Please set the GOOGLE_AI_API_KEY environment variable"
            )

        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)
        self.model = model
        self.rate_limiter = None

    def prompt(self, prompt: str, max_number_of_tokens: str = 100, **kwargs) -> str:
        self.rate_limiter.add_request(request_text=prompt)
        response = self.client.generate_content(prompt)
        response_text = response.text
        request_token_count = response.usage_metadata.total_token_count
        self.rate_limiter.add_request(request_token_count=request_token_count)
        self._update_cost_and_token_usage(response)

        return response_text

    def _update_cost_and_token_usage(self, response):
        """Update the cost and token usage statistics based on the LLM response.

        Args:
            response: The response object from the LLM, which should contain token usage information.
        """
        token_usage = response.usage_metadata
        usage = {
            "prompt_tokens": token_usage.prompt_token_count,
            "completion_tokens": token_usage.candidates_token_count,
            "total_tokens": token_usage.total_token_count,
        }
        cost = estimate_cost(
            {
                "input_cost_per_1000_tokens": self.llm_settings.get(
                    "input_cost_per_1000_tokens", 0
                ),
                "output_cost_per_1000_tokens": self.llm_settings.get(
                    "output_cost_per_1000_tokens", 0
                ),
            },
            usage,
        )
        logger.debug(f"Cost of request: {cost}")
        logger.debug(f"Token usage: {usage}")
        self.statistics.update_cost(cost)
        self.statistics.update_token_usage(usage)
