import inspect
import logging

from typing import Dict
from random import randrange, uniform
from mohollm.llm.llm import LLMInterface

logger = logging.getLogger("Dummy_LLM")


class DUMMY_NB201(LLMInterface):
    def __init__(self, model: str = "", forceCPU: bool = False) -> None:
        super().__init__()
        self.model = model

    def prompt(self, prompt: str, max_number_of_tokens: str = 100, **kwargs) -> Dict:
        stack = inspect.stack()
        called_by = stack[1][0].f_locals["self"].__class__.__name__

        response = ""
        match called_by:
            case "LLM_SAMPLER":
                response = f'{{"op_0_to_1": {randrange(5)},"op_0_to_2": {randrange(5)},"op_0_to_3": {randrange(5)},"op_1_to_2": {randrange(5)},"op_1_to_3": {randrange(5)},"op_2_to_3": {randrange(5)}}}'
            case "SURROGATE_MODEL":
                response = f'{{"Performance": {uniform(8.0, 12.0)},"Latency": {uniform(1.2, 5.1)}}}'
            case "ZERO_SHOT_WARMSTARTER":
                response = "["
                for i in range(self.initial_samples):
                    response += f'{{"op_0_to_1": {randrange(5)},"op_0_to_2": {randrange(5)},"op_0_to_3": {randrange(5)},"op_1_to_2": {randrange(5)},"op_1_to_3": {randrange(5)},"op_2_to_3": {randrange(5)}}}'
                    if i != self.initial_samples - 1:
                        response += ", "
                response += "]"
            case _:
                response = ""

        return response
