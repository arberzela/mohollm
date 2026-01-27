def estimate_cost(config: dict, usage) -> dict:
    """
    Estimate the total cost required for a trial based on the number of tokens and cost.

    Args:
        config (Dict): A dictionary containing configuration parameters. Keys include:
            - "input_cost_per_1000_tokens" (float): The cost per 1000 tokens for input.
            - "output_cost_per_1000_tokens" (float): The cost per 1000 tokens for output
        usage (CompletionUsage): A dictionary containing the usage of a open ai response.

    Returns:
        Dict: A dictionary containing prompt_cost, completion_cost, and total_cost.
    """
    input_cost_per_1000_tokens = config.get("input_cost_per_1000_tokens", 0)
    output_cost_per_1000_tokens = config.get("output_cost_per_1000_tokens", 0)

    prompt_cost = usage.get("prompt_tokens", 0) * input_cost_per_1000_tokens / 1000
    completion_cost = usage.get("completion_tokens", 0) * output_cost_per_1000_tokens / 1000

    total_cost = prompt_cost + completion_cost

    cost = {
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "total_cost": total_cost,
    }

    return cost
