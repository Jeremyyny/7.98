"""Explicit, validated generation settings shared by advisor serving and replay."""
import math


def normalize_generation(settings=None):
    if settings is not None and not isinstance(settings, dict):
        raise ValueError("advisor_generation must be an object")
    values = {"temperature": 0.0, "seed": 42, **(settings or {})}
    ranges = {"temperature": (0, 2), "top_p": (0, 1), "min_p": (0, 1),
              "presence_penalty": (0, 2), "repetition_penalty": (0, 2)}
    if set(values) - (set(ranges) | {"top_k", "seed"}):
        raise ValueError("Unsupported advisor generation parameter")
    for key, value in values.items():
        if key in {"top_k", "seed"}:
            maximum = 2**32 - 1 if key == "seed" else 1000000
            if type(value) is not int or not 0 <= value <= maximum:
                raise ValueError(f"Invalid {key}")
        else:
            low, high = ranges[key]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not low <= value <= high
                    or (key in {"top_p", "repetition_penalty"} and value == 0)):
                raise ValueError(f"Invalid {key}")
    return values


def presence_penalty(prompt_length, penalty):
    class GeneratedPresencePenalty:
        def __call__(self, input_ids, scores):
            adjusted = scores.clone()
            for batch in range(input_ids.shape[0]):
                seen = input_ids[batch, prompt_length:].unique()
                adjusted[batch, seen] -= penalty
            return adjusted
    return GeneratedPresencePenalty()


def generation_kwargs(settings, prompt_length):
    settings = normalize_generation(settings)
    sampled = settings["temperature"] > 0
    options = {"do_sample": sampled}
    if sampled:
        options.update({key: settings[key] for key in
                        ("temperature", "top_p", "top_k", "min_p") if key in settings})
    if "repetition_penalty" in settings:
        options["repetition_penalty"] = settings["repetition_penalty"]
    if settings.get("presence_penalty", 0):
        from transformers import LogitsProcessorList
        options["logits_processor"] = LogitsProcessorList([
            presence_penalty(prompt_length, settings["presence_penalty"])])
    return options
