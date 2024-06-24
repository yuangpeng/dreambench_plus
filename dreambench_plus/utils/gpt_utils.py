import base64
import inspect
import io
import random
import re
import time
from functools import wraps
from typing import Literal, Union, get_args, get_origin

import numpy
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from ..constants import OPENAI_API_KEYS, OPENAI_BASE_URL, OPENAI_MODEL
from .image_utils import ImageType, load_image
from .loguru import logger
from .misc import truly_random_seed


def retry(total_tries=5, initial_wait=1, backoff_factor=2, max_wait=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            wait_time = initial_wait
            for i in range(total_tries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # If this was the last attempt
                    if i == total_tries - 1:
                        raise ValueError(f"Function failed after {total_tries} attempts") from e
                    logger.error(f"Function failed with error: `{e}`, retry in {wait_time} seconds...")
                    time.sleep(wait_time)
                    # Exponential backoff
                    wait_time *= backoff_factor
                    if wait_time > max_wait and max_wait is not None:
                        wait_time = max_wait

        return wrapper

    return decorator


@retry(total_tries=10, initial_wait=0.1, backoff_factor=1, max_wait=10)
def call_gpt(
    messages: list[ChatCompletionMessageParam],
    *,
    api_keys: str | list[str] = OPENAI_API_KEYS,
    base_url: str = OPENAI_BASE_URL,
    model: str = OPENAI_MODEL,
    return_all: bool = False,
    **kwargs,
):
    if not isinstance(api_keys, list):
        api_keys = [api_keys]
    rng = random.Random(truly_random_seed())
    api_key = rng.choice(api_keys)
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Random sleep about 0.22s
    time.sleep(abs(numpy.random.normal(loc=0.12, scale=0.04)) + 0.1)

    completion: ChatCompletion = client.chat.completions.create(messages=messages, model=model, **kwargs)
    content = completion.choices[0].message.content
    if return_all:
        return completion
    return content


def encode_image_into_base64(image: ImageType):
    image = load_image(image)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def gpt_cli():
    def _check_literal_in_function(func, specific_param_name=None):
        signature = inspect.signature(func)
        params = []
        literal_values_list = []
        for param_name, param in signature.parameters.items():
            if specific_param_name is not None:
                if param_name != specific_param_name:
                    continue

            param_annotation = param.annotation
            if isinstance(param_annotation, str):
                param_annotation = eval(param_annotation)
            origin = get_origin(param_annotation)
            if origin is Union:
                union_args = get_args(param_annotation)
                literal_values_tuple = ()
                for arg in union_args:
                    if get_origin(arg) is Literal:
                        literal_values_tuple += get_args(arg)
                if len(literal_values_tuple) > 0:
                    params.append(param_name)
                    literal_values_list.append(literal_values_tuple)

            elif get_origin(param_annotation) is Literal:
                literal_values = get_args(param_annotation)
                params.append(param_name)
                literal_values_list.append(literal_values)

        return params, literal_values_list

    _, available_models = _check_literal_in_function(OpenAI(api_key="").chat.completions.create, "model")
    available_models = available_models[0]
    while True:
        print("Input \033[1;31;1m[exit]\033[0m to exit.\nInput \033[1;31;1m[restart]\033[0m to restart conversation.")

        while True:
            print(f"Choose the model from {available_models}\nModel: ", end="")
            ret = input()
            if ret == "[exit]":
                break
            elif ret == "[restart]":
                continue
            if ret not in available_models:
                print(f"No model named {ret}.")
                continue
            else:
                model_name = ret
                break
        if ret == "[exit]":
            break
        elif ret == "[restart]":
            continue

        print("Do you want to set the behavior of the assistant [yes/NO]: ", end="")
        ret = input()
        if ret == "[exit]":
            break
        elif ret == "[restart]":
            continue

        messages = []
        if ret == "yes" or ret == "YES" or ret == "y" or ret == "Y":
            print("The conversation is formatted with this system message (e.g. You are a helpful assistant.): ", end="")
            content = input()
            if content == "[exit]":
                break
            elif content == "[restart]":
                continue
            messages.append({"role": "system", "content": content})
            print(f"\033[1;35;1msystem:\033[0m {content}")
        elif ret == "no" or ret == "NO" or ret == "n" or ret == "N" or ret == "":
            pass
        else:
            raise Exception("Please enter yes/y/YES/Y or no/n/NO/N.")

        while True:
            print("\033[1;34;1muser:\033[0m ", end="")
            content = input()
            if content == "[exit]":
                break
            elif content == "[restart]":
                break

            pattern = r"\[([^]]+)\]"
            match = re.search(pattern, content)
            if match:
                matched = match.group(1)
                paths = [path.strip() for path in matched.split(",")]
                text_prompt = re.sub(pattern, "", content).strip()
                content = [{"type": "text", "text": text_prompt}]
                for path in paths:
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_into_base64(path)}"}})

            messages.append({"role": "user", "content": content})

            completion = call_gpt(messages, model=model_name, return_all=True)
            role = completion.choices[0].message.role
            gpt_content = completion.choices[0].message.content.strip("\n")

            print(f"\033[1;32;1m{role}:\033[0m {gpt_content}")
            messages.append({"role": role, "content": gpt_content})

        if content == "[exit]":
            break


if __name__ == "__main__":
    gpt_cli()
