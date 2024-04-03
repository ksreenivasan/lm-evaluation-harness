from pathlib import Path
from typing import Literal, Optional
from tqdm.auto import tqdm
# from kartik_api import api_generate
from api_helper import api_generate

from concurrent.futures import ThreadPoolExecutor

from tqdm.auto import tqdm

import typer

import json


def read_gsm8k_problems(task_id='gsm8k_cot'):
    if task_id == 'gsm8k_cot':
        input_file = 'lm_eval_harness_gsm8k_cot_problem_list.json'
    elif task_id == 'gsm8k':
        input_file = 'lm_eval_harness_gsm8k_problem_list.json'
    else:
        raise ValueError(f"{task_id=} not supported")
    
    with open(input_file) as f:
        return json.load(f)


def format_instruction(prompt, option=None):
    if option is None:
        # default behavior is Jose's human-eval prompt
        formatted_prompt = (
            "Please generate code to complete the following problem wrapped in a Python markdown block:"
            f"\n```python\n{prompt.strip()}\n```\n"
        )
    elif option == "gsm8k-cot-force-format-following":
        # formatted_prompt = (
        #     "Please solve the final math problem in the list given below. Follow the format specified in the few-shot samples given before the final math problem. In particular, make sure that you end your solution with the statement 'The answer is <answer>.' where <answer> is the final answer."
        #     f"\n\n{prompt.strip()}"
        # )
        # formatted_prompt = (
        #     "Below are 8 example math problems followed by their solutions. Think carefully, step by step and answer the final problem."
        #     f"\n\n{prompt.strip()}"
        # )
        # NOTE: trying to make it 0-shot
        final_problem = prompt.split("\n\n")[-1]
        formatted_prompt = (
            "Below is a math problem. Think carefully, step by step and answer it. Make sure to end your solution with '<answer>' on a newline where <answer> is the final numeric answer."
            f"\n\n{final_problem.strip()}"
        )
    elif option == "gsm8k-force-format-following":
        formatted_prompt = (
            "Please solve the final math problem in the list given below. Follow the format specified in the few-shot samples given before the final math problem. In particular, make sure that you end your solution with '#### <answer>' on a newline where <answer> is the final answer."
            f"\n\n{prompt.strip()}"
        )
    else:
        raise ValueError(f"{option=} not supported")
    return formatted_prompt


def format_prompt(prompt: str, mode: str, option: Optional[str]) -> str:
    if mode == "completion":
        return prompt
    elif mode == "custom_instruction":
        return format_instruction(prompt, option=option)
    else:
        raise ValueError(f"{mode=} not supported")


def generate(params):
    # TODO: maybe i should try splitting the prompt into instruction and problem
    # if params["mode"] == "custom_instruction":
    #     # split prompt into instruction and problem
    #     instruction, problem = params["prompt"].split("\n\n", 1)
    completion = api_generate(
        model=params["model"],
        prompt=params["prompt"],
        tenacity=False,
        **params.get("generation_kwargs", {}),
    )
    return params | {"completion": completion}


def main(
    model: str,
    mode: str = 'completion', #Literal["completion", "instruction", "custom_instruction"],
    temperature: float = 0.0,
    max_tokens: int = 512,
    generations_per_sample: int = 1,
    output_file: Optional[str] = None,
    workers: int = 5,
    task_id: str = 'gsm8k_cot',
    debug_mode: Optional[bool] = None,
):
    problems = read_gsm8k_problems(task_id=task_id)
    if debug_mode:
        problems = problems[:100]

    # TODO: calculate option based on task_id and pass it to format_prompt
    if mode == "custom_instruction":
        # compute option based on task_id
        if task_id == 'gsm8k_cot':
            prompt_format_option = "gsm8k-cot-force-format-following"
        elif task_id == 'gsm8k':
            prompt_format_option = "gsm8k-force-format-following"
        else:
            raise ValueError(f"{task_id=} not supported in custom_instruction mode")
    else:
        prompt_format_option = None
    params = [
        {
            "task_id": task_id,
            "prompt": format_prompt(problem["prompt"], mode=mode, option=prompt_format_option),
            "model": model,
            "generation_kwargs": {
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            "prompt_prefix": format_prompt("", mode=mode, option=prompt_format_option),
            "mode": mode,
        }
        for problem in problems
    ] * generations_per_sample

    debug_mode_str = "debug_" if debug_mode else ""
    if output_file is None:
        output_file = f"{task_id}_outputs/{debug_mode_str}{model}_{mode}_temp{temperature}.jsonl"
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        with open(output_file, "r") as f:
            existing = [json.loads(line) for line in f]
        for i, vals in enumerate(existing):
            assert vals['task_id'] == params[i]['task_id']
        params = params[len(existing):]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in tqdm(executor.map(generate, params), total=len(params)):
            with open(output_file, "a") as f:
                print(json.dumps(result), file=f)


if __name__ == "__main__":
    import typer
    typer.run(main)

## Example usage: # python gsm8k_eval.py 'mixtral-instruct' --task-id gsm8k --debug-mode
## Mixtral-instruct runs (once these are verified, I'll move to other APIs)
# python gsm8k_eval.py 'mixtral-instruct' --task-id gsm8k_cot
# python gsm8k_eval.py 'mixtral-instruct' --task-id gsm8k

