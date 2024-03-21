from pathlib import Path
from typing import Literal, Optional
from tqdm.auto import tqdm
from kartik_api import api_generate

from human_eval.data import read_problems

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


def format_instruction(prompt):
    return (
        "Please generate code to complete the following problem wrapped in a Python markdown block:"
        f"\n```python\n{prompt.strip()}\n```\n"
    )


def format_prompt(prompt: str, mode: str) -> str:
    if mode == "completion":
        return prompt
    elif mode == "instruction":
        return format_instruction(prompt)
    else:
        raise ValueError(f"{mode=} not supported")


def generate(params):
    completion = api_generate(
        model=params["model"],
        prompt=params["prompt"],
        tenacity=False,
        **params.get("generation_kwargs", {}),
    )
    return params | {"completion": completion}


def main(
    model: str,
    mode: str = 'completion', #Literal["completion", "instruction"],
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

    params = [
        {
            "task_id": task_id,
            "prompt": format_prompt(problem["prompt"], mode=mode),
            "model": model,
            "generation_kwargs": {
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
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
