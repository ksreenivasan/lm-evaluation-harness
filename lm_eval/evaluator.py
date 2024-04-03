import itertools
import logging
import random
import time
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import torch

import lm_eval.api.metrics
import lm_eval.api.registry
import lm_eval.models
from lm_eval.caching.cache import delete_cache
from lm_eval.evaluator_utils import (
    consolidate_results,
    get_sample_size,
    get_task_list,
    prepare_print_tasks,
    print_writeout,
    run_task_tests,
)
from lm_eval.logging_utils import add_env_info, get_git_commit_hash
from lm_eval.tasks import TaskManager, get_task_dict
from lm_eval.utils import eval_logger, positional_deprecated, simple_parse_args_string


if TYPE_CHECKING:
    from lm_eval.api.model import LM
    from lm_eval.tasks import Task


@positional_deprecated
def simple_evaluate(
    model,
    model_args: Optional[Union[str, dict]] = None,
    tasks: Optional[List[Union[str, dict, object]]] = None,
    num_fewshot: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    device: Optional[str] = None,
    use_cache: Optional[str] = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    delete_requests_cache: bool = False,
    limit: Optional[Union[int, float]] = None,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    write_out: bool = False,
    log_samples: bool = True,
    gen_kwargs: Optional[str] = None,
    task_manager: Optional[TaskManager] = None,
    verbosity: str = "INFO",
    predict_only: bool = False,
    random_seed: int = 0,
    numpy_random_seed: int = 1234,
    torch_random_seed: int = 1234,
    save_problem_list: Optional[bool] = False,
    score_generations: Optional[bool] = False,
    custom_eval_debug_mode: Optional[bool] = False,
    custom_eval_mode: Optional[str] = None,
    custom_eval_temperature: Optional[float] = 0.0,
    custom_eval_model_name: Optional[str] = None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str, dict]
        String or dict arguments for each model class, see LM.create_from_arg_string and LM.create_from_arg_object.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, dict, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int or str, optional
        Batch size for model
    :param max_batch_size: int, optional
        Maximal batch size to try with automatic batch size detection
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param use_cache: str, optional
        A path to a sqlite db file for caching model responses. `None` if not caching.
    :param cache_requests: bool, optional
        Speed up evaluation by caching the building of dataset requests. `None` if not caching.
    :param rewrite_requests_cache: bool, optional
        Rewrites all of the request cache if set to `True`. `None` if not desired.
    :param delete_requests_cache: bool, optional
        Deletes all of the request cache if set to `True`. `None` if not desired.
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param gen_kwargs: str
        String arguments for model generation
        Ignored for all tasks with loglikelihood output_type
    :param predict_only: bool
        If true only model outputs will be generated and returned. Metrics will not be evaluated
    :param random_seed: int
        Random seed for python's random module. If set to None, the seed will not be set.
    :param numpy_random_seed: int
        Random seed for numpy. If set to None, the seed will not be set.
    :param torch_random_seed: int
        Random seed for torch. If set to None, the seed will not be set.

    :return
        Dictionary of results
    """
    eval_logger.setLevel(getattr(logging, f"{verbosity}"))
    start_date = time.time()

    if delete_requests_cache:
        eval_logger.info("Deleting requests cache...")
        delete_cache()

    seed_message = []
    if random_seed is not None:
        # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1412
        seed_message.append(f"Setting random seed to {random_seed}")
        random.seed(random_seed)

    if numpy_random_seed is not None:
        seed_message.append(f"Setting numpy seed to {numpy_random_seed}")
        np.random.seed(numpy_random_seed)

    if torch_random_seed is not None:
        seed_message.append(f"Setting torch manual seed to {torch_random_seed}")
        torch.manual_seed(torch_random_seed)

    if seed_message:
        eval_logger.info(" | ".join(seed_message))

    if tasks is None:
        tasks = []
    if len(tasks) == 0:
        raise ValueError(
            "No tasks specified, or no tasks found. Please verify the task names."
        )

    if gen_kwargs is not None:
        gen_kwargs = simple_parse_args_string(gen_kwargs)
        eval_logger.warning(
            "generation_kwargs specified through cli, these settings will update set parameters in yaml tasks. "
            "Ensure 'do_sample=True' for non-greedy decoding!"
        )
        if gen_kwargs == "":
            gen_kwargs = None

    if isinstance(model, str):
        if model_args is None:
            model_args = ""

        if isinstance(model_args, dict):
            lm = lm_eval.api.registry.get_model(model).create_from_arg_obj(
                model_args,
                {
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    "device": device,
                },
            )

        else:
            lm = lm_eval.api.registry.get_model(model).create_from_arg_string(
                model_args,
                {
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    "device": device,
                },
            )
    else:
        if not isinstance(model, lm_eval.api.model.LM):
            raise TypeError
        lm = model

    if use_cache is not None:
        eval_logger.info(f"Using cache at {use_cache + '_rank' + str(lm.rank) + '.db'}")
        lm = lm_eval.api.model.CachingLM(
            lm,
            use_cache
            # each rank receives a different cache db.
            # necessary to avoid multiple writes to cache at once
            + "_rank"
            + str(lm.rank)
            + ".db",
        )

    if task_manager is None:
        task_manager = TaskManager(verbosity)

    task_dict = get_task_dict(tasks, task_manager)
    for task_name in task_dict.keys():
        task_obj = task_dict[task_name]
        if isinstance(task_obj, tuple):
            _, task_obj = task_obj
            if task_obj is None:
                continue

        if task_obj.get_config("output_type") == "generate_until":
            if gen_kwargs is not None:
                task_obj.set_config(
                    key="generation_kwargs", value=gen_kwargs, update=True
                )

        if predict_only:
            log_samples = True
            eval_logger.info(
                f"Processing {task_name} in output-only mode. Metrics will not be calculated!"
            )
            # we have to change the class properties post-hoc. This is pretty hacky.
            task_obj.override_metric(metric_name="bypass")

        # override tasks' fewshot values to the provided num_fewshot arg value
        # except if tasks have it set to 0 manually in their configs--then we should never overwrite that
        if num_fewshot is not None:
            if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                eval_logger.info(
                    f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored."
                )
            else:
                eval_logger.warning(
                    f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}"
                )
                task_obj.set_config(key="num_fewshot", value=num_fewshot)
        else:
            # if num_fewshot not provided, and the task does not define a default one, default to 0
            if (default_num_fewshot := task_obj.get_config("num_fewshot")) is None:
                task_obj.set_config(key="num_fewshot", value=0)

    if check_integrity:
        run_task_tests(task_list=tasks)

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        cache_requests=cache_requests,
        rewrite_requests_cache=rewrite_requests_cache,
        bootstrap_iters=bootstrap_iters,
        write_out=write_out,
        log_samples=log_samples,
        verbosity=verbosity,
        save_problem_list=save_problem_list,
        score_generations=score_generations,
        custom_eval_debug_mode=custom_eval_debug_mode,
        custom_eval_mode=custom_eval_mode,
        custom_eval_temperature=custom_eval_temperature,
        custom_eval_model_name=custom_eval_model_name,
    )

    if lm.rank == 0:
        if isinstance(model, str):
            model_name = model
        elif hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            model_name = model.config._name_or_path
        else:
            model_name = type(model).__name__

        # add info about the model and few shot config
        results["config"] = {
            "model": model_name,
            "model_args": model_args,
            "batch_size": batch_size,
            "batch_sizes": (
                list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else []
            ),
            "device": device,
            "use_cache": use_cache,
            "limit": limit,
            "bootstrap_iters": bootstrap_iters,
            "gen_kwargs": gen_kwargs,
        }
        results["git_hash"] = get_git_commit_hash()
        results["date"] = start_date
        add_env_info(results)  # additional environment info to results
        return results
    else:
        return None


@positional_deprecated
def evaluate(
    lm: "LM",
    task_dict,
    limit: Optional[int] = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    bootstrap_iters: Optional[int] = 100000,
    write_out: bool = False,
    log_samples: bool = True,
    verbosity: str = "INFO",
    save_problem_list: Optional[bool] = False,
    score_generations: Optional[bool] = False,
    custom_eval_debug_mode: Optional[bool] = False,
    custom_eval_mode: Optional[str] = None,
    custom_eval_temperature: Optional[float] = 0.0,
    custom_eval_model_name: Optional[str] = None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name type(task).config.task .
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :return
        Dictionary of results
    """

    eval_logger.setLevel(getattr(logging, f"{verbosity}"))

    # tracks all Instances/requests a model must generate output on.
    requests = defaultdict(list)
    # stores the amount to pad out reqs per req. type so that
    # number of fwd passes per distributed rank is equal
    padding_requests = defaultdict(int)

    # get lists of group hierarchy and each type of request
    task_hierarchy, eval_tasks = get_task_list(task_dict)
    if not log_samples:
        if not all(
            "bypass" not in getattr(task_output.task, "_metric_fn_list", {}).keys()
            for task_output in eval_tasks
        ):
            raise ValueError("log_samples must be True for 'bypass' metric-only tasks")
    for task_output in eval_tasks:
        task: Task = task_output.task
        limit = get_sample_size(task, limit)
        task.build_all_requests(
            limit=limit,
            rank=lm.rank,
            world_size=lm.world_size,
            cache_requests=cache_requests,
            rewrite_requests_cache=rewrite_requests_cache,
        )
        eval_logger.debug(
            f"Task: {task_output.task_name}; number of requests on this rank: {len(task.instances)}"
        )

        if write_out:
            print_writeout(task)
        # aggregate Instances by LM method requested to get output.
        for instance in task.instances:
            reqtype = instance.request_type
            requests[reqtype].append(instance)

        if lm.world_size > 1:
            instances_rnk = torch.tensor(len(task._instances), device=lm.device)
            gathered_item = (
                lm.accelerator.gather(instances_rnk).cpu().detach().numpy().tolist()
            )
            # "multiple_choice" task types dispatch (several) "loglikelihood" request types
            reqtype = (
                "loglikelihood"
                if task.OUTPUT_TYPE == "multiple_choice"
                else task.OUTPUT_TYPE
            )
            # compute number of pseudo-batches to pad with (FSDP/DDP require even batches among ranks)
            numpad = max(gathered_item) - gathered_item[lm.rank]
            # todo: may not account for padding in cases like SquadV2 which has multiple req types
            padding_requests[reqtype] += numpad

    ### Run LM on inputs, get all outputs ###
    # execute each type of request

    # @KS: NOTE: Use these flags to decide which mode to run lm-eval-harness in.
    SAVE_PROBLEM_LIST = save_problem_list
    SCORE_GENERATIONS = score_generations
    # TODO: check kartik args DONE
    # print("TODO: check kartik args here before running anything else.")
    # import ipdb; ipdb.set_trace()
    import json

    if SAVE_PROBLEM_LIST:
        print("-"*200)
        print("Kartik: Running eval in SAVE_PROBLEM_LIST mode. Will just run through gsm8k and construct prompts.")
        print("-"*200)
    elif SCORE_GENERATIONS:
        print("-"*200)
        print("Kartik: Running eval in SCORE_GENERATIONS mode. Will use stored generations and score them.")
        print("-"*200)
    else:
        print("-"*200)
        print("Kartik: Running eval in regular mode. Default lm-eval-harness behavior.")
        print("-"*200)
    cached_response = None
    problem_list = []
    for reqtype, reqs in requests.items():
        eval_logger.info(f"Running {reqtype} requests")
        # create `K` copies of each request `req` based off `K = req.repeats`
        cloned_reqs = []
        for req in reqs:
            cloned_reqs.extend([req] * req.repeats)

        if (lm.world_size > 1) and (padding_requests[reqtype] > 0):
            for _ in range(padding_requests[reqtype]):
                cloned_reqs.extend([req] * req.repeats)

        # import ipdb; ipdb.set_trace()
        # run requests through model
        # @KS: TODO: this is where the model is called.
        # This is also probably where I will stick my logic for putting the generations back in.
        if SAVE_PROBLEM_LIST or SCORE_GENERATIONS:
            print("Skipping model call for now, so I can extract prompts, answers, etc.")
            if cached_response is None:
                dummy_response = getattr(lm, reqtype)(cloned_reqs[0:1])
                resps = [dummy_response[0] for _ in cloned_reqs]
                cached_response = resps
            else:
                resps = cached_response
        else:
            resps = getattr(lm, reqtype)(cloned_reqs)

        if SAVE_PROBLEM_LIST:
            for req in cloned_reqs:
                problem = {'prompt': req.args[0], 'doc': req.doc, 'doc_id': req.doc_id,
                        'filtered_resps': req.filtered_resps, 'resps': req.resps,
                        'idx': req.idx, 'metadata': req.metadata,
                        'request_type': req.request_type, 'repeats': req.repeats,
                        'task_name': req.task_name}
                problem_list.append(problem)
        # import ipdb; ipdb.set_trace()
        # put responses from model into a list of length K for each request.

        if SCORE_GENERATIONS:
            # read generations file and fill in resps with them
            # temperature = 0.0
            # mode = 'completion'
            # model = 'mixtral-instruct'
            task_id = cloned_reqs[0].task_name

            debug_mode_str = 'debug_' if custom_eval_debug_mode else ''
            generations_file = f'{task_id}_outputs/{debug_mode_str}{custom_eval_model_name}_{custom_eval_mode}_temp{custom_eval_temperature}.jsonl'
            # generations_file = f'{task_id}_outputs/200_max_tokens_{debug_mode_str}{custom_eval_model_name}_{custom_eval_mode}_temp{custom_eval_temperature}.jsonl'
            # generations_file = f'gsm8k_outputs/debug_mixtral-instruct_completion_temp0.0.jsonl'
            print(f'Gonna fill in the responses from {generations_file}.')
            with open(generations_file) as f:
                stored_generations = [json.loads(line) for line in f]

            # extract model name from stored_generations here and update it wherever it's used in the code
            def apply_generate_until_filter(stored_generations):
                generate_until_keywords = task.config.generation_kwargs['until']
                for generation in stored_generations:
                    response = generation['completion']
                    if response is not None:
                        stop_idxs = [response.find(keyword) for keyword in generate_until_keywords]
                        # remove -1s
                        stop_idxs = [x for x in stop_idxs if x > 0]
                        if len(stop_idxs) > 0 and min(stop_idxs) > 0:
                            # trim to first stop_idx
                            response = response[:min(stop_idxs)].strip()
                    else:
                        print("Warning: None response found in stored generations. This shouldn't have happened.")
                        response = "None"
                    generation['processed_completion'] = response
                return stored_generations

            # apply generation_until functionality
            processed_generations = apply_generate_until_filter(stored_generations)

            # this will only work if the order of the generations is the same as the order of the requests I saw earlier. fingers crossed
            for idx, req in enumerate(cloned_reqs):
                if custom_eval_debug_mode and idx >= len(processed_generations):
                    print("Warning: Running in debug mode and ran out of stored generations.")
                    break

                if custom_eval_mode == 'custom_instruction':
                    # need to strip the prompt_prefix from the prompt
                    prompt_without_prefix = processed_generations[idx]['prompt'].strip(processed_generations[idx]['prompt_prefix'])
                else:
                    prompt_without_prefix = processed_generations[idx]['prompt']

                if req.arguments[0] == prompt_without_prefix:
                    resps[idx] = processed_generations[idx]['processed_completion']
                elif req.arguments[0].split("\n\n")[-1] == prompt_without_prefix:
                    # specifically for the 0-shot prompting that I'm trying out
                    resps[idx] = processed_generations[idx]['processed_completion']
                else:
                    print(f"Prompt mismatch at {idx}. Something went wrong.")
                    resps[idx] = 'Kartik ERROR: Something went wrong with my script. Prompt mismatch, so I lost the completion.'
                    raise ValueError(f"Prompt mismatch at {idx}. Something went wrong.")

        for x, req in zip(resps, cloned_reqs):
            req.resps.append(x)

        if lm.world_size > 1:
            lm.accelerator.wait_for_everyone()

    if SAVE_PROBLEM_LIST:
        with open(f'lm_eval_harness_{problem_list[0]["task_name"]}_problem_list.json', 'w') as f:
            json.dump(problem_list, f)

    RANK = lm.rank
    WORLD_SIZE = lm.world_size
    ### Postprocess outputs ###
    # TODO: del model here, maybe (idea: allow user to specify device of e.g. reward model separately)
    for task_output in eval_tasks:
        task = task_output.task
        # import ipdb; ipdb.set_trace()
        task.apply_filters()

        ### Collect values of metrics on all datapoints ###
        # # unpack results and sort back in order and return control to Task
        # TODO: make it possible to use a different metric per filter
        # Pre-process task.instances to group by doc_id
        instances_by_doc_id = defaultdict(list)
        for instance in task.instances:
            instances_by_doc_id[instance.doc_id].append(instance)
        # Sort instances within each group
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)
        # iterate over different filters used
        # @ks: NOTE: filter_key is strict-match and flexible-extract. The solution computation is already done at this stage.
        # import ipdb; ipdb.set_trace()
        for filter_key in task.instances[0].filtered_resps.keys():
            doc_iterator = task.doc_iterator(
                rank=RANK, limit=limit, world_size=WORLD_SIZE
            )
            for doc_id, doc in doc_iterator:
                requests = instances_by_doc_id[doc_id]
                metrics = task.process_results(
                    doc, [req.filtered_resps[filter_key] for req in requests]
                )
                if log_samples:
                    target = task.doc_to_target(doc)
                    example = {
                        "doc_id": doc_id,
                        "doc": doc,
                        "target": target,
                        "arguments": [req.args for req in requests],
                        "resps": [req.resps for req in requests],
                        "filtered_resps": [
                            req.filtered_resps[filter_key] for req in requests
                        ],
                    }
                    example.update(metrics)
                    task_output.logged_samples.append(example)
                for metric, value in metrics.items():
                    task_output.sample_metrics[(metric, filter_key)].append(value)

    if WORLD_SIZE > 1:
        # if multigpu, then gather data across all ranks to rank 0
        # first gather logged samples across all ranks
        for task_output in eval_tasks:
            if log_samples:
                # for task_name, task_samples in list(samples.items()):
                full_samples = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    obj=task_output.logged_samples,
                    object_gather_list=full_samples,
                    dst=0,
                )

                if RANK == 0:
                    task_output.logged_samples = list(
                        itertools.chain.from_iterable(full_samples)
                    )

            # then collect metrics across all ranks
            for metrics in task_output.sample_metrics:
                metric_list = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    obj=task_output.sample_metrics[metrics],
                    object_gather_list=metric_list,
                    dst=0,
                )
                if RANK == 0:
                    task_output.sample_metrics[metrics] = list(
                        itertools.chain.from_iterable(metric_list)
                    )
    # import ipdb; ipdb.set_trace()
    if RANK == 0:
        ### Aggregate results over all datapoints ###
        # aggregate results ; run bootstrap CIs
        for task_output in eval_tasks:
            task_output.calculate_aggregate_metric(bootstrap_iters=bootstrap_iters)
        results, samples, configs, versions, num_fewshot = consolidate_results(
            eval_tasks
        )

        ### Calculate group metrics ###
        if bool(results):
            for group, task_list in reversed(task_hierarchy.items()):
                if len(task_list) == 0:
                    # task_hierarchy entries are either
                    # `group_name: [subtask1, subtask2, ...]`
                    # or `task_name: []`.
                    # we only want to operate on groups here.
                    continue
                metric_list = list(
                    {
                        key
                        for task in task_list
                        for key in results[task].keys()
                        if "_stderr" not in key and key not in ["alias", "samples"]
                    }
                )
                for metric in metric_list:
                    stderr = "_stderr,".join(metric.split(","))

                    # gather metrics, sizes, and stderrs from subtasks
                    metrics = [
                        results[task][metric]
                        for task in task_list
                        if metric in results[task]
                    ]  # TODO: copy?
                    stderrs = [
                        results[task][stderr]
                        for task in task_list
                        if stderr in results[task]
                    ]
                    sizes = [
                        results[task]["samples"]
                        for task in task_list
                        if metric in results[task]
                    ]

                    # compute group's pooled metric and stderr
                    results[group][
                        metric
                    ] = lm_eval.api.metrics.aggregate_subtask_metrics(metrics, sizes)
                    # TODO: calculate grouped metric using aggregation fn
                    if "N/A" in stderrs:
                        results[group][stderr] = "N/A"
                    else:
                        results[group][
                            stderr
                        ] = lm_eval.api.metrics.pooled_sample_stderr(stderrs, sizes)
                        # TODO: allow GroupConfigs to choose which variance formula is used, for back-compatibility
                        # To use the old (likely incorrect) variance formula, comment out the above and uncomment this line:
                        # results[group][stderr] = lm_eval.api.metrics.combined_sample_stderr(stderrs, sizes, metrics=metrics)

                    results[group]["samples"] = sum(sizes)

        results_agg = defaultdict(dict)
        groups_agg = defaultdict(dict)
        all_tasks_list = list(task_hierarchy.keys())
        while True:
            add_tasks_list = list(k for k in results_agg.keys())
            left_tasks_list = sorted(list(set(all_tasks_list) - set(add_tasks_list)))
            if len(left_tasks_list) == 0:
                break

            _task_hierarchy = {
                k: v for k, v in task_hierarchy.items() if k in left_tasks_list
            }
            _results_agg, _groups_agg = prepare_print_tasks(_task_hierarchy, results)

            results_agg = {**results_agg, **_results_agg}
            groups_agg = {**groups_agg, **_groups_agg}

        for group_name, task_list in task_hierarchy.items():
            if task_list:
                num_fewshot[group_name] = num_fewshot[
                    task_list[0]
                ]  # TODO: validate this

        results_dict = {
            "results": dict(results_agg.items()),
            **({"groups": dict(groups_agg.items())} if bool(groups_agg) else {}),
            "group_subtasks": dict(reversed(task_hierarchy.items())),
            "configs": dict(sorted(configs.items())),
            "versions": dict(sorted(versions.items())),
            "n-shot": dict(sorted(num_fewshot.items())),
        }

        # TODO: @ks: I think I can stick model in here
        # import ipdb; ipdb.set_trace()
        results_dict["custom_eval_metadata"] = {'custom_eval_model_name': custom_eval_model_name,
                                           'custom_eval_debug_mode': custom_eval_debug_mode,
                                           'custom_eval_mode': custom_eval_mode,
                                           'custom_eval_temperature': custom_eval_temperature,
                                           'save_problem_list': save_problem_list,
                                           'score_generations': score_generations
                                        }
        if log_samples:
            results_dict["samples"] = dict(samples)

        return results_dict

    else:
        return None


def request_caching_arg_to_dict(cache_requests: str) -> dict:
    request_caching_args = {
        "cache_requests": cache_requests in {"true", "refresh"},
        "rewrite_requests_cache": cache_requests == "refresh",
        "delete_requests_cache": cache_requests == "delete",
    }

    return request_caching_args
