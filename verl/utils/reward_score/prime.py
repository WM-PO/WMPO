import asyncio
import json
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Optional

import math

from .evaluation_utils.code_util import evaluate_code
from .evaluation_utils.math_util import evaluate_math
from tqdm.asyncio import tqdm


# def process_completion(completion, task, reference):
#     if task == "code":
#         return evaluate_code(completion, reference)
#     elif task == "math":
#         return evaluate_math(completion, str(reference))
#     else:
#         print('task')
#         raise NotImplementedError

# def compute_score(completions, references, tasks):
#     # Process results
#     scores = []
#     format = []
#     results = process_completion(completions, tasks, references)
#     for result, completion, reference, task in zip(results, completions, references, tasks):
#         if isinstance(result, Exception) or result is None:
#             # Handle failed or timed-out tasks
#             scores.append(0.0)
#             format.append(0.0)
#             continue

#         try:
#             # Process result based on task type
#             if task == 'code' and not result[0]: # if task is code, the reference should be json string
#                 correct = 0
#                 total = min(
#                     len(json.loads(reference)['inputs'] if not isinstance(reference, dict) else reference['inputs']),
#                     10)
#                 for run in result[1]:
#                     if 'test_case' in run and 'res' in run['test_case'] and run['test_case']['res'] == '[True]':
#                         correct += 1
#                 scores.append(correct / total)
#                 format.append(correct / total)
#             elif task == 'code':
#                 scores.append(float(int(result[0])))
#                 format.append(float(int(result[0])))
#             else:
#                 scores.append(float(int(result[0])))
#                 format.append(float(int(result[1])))
            
#         except Exception as e:
#             print(f"Error processing result for row: {completion[:10]}, Error: {e}")
#             scores.append(0.0)
#             format.append(0.0)
            
#     return scores, format








# #############################################################
# def process_completion(completion, task, reference):
#     if task == "code":
#         return evaluate_code(completion, reference)
#     elif task == "math":
#         return evaluate_math(completion, str(reference))
#     else:
#         print('task')
#         raise NotImplementedError

# async def process_row_with_timeout(completion, reference, task, executor, timeout=30.0):
#     """
#     Process a single row with a timeout.
#     """
#     loop = asyncio.get_running_loop()
#     try:
#         result = await asyncio.wait_for(
#             loop.run_in_executor(
#                 executor,
#                 partial(process_completion, completion, task, reference)
#             ),
#             timeout=timeout
#         )
#         return result
#     except asyncio.TimeoutError:
#         print(f"Timeout occurred for completion: {completion[:10]}, reference: {reference[:10]}")
#         return None  # Default value for timed-out rows
#     except Exception as e:
#         print(f"Error processing completion: {completion[:10]}, Error: {e}")
#         return None  # Default value for failed rows

# async def parallel_evaluate_continual_async(completions, references, tasks, num_processes, task_timeout=30.0):
#     """
#     Evaluate rows in parallel with a process pool and timeout handling.
#     """
#     scores = []
#     format = []
#     with ProcessPoolExecutor(max_workers=num_processes) as executor:
#         # Create tasks for all rows
#         tasks_async = [
#             process_row_with_timeout(completion, reference, task, executor, timeout=task_timeout)
#             for completion, reference, task in zip(completions, references, tasks)
#         ]
#         # Use tqdm for progress tracking
#         try:
#             results = await asyncio.gather(*tasks_async, return_exceptions=False)
#         except Exception as exc:
#             print('Error encountered in parallel evaluation! now shutting all evaluation workers down to prevent starvation')
#             for pid, proc in executor._processes.items():
#                 try:
#                     proc.kill()
#                 except Exception as kill_err:
#                     print('shut down failed: '+str(kill_err))

#     # Process results
#     for result, completion, reference, task in zip(results, completions, references, tasks):
#         if isinstance(result, Exception) or result is None:
#             # Handle failed or timed-out tasks
#             scores.append(0.0)
#             format.append(0.0)
#             continue

#         try:
#             # Process result based on task type
#             if task == 'code' and not result[0]: # if task is code, the reference should be json string
#                 correct = 0
#                 total = min(
#                     len(json.loads(reference)['inputs'] if not isinstance(reference, dict) else reference['inputs']),
#                     10)
#                 for run in result[1]:
#                     if 'test_case' in run and 'res' in run['test_case'] and run['test_case']['res'] == '[True]':
#                         correct += 1
#                 scores.append(correct / total)
#                 format.append(correct / total)
#             elif task == 'code':
#                 scores.append(float(int(result[0])))
#                 format.append(float(int(result[0])))
#             else:
#                 scores.append(float(int(result[0])))
#                 format.append(float(int(result[1])))
            
#         except Exception as e:
#             print(f"Error processing result for row: {completion[:10]}, Error: {e}")
#             scores.append(0.0)
#             format.append(0.0)
            
#     return scores, format

# def compute_score(completions, references, tasks):
#     # three lists should have identical length
#     # TODO: make this one completely asynchronous, which means the main process can do other things(e.g., forwarding reward model) while computing score
#     assert len(completions) == len(references) == len(tasks)
#     try:
#         return asyncio.run(parallel_evaluate_continual_async(completions, references, tasks, num_processes=64))
#     except asyncio.TimeoutError as e:
#         print('Global timeout in reward computing! Setting all as 0.5.')
#         return [0.5 for _ in range(len(completions))], [0.5 for _ in range(len(completions))]
#     except Exception as e:
#         print(f"Unexpected error: {e}")
#         return [0.5 for _ in range(len(completions))], [0.5 for _ in range(len(completions))]


import asyncio
import json
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Optional

import math

from .evaluation_utils.code_util import evaluate_code
from .evaluation_utils.math_util import evaluate_math
from .evaluation_utils.libero_util import evaluate_libero
from tqdm.asyncio import tqdm

def process_completion(completion, task, reference):
    if task == "code":
        return evaluate_code(completion, reference)
    elif task == "math":
        return evaluate_math(completion, str(reference))
    elif task == "libero":
        return evaluate_libero(completion)
    else:
        print('task')
        raise NotImplementedError

async def process_row_with_timeout(completion, reference, task, executor, timeout=30.0):
    """
    Process a single row with a timeout.
    """
    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                partial(process_completion, completion, task, reference)
            ),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        print(f"Timeout occurred for completion: {completion[:10]}, reference: {reference[:10]}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"Error processing completion: {completion[:10]}, Error: {e}")
        return None  # Default value for failed rows

async def parallel_evaluate_continual_async(completions, references, tasks, num_processes, task_timeout=600.0):
    """
    Evaluate rows in parallel with a process pool and timeout handling.
    """
    scores = []
    format = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Create tasks for all rows
        tasks_async = [
            process_row_with_timeout(completion, reference, task, executor, timeout=task_timeout)
            for completion, reference, task in zip(completions, references, tasks)
        ]
        # Use tqdm for progress tracking
        try:
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except Exception as exc:
            print('Error encountered in parallel evaluation! now shutting all evaluation workers down to prevent starvation')
            for pid, proc in executor._processes.items():
                try:
                    proc.kill()
                except Exception as kill_err:
                    print('shut down failed: '+str(kill_err))

    # Process results
    for result, completion, reference, task in zip(results, completions, references, tasks):
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append(0.0)
            format.append(0.0)
            continue

        try:
            # Process result based on task type
            if task == 'code' and not result[0]: # if task is code, the reference should be json string
                correct = 0
                total = min(
                    len(json.loads(reference)['inputs'] if not isinstance(reference, dict) else reference['inputs']),
                    10)
                for run in result[1]:
                    if 'test_case' in run and 'res' in run['test_case'] and run['test_case']['res'] == '[True]':
                        correct += 1
                scores.append(correct / total)
                #format.append(correct / total)
                format.append(float(int(result[2])))
            elif task == 'code':
                scores.append(float(int(result[0])))
                format.append(float(int(result[2])))
            else:
                scores.append(float(int(result[0])))
                format.append(float(int(result[1])))
            
        except Exception as e:
            print(f"Error processing result for row: {completion[:10]}, Error: {e}")
            scores.append(0.0)
            format.append(0.0)
            
    return scores, format

def compute_score(completions, references, tasks):
    # three lists should have identical length
    # TODO: make this one completely asynchronous, which means the main process can do other things(e.g., forwarding reward model) while computing score
    assert len(completions) == len(references) == len(tasks)
    try:
        return asyncio.run(parallel_evaluate_continual_async(completions, references, tasks, num_processes=128))
    except asyncio.TimeoutError as e:
        print('Global timeout in reward computing! Setting all as 0.5.')
        return [0.5 for _ in range(len(completions))], [0.5 for _ in range(len(completions))]
    except Exception as e:
        print(f"Unexpected error: {e}")
        return [0.5 for _ in range(len(completions))], [0.5 for _ in range(len(completions))]