# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements the evaluator for the second autocorrelation inequality problem.
#
# ===--------------------------------------------------------------------------------------===#
#
# Some of the code in this file is adapted from:
#
# google-deepmind/alphaevolve_results:
# Licensed under the Apache License v2.0.
#
# ===--------------------------------------------------------------------------------------===#

import multiprocessing
import sys
import os
import time
import json
import numpy as np
from importlib import __import__

BENCHMARK = 0.962
HARD_TIMEOUT = 60

def verify_c2_solution(f_values: np.ndarray):
    """
    Verifies the C2 lower bound solution using the rigorous, unitless, piecewise linear integral method.
    """
    n_points = len(f_values)
    if n_points == 0 or f_values is None:
        raise ValueError("Received empty function values.")
    if f_values.shape != (n_points,):
        raise ValueError(f"Expected function values shape {(n_points,)}. Got {f_values.shape}.")
    if np.any(f_values < -1e-6):  # Allow for small floating point errors
        raise ValueError("Function must be non-negative.")
    
    f_nonneg = np.maximum(f_values, 0.0)
    
    # The raw, unscaled convolution is used
    convolution = np.convolve(f_nonneg, f_nonneg, mode="full")
    
    # Calculate the L2-norm squared: ||f*f||_2^2 via piecewise linear integration
    num_conv_points = len(convolution)
    x_points = np.linspace(-0.5, 0.5, num_conv_points + 2)
    x_intervals = np.diff(x_points)
    y_points = np.concatenate(([0], convolution, [0]))
    
    l2_norm_squared = 0.0
    for i in range(len(convolution) + 1):
        y1, y2, h = y_points[i], y_points[i + 1], x_intervals[i]
        interval_l2_squared = (h / 3) * (y1**2 + y1 * y2 + y2**2)
        l2_norm_squared += interval_l2_squared
    
    # Calculate the L1-norm: ||f*f||_1
    # This is an approximation of the integral of the absolute value of the autoconvolution
    norm_1 = np.sum(np.abs(convolution)) / (len(convolution) + 1)
    
    # Calculate the infinity-norm: ||f*f||_inf
    norm_inf = np.max(np.abs(convolution))
    
    # Check for division by zero
    if norm_1 * norm_inf < 1e-12:
        raise ValueError(f"Norm product too close to zero: norm_1={norm_1}, norm_inf={norm_inf}")
    
    computed_c2 = l2_norm_squared / (norm_1 * norm_inf)
    return computed_c2


def _run_in_subprocess(program_path: str, result_queue: multiprocessing.Queue):
    """Run the evaluation inside a subprocess so it can be hard-killed on timeout."""
    try:
        abs_program_path = os.path.abspath(program_path)
        program_dir = os.path.dirname(abs_program_path)
        module_name = os.path.splitext(os.path.basename(program_path))[0]

        try:
            sys.path.insert(0, program_dir)
            program = __import__(module_name)
            start_time = time.time()
            f_values_list = program.construct_function()
            eval_time = time.time() - start_time

            # Convert to numpy array
            if not isinstance(f_values_list, (list, np.ndarray)):
                raise ValueError(f"construct_function must return list or np.ndarray, got {type(f_values_list)}")
            f_values = np.array(f_values_list, dtype=float)

        except Exception as err:
            result_queue.put({"combined_score": 0.0, "error": str(err)})
            return
        finally:
            if program_dir in sys.path:
                sys.path.remove(program_dir)

        c2 = verify_c2_solution(f_values)

        result_queue.put({
            "combined_score": float(c2) / BENCHMARK,
            "c2": float(c2),
            "eval_time": float(eval_time),
        })
    except Exception as e:
        result_queue.put({"combined_score": 0.0, "error": str(e)})


def evaluate(program_path: str):
    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_run_in_subprocess, args=(program_path, result_queue))
    proc.start()
    proc.join(timeout=HARD_TIMEOUT)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return {"combined_score": 0.0, "error": f"Hard timeout after {HARD_TIMEOUT}s"}

    if not result_queue.empty():
        return result_queue.get_nowait()

    return {"combined_score": 0.0, "error": f"Subprocess exited with code {proc.exitcode} (no result)"}