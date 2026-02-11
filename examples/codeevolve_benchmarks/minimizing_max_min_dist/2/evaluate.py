# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements the evaluator for problem of minimizing the ratio of maximum
# to minimum distance on dimension 2 and with 16 points.
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
from importlib import __import__
import scipy as sp
import time
import numpy as np
import json

NUM_POINTS = 16
DIMENSION = 2
BENCHMARK = 1 / 12.889266112
HARD_TIMEOUT = 60


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
            points = program.min_max_dist_dim2_16()
            eval_time = time.time() - start_time
        except Exception as err:
            result_queue.put({"combined_score": 0.0, "error": str(err)})
            return
        finally:
            if program_dir in sys.path:
                sys.path.remove(program_dir)

        if not isinstance(points, np.ndarray):
            points = np.array(points)

        if points.shape != (NUM_POINTS, DIMENSION):
            raise ValueError(
                f"Invalid shapes: points = {points.shape}, expected {(NUM_POINTS, DIMENSION)}"
            )

        pairwise_distances = sp.spatial.distance.pdist(points)
        min_distance = np.min(pairwise_distances)
        max_distance = np.max(pairwise_distances)

        inv_ratio_squared = (min_distance / max_distance) ** 2 if max_distance > 0 else 0

        result_queue.put({
            "combined_score": float(inv_ratio_squared / BENCHMARK),
            "min_max_ratio": float(inv_ratio_squared),
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