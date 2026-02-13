# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements the evaluator for the circle packing problem on a rectangle
# of perimeter 4.
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
import time
import numpy as np
import sys
import json
import os
from importlib import __import__

BENCHMARK = 2.3658321334167627
NUM_CIRCLES = 21
TOL = 1e-6
HARD_TIMEOUT = 60


def minimum_circumscribing_rectangle(circles: np.ndarray):
    """Returns the width and height of the minimum circumscribing rectangle.

    Args:
    circles: A numpy array of shape (num_circles, 3), where each row is of the
        form (x, y, radius), specifying a circle.

    Returns:
    A tuple (width, height) of the minimum circumscribing rectangle.
    """
    min_x = np.min(circles[:, 0] - circles[:, 2])
    max_x = np.max(circles[:, 0] + circles[:, 2])
    min_y = np.min(circles[:, 1] - circles[:, 2])
    max_y = np.max(circles[:, 1] + circles[:, 2])
    return max_x - min_x, max_y - min_y


def validate_packing_radii(radii: np.ndarray) -> None:
    n = len(radii)
    for i in range(n):
        if radii[i] < 0:
            raise ValueError(f"Circle {i} has negative radius {radii[i]}")
        elif np.isnan(radii[i]):
            raise ValueError(f"Circle {i} has nan radius")


def validate_packing_overlap_wtol(circles: np.ndarray, tol: float = 1e-6) -> None:
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((circles[i, :2] - circles[j, :2]) ** 2))
            if dist < circles[i, 2] + circles[j, 2] - tol:
                raise ValueError(
                    f"Circles {i} and {j} overlap: dist={dist}, r1+r2={circles[i,2]+circles[j,2]}"
                )


def validate_packing_inside_rect_wtol(circles: np.array, tol: float = 1e-6) -> None:
    width, height = minimum_circumscribing_rectangle(circles)
    if width + height > (2 + tol):
        raise ValueError("Circles are not contained inside a rectangle of perimeter 4.")


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
            circles = program.circle_packing21()
            eval_time = time.time() - start_time
        except Exception as err:
            result_queue.put({"combined_score": 0.0, "error": str(err)})
            return
        finally:
            if program_dir in sys.path:
                sys.path.remove(program_dir)

        if not isinstance(circles, np.ndarray):
            circles = np.array(circles)

        if circles.shape != (NUM_CIRCLES, 3):
            raise ValueError(
                f"Invalid shapes: circles = {circles.shape}, expected {(NUM_CIRCLES, 3)}"
            )
        assert bool(np.isnan(circles).any()) is False, "nan entry found in answer!"
        validate_packing_radii(circles[:, -1])
        validate_packing_overlap_wtol(circles, TOL)
        validate_packing_inside_rect_wtol(circles, TOL)

        radii_sum = np.sum(circles[:, -1])

        result_queue.put({
            "combined_score": float(radii_sum / BENCHMARK),
            "radii_sum": float(radii_sum),
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

