from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "arc_utils_task_6",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()
find_components = _ARC_UTILS.find_components
in_bounds = _ARC_UTILS.in_bounds


BG = 0
BLUE = 1
RED = 2
EDGE_ORDER = ("top", "bottom", "left", "right")

GRID_BOUNDS = (9, 12)
LINE_COUNT_CHOICES = [1, 2]
LINE_COUNT_WEIGHTS = [3, 5]
TRIGGER_CHOICES = ["none", "top", "right", "corner"]
TRIGGER_WEIGHTS = [3, 4, 4, 2]
BAR_COUNT_CHOICES = [0, 1, 2]
BAR_COUNT_WEIGHTS = [3, 5, 4]
GENERATION_ATTEMPTS = 600
PLACEMENT_ATTEMPTS = 80


def _edge_set(trigger: str) -> set[str]:
    if trigger == "top":
        return {"top"}
    if trigger == "right":
        return {"right"}
    if trigger == "corner":
        return {"top", "right"}
    return set()


def _edge_bars(grid: np.ndarray) -> list[str]:
    bars: list[str] = []
    if np.all(grid[0, :] == RED):
        bars.append("top")
    if np.all(grid[-1, :] == RED):
        bars.append("bottom")
    if np.all(grid[:, 0] == RED):
        bars.append("left")
    if np.all(grid[:, -1] == RED):
        bars.append("right")
    return bars


def _apply_full_bar(grid: np.ndarray, edge: str) -> None:
    if edge == "top":
        grid[0, :] = RED
    elif edge == "bottom":
        grid[-1, :] = RED
    elif edge == "left":
        grid[:, 0] = RED
    elif edge == "right":
        grid[:, -1] = RED
    else:
        raise ValueError(f"unknown edge {edge}")


def _grow_existing_bar(grid: np.ndarray, edge: str) -> None:
    if edge == "top" and grid.shape[0] >= 2:
        mask = grid[1, :] != BLUE
        grid[1, mask] = RED
    elif edge == "bottom" and grid.shape[0] >= 2:
        mask = grid[-2, :] != BLUE
        grid[-2, mask] = RED
    elif edge == "left" and grid.shape[1] >= 2:
        mask = grid[:, 1] != BLUE
        grid[mask, 1] = RED
    elif edge == "right" and grid.shape[1] >= 2:
        mask = grid[:, -2] != BLUE
        grid[mask, -2] = RED


def _segment_cells_from_bounds(anti: int, c0: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(int(c0), int(c1) + 1):
        cells.append((int(anti - c), int(c)))
    return cells


def _inspect_blue_segments(grid: np.ndarray) -> list[dict]:
    comps = find_components(grid, bg=BG, colors=[BLUE], connectivity=8, monochromatic=True)
    segments: list[dict] = []
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        sums = coords[:, 0] + coords[:, 1]
        assert np.all(sums == sums[0]), "blue component must lie on a single anti-diagonal"
        anti = int(sums[0])
        cols = np.sort(coords[:, 1])
        c0 = int(cols[0])
        c1 = int(cols[-1])
        expected_cols = np.arange(c0, c1 + 1, dtype=int)
        assert np.array_equal(cols, expected_cols), "blue anti-diagonal segment must be contiguous"
        expected_cells = _segment_cells_from_bounds(anti, c0, c1)
        actual_cells = sorted((int(r), int(c)) for r, c in coords.tolist())
        assert actual_cells == sorted(expected_cells), "blue segment cells must match the implied anti-diagonal span"
        segments.append(
            {
                "anti": anti,
                "c0": c0,
                "c1": c1,
                "length": int(c1 - c0 + 1),
                "cells": expected_cells,
            }
        )
    segments.sort(key=lambda seg: (int(seg["c0"]), int(seg["c1"]), int(seg["anti"])))
    return segments


def _advanced_cells(segment: dict, shape: tuple[int, int]) -> list[tuple[int, int]]:
    anti = int(segment["anti"])
    c1 = int(segment["c1"])
    length = int(segment["length"])
    cells: list[tuple[int, int]] = []
    for c in range(c1 + 1, c1 + length + 2):
        r = anti - c
        if in_bounds(shape, r, c):
            cells.append((int(r), int(c)))
    return cells


def _advanced_endpoint_edges(cells: list[tuple[int, int]], shape: tuple[int, int]) -> list[str]:
    if not cells:
        return []
    endpoints = [cells[0], cells[-1]]
    h, w = map(int, shape)
    edges: list[str] = []
    for r, c in endpoints:
        if r == 0 and "top" not in edges:
            edges.append("top")
        if r == h - 1 and "bottom" not in edges:
            edges.append("bottom")
        if c == 0 and "left" not in edges:
            edges.append("left")
        if c == w - 1 and "right" not in edges:
            edges.append("right")
    return edges


def _expected_from_specs(shape: tuple[int, int], specs: list[dict], existing_bars: list[str]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for edge in existing_bars:
        _apply_full_bar(out, edge)
    for spec in sorted(specs, key=lambda s: (int(s["c0"]), int(s["c1"]), int(s["anti"]))):
        new_cells = list(spec["new_cells"])
        for r, c in spec["old_cells"]:
            out[r, c] = BG
        for r, c in new_cells:
            out[r, c] = BLUE
        for edge in spec["trigger_edges"]:
            _apply_full_bar(out, edge)
    for edge in existing_bars:
        _grow_existing_bar(out, edge)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance each blue anti-diagonal segment and trigger full red edge bars when the advanced endpoint touches a border.

    Input structure:
    - The grid is a black background with one or two blue anti-diagonal segments, each forming one 8-connected line on a single anti-diagonal.
    - Each blue segment stays at least one cell away from the border and has length 2 to 4.
    - Some outer edges may already be full red bars that occupy an entire border row or column.

    Rule:
    - Process blue segments from left to right.
    - Erase each old segment, then draw a longer segment on the same anti-diagonal using columns `c1+1..c1+L+1`, clipped to the grid if needed.
    - If either endpoint of the new segment lies on a grid edge, paint that whole edge red, overwriting everything on that row or column.
    - After all blue segments are processed, each red edge bar that was already present in the input grows inward by one cell, but this inward growth does not overwrite blue pixels.

    Key insight:
    - The blue objects follow a geometric anti-diagonal motion, while the red bars come from a separate trigger: only the advanced endpoint location decides whether a whole border suddenly fills.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    existing_bars = _edge_bars(g)
    segments = _inspect_blue_segments(g)
    assert 1 <= len(segments) <= 2, "expected one or two blue segments"

    out = g.copy()
    for segment in segments:
        for r, c in segment["cells"]:
            out[r, c] = BG
        new_cells = _advanced_cells(segment, g.shape)
        for r, c in new_cells:
            out[r, c] = BLUE
        for edge in _advanced_endpoint_edges(new_cells, g.shape):
            _apply_full_bar(out, edge)

    for edge in existing_bars:
        _grow_existing_bar(out, edge)
    return out


def _touches(existing: set[tuple[int, int]], cells: list[tuple[int, int]], margin: int = 1) -> bool:
    for r, c in cells:
        for dr in range(-margin, margin + 1):
            for dc in range(-margin, margin + 1):
                if (r + dr, c + dc) in existing:
                    return True
    return False


def _build_spec(h: int, w: int, length: int, trigger: str, re: int, ce: int) -> dict:
    old_cells = [(re + length + 1 + k, ce - length - 1 - k) for k in range(length)]
    new_cells = [(re + k, ce - k) for k in range(length + 1)]
    anti = int(re + ce)
    c0 = int(ce - 2 * length)
    c1 = int(ce - length - 1)

    for r, c in old_cells:
        assert 1 <= r <= h - 2 and 1 <= c <= w - 2, "old segment must stay one cell from the border"
    for r, c in new_cells:
        assert in_bounds((h, w), r, c), "new segment must stay in bounds"

    trigger_edges = _advanced_endpoint_edges(new_cells, (h, w))
    assert set(trigger_edges) == _edge_set(trigger), "trigger type and endpoint must agree"
    return {
        "anti": anti,
        "c0": c0,
        "c1": c1,
        "length": int(length),
        "old_cells": old_cells,
        "new_cells": new_cells,
        "trigger": trigger,
        "trigger_edges": trigger_edges,
    }


def _sample_trigger_plan(line_count: int) -> list[str]:
    for _ in range(200):
        plan: list[str] = []
        used_edges: set[str] = set()
        for _idx in range(line_count):
            options = [t for t in TRIGGER_CHOICES if not (_edge_set(t) & used_edges)]
            weights = [TRIGGER_WEIGHTS[TRIGGER_CHOICES.index(t)] for t in options]
            trigger = random.choices(options, weights=weights, k=1)[0]
            plan.append(trigger)
            used_edges |= _edge_set(trigger)
        if used_edges or random.random() < 0.35:
            return plan
    return ["none"] * line_count


def _sample_existing_bars(blocked_edges: set[str]) -> list[str]:
    available = [edge for edge in EDGE_ORDER if edge not in blocked_edges]
    if not available:
        return []
    max_count = min(2, len(available))
    choices = [count for count in BAR_COUNT_CHOICES if count <= max_count]
    weights = BAR_COUNT_WEIGHTS[: len(choices)]
    count = random.choices(choices, weights=weights, k=1)[0]
    if count == 0:
        return []
    return sorted(random.sample(available, count), key=EDGE_ORDER.index)


def _sample_line_spec(h: int, w: int, trigger: str, placed_old: set[tuple[int, int]], placed_new: set[tuple[int, int]]) -> dict | None:
    max_len = min(4, (h - 2) // 2, (w - 2) // 2)
    if max_len < 2:
        return None
    lengths = list(range(2, max_len + 1))

    for _ in range(PLACEMENT_ATTEMPTS):
        length = random.choice(lengths)
        if trigger == "top":
            re_values = [0]
            ce_values = list(range(2 * length + 1, w - 1))
        elif trigger == "right":
            re_values = list(range(1, h - 2 * length - 1))
            ce_values = [w - 1]
        elif trigger == "corner":
            re_values = [0]
            ce_values = [w - 1]
        else:
            re_values = list(range(1, h - 2 * length - 1))
            ce_values = list(range(2 * length + 1, w - 1))

        if not re_values or not ce_values:
            continue
        re = random.choice(re_values)
        ce = random.choice(ce_values)
        spec = _build_spec(h, w, length, trigger, re, ce)

        if _touches(placed_old, spec["old_cells"], margin=1):
            continue
        if _touches(placed_new, spec["new_cells"], margin=1):
            continue
        if any((r + c) == spec["anti"] for r, c in placed_old):
            continue
        return spec
    return None


def _validate_input(grid: np.ndarray, specs: list[dict], existing_bars: list[str]) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    actual_bars = _edge_bars(grid)
    assert actual_bars == existing_bars, "existing red bars must be recognizable from the input border"

    segments = _inspect_blue_segments(grid)
    assert len(segments) == len(specs), "input must expose the intended number of blue segments"
    expected_old_sets = {frozenset(spec["old_cells"]) for spec in specs}
    actual_old_sets = {frozenset(seg["cells"]) for seg in segments}
    assert actual_old_sets == expected_old_sets, "input blue segments must match the generated anti-diagonal spans"

    used_edges: set[str] = set()
    for spec in specs:
        assert not (_edge_set(spec["trigger"]) & used_edges), "different lines must not target the same edge"
        used_edges |= _edge_set(spec["trigger"])


def _validate_output(grid: np.ndarray, out: np.ndarray, specs: list[dict], existing_bars: list[str]) -> None:
    expected = _expected_from_specs(grid.shape, specs, existing_bars)
    assert np.array_equal(out, expected), "main() must match the rule-level expected output"

    for edge in existing_bars:
        if edge == "top" and grid.shape[0] >= 2:
            blue_mask = out[1, :] == BLUE
            assert np.all(out[1, ~blue_mask] == RED), "top bar must grow inward without overwriting blue"
        elif edge == "bottom" and grid.shape[0] >= 2:
            blue_mask = out[-2, :] == BLUE
            assert np.all(out[-2, ~blue_mask] == RED), "bottom bar must grow inward without overwriting blue"
        elif edge == "left" and grid.shape[1] >= 2:
            blue_mask = out[:, 1] == BLUE
            assert np.all(out[~blue_mask, 1] == RED), "left bar must grow inward without overwriting blue"
        elif edge == "right" and grid.shape[1] >= 2:
            blue_mask = out[:, -2] == BLUE
            assert np.all(out[~blue_mask, -2] == RED), "right bar must grow inward without overwriting blue"


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        h = random.randint(*GRID_BOUNDS)
        w = random.randint(*GRID_BOUNDS)
        line_count = random.choices(LINE_COUNT_CHOICES, weights=LINE_COUNT_WEIGHTS, k=1)[0]
        trigger_plan = _sample_trigger_plan(line_count)

        specs: list[dict] = []
        placed_old: set[tuple[int, int]] = set()
        placed_new: set[tuple[int, int]] = set()

        ok = True
        for trigger in trigger_plan:
            spec = _sample_line_spec(h, w, trigger, placed_old, placed_new)
            if spec is None:
                ok = False
                break
            specs.append(spec)
            placed_old |= set(spec["old_cells"])
            placed_new |= set(spec["new_cells"])
        if not ok:
            continue

        blocked_edges = set().union(*(_edge_set(spec["trigger"]) for spec in specs)) if specs else set()
        existing_bars = _sample_existing_bars(blocked_edges)

        grid = np.full((h, w), BG, dtype=int)
        for edge in existing_bars:
            _apply_full_bar(grid, edge)
        for spec in specs:
            for r, c in spec["old_cells"]:
                grid[r, c] = BLUE

        try:
            _validate_input(grid, specs, existing_bars)
            out = main(grid)
            _validate_output(grid, out, specs, existing_bars)
        except AssertionError:
            continue
        return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
