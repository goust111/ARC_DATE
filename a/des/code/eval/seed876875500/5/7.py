from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
GREEN = 3

MIN_HEIGHT = 7
MAX_HEIGHT = 13
MIN_WIDTH = 15
MAX_WIDTH = 30
MIN_TREES = 2
MAX_TREES = 4
MIN_TRUNK_HEIGHT = 4
MAX_TRUNK_HEIGHT = 8
MAX_BRANCH_LEN = 3
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        coords: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            coords.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc, height, width):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(coords)
    return comps


def _tree_info(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, object]:
    cells = {(int(r), int(c)) for r, c in coords}
    seed_row = max(r for r, _ in cells)
    seed_cells = [(r, c) for r, c in cells if r == seed_row]
    assert len(seed_cells) == 1, "each tree must have one seed pixel"
    seed_r, seed_c = seed_cells[0]
    seed_color = int(grid[seed_r, seed_c])
    assert seed_color in (GREEN, RED), "seed must be green or red"

    trunk_top = seed_r
    while trunk_top - 1 >= 0 and (trunk_top - 1, seed_c) in cells:
        trunk_top -= 1
    trunk_height = seed_r - trunk_top + 1
    assert trunk_height >= 2, "trunk must be at least two cells tall"

    covered = {(r, seed_c) for r in range(trunk_top, seed_r + 1)}
    assert covered <= cells, "trunk must be contiguous"

    branches: dict[int, tuple[int, int]] = {}
    for row in range(trunk_top, seed_r + 1):
        left_len = 0
        col = seed_c - 1
        while (row, col) in cells:
            left_len += 1
            col -= 1

        right_len = 0
        col = seed_c + 1
        while (row, col) in cells:
            right_len += 1
            col += 1

        assert not (left_len and right_len), "a branch height cannot split both ways"
        level = seed_r - row
        if left_len:
            branches[level] = (-1, left_len)
            for step in range(1, left_len + 1):
                covered.add((row, seed_c - step))
        elif right_len:
            branches[level] = (1, right_len)
            for step in range(1, right_len + 1):
                covered.add((row, seed_c + step))

    assert covered == cells, "tree pixels must be trunk cells or horizontal branches"

    non_seed_colors = {int(grid[r, c]) for r, c in cells if (r, c) != (seed_r, seed_c)}
    assert non_seed_colors <= {GREEN}, "only the seed may be red"
    return {
        "coords": coords,
        "cells": cells,
        "seed_row": seed_r,
        "seed_col": seed_c,
        "seed_color": seed_color,
        "trunk_top": trunk_top,
        "trunk_height": trunk_height,
        "branches": branches,
    }


def _draw_tree(
    grid: np.ndarray,
    seed_row: int,
    seed_col: int,
    trunk_height: int,
    branches: dict[int, tuple[int, int]],
    seed_color: int,
) -> None:
    trunk_top = seed_row - trunk_height + 1
    grid[trunk_top : seed_row + 1, seed_col] = GREEN
    grid[seed_row, seed_col] = seed_color
    for level, (direction, length) in branches.items():
        row = seed_row - level
        for step in range(1, length + 1):
            grid[row, seed_col + direction * step] = GREEN


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align every non-template tree to the branch pattern of the red-seeded template tree.

    The input contains several green trees on a black background. Each tree has a
    vertical trunk grown from a bottom seed and optional horizontal branches. The
    unique tree whose seed is red is the template. Every other tree is rewritten so
    that its relative branch heights, directions, and lengths exactly match the
    template, while keeping its own trunk placement and green seed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    colors = set(map(int, np.unique(g)))
    assert colors <= {BG, RED, GREEN}, "unexpected colors in grid"
    comps = _components(g != BG)
    assert len(comps) >= 2, "expected at least two trees"

    trees = [_tree_info(g, comp) for comp in comps]
    template_trees = [tree for tree in trees if int(tree["seed_color"]) == RED]
    assert len(template_trees) == 1, "expected exactly one red-seeded template tree"
    template = template_trees[0]
    trunk_height = int(template["trunk_height"])
    assert all(int(tree["trunk_height"]) == trunk_height for tree in trees), "all trunks must share one height"

    template_branches = dict(template["branches"])
    out = g.copy()
    for tree in trees:
        if tree is template:
            continue
        for r, c in tree["coords"]:
            out[int(r), int(c)] = BG
        _draw_tree(
            out,
            seed_row=int(tree["seed_row"]),
            seed_col=int(tree["seed_col"]),
            trunk_height=trunk_height,
            branches=template_branches,
            seed_color=int(tree["seed_color"]),
        )

    return out


def _random_branch(direction: int | None = None, length: int | None = None) -> tuple[int, int]:
    if direction is None:
        direction = random.choice([-1, 1])
    if length is None:
        length = random.randint(1, MAX_BRANCH_LEN)
    return int(direction), int(length)


def _sample_template(levels: list[int]) -> dict[int, tuple[int, int]]:
    branch_count = random.randint(1, len(levels) - 1)
    chosen = random.sample(levels, branch_count)
    return {level: _random_branch() for level in chosen}


def _replace_branch(spec: tuple[int, int]) -> tuple[int, int]:
    direction, length = spec
    options = []
    if length > 1:
        options.append((direction, random.randint(1, length - 1)))
    if length < MAX_BRANCH_LEN:
        options.append((direction, random.randint(length + 1, MAX_BRANCH_LEN)))
    options.append((-direction, random.randint(1, MAX_BRANCH_LEN)))
    options = [opt for opt in options if opt != spec]
    return random.choice(options)


def _mutate_pattern(
    template: dict[int, tuple[int, int]], levels: list[int], mode: str
) -> dict[int, tuple[int, int]]:
    pattern = dict(template)
    present = list(template)
    absent = [level for level in levels if level not in template]

    if mode == "missing":
        remove_count = random.randint(1, len(present))
        for level in random.sample(present, remove_count):
            pattern.pop(level, None)
        return pattern

    if mode == "extra":
        add_count = random.randint(1, len(absent))
        for level in random.sample(absent, add_count):
            pattern[level] = _random_branch()
        return pattern

    if mode == "replace":
        change_count = random.randint(1, len(present))
        for level in random.sample(present, change_count):
            pattern[level] = _replace_branch(template[level])
        return pattern

    if mode == "mixed":
        applied = False
        if present and random.random() < 0.75:
            remove_count = random.randint(1, len(present))
            for level in random.sample(present, remove_count):
                pattern.pop(level, None)
            applied = True
        remaining_present = [level for level in template if level in pattern]
        if remaining_present and random.random() < 0.75:
            change_count = random.randint(1, len(remaining_present))
            for level in random.sample(remaining_present, change_count):
                pattern[level] = _replace_branch(template[level])
            applied = True
        remaining_absent = [level for level in levels if level not in pattern]
        if remaining_absent and random.random() < 0.75:
            add_count = random.randint(1, len(remaining_absent))
            for level in random.sample(remaining_absent, add_count):
                pattern[level] = _random_branch()
            applied = True
        if applied and pattern != template:
            return pattern
        return _mutate_pattern(template, levels, random.choice(["missing", "extra", "replace"]))

    raise AssertionError(f"unknown mode: {mode}")


def _tree_extents(branches: dict[int, tuple[int, int]]) -> tuple[int, int]:
    left = 0
    right = 0
    for direction, length in branches.values():
        if direction < 0:
            left = max(left, length)
        else:
            right = max(right, length)
    return left, right


def _expected_output(inp: np.ndarray) -> np.ndarray:
    return main(inp)


def _sample_specs() -> list[dict[str, object]] | None:
    tree_count = random.randint(MIN_TREES, MAX_TREES)
    trunk_height = random.randint(MIN_TRUNK_HEIGHT, MAX_TRUNK_HEIGHT)
    height_min = max(MIN_HEIGHT, trunk_height + 1)
    height_max = max(height_min, min(MAX_HEIGHT, trunk_height + 4))
    height = random.randint(height_min, height_max)

    levels = list(range(1, trunk_height))
    if len(levels) < 2:
        return None
    template_branches = _sample_template(levels)
    template_left, template_right = _tree_extents(template_branches)

    modes = ["missing", "extra", "replace", "mixed"]
    random.shuffle(modes)
    non_template_count = tree_count - 1
    chosen_modes = [modes[i % len(modes)] for i in range(non_template_count)]
    random.shuffle(chosen_modes)

    specs: list[dict[str, object]] = []
    for mode in chosen_modes:
        pattern = _mutate_pattern(template_branches, levels, mode)
        own_left, own_right = _tree_extents(pattern)
        specs.append(
            {
                "seed_color": GREEN,
                "seed_row": height - 1 - random.randint(0, 1),
                "trunk_height": trunk_height,
                "input_branches": pattern,
                "reserve_left": max(template_left, own_left),
                "reserve_right": max(template_right, own_right),
            }
        )

    template_spec = {
        "seed_color": RED,
        "seed_row": height - 1 - random.randint(0, 1),
        "trunk_height": trunk_height,
        "input_branches": template_branches,
        "reserve_left": template_left,
        "reserve_right": template_right,
    }
    insert_at = random.randint(0, len(specs))
    specs.insert(insert_at, template_spec)

    gap_choices = [random.randint(2, 4) for _ in range(len(specs) - 1)]
    left_margin = random.randint(0, 2)
    right_margin = random.randint(0, 2)

    cols: list[int] = []
    start = left_margin
    for i, spec in enumerate(specs):
        col = start + int(spec["reserve_left"])
        cols.append(col)
        end = col + int(spec["reserve_right"])
        if i + 1 < len(specs):
            start = end + gap_choices[i] + 1
    width = cols[-1] + int(specs[-1]["reserve_right"]) + right_margin + 1
    if not (MIN_WIDTH <= width <= MAX_WIDTH):
        return None

    for spec, col in zip(specs, cols):
        spec["seed_col"] = col

    if not any(spec["seed_color"] == GREEN and spec["input_branches"] != template_branches for spec in specs):
        return None

    for spec in specs:
        seed_row = int(spec["seed_row"])
        if seed_row - trunk_height + 1 < 0:
            return None

    grid = np.full((height, width), BG, dtype=int)
    for spec in specs:
        _draw_tree(
            grid,
            seed_row=int(spec["seed_row"]),
            seed_col=int(spec["seed_col"]),
            trunk_height=trunk_height,
            branches=dict(spec["input_branches"]),
            seed_color=int(spec["seed_color"]),
        )

    out = _expected_output(grid)
    for spec in specs:
        if int(spec["seed_color"]) == RED:
            continue
        expected = template_branches
        actual = dict(spec["input_branches"])
        if actual == expected:
            return None

    return grid if np.array_equal(main(grid), out) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _sample_specs()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid tree-alignment input")
