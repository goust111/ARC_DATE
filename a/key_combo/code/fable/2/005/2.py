from __future__ import annotations

from collections import Counter
from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_fable_2_005_2",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()
Color = _ARC_UTILS.Color


_GEN_ATTEMPTS = 400
_GEN_PROFILE_COUNT = 4
_GEN_COLOR_POOL = tuple(color for color in Color.ALL_COLORS if color != Color.BLACK)
_GEN_MAX_TOTAL_DEPTH = len(_GEN_COLOR_POOL)
_GEN_MAX_GROUPS = 5

_GEN_CALL_COUNT = 0
_GEN_SEEN_INPUTS: set[tuple[tuple[int, int], bytes]] = set()
_GEN_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _grid_signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(grid, dtype=int)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _foreground_tokens(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] == 1, "expected a single-row grid"
    return [int(color) for color in g[0] if int(color) != Color.BLACK]


def _parse_groups(tokens: list[int]) -> list[tuple[int, ...]]:
    assert tokens, "expected at least one foreground token"

    counts = Counter(tokens)
    assert all(count == 2 for count in counts.values()), "every foreground color must appear exactly twice"

    groups: list[tuple[int, ...]] = []
    stack: list[int] = []
    current_group: list[int] = []
    closing_started = False

    for color in tokens:
        if color not in counts or color == Color.BLACK:
            raise AssertionError("tokens must be non-black colors")

        if stack and stack[-1] == color:
            stack.pop()
            closing_started = True
            if not stack:
                groups.append(tuple(current_group))
                current_group = []
                closing_started = False
            continue

        if closing_started:
            raise AssertionError("top-level groups may be siblings, but each group must be a single nesting chain")

        if current_group.count(color) != 0:
            raise AssertionError("a color may not reopen after it has closed")

        stack.append(color)
        current_group.append(color)

    assert not stack, "all opened colors must close"
    assert not current_group, "group collection must end cleanly"
    assert groups, "expected at least one top-level group"
    return groups


def _draw_tower(group: tuple[int, ...]) -> np.ndarray:
    depth = len(group)
    side = 2 * depth - 1
    tower = np.full((side, side), Color.BLACK, dtype=int)
    for layer, color in enumerate(group):
        tower[layer : side - layer, layer : side - layer] = int(color)
    return tower


def _render_groups(groups: list[tuple[int, ...]]) -> np.ndarray:
    heights = [2 * len(group) - 1 for group in groups]
    out_h = max(heights)
    out_w = sum(heights)
    out = np.full((out_h, out_w), Color.BLACK, dtype=int)

    c0 = 0
    for group, side in zip(groups, heights):
        tower = _draw_tower(group)
        out[:side, c0 : c0 + side] = tower
        c0 += side
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Parse the color pairs in the single-row input as nested brackets and draw one concentric square tower per top-level group.

    Input structure:
    - The grid is a single row on a black background.
    - Every foreground color appears exactly twice, and identical colors act as the two endpoints of one bracket pair.
    - The pairs are properly nested without crossings, and each top-level group is a single nesting chain while multiple top-level groups may appear side by side.

    Rule:
    - Read the row from left to right and match each color with its second occurrence as a bracket pair.
    - Split the sequence into top-level groups; for one group of depth d, draw a (2d-1)x(2d-1) concentric square whose outer ring through center use that group's colors from outermost pair to innermost pair.
    - Place the resulting squares from left to right in the same top-level group order, top-aligned and touching with no gap; fill all remaining cells with black.

    Key insight:
    - The output layers are determined by bracket nesting depth, not by local symmetry in the line: same-color pairs define a parse tree, and each top-level chain becomes one concentric tower.
    """
    tokens = _foreground_tokens(grid)
    groups = _parse_groups(tokens)
    return _render_groups(groups)


def _random_composition(total: int, parts: int) -> list[int]:
    assert 1 <= parts <= total
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    bounds = [0] + cuts + [total]
    return [bounds[i + 1] - bounds[i] for i in range(parts)]


def _depths_fit(depths: list[int]) -> bool:
    total_depth = sum(depths)
    input_w = 2 * total_depth
    output_h = max(2 * depth - 1 for depth in depths)
    output_w = sum(2 * depth - 1 for depth in depths)
    return (
        depths
        and all(depth >= 1 for depth in depths)
        and total_depth <= _GEN_MAX_TOTAL_DEPTH
        and len(depths) <= _GEN_MAX_GROUPS
        and input_w <= 30
        and output_h <= 30
        and output_w <= 30
    )


def _sample_depths(profile: int) -> list[int]:
    for _ in range(_GEN_ATTEMPTS):
        if profile == 0:
            parts = random.randint(3, 4)
            total = random.randint(max(parts + 2, 5), _GEN_MAX_TOTAL_DEPTH)
            depths = _random_composition(total, parts)
            if 1 not in depths or max(depths) < 3 or depths.index(max(depths)) == 0:
                continue
        elif profile == 1:
            depths = [random.randint(4, 6)]
        elif profile == 2:
            parts = random.randint(3, 5)
            depths = [random.randint(1, 2) for _ in range(parts)]
            if 2 not in depths:
                continue
        else:
            parts = random.randint(1, 4)
            total = random.randint(max(parts, 3), _GEN_MAX_TOTAL_DEPTH)
            depths = _random_composition(total, parts)
            random.shuffle(depths)

        if _depths_fit(depths):
            return depths
    raise RuntimeError("failed to sample valid group depths")


def _sample_groups(depths: list[int]) -> list[tuple[int, ...]]:
    colors = random.sample(_GEN_COLOR_POOL, sum(depths))
    groups: list[tuple[int, ...]] = []
    idx = 0
    for depth in depths:
        group_colors = colors[idx : idx + depth]
        idx += depth
        random.shuffle(group_colors)
        groups.append(tuple(group_colors))
    return groups


def _groups_to_input(groups: list[tuple[int, ...]]) -> np.ndarray:
    tokens: list[int] = []
    for group in groups:
        tokens.extend(int(color) for color in group)
        tokens.extend(int(color) for color in reversed(group))
    return np.asarray([tokens], dtype=int)


def _generator_profile() -> int:
    global _GEN_CALL_COUNT
    profile = _GEN_CALL_COUNT % _GEN_PROFILE_COUNT
    _GEN_CALL_COUNT += 1
    return profile


def _attempt_input(profile: int) -> np.ndarray | None:
    depths = _sample_depths(profile)
    groups = _sample_groups(depths)
    inp = _groups_to_input(groups)
    expected = _render_groups(groups)

    if not np.array_equal(main(inp), expected):
        return None

    inp_sig = _grid_signature(inp)
    out_sig = _grid_signature(expected)
    if inp_sig in _GEN_SEEN_INPUTS or out_sig in _GEN_SEEN_OUTPUTS:
        return None

    _GEN_SEEN_INPUTS.add(inp_sig)
    _GEN_SEEN_OUTPUTS.add(out_sig)
    return inp


def generate_input() -> np.ndarray:
    profile = _generator_profile()
    for _ in range(_GEN_ATTEMPTS):
        grid = _attempt_input(profile)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a fresh valid input for puzzle 2")
