from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg


Block = dict
Coord = tuple[int, int]

BG = Color.BLACK
START_COLOR = Color.GREY
NONSTART_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GRID_MIN = 18
GRID_MAX = 29
BLOCK_SIZES = (1, 2, 3)
PATH_LENGTHS = (4, 5, 6)
MAX_GENERATE_ATTEMPTS = 500


def _top_left(block: Block) -> Coord:
    r0, c0, _r1, _c1 = block["bbox"]
    return int(r0), int(c0)


def _extract_blocks(grid: np.ndarray) -> tuple[int, Block, list[Block], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = int(get_bg(g))
    blocks = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert len(blocks) >= 3, "expected at least one start block and two colored blocks"

    dims = {(bbox_h(block["bbox"]), bbox_w(block["bbox"])) for block in blocks}
    assert len(dims) == 1, "all blocks must have the same size"
    block_h, block_w = next(iter(dims))

    starts = [block for block in blocks if int(block["color"]) == START_COLOR]
    others = [block for block in blocks if int(block["color"]) != START_COLOR]
    assert len(starts) == 1, "expected exactly one grey start block"
    assert others, "expected at least one non-start block"

    colors = [int(block["color"]) for block in others]
    assert len(colors) == len(set(colors)), "non-start block colors must be unique"
    assert START_COLOR not in colors and bg not in colors

    for block in blocks:
        assert int(block["area"]) == block_h * block_w, "every block must be a solid rectangle"

    return bg, starts[0], others, (int(block_h), int(block_w))


def _visible_step(grid: np.ndarray, bg: int, a: Block, b: Block) -> tuple[str, int] | None:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]

    if ar0 == br0 and ar1 == br1:
        c0 = min(ac1, bc1) + 1
        c1 = max(ac0, bc0) - 1
        if c0 <= c1 and not np.all(grid[ar0 : ar1 + 1, c0 : c1 + 1] == bg):
            return None
        return "h", abs(int(bc0) - int(ac0))

    if ac0 == bc0 and ac1 == bc1:
        r0 = min(ar1, br1) + 1
        r1 = max(ar0, br0) - 1
        if r0 <= r1 and not np.all(grid[r0 : r1 + 1, ac0 : ac1 + 1] == bg):
            return None
        return "v", abs(int(br0) - int(ar0))

    return None


def _visible_candidates(grid: np.ndarray, bg: int, current: Block, unvisited: list[Block]) -> list[tuple[int, str, Block]]:
    candidates: list[tuple[int, str, Block]] = []
    for block in unvisited:
        step = _visible_step(grid, bg, current, block)
        if step is None:
            continue
        axis, distance = step
        candidates.append((int(distance), axis, block))
    return candidates


def _walk_sequence(grid: np.ndarray, bg: int, start: Block, others: list[Block]) -> list[Block]:
    sequence = [start]
    current = start
    unvisited = list(others)

    while True:
        candidates = _visible_candidates(grid, bg, current, unvisited)
        if not candidates:
            break

        min_distance = min(distance for distance, _axis, _block in candidates)
        nearest = [candidate for candidate in candidates if candidate[0] == min_distance]
        assert len(nearest) == 1, "nearest visible unvisited block must be unique"

        _distance, _axis, block = nearest[0]
        sequence.append(block)
        unvisited = [candidate for candidate in unvisited if candidate is not block]
        current = block

    return sequence


def _fill_block(out: np.ndarray, block: Block, color: int) -> None:
    r0, c0, r1, c1 = block["bbox"]
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate block colors along a hidden greedy walk from the grey start block.

    Input structure:
    - The background is black, with several separated solid blocks of one shared size.
    - Exactly one block is grey and serves as the start; every other block has a distinct non-grey color.
    - Two blocks are in the same horizontal or vertical band only when their full row-span or full column-span matches.

    Rule:
    - Start at the grey block.
    - Repeatedly move to the unique nearest unvisited block that is visible in the same row-band or column-band; any colored pixels between two aligned blocks block visibility.
    - Stop when no visible unvisited block remains.
    - Do not draw the walk; instead, cyclically shift colors forward along the visited sequence so each visited block takes its predecessor's color and the start takes the last visited color.
    - Blocks never reached by the walk stay unchanged.

    Key insight:
    - The recoloring order is not spatial sorting but the order of a hidden nearest-visible-neighbor traversal, so the walk must be inferred before any color can be reassigned.
    """
    g = np.asarray(grid, dtype=int)
    bg, start, others, _shape = _extract_blocks(g)
    sequence = _walk_sequence(g, bg, start, others)

    out = g.copy()
    original_colors = [int(block["color"]) for block in sequence]
    shifted_colors = [original_colors[-1], *original_colors[:-1]]
    for block, color in zip(sequence, shifted_colors):
        _fill_block(out, block, color)
    return out


def _path_row_count(length: int, start_axis: str) -> int:
    if start_axis == "h":
        return (length + 1) // 2
    return length // 2 + 1


def _path_col_count(length: int, start_axis: str) -> int:
    if start_axis == "h":
        return length // 2 + 1
    return (length + 1) // 2


def _build_path(rows_path: list[int], cols_path: list[int], start_axis: str, length: int) -> list[Coord]:
    path = [(rows_path[0], cols_path[0])]
    row_index = 0
    col_index = 0
    axis = start_axis
    for _ in range(length - 1):
        if axis == "h":
            col_index += 1
        else:
            row_index += 1
        path.append((rows_path[row_index], cols_path[col_index]))
        axis = "v" if axis == "h" else "h"
    return path


def _render_grid(height: int, width: int, block_size: int, colors_by_pos: dict[Coord, int]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for (r0, c0), color in colors_by_pos.items():
        grid[r0 : r0 + block_size, c0 : c0 + block_size] = int(color)
    return grid


def _make_candidate(rng: random.Random) -> tuple[np.ndarray, dict] | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    block_size = rng.choice(BLOCK_SIZES)
    step = block_size + 1

    row_slots = list(range(1, height - block_size + 1, step))
    col_slots = list(range(1, width - block_size + 1, step))
    if len(row_slots) < 4 or len(col_slots) < 4:
        return None

    feasible = []
    for start_axis in ("h", "v"):
        for length in PATH_LENGTHS:
            rows_needed = _path_row_count(length, start_axis) + 1
            cols_needed = _path_col_count(length, start_axis) + 1
            for isolated in (1, 0):
                if length + 2 + isolated > 9:
                    continue
                if len(row_slots) >= rows_needed + isolated and len(col_slots) >= cols_needed + isolated:
                    feasible.append((start_axis, length, isolated))
    if not feasible:
        return None

    start_axis, length, isolated = rng.choice(feasible)
    rows_path_needed = _path_row_count(length, start_axis)
    cols_path_needed = _path_col_count(length, start_axis)

    if start_axis == "h":
        col_patterns = [(idx, direction) for idx in range(len(col_slots)) for direction in (-1, 1) if 0 <= idx + 3 * direction < len(col_slots)]
        row_patterns = [(idx, direction) for idx in range(len(row_slots)) for direction in (-1, 1) if 0 <= idx + 2 * direction < len(row_slots)]
        if not col_patterns or not row_patterns:
            return None

        col_start_idx, col_direction = rng.choice(col_patterns)
        row_start_idx, row_direction = rng.choice(row_patterns)

        start_col = col_slots[col_start_idx]
        second_col = col_slots[col_start_idx + col_direction]
        hidden_col = col_slots[col_start_idx + 3 * col_direction]

        start_row = row_slots[row_start_idx]
        second_row = row_slots[row_start_idx + row_direction]
        branch_row = row_slots[row_start_idx + 2 * row_direction]

        used_col_indices = {col_start_idx, col_start_idx + col_direction, col_start_idx + 3 * col_direction}
        used_row_indices = {row_start_idx, row_start_idx + row_direction, row_start_idx + 2 * row_direction}
        remaining_cols = [col_slots[idx] for idx in range(len(col_slots)) if idx not in used_col_indices]
        remaining_rows = [row_slots[idx] for idx in range(len(row_slots)) if idx not in used_row_indices]
        extra_cols = rng.sample(remaining_cols, cols_path_needed - 2)
        extra_rows = rng.sample(remaining_rows, rows_path_needed - 2)
        cols_path = [start_col, second_col, *extra_cols]
        rows_path = [start_row, second_row, *extra_rows]
        free_cols = [c for c in remaining_cols if c not in extra_cols]
        free_rows = [r for r in remaining_rows if r not in extra_rows]
        branch = (branch_row, start_col)
        hidden = (start_row, hidden_col)
    else:
        row_patterns = [(idx, direction) for idx in range(len(row_slots)) for direction in (-1, 1) if 0 <= idx + 3 * direction < len(row_slots)]
        col_patterns = [(idx, direction) for idx in range(len(col_slots)) for direction in (-1, 1) if 0 <= idx + 2 * direction < len(col_slots)]
        if not row_patterns or not col_patterns:
            return None

        row_start_idx, row_direction = rng.choice(row_patterns)
        col_start_idx, col_direction = rng.choice(col_patterns)

        start_row = row_slots[row_start_idx]
        second_row = row_slots[row_start_idx + row_direction]
        hidden_row = row_slots[row_start_idx + 3 * row_direction]

        start_col = col_slots[col_start_idx]
        second_col = col_slots[col_start_idx + col_direction]
        branch_col = col_slots[col_start_idx + 2 * col_direction]

        used_row_indices = {row_start_idx, row_start_idx + row_direction, row_start_idx + 3 * row_direction}
        used_col_indices = {col_start_idx, col_start_idx + col_direction, col_start_idx + 2 * col_direction}
        remaining_rows = [row_slots[idx] for idx in range(len(row_slots)) if idx not in used_row_indices]
        remaining_cols = [col_slots[idx] for idx in range(len(col_slots)) if idx not in used_col_indices]
        extra_rows = rng.sample(remaining_rows, rows_path_needed - 2)
        extra_cols = rng.sample(remaining_cols, cols_path_needed - 2)
        rows_path = [start_row, second_row, *extra_rows]
        cols_path = [start_col, second_col, *extra_cols]
        free_rows = [r for r in remaining_rows if r not in extra_rows]
        free_cols = [c for c in remaining_cols if c not in extra_cols]
        branch = (start_row, branch_col)
        hidden = (hidden_row, start_col)

    path = _build_path(rows_path, cols_path, start_axis, length)
    positions = list(path) + [branch, hidden]
    isolated_pos = None
    if isolated:
        if start_axis == "h" and free_rows and free_cols:
            isolated_pos = (free_rows[0], free_cols[0])
        elif start_axis == "v" and free_rows and free_cols:
            isolated_pos = (free_rows[0], free_cols[0])
        else:
            return None
        if isolated_pos in positions:
            return None
        positions.append(isolated_pos)

    if len(set(positions)) != len(positions):
        return None

    shuffled_colors = rng.sample(NONSTART_COLORS, len(positions) - 1)
    colors_by_pos = {path[0]: START_COLOR}
    for pos, color in zip(positions[1:], shuffled_colors):
        colors_by_pos[pos] = color

    grid = _render_grid(height, width, block_size, colors_by_pos)
    meta = {
        "path": path,
        "branch": branch,
        "hidden": hidden,
        "isolated": isolated_pos,
        "start_axis": start_axis,
    }
    return grid, meta


def _validate_candidate(grid: np.ndarray, meta: dict) -> bool:
    try:
        bg, start, others, _shape = _extract_blocks(grid)
        block_map = {_top_left(block): block for block in [start, *others]}
        expected_positions = {
            tuple(pos)
            for pos in [*meta["path"], meta["branch"], meta["hidden"], *([meta["isolated"]] if meta["isolated"] is not None else [])]
        }
        if set(block_map) != expected_positions:
            return False

        sequence = _walk_sequence(grid, bg, start, others)
        actual_path = [_top_left(block) for block in sequence]
        if actual_path != meta["path"]:
            return False

        unvisited_positions = set(block_map) - set(actual_path)
        if meta["branch"] not in unvisited_positions or meta["hidden"] not in unvisited_positions:
            return False

        start_candidates = _visible_candidates(
            grid,
            bg,
            block_map[meta["path"][0]],
            [block_map[pos] for pos in unvisited_positions | {meta["path"][1]} if pos != meta["path"][0]],
        )
        start_visible = {_top_left(block): distance for distance, _axis, block in start_candidates}
        if meta["path"][1] not in start_visible or meta["branch"] not in start_visible:
            return False
        if meta["hidden"] in start_visible:
            return False
        if start_visible[meta["path"][1]] >= start_visible[meta["branch"]]:
            return False

        second_candidates = _visible_candidates(
            grid,
            bg,
            block_map[meta["path"][1]],
            [block_map[pos] for pos in unvisited_positions | {meta["path"][2]} if pos != meta["path"][1]],
        )
        second_visible = {_top_left(block): distance for distance, _axis, block in second_candidates}
        if meta["path"][2] not in second_visible or meta["hidden"] not in second_visible:
            return False
        if second_visible[meta["path"][2]] >= second_visible[meta["hidden"]]:
            return False

        if np.array_equal(main(grid), grid):
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATE_ATTEMPTS):
        candidate = _make_candidate(rng)
        if candidate is None:
            continue
        grid, meta = candidate
        if _validate_candidate(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid hidden-patrol recoloring instance")
