from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BLACK = 0
GREY = 5
PINK = 6
BAR_COLORS = (1, 2, 3, 4, 7, 8, 9)

GRID_ROWS_RANGE = (21, 24)
GRID_COLS_RANGE = (19, 24)
PAIR_COUNT_CHOICES = (4, 5)
SELECTED_COUNT_CHOICES = (2, 3)
BAR_HEIGHT_CHOICES = (2, 3, 4)
BAR_GAP_RANGE = (3, 6)
GEN_RETRIES = 500
LAYOUT_RETRIES = 150
PATH_RETRIES = 120


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels = np.zeros(mask.shape, dtype=int)
    comp_id = 0
    height, width = mask.shape
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            comp_id += 1
            labels[r, c] = comp_id
            stack = [(r, c)]
            while stack:
                rr, cc = stack.pop()
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = comp_id
                        stack.append((nr, nc))
    return labels, comp_id


def _find_vertical_pairs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(color) for color in np.unique(g) if color not in (BLACK, GREY))
    assert colors, "expected colored vertical bar pairs"
    assert PINK not in colors, "pink is reserved for overlap cells in the output"

    pairs: list[dict] = []
    for color in colors:
        coords = np.argwhere(g == color)
        cols = sorted(set(int(c) for c in coords[:, 1]))
        assert len(cols) == 2, "each source color must form exactly two vertical bars"

        bars = []
        for col in cols:
            rows = sorted(int(r) for r in coords[coords[:, 1] == col, 0])
            assert rows == list(range(rows[0], rows[-1] + 1)), "bar pixels must be vertically contiguous"
            bars.append(
                {
                    "col": int(col),
                    "r0": int(rows[0]),
                    "r1": int(rows[-1]),
                    "cells": [(int(r), int(col)) for r in rows],
                }
            )

        assert bars[0]["r0"] == bars[1]["r0"] and bars[0]["r1"] == bars[1]["r1"], "paired bars must share one row interval"
        pairs.append(
            {
                "color": int(color),
                "r0": int(bars[0]["r0"]),
                "r1": int(bars[0]["r1"]),
                "height": int(bars[0]["r1"] - bars[0]["r0"] + 1),
                "left_col": int(bars[0]["col"]),
                "right_col": int(bars[1]["col"]),
                "bars": bars,
            }
        )
    return pairs


def _connected_pair_indices(grid: np.ndarray, pairs: list[dict]) -> list[int]:
    g = np.asarray(grid, dtype=int)
    labels, n_comp = _label_components(g != BLACK)
    comp_has_grey = [False] * (n_comp + 1)
    for r, c in np.argwhere(g == GREY):
        comp_has_grey[int(labels[r, c])] = True

    selected = []
    for idx, pair in enumerate(pairs):
        left_id = int(labels[pair["r0"], pair["left_col"]])
        right_id = int(labels[pair["r0"], pair["right_col"]])
        if left_id != 0 and left_id == right_id and comp_has_grey[left_id]:
            selected.append(idx)
    return selected


def _outward_offset(row: int, r0: int, r1: int, step: int) -> int:
    if row < r0:
        return ((r0 - 1 - row) // step) + 1
    if row > r1:
        return ((row - r1 - 1) // step) + 1
    return 0


def _render_output(grid: np.ndarray, pairs: list[dict], selected: list[int], step: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape

    claims: dict[tuple[int, int], set[int]] = defaultdict(set)
    for idx in selected:
        pair = pairs[idx]
        for row in range(height):
            offset = _outward_offset(row, pair["r0"], pair["r1"], step)
            left = pair["left_col"] - offset
            right = pair["right_col"] + offset
            if 0 <= left < width:
                claims[(row, left)].add(int(pair["color"]))
            if 0 <= right < width:
                claims[(row, right)].add(int(pair["color"]))

    out = np.full_like(g, BLACK)
    for (row, col), colors in claims.items():
        out[row, col] = next(iter(colors)) if len(colors) == 1 else PINK
    out[g == GREY] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the vertical bar pairs jointly connected to the grey path network, then extend them with a step size equal to the number of connected pairs.

    Input structure:
    - The grid has a black background, one grey 8-connected path network, and 4-5 same-color vertical bar pairs.
    - Each non-grey color appears as exactly two 1-pixel-wide vertical bars of the same height and on distinct columns.
    - Some bars touch the grey network; others remain isolated from it.

    Rule:
    - Treat all non-black cells as an 8-connected graph. A bar pair is active iff its two bars lie in the same connected component and that component contains grey path cells.
    - Erase every inactive pair.
    - Let N be the number of active pairs.
    - For each active pair, keep its original two columns inside the bar row interval. Above and below that interval, every N rows the left bar shifts one more column left and the right bar shifts one more column right.
    - Paint all in-bounds emitted cells in the pair's color; cells emitted by multiple different colors become pink.
    - Restore the grey path network unchanged.

    Key insight:
    - The grey path has a double role: it filters which bar pairs survive, and the number of survivors becomes the global stepping period that changes every surviving pair's expansion geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pairs = _find_vertical_pairs(g)
    selected = _connected_pair_indices(g, pairs)
    assert selected, "expected at least one grey-connected pair"
    return _render_output(g, pairs, selected, len(selected))


def _bar_cells(spec: dict) -> set[tuple[int, int]]:
    return {(r, spec["left_col"]) for r in range(spec["r0"], spec["r1"] + 1)} | {
        (r, spec["right_col"]) for r in range(spec["r0"], spec["r1"] + 1)
    }


def _all_bar_cells(specs: list[dict]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for spec in specs:
        cells |= _bar_cells(spec)
    return cells


def _candidate_valid(spec: dict, specs: list[dict]) -> bool:
    cand = _bar_cells(spec)
    for other in specs:
        other_cells = _bar_cells(other)
        for r, c in cand:
            for rr, cc in other_cells:
                if max(abs(r - rr), abs(c - cc)) <= 1:
                    return False
    return True


def _random_pair(height: int, width: int, bar_height: int) -> dict:
    gap = random.randint(*BAR_GAP_RANGE)
    left = random.randint(2, width - gap - 4)
    row0 = random.randint(2, height - bar_height - 3)
    return {
        "r0": row0,
        "r1": row0 + bar_height - 1,
        "left_col": left,
        "right_col": left + gap,
    }


def _sample_specs(height: int, width: int, pair_count: int, selected_count: int, bar_height: int, want_overlap: bool):
    for _ in range(LAYOUT_RETRIES):
        specs: list[dict] = []

        if want_overlap:
            row0 = random.randint(3, height - bar_height - 4)
            gap1 = random.randint(3, 5)
            gap2 = random.randint(3, 5)
            max_left = width - (gap1 + gap2 + 9)
            if max_left < 3:
                continue
            left1 = random.randint(3, max_left)
            spec_a = {"r0": row0, "r1": row0 + bar_height - 1, "left_col": left1, "right_col": left1 + gap1}
            spec_b = {
                "r0": row0,
                "r1": row0 + bar_height - 1,
                "left_col": spec_a["right_col"] + 2,
                "right_col": spec_a["right_col"] + 2 + gap2,
            }
            if spec_b["right_col"] > width - 3:
                continue
            specs.extend([spec_a, spec_b])

        while len(specs) < pair_count:
            spec = _random_pair(height, width, bar_height)
            if _candidate_valid(spec, specs):
                specs.append(spec)

        if len(specs) != pair_count:
            continue

        if want_overlap:
            selected = list(range(selected_count))
        else:
            selected = sorted(random.sample(range(pair_count), selected_count))

        if want_overlap and not {0, 1}.issubset(selected):
            continue

        return specs, selected
    return None, None


def _build_path(height: int, width: int, specs: list[dict], selected: list[int], decoy_idx: int | None):
    bar_id_to_cells: dict[tuple[int, int], set[tuple[int, int]]] = {}
    bar_touch_map: dict[tuple[int, int], set[tuple[int, int]]] = {}
    blocked = np.zeros((height, width), dtype=bool)
    for idx, spec in enumerate(specs):
        for which_bar, col in enumerate((spec["left_col"], spec["right_col"])):
            bar_id = (idx, which_bar)
            cells = {(row, col) for row in range(spec["r0"], spec["r1"] + 1)}
            bar_id_to_cells[bar_id] = cells
            for row, col0 in cells:
                blocked[row, col0] = True
                for nr, nc in _neighbors8(row, col0):
                    if 0 <= nr < height and 0 <= nc < width:
                        blocked[nr, nc] = True
                        bar_touch_map.setdefault((nr, nc), set()).add(bar_id)

    target_bars: list[tuple[int, int]] = []
    for idx in selected:
        target_bars.append((idx, 0))
        target_bars.append((idx, 1))
    if decoy_idx is not None:
        target_bars.append((decoy_idx, random.randint(0, 1)))

    def anchor_options(bar_id: tuple[int, int]) -> list[tuple[int, int]]:
        options = []
        for row, col in sorted(bar_id_to_cells[bar_id]):
            for nr, nc in _neighbors4(row, col):
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                if (nr, nc) in bar_id_to_cells[bar_id]:
                    continue
                if bar_touch_map.get((nr, nc), set()) == {bar_id}:
                    options.append((nr, nc))
        random.shuffle(options)
        return list(dict.fromkeys(options))

    def bfs_to_path(start: tuple[int, int], targets: set[tuple[int, int]], allowed: np.ndarray):
        queue = [start]
        prev = {start: None}
        head = 0
        while head < len(queue):
            cell = queue[head]
            head += 1
            if cell in targets:
                path = []
                cur = cell
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                return path[::-1]
            nbs = list(_neighbors4(*cell))
            random.shuffle(nbs)
            for nr, nc in nbs:
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                if (nr, nc) in prev:
                    continue
                if not allowed[nr, nc] and (nr, nc) not in targets:
                    continue
                prev[(nr, nc)] = cell
                queue.append((nr, nc))
        return None

    for _ in range(PATH_RETRIES):
        chosen_anchors = {}
        ok = True
        for bar_id in target_bars:
            options = [cell for cell in anchor_options(bar_id) if cell not in chosen_anchors.values()]
            if not options:
                ok = False
                break
            options.sort(key=lambda cell: abs(cell[0] - height // 2) + abs(cell[1] - width // 2))
            chosen_anchors[bar_id] = random.choice(options[: min(4, len(options))])
        if not ok:
            continue

        allowed = ~blocked
        for anchor in chosen_anchors.values():
            allowed[anchor] = True

        hub_candidates = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if allowed[r, c]
        ]
        hub_candidates.sort(key=lambda cell: abs(cell[0] - height // 2) + abs(cell[1] - width // 2))
        if not hub_candidates:
            continue
        hub = random.choice(hub_candidates[: min(10, len(hub_candidates))])

        path = {hub}
        anchor_order = sorted(chosen_anchors.values(), key=lambda cell: -abs(cell[0] - hub[0]) - abs(cell[1] - hub[1]))
        for anchor in anchor_order:
            found = bfs_to_path(anchor, path, allowed)
            if found is None:
                ok = False
                break
            path.update(found)
        if not ok:
            continue
        return path
    return None


def _touch_counts(path: set[tuple[int, int]], specs: list[dict]) -> list[int]:
    counts = []
    for spec in specs:
        bar_counts = 0
        for bar_col in (spec["left_col"], spec["right_col"]):
            touched = False
            for row in range(spec["r0"], spec["r1"] + 1):
                for nr, nc in _neighbors8(row, bar_col):
                    if (nr, nc) in path:
                        touched = True
                        break
                if touched:
                    break
            if touched:
                bar_counts += 1
        counts.append(bar_counts)
    return counts


def _specs_to_grid(height: int, width: int, specs: list[dict], colors: list[int], path: set[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for spec, color in zip(specs, colors, strict=True):
        grid[spec["r0"] : spec["r1"] + 1, spec["left_col"]] = int(color)
        grid[spec["r0"] : spec["r1"] + 1, spec["right_col"]] = int(color)
        spec["color"] = int(color)
        spec["height"] = int(spec["r1"] - spec["r0"] + 1)
    for r, c in path:
        grid[r, c] = GREY
    return grid


def _claims_for_pair(pair: dict, step: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    height, width = shape
    claims: set[tuple[int, int]] = set()
    for row in range(height):
        offset = _outward_offset(row, pair["r0"], pair["r1"], step)
        left = pair["left_col"] - offset
        right = pair["right_col"] + offset
        if 0 <= left < width:
            claims.add((row, left))
        if 0 <= right < width:
            claims.add((row, right))
    return claims


def _validate_instance(
    grid: np.ndarray,
    specs: list[dict],
    selected: list[int],
    bar_height: int,
    decoy_idx: int | None,
    want_overlap: bool,
) -> bool:
    try:
        pairs = _find_vertical_pairs(grid)
    except AssertionError:
        return False

    by_color = {pair["color"]: pair for pair in pairs}
    selected_colors = {int(specs[idx]["color"]) for idx in selected}
    detected_selected = _connected_pair_indices(grid, pairs)
    detected_colors = {int(pairs[idx]["color"]) for idx in detected_selected}
    if detected_colors != selected_colors or len(detected_selected) != len(selected):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    touch_counts = _touch_counts(set(map(tuple, np.argwhere(grid == GREY))), specs)
    if any(touch_counts[idx] != 2 for idx in selected):
        return False
    if decoy_idx is not None and touch_counts[decoy_idx] != 1:
        return False
    untouched = [idx for idx in range(len(specs)) if idx not in selected and idx != decoy_idx and touch_counts[idx] == 0]
    if not untouched:
        return False

    pink_count = int(np.count_nonzero(out == PINK))
    if want_overlap and pink_count == 0:
        return False
    if not want_overlap and pink_count != 0:
        return False

    if bar_height == len(selected):
        return False
    wrong_step = _render_output(grid, pairs, detected_selected, bar_height)
    if np.array_equal(out, wrong_step):
        return False

    wrong_keep_all = _render_output(grid, pairs, list(range(len(pairs))), len(selected))
    if np.array_equal(out, wrong_keep_all):
        return False

    for idx in selected:
        pair = by_color[int(specs[idx]["color"])]
        claims = _claims_for_pair(pair, len(selected), grid.shape)
        if not any(row < pair["r0"] or row > pair["r1"] for row, _ in claims):
            return False

    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_ROWS_RANGE)
    width = random.randint(*GRID_COLS_RANGE)
    selected_count = random.choice(SELECTED_COUNT_CHOICES)
    if selected_count == 3:
        pair_count = 5
    else:
        pair_count = random.choice([count for count in PAIR_COUNT_CHOICES if count >= selected_count + 1])
    want_overlap = bool(random.random() < 0.55)
    if selected_count == 3:
        want_overlap = False

    if want_overlap and selected_count < 2:
        return None

    bar_height_choices = [value for value in BAR_HEIGHT_CHOICES if value != selected_count]
    bar_height = random.choice(bar_height_choices)

    specs, selected = _sample_specs(height, width, pair_count, selected_count, bar_height, want_overlap)
    if specs is None or selected is None:
        return None

    remaining = [idx for idx in range(pair_count) if idx not in selected]
    can_have_decoy = selected_count == 2 and len(remaining) >= 2
    decoy_idx = random.choice(remaining) if can_have_decoy and random.random() < 0.65 else None

    path = _build_path(height, width, specs, selected, decoy_idx)
    if path is None:
        return None

    colors = random.sample(BAR_COLORS, pair_count)
    grid = _specs_to_grid(height, width, specs, colors, path)
    if not _validate_instance(grid, specs, selected, bar_height, decoy_idx, want_overlap):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-filtered staircase input")
