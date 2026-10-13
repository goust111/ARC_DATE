from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
FRAME_COLORS = (1, 2, 3)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_from_component(grid: np.ndarray, color: int, comp: list[tuple[int, int]]) -> dict | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    outer_h = r1 - r0 + 1
    outer_w = c1 - c0 + 1
    if outer_h < 4 or outer_w < 4:
        return None

    expected = {
        (rr, cc)
        for rr in range(r0, r1 + 1)
        for cc in range(c0, c1 + 1)
        if rr in (r0, r1) or cc in (c0, c1)
    }
    comp_set = set(comp)
    if comp_set != expected:
        return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "outer_shape": (outer_h, outer_w),
        "inner_shape": tuple(map(int, inner.shape)),
        "orange_count": int(np.count_nonzero(inner == ORANGE)),
        "grey_mask": inner == GREY,
        "top_left": (r0, c0),
    }


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(v) for v in np.unique(g) if int(v) not in (BG, GREY, ORANGE))
    frames: list[dict] = []
    for color in colors:
        for comp in _color_components(g, color):
            frame = _frame_from_component(g, color, comp)
            if frame is not None:
                frames.append(frame)
    frames.sort(key=lambda f: (f["top_left"][0], f["top_left"][1], f["color"]))
    return frames


def _solve_frames(grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    groups: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        groups[int(frame["color"])].append(frame)

    order = sorted(
        groups,
        key=lambda color: min((f["top_left"][0], f["top_left"][1]) for f in groups[color]),
    )
    for color in order:
        group = groups[color]
        assert len(group) == 3, f"expected 3 frames for color {color}, got {len(group)}"
        inner_shapes = {tuple(f["inner_shape"]) for f in group}
        assert len(inner_shapes) == 1, f"same-color frames must share size: {group}"

        orange_counts = [int(f["orange_count"]) for f in group]
        fewest = min(orange_counts)
        anomaly_indices = [i for i, count in enumerate(orange_counts) if count == fewest]
        assert len(anomaly_indices) == 1, f"expected unique anomaly by orange count: {orange_counts}"
        anomaly_i = anomaly_indices[0]
        survivors = [f for i, f in enumerate(group) if i != anomaly_i]
        assert int(survivors[0]["orange_count"]) == int(survivors[1]["orange_count"])
        assert int(survivors[0]["orange_count"]) >= int(group[anomaly_i]["orange_count"]) + 2

        union_mask = np.zeros(group[0]["inner_shape"], dtype=bool)
        for frame in survivors:
            r0, c0, r1, c1 = frame["bbox"]
            union_mask |= grid[r0 + 1 : r1, c0 + 1 : c1] == GREY

        ar0, ac0, ar1, ac1 = group[anomaly_i]["bbox"]
        out[ar0 : ar1 + 1, ac0 : ac1 + 1] = BG

        for frame in survivors:
            r0, c0, r1, c1 = frame["bbox"]
            inner = out[r0 + 1 : r1, c0 + 1 : c1]
            draw_mask = union_mask & (inner != ORANGE)
            inner[draw_mask] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the least-marked frame in each border-color group, then union-complete the surviving motifs.

    Input structure:
    - The background is black, and the foreground contains several non-overlapping hollow rectangular frames.
    - Frame borders use repeated non-background colors, while frame interiors contain black, grey motif cells, and orange marker cells.
    - Frames of the same border color share the same interior size and are separated by at least one background cell.

    Rule:
    - Group frames by border color and count the orange interior cells in each frame.
    - In every color group, the unique frame with the fewest orange cells is the anomaly and is erased completely to black.
    - Using only the two surviving frames of that color, take the union of grey motif positions in interior-relative coordinates.
    - Paint that grey union into both surviving interiors without overwriting any existing orange cells.

    Key insight:
    - Orange cells are not output targets; they are the evidence used to identify which same-colored frame must be removed before motif completion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    frames = _extract_frames(g)
    assert frames, "no frames detected"
    return _solve_frames(g, frames)


OUTER_DIM_CHOICES = [
    (4, 5),
    (4, 6),
    (5, 4),
    (5, 5),
    (5, 6),
    (6, 4),
    (6, 5),
    (6, 6),
]
CELL_STEP = 9
CELL_OFFSET_CHOICES = (0, 1, 2)
MIN_ATTEMPTS = 400


def _relative_cells(shape: tuple[int, int]) -> list[tuple[int, int]]:
    return [(r, c) for r in range(shape[0]) for c in range(shape[1])]


def _sample_motif_and_markers(inner_shape: tuple[int, int]) -> dict | None:
    area = inner_shape[0] * inner_shape[1]
    if area < 6:
        return None

    cells = _relative_cells(inner_shape)
    full_count_max = area - 3
    if full_count_max < 3:
        return None
    full_count = random.randint(3, full_count_max)
    full = set(random.sample(cells, full_count))

    common_count = random.randint(0, full_count - 2)
    common = set(random.sample(list(full), common_count)) if common_count else set()
    remaining = list(full - common)
    a_only_count = random.randint(1, len(remaining) - 1)
    a_only = set(random.sample(remaining, a_only_count))
    b_only = set(remaining) - a_only
    motif_a = common | a_only
    motif_b = common | b_only
    if motif_a == full or motif_b == full:
        return None

    anomaly_size = random.randint(1, full_count - 1)
    motif_c = set(random.sample(list(full), anomaly_size))
    if motif_c == full:
        return None

    complement = [cell for cell in cells if cell not in full]
    if len(complement) < 3:
        return None
    survivor_count = random.randint(3, min(5, len(complement)))
    anomaly_max = min(len(complement), survivor_count - 2)
    if anomaly_max < 1:
        return None
    anomaly_count = random.randint(1, anomaly_max)

    oranges_a = set(random.sample(complement, survivor_count))
    oranges_b = set(random.sample(complement, survivor_count))
    oranges_c = set(random.sample(complement, anomaly_count))

    return {
        "full": full,
        "motif_a": motif_a,
        "motif_b": motif_b,
        "motif_c": motif_c,
        "oranges": (oranges_a, oranges_b, oranges_c),
    }


def _paint_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    outer_shape: tuple[int, int],
    color: int,
    grey_cells: set[tuple[int, int]],
    orange_cells: set[tuple[int, int]],
) -> None:
    h, w = outer_shape
    grid[top, left : left + w] = color
    grid[top + h - 1, left : left + w] = color
    grid[top : top + h, left] = color
    grid[top : top + h, left + w - 1] = color
    for rr, cc in grey_cells:
        grid[top + 1 + rr, left + 1 + cc] = GREY
    for rr, cc in orange_cells:
        grid[top + 1 + rr, left + 1 + cc] = ORANGE


def _coarse_region(cell_index: int) -> tuple[int, int]:
    return divmod(cell_index, 3)


def _make_cell_assignment() -> dict[int, dict]:
    anomaly_cols = list(range(3))
    random.shuffle(anomaly_cols)
    anomaly_cells = [row * 3 + anomaly_cols[row] for row in range(3)]
    color_order = list(FRAME_COLORS)
    random.shuffle(color_order)

    assignment: dict[int, dict] = {}
    remaining = [idx for idx in range(9) if idx not in anomaly_cells]
    random.shuffle(remaining)

    for i, color in enumerate(color_order):
        cells = [anomaly_cells[i], remaining[2 * i], remaining[2 * i + 1]]
        random.shuffle(cells[1:])
        assignment[color] = {
            "cells": cells,
            "anomaly_cell": anomaly_cells[i],
        }
    return assignment


def _grid_and_origins() -> tuple[int, int, list[int], list[int]]:
    row_margin = random.randint(0, 1)
    col_margin = random.randint(0, 1)
    h = random.randint(27 + row_margin, 30)
    w = random.randint(27 + col_margin, 30)
    row_origins = [row_margin + i * CELL_STEP for i in range(3)]
    col_origins = [col_margin + i * CELL_STEP for i in range(3)]
    return h, w, row_origins, col_origins


def _semantic_check_input(grid: np.ndarray) -> bool:
    try:
        frames = _extract_frames(grid)
        if len(frames) != 9:
            return False
        groups: dict[int, list[dict]] = defaultdict(list)
        for frame in frames:
            groups[int(frame["color"])].append(frame)
        if set(groups) != set(FRAME_COLORS):
            return False

        anomaly_regions = set()
        for color, group in groups.items():
            if len(group) != 3:
                return False
            if len({tuple(f["outer_shape"]) for f in group}) != 1:
                return False
            if len({tuple(f["inner_shape"]) for f in group}) != 1:
                return False

            counts = [int(f["orange_count"]) for f in group]
            least = min(counts)
            if counts.count(least) != 1:
                return False
            survivors = [group[i] for i in range(3) if counts[i] != least]
            anomaly = group[counts.index(least)]
            if int(survivors[0]["orange_count"]) != int(survivors[1]["orange_count"]):
                return False
            if int(survivors[0]["orange_count"]) < int(anomaly["orange_count"]) + 2:
                return False

            union = survivors[0]["grey_mask"] | survivors[1]["grey_mask"]
            if not union.any():
                return False
            if np.array_equal(union, survivors[0]["grey_mask"]) or np.array_equal(union, survivors[1]["grey_mask"]):
                return False
            if not np.all(anomaly["grey_mask"] <= union):
                return False
            if np.array_equal(anomaly["grey_mask"], union):
                return False

            r0, c0 = anomaly["top_left"]
            anomaly_regions.add((min(r0 // CELL_STEP, 2), min(c0 // CELL_STEP, 2)))
        return len(anomaly_regions) == 3
    except Exception:
        return False


def _attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h, w, row_origins, col_origins = _grid_and_origins()
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    assignment = _make_cell_assignment()

    group_specs: dict[int, dict] = {}
    for color in FRAME_COLORS:
        outer_shape = random.choice(OUTER_DIM_CHOICES)
        inner_shape = (outer_shape[0] - 2, outer_shape[1] - 2)
        motif = _sample_motif_and_markers(inner_shape)
        if motif is None:
            return None
        group_specs[color] = {
            "outer_shape": outer_shape,
            "inner_shape": inner_shape,
            **motif,
        }

    top_lefts: dict[int, tuple[int, int]] = {}
    for color, info in assignment.items():
        outer_h, outer_w = group_specs[color]["outer_shape"]
        for cell in info["cells"]:
            rr, cc = _coarse_region(cell)
            top = row_origins[rr] + random.choice(CELL_OFFSET_CHOICES)
            left = col_origins[cc] + random.choice(CELL_OFFSET_CHOICES)
            if top + outer_h > h or left + outer_w > w:
                return None
            top_lefts[cell] = (top, left)

    for color in FRAME_COLORS:
        spec = group_specs[color]
        cells = assignment[color]["cells"]
        anomaly_cell = assignment[color]["anomaly_cell"]
        survivors = [cell for cell in cells if cell != anomaly_cell]
        random.shuffle(survivors)
        partials = {
            survivors[0]: (spec["motif_a"], spec["oranges"][0]),
            survivors[1]: (spec["motif_b"], spec["oranges"][1]),
            anomaly_cell: (spec["motif_c"], spec["oranges"][2]),
        }
        outputs = {
            survivors[0]: (spec["full"], spec["oranges"][0]),
            survivors[1]: (spec["full"], spec["oranges"][1]),
        }

        for cell in cells:
            top, left = top_lefts[cell]
            grey_cells, orange_cells = partials[cell]
            _paint_frame(inp, top, left, spec["outer_shape"], color, grey_cells, orange_cells)
        for cell in survivors:
            top, left = top_lefts[cell]
            grey_cells, orange_cells = outputs[cell]
            _paint_frame(out, top, left, spec["outer_shape"], color, grey_cells, orange_cells)

    if not _semantic_check_input(inp):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MIN_ATTEMPTS):
        built = _attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input for puzzle 4")
