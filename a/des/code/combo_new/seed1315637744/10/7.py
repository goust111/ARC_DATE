from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
ORANGE = 7
PATTERN_COLORS = (1, 4, 5, 6, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _adjacent(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    return any(mask[nr, nc] for nr, nc in _neighbors4(r, c, h, w))


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _detect_marker(grid: np.ndarray) -> tuple[str, bool]:
    red_pos = np.argwhere(grid == RED)
    green_pos = np.argwhere(grid == GREEN)
    assert red_pos.shape[0] == 1 and green_pos.shape[0] == 1, "expected one red and one green marker pixel"

    rr, rc = map(int, red_pos[0])
    gr, gc = map(int, green_pos[0])

    if rr == gr and abs(rc - gc) == 1:
        return "vertical", rc < gc
    if rc == gc and abs(rr - gr) == 1:
        return "horizontal", rr < gr
    raise AssertionError("marker pixels must form a 2x1 corner marker")


def _pattern_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    colors = [int(color) for color in np.unique(grid) if int(color) not in {bg, RED, GREEN}]
    assert colors, "expected one non-marker pattern color"
    counts = {color: int(np.count_nonzero(grid == color)) for color in colors}
    pattern_color = max(counts, key=counts.get)
    largest = counts[pattern_color]
    assert list(counts.values()).count(largest) == 1, "pattern color must be uniquely largest"
    return grid == pattern_color


def _mirror(axis_kind: str, axis2: int, r: int, c: int) -> tuple[int, int]:
    if axis_kind == "vertical":
        return r, axis2 - c
    assert axis_kind == "horizontal"
    return axis2 - r, c


def _side(axis_kind: str, axis2: int, r: int, c: int) -> int:
    coord = c if axis_kind == "vertical" else r
    delta = 2 * coord - axis2
    if delta < 0:
        return -1
    if delta > 0:
        return 1
    return 0


def _axis_score(mask: np.ndarray, axis_kind: str, axis2: int) -> tuple[int, int, int]:
    h, w = mask.shape
    coords = np.argwhere(mask)
    center2 = (
        int(coords[:, 1].min() + coords[:, 1].max())
        if axis_kind == "vertical"
        else int(coords[:, 0].min() + coords[:, 0].max())
    )

    matched_pairs = 0
    unmatched_cells = 0
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        side = _side(axis_kind, axis2, r, c)
        if side == 0:
            continue
        rr, cc = _mirror(axis_kind, axis2, r, c)
        if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
            if side < 0:
                matched_pairs += 1
        else:
            unmatched_cells += 1

    return matched_pairs, -unmatched_cells, -abs(axis2 - center2)


def _best_axis(mask: np.ndarray, axis_kind: str) -> int:
    length = mask.shape[1] if axis_kind == "vertical" else mask.shape[0]
    best_axis2 = None
    best_score = None
    for axis2 in range(2 * length - 1):
        score = _axis_score(mask, axis_kind, axis2)
        if best_score is None or score > best_score:
            best_score = score
            best_axis2 = axis2
    assert best_axis2 is not None
    return int(best_axis2)


def _selected_unique_coords(mask: np.ndarray, axis_kind: str, axis2: int, keep_negative: bool) -> set[tuple[int, int]]:
    h, w = mask.shape
    out: set[tuple[int, int]] = set()
    for r_raw, c_raw in np.argwhere(mask):
        r = int(r_raw)
        c = int(c_raw)
        side = _side(axis_kind, axis2, r, c)
        if side == 0 or (keep_negative and side > 0) or (not keep_negative and side < 0):
            continue
        rr, cc = _mirror(axis_kind, axis2, r, c)
        if not (0 <= rr < h and 0 <= cc < w and mask[rr, cc]):
            out.add((r, c))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the best horizontal or vertical symmetry axis and keep only the marked side's unmatched pixels.

    Input structure:
    - The background is black.
    - One large connected single-color shape is approximately symmetric about a horizontal or vertical axis.
    - A red/green 2x1 corner marker is oriented perpendicular to that axis and indicates which side to keep.

    Rule:
    - Infer the axis that maximizes mirrored overlap between the two sides of the large shape.
    - Compare the shape against its mirror and collect pixels that appear on exactly one side.
    - Use the marker order to choose the left/top side when red is left/above green, otherwise the right/bottom side.
    - Output a same-size black grid containing only those selected unmatched pixels, recolored orange.

    Key insight:
    - The marker does not locate the axis; it only says which side's asymmetry survives after the hidden best-fit reflection axis is inferred.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    axis_kind, keep_negative = _detect_marker(g)
    mask = _pattern_mask(g, bg)
    axis2 = _best_axis(mask, axis_kind)

    out = np.full_like(g, bg)
    for r, c in _selected_unique_coords(mask, axis_kind, axis2, keep_negative):
        out[r, c] = ORANGE
    return out


# --------------- generator (o5x) ---------------

GRID_HEIGHT_RANGE = (12, 24)
GRID_WIDTH_RANGE = (12, 24)
SHAPE_MARGIN = 2
MIN_SIDE_DEPTH = 3
CORE_LEN_RANGE = (3, 6)
BASE_TARGET_RANGE = (18, 44)
BRANCH_SIZE_RANGE = (1, 6)
GENERATION_ATTEMPTS = 500


def _in_interior(shape: tuple[int, int], r: int, c: int, margin: int) -> bool:
    h, w = shape
    return margin <= r < h - margin and margin <= c < w - margin


def _axis_candidates(length: int, margin: int, min_side_depth: int, parity: int) -> list[int]:
    candidates = []
    interior = list(range(margin, length - margin))
    for axis2 in range(2 * length - 1):
        if axis2 % 2 != parity:
            continue
        neg = sum(1 for idx in interior if 2 * idx < axis2)
        pos = sum(1 for idx in interior if 2 * idx > axis2)
        if neg >= min_side_depth and pos >= min_side_depth:
            candidates.append(axis2)
    return candidates


def _seed_core(mask: np.ndarray, axis_kind: str, axis2: int, margin: int) -> bool:
    h, w = mask.shape
    ortho_length = h if axis_kind == "vertical" else w
    max_len = min(CORE_LEN_RANGE[1], ortho_length - 2 * margin)
    if max_len < CORE_LEN_RANGE[0]:
        return False

    core_len = random.randint(CORE_LEN_RANGE[0], max_len)
    start = random.randint(margin, ortho_length - margin - core_len)

    for offset in range(core_len):
        if axis_kind == "vertical":
            r = start + offset
            if axis2 % 2 == 0:
                mask[r, axis2 // 2] = True
            else:
                c_left = axis2 // 2
                mask[r, c_left] = True
                mask[r, c_left + 1] = True
        else:
            c = start + offset
            if axis2 % 2 == 0:
                mask[axis2 // 2, c] = True
            else:
                r_top = axis2 // 2
                mask[r_top, c] = True
                mask[r_top + 1, c] = True
    return True


def _pair_candidates(mask: np.ndarray, axis_kind: str, axis2: int, margin: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    candidates = []
    for r in range(margin, h - margin):
        for c in range(margin, w - margin):
            if mask[r, c] or _side(axis_kind, axis2, r, c) != -1:
                continue
            rr, cc = _mirror(axis_kind, axis2, r, c)
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if not _in_interior(mask.shape, rr, cc, margin):
                continue
            if mask[rr, cc]:
                continue
            if _adjacent(mask, r, c) or _adjacent(mask, rr, cc):
                candidates.append((r, c))
    return candidates


def _grow_symmetric_base(mask: np.ndarray, axis_kind: str, axis2: int, margin: int) -> bool:
    interior_area = (mask.shape[0] - 2 * margin) * (mask.shape[1] - 2 * margin)
    max_target = min(BASE_TARGET_RANGE[1], interior_area // 2 + int(mask.sum()))
    min_target = min(BASE_TARGET_RANGE[0], max_target)
    if max_target < min_target:
        return False

    target = random.randint(min_target, max_target)
    while int(mask.sum()) < target:
        candidates = _pair_candidates(mask, axis_kind, axis2, margin)
        if not candidates:
            break
        r, c = random.choice(candidates)
        rr, cc = _mirror(axis_kind, axis2, r, c)
        mask[r, c] = True
        mask[rr, cc] = True
    return int(mask.sum()) >= min_target


def _branch_cell_valid(mask: np.ndarray, axis_kind: str, axis2: int, side_sign: int, r: int, c: int, margin: int) -> bool:
    h, w = mask.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    if mask[r, c] or _side(axis_kind, axis2, r, c) != side_sign:
        return False
    if not _in_interior(mask.shape, r, c, margin):
        return False

    rr, cc = _mirror(axis_kind, axis2, r, c)
    if not (0 <= rr < h and 0 <= cc < w):
        return False
    if not _in_interior(mask.shape, rr, cc, margin):
        return False
    return not mask[rr, cc]


def _grow_branch(mask: np.ndarray, axis_kind: str, axis2: int, side_sign: int, size: int, margin: int) -> set[tuple[int, int]] | None:
    h, w = mask.shape
    starts = []
    for r in range(margin, h - margin):
        for c in range(margin, w - margin):
            if _branch_cell_valid(mask, axis_kind, axis2, side_sign, r, c, margin) and _adjacent(mask, r, c):
                starts.append((r, c))
    if not starts:
        return None

    branch = {random.choice(starts)}
    while len(branch) < size:
        frontier: set[tuple[int, int]] = set()
        for r, c in branch:
            for nr, nc in _neighbors4(r, c, h, w):
                if (nr, nc) in branch:
                    continue
                if _branch_cell_valid(mask, axis_kind, axis2, side_sign, nr, nc, margin):
                    frontier.add((nr, nc))
        if not frontier:
            return None
        branch.add(random.choice(tuple(frontier)))

    for r, c in branch:
        mask[r, c] = True
    return branch


def _place_marker(grid: np.ndarray, axis_kind: str, keep_negative: bool) -> None:
    h, w = grid.shape
    corner = random.choice(("tl", "tr", "bl", "br"))

    if axis_kind == "vertical":
        if corner == "tl":
            cells = ((0, 0), (0, 1))
        elif corner == "tr":
            cells = ((0, w - 2), (0, w - 1))
        elif corner == "bl":
            cells = ((h - 1, 0), (h - 1, 1))
        else:
            cells = ((h - 1, w - 2), (h - 1, w - 1))
        left_cell, right_cell = cells
        grid[left_cell] = RED if keep_negative else GREEN
        grid[right_cell] = GREEN if keep_negative else RED
    else:
        if corner == "tl":
            cells = ((0, 0), (1, 0))
        elif corner == "tr":
            cells = ((0, w - 1), (1, w - 1))
        elif corner == "bl":
            cells = ((h - 2, 0), (h - 1, 0))
        else:
            cells = ((h - 2, w - 1), (h - 1, w - 1))
        top_cell, bottom_cell = cells
        grid[top_cell] = RED if keep_negative else GREEN
        grid[bottom_cell] = GREEN if keep_negative else RED


def _mask_span(mask: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(mask)
    return int(coords[:, 0].max() - coords[:, 0].min() + 1), int(coords[:, 1].max() - coords[:, 1].min() + 1)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    axis_kind = random.choice(("vertical", "horizontal"))
    keep_negative = random.choice((True, False))
    parity = random.choice((0, 1))

    axis_length = w if axis_kind == "vertical" else h
    axis_choices = _axis_candidates(axis_length, SHAPE_MARGIN, MIN_SIDE_DEPTH, parity)
    if not axis_choices:
        return None
    axis2 = random.choice(axis_choices)

    mask = np.zeros((h, w), dtype=bool)
    if not _seed_core(mask, axis_kind, axis2, SHAPE_MARGIN):
        return None
    if not _grow_symmetric_base(mask, axis_kind, axis2, SHAPE_MARGIN):
        return None

    sizes = random.sample(range(BRANCH_SIZE_RANGE[0], BRANCH_SIZE_RANGE[1] + 1), 2)
    branches: dict[int, set[tuple[int, int]]] = {}
    for side_sign in random.sample([-1, 1], 2):
        branch = _grow_branch(mask, axis_kind, axis2, side_sign, sizes[0] if side_sign == -1 else sizes[1], SHAPE_MARGIN)
        if branch is None:
            return None
        branches[side_sign] = branch

    if not _is_connected(mask):
        return None

    row_span, col_span = _mask_span(mask)
    if row_span < 5 or col_span < 5:
        return None

    if _best_axis(mask, axis_kind) != axis2:
        return None

    neg_uniques = _selected_unique_coords(mask, axis_kind, axis2, True)
    pos_uniques = _selected_unique_coords(mask, axis_kind, axis2, False)
    if neg_uniques != branches[-1] or pos_uniques != branches[1]:
        return None

    pattern_color = random.choice(PATTERN_COLORS)
    inp = np.full((h, w), BG, dtype=int)
    inp[mask] = pattern_color
    _place_marker(inp, axis_kind, keep_negative)

    expected = np.full((h, w), BG, dtype=int)
    for r, c in branches[-1 if keep_negative else 1]:
        expected[r, c] = ORANGE

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
