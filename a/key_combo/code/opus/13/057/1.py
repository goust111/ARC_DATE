from __future__ import annotations

import random
from typing import Callable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
TEAL = 8
CONTROL_COLORS = (BLUE, GREEN, TEAL)

_GEN_RETRIES = 400
_SIDE_MARGIN = 1
_MIN_RECT_GAP = 1


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _rectangles_overlap_or_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_separated = ar1 + pad < br0 or br1 + pad < ar0
    col_separated = ac1 + pad < bc0 or bc1 + pad < ac0
    return not (row_separated or col_separated)


def _detect_gray_rectangles(grid: np.ndarray) -> list[dict]:
    h, w = map(int, grid.shape)
    visited = np.zeros((h, w), dtype=bool)
    rects: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != GREY:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == GREY:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            rect_h = r1 - r0 + 1
            rect_w = c1 - c0 + 1
            assert rect_h >= 3 and rect_w >= 3, "gray components must have an interior"
            area = rect_h * rect_w
            assert len(cells) == area, "gray component must fill its bbox"
            assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY), "gray rectangle bbox must be solid"
            rects.append({"bbox": (r0, c0, r1, c1), "hollowed": False})

    assert rects, "expected at least one gray rectangle"
    return rects


def _rect_index_at(rects: list[dict], r: int, c: int) -> int | None:
    for idx, rect in enumerate(rects):
        r0, c0, r1, c1 = rect["bbox"]
        if r0 <= r <= r1 and c0 <= c <= c1:
            return idx
    return None


def _controls_from_bottom_row(grid: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = map(int, grid.shape)
    controls = []
    for c in range(w):
        color = int(grid[h - 1, c])
        if color in CONTROL_COLORS:
            controls.append((h - 1, c, color))
        else:
            assert color == BLACK, "bottom row may only contain black or control pixels"
    assert 2 <= len(controls) <= len(CONTROL_COLORS), "generator uses 2-3 distinct control pixels"
    used_colors = [color for _, _, color in controls]
    assert len(set(used_colors)) == len(used_colors), "control colors must be distinct"
    return controls


def _simulate(grid: np.ndarray, order: list[int] | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, _w = map(int, g.shape)

    rects = _detect_gray_rectangles(g)
    controls = sorted(_controls_from_bottom_row(g), key=lambda item: item[1])
    if order is None:
        order = list(range(len(controls)))

    out = g.copy()
    for control_i in order:
        _sr, c, current_color = controls[control_i]
        r = h - 1
        while True:
            nr = r - 1
            if nr < 0:
                break
            cell = int(out[nr, c])
            if cell == BLACK:
                out[nr, c] = current_color
                r = nr
                continue
            if cell == GREY:
                rect_idx = _rect_index_at(rects, nr, c)
                assert rect_idx is not None, "every gray cell must belong to a tracked rectangle"
                rect = rects[rect_idx]
                r0, c0, r1, c1 = rect["bbox"]
                if rect["hollowed"]:
                    break
                assert c0 < c < c1, "path must hit a rectangle through an interior column"
                out[r0 + 1 : r1, c0 + 1 : c1] = BLACK
                for rr in range(r0 + 1, r1):
                    out[rr, c] = current_color
                rect["hollowed"] = True
                current_color = RED
                r = r0
                continue
            break

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each bottom control upward, hollowing the first solid gray rectangles it hits so the path can pass through them.

    Input structure:
    - Black is the background.
    - The bottom row contains 2 or 3 distinct control pixels chosen from blue, green, and teal.
    - Above them are several non-touching solid gray axis-aligned rectangles, each at least 3 by 3 and not touching the bottom row.

    Rule:
    - Process the control pixels from left to right.
    - Each path grows straight upward in its own column, painting black cells with its current color.
    - When a path first hits an un-hollowed gray rectangle, hollow that rectangle in place, paint the path through the cleared interior column, keep the gray border intact, and switch the path's later color to red after it exits the top border.
    - If a path later reaches a gray border from a rectangle that was already hollowed, or any non-black non-gray cell, or the top edge, it stops.

    Key insight:
    - Hollowing is triggered dynamically by the first path that collides with a solid rectangle, so rectangle-sharing columns create order-dependent competition and only collided rectangles become frames.
    """
    return _simulate(grid)


def _choose_controls(rng: random.Random, width: int, n_controls: int) -> tuple[list[int], int]:
    candidates: list[tuple[list[int], int]] = []
    if n_controls == 2:
        for c0 in range(2, width - 5):
            for gap in range(2, 5):
                c1 = c0 + gap
                if c1 <= width - 3:
                    candidates.append(([c0, c1], 0))
    else:
        for pair_start in (0, 1):
            if pair_start == 0:
                for c0 in range(2, width - 9):
                    for gap in range(2, 5):
                        c1 = c0 + gap
                        for c2 in range(c1 + 4, width - 2):
                            candidates.append(([c0, c1, c2], pair_start))
            else:
                for c0 in range(2, width - 9):
                    for c1 in range(c0 + 4, width - 5):
                        for gap in range(2, 5):
                            c2 = c1 + gap
                            if c2 <= width - 3:
                                candidates.append(([c0, c1, c2], pair_start))
    assert candidates, "no feasible control layout"
    return rng.choice(candidates)


def _pick_rect(
    rng: random.Random,
    shape: tuple[int, int],
    existing: list[tuple[int, int, int, int]],
    cover_cols: list[int],
    h_range: tuple[int, int],
    w_range: tuple[int, int],
    predicate: Callable[[tuple[int, int, int, int]], bool] | None = None,
) -> tuple[int, int, int, int] | None:
    h_grid, w_grid = shape
    candidates = []
    for rect_h in range(h_range[0], h_range[1] + 1):
        for rect_w in range(w_range[0], w_range[1] + 1):
            for r0 in range(1, h_grid - rect_h):
                r1 = r0 + rect_h - 1
                if r1 >= h_grid - 1:
                    continue
                for c0 in range(_SIDE_MARGIN, w_grid - rect_w):
                    c1 = c0 + rect_w - 1
                    if c1 >= w_grid - _SIDE_MARGIN:
                        continue
                    if any(not (c0 < col < c1) for col in cover_cols):
                        continue
                    bbox = (r0, c0, r1, c1)
                    if any(_rectangles_overlap_or_touch(bbox, other, _MIN_RECT_GAP) for other in existing):
                        continue
                    if predicate is not None and not predicate(bbox):
                        continue
                    candidates.append(bbox)
    if not candidates:
        return None
    return rng.choice(candidates)


def _draw_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _sample_layout(rng: random.Random) -> tuple[np.ndarray, dict] | None:
    n_controls = rng.choice([2, 3, 3])
    height = rng.randint(15, 19)
    width = rng.randint(18 if n_controls == 3 else 16, 24)
    control_cols, pair_start = _choose_controls(rng, width, n_controls)

    shared_left = pair_start
    shared_right = pair_start + 1
    early_shared_col = control_cols[shared_left]
    late_shared_col = control_cols[shared_right]
    unshared_cols = [control_cols[i] for i in range(n_controls) if i not in (shared_left, shared_right)]

    bboxes: list[tuple[int, int, int, int]] = []
    named_rects: dict[str, tuple[int, int, int, int]] = {}

    shared = _pick_rect(
        rng,
        (height, width),
        bboxes,
        [early_shared_col, late_shared_col],
        (4, 6),
        (5, 9),
        predicate=lambda b: b[0] >= 5 and height - 7 <= b[2] <= height - 4,
    )
    if shared is None:
        return None
    bboxes.append(shared)
    named_rects["shared"] = shared

    def upper_predicate(bbox: tuple[int, int, int, int]) -> bool:
        r0, c0, r1, c1 = bbox
        _sr0, _sc0, shared_r0, _sc1 = named_rects["shared"]
        return r1 <= shared_r0 - 2 and not (c0 <= late_shared_col <= c1)

    stacked = _pick_rect(
        rng,
        (height, width),
        bboxes,
        [early_shared_col],
        (3, 5),
        (3, 5),
        predicate=upper_predicate,
    )
    if stacked is None:
        return None
    bboxes.append(stacked)
    named_rects["stacked"] = stacked

    unique_rects: list[tuple[int, int, int, int]] = []
    for idx, col in enumerate(unshared_cols):
        rect = _pick_rect(
            rng,
            (height, width),
            bboxes,
            [col],
            (3, 5),
            (3, 6),
            predicate=lambda b, h=height: 2 <= b[0] and b[2] <= h - 4,
        )
        if rect is None:
            return None
        bboxes.append(rect)
        unique_rects.append(rect)
        named_rects[f"unique_{idx}"] = rect

    total_rectangles = rng.randint(3, 5)
    filler_needed = max(1, total_rectangles - len(bboxes))
    filler_rects = []
    all_control_cols = set(control_cols)
    while len(filler_rects) < filler_needed:
        rect = _pick_rect(
            rng,
            (height, width),
            bboxes,
            [],
            (3, 5),
            (3, 6),
            predicate=lambda b, control_cols=all_control_cols: all(not (b[1] <= col <= b[3]) for col in control_cols),
        )
        if rect is None:
            return None
        bboxes.append(rect)
        filler_rects.append(rect)
        named_rects[f"filler_{len(filler_rects) - 1}"] = rect

    grid = np.full((height, width), BLACK, dtype=int)
    for bbox in bboxes:
        _draw_rect(grid, bbox, GREY)

    control_colors = rng.sample(list(CONTROL_COLORS), n_controls)
    for col, color in zip(control_cols, control_colors):
        grid[height - 1, col] = color

    meta = {
        "control_cols": control_cols,
        "pair_start": pair_start,
        "named_rects": named_rects,
        "filler_rects": filler_rects,
    }
    return grid, meta


def _check_hollowed(output: np.ndarray, bbox: tuple[int, int, int, int], path_col: int, expected_color: int) -> bool:
    r0, c0, r1, c1 = bbox
    interior = output[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0:
        return False
    if np.any(interior == GREY):
        return False
    if not np.all(output[r0 : r1 + 1, c0] == GREY):
        return False
    if not np.all(output[r0 : r1 + 1, c1] == GREY):
        return False
    if not np.all(output[r0, c0 : c1 + 1] == GREY):
        return False
    if not np.all(output[r1, c0 : c1 + 1] == GREY):
        return False
    return np.all(output[r0 + 1 : r1, path_col] == expected_color)


def _semantic_checks(inp: np.ndarray, out: np.ndarray, meta: dict) -> bool:
    control_cols = meta["control_cols"]
    pair_start = meta["pair_start"]
    named_rects = meta["named_rects"]

    if np.array_equal(inp, out):
        return False

    shared = named_rects["shared"]
    stacked = named_rects["stacked"]
    shared_early_col = control_cols[pair_start]
    shared_late_col = control_cols[pair_start + 1]

    if not _check_hollowed(out, shared, shared_early_col, int(inp[-1, shared_early_col])):
        return False
    if not _check_hollowed(out, stacked, shared_early_col, RED):
        return False

    shared_r0, _sc0, shared_r1, _sc1 = shared
    if out[shared_r0 - 1, shared_early_col] != RED:
        return False
    if out[shared_r1 + 1, shared_late_col] == BLACK:
        return False
    if out[shared_r1, shared_late_col] != GREY:
        return False

    for name, bbox in named_rects.items():
        if name.startswith("unique_"):
            path_col = next(col for col in control_cols if bbox[1] < col < bbox[3])
            if not _check_hollowed(out, bbox, path_col, int(inp[-1, path_col])):
                return False

    for bbox in meta["filler_rects"]:
        r0, c0, r1, c1 = bbox
        if not np.all(out[r0 : r1 + 1, c0 : c1 + 1] == GREY):
            return False

    reversed_out = _simulate(inp, order=list(reversed(range(len(control_cols)))))
    if np.array_equal(out, reversed_out):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        rng = random.Random(random.randrange(1 << 30))
        built = _sample_layout(rng)
        if built is None:
            continue
        grid, meta = built
        try:
            out = main(grid)
        except AssertionError:
            continue
        if _semantic_checks(grid, out, meta):
            return grid
    raise RuntimeError("failed to generate a valid path-hollowing ARC input")

