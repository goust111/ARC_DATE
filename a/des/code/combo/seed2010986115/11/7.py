from __future__ import annotations

import random

import numpy as np


BG = 0
INPUT_COLORS = (1, 2, 3)
YELLOW = 4
BRANCH_TYPES = ("mono", "diag", "side", "mixed")

MIN_RECTS = 2
MAX_RECTS = 4
MIN_SIDE = 3
MAX_SIDE = 8
MIN_GAP = 1
MAX_GAP = 3
MIN_MARGIN = 1
MAX_MARGIN = 3
GRID_MIN = 10
GRID_MAX = 30
GEN_RETRIES = 300


def _anchor_map(grid: np.ndarray) -> dict[tuple[int, int], int]:
    anchors = {}
    for r, c in np.argwhere(grid != BG):
        anchors[(int(r), int(c))] = int(grid[r, c])
    return anchors


def _boxes_overlap(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    rows_disjoint = ar1 < br0 or br1 < ar0
    cols_disjoint = ac1 < bc0 or bc1 < ac0
    return not (rows_disjoint or cols_disjoint)


def _candidate_rectangles(
    anchors: dict[tuple[int, int], int]
) -> list[dict[str, object]]:
    row_to_cols: dict[int, set[int]] = {}
    for r, c in anchors:
        row_to_cols.setdefault(r, set()).add(c)

    candidates: list[dict[str, object]] = []
    rows = sorted(row_to_cols)
    for i, top in enumerate(rows):
        top_cols = row_to_cols[top]
        for bottom in rows[i + 1 :]:
            common_cols = sorted(top_cols & row_to_cols[bottom])
            for j, left in enumerate(common_cols):
                for right in common_cols[j + 1 :]:
                    corners = (
                        (top, left),
                        (top, right),
                        (bottom, left),
                        (bottom, right),
                    )
                    bbox = (top, left, bottom, right)
                    candidates.append(
                        {
                            "bbox": bbox,
                            "corners": corners,
                            "colors": tuple(anchors[pos] for pos in corners),
                            "area": (bottom - top + 1) * (right - left + 1),
                        }
                    )

    candidates.sort(key=lambda item: (item["area"], item["bbox"]))
    return candidates


def _recover_rectangles(grid: np.ndarray) -> list[dict[str, object]]:
    anchors = _anchor_map(grid)
    anchor_positions = sorted(anchors)
    assert anchor_positions, "expected at least one anchor"
    assert len(anchor_positions) % 4 == 0, "anchor count must be divisible by 4"

    candidates = _candidate_rectangles(anchors)
    assert candidates, "expected at least one candidate rectangle"

    anchor_to_candidates: dict[tuple[int, int], list[int]] = {pos: [] for pos in anchor_positions}
    for idx, candidate in enumerate(candidates):
        for pos in candidate["corners"]:
            anchor_to_candidates[pos].append(idx)
    assert all(anchor_to_candidates[pos] for pos in anchor_positions), "every anchor must belong to a rectangle"

    target_count = len(anchor_positions) // 4
    used: set[tuple[int, int]] = set()
    chosen_idxs: list[int] = []
    chosen_boxes: list[tuple[int, int, int, int]] = []
    best_solution: list[dict[str, object]] | None = None
    best_key: tuple[object, ...] | None = None

    def compatible(idx: int) -> bool:
        candidate = candidates[idx]
        if any(pos in used for pos in candidate["corners"]):
            return False
        bbox = candidate["bbox"]
        return all(not _boxes_overlap(bbox, other) for other in chosen_boxes)

    def search() -> None:
        nonlocal best_solution, best_key
        if len(used) == len(anchor_positions):
            if len(chosen_idxs) != target_count:
                return
            solution = [candidates[idx] for idx in chosen_idxs]
            solution.sort(key=lambda item: item["bbox"])
            key = (
                sum(int(item["area"]) for item in solution),
                tuple(item["bbox"] for item in solution),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_solution = solution
            return

        if len(chosen_idxs) >= target_count:
            return

        pivot = None
        options = None
        for pos in anchor_positions:
            if pos in used:
                continue
            valid = [idx for idx in anchor_to_candidates[pos] if compatible(idx)]
            if not valid:
                return
            valid.sort(key=lambda idx: (candidates[idx]["area"], candidates[idx]["bbox"]))
            if options is None or len(valid) < len(options):
                pivot = pos
                options = valid
                if len(options) == 1:
                    break

        assert pivot is not None and options is not None
        for idx in options:
            candidate = candidates[idx]
            chosen_idxs.append(idx)
            chosen_boxes.append(candidate["bbox"])
            used.update(candidate["corners"])
            search()
            for pos in candidate["corners"]:
                used.remove(pos)
            chosen_boxes.pop()
            chosen_idxs.pop()

    search()
    assert best_solution is not None, "failed to recover a rectangle partition"
    return best_solution


def _paint_border(
    out: np.ndarray, bbox: tuple[int, int, int, int], color: int
) -> None:
    top, left, bottom, right = bbox
    out[top, left : right + 1] = color
    out[bottom, left : right + 1] = color
    out[top : bottom + 1, left] = color
    out[top : bottom + 1, right] = color


def _restore_corners(
    out: np.ndarray,
    corners: tuple[tuple[int, int], ...],
    colors: tuple[int, int, int, int],
) -> None:
    for pos, color in zip(corners, colors, strict=True):
        out[pos] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover rectangles from scattered corner anchors and fill each rectangle by corner-color pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rectangles = _recover_rectangles(g)
    out = np.full_like(g, BG)

    for rect in rectangles:
        top, left, bottom, right = rect["bbox"]
        corners = rect["corners"]
        tl, tr, bl, br = rect["colors"]
        unique = {tl, tr, bl, br}

        if len(unique) == 1:
            _paint_border(out, (top, left, bottom, right), tl)
        elif len(unique) == 2 and tl == br and tr == bl and tl != tr:
            mid = (top + bottom) // 2
            out[top : mid + 1, left : right + 1] = tl
            out[mid + 1 : bottom + 1, left : right + 1] = tr
        elif len(unique) == 2 and tl == bl and tr == br and tl != tr:
            mid = (left + right) // 2
            out[top : bottom + 1, left : mid + 1] = tl
            out[top : bottom + 1, mid + 1 : right + 1] = tr
        else:
            if bottom - top >= 2 and right - left >= 2:
                out[top + 1 : bottom, left + 1 : right] = YELLOW

        _restore_corners(out, corners, rect["colors"])

    return out


def _sample_axis(n_rects: int) -> tuple[list[tuple[int, int]], int] | None:
    lengths = [random.randint(MIN_SIDE, MAX_SIDE) for _ in range(n_rects)]
    gaps = [random.randint(MIN_GAP, MAX_GAP) for _ in range(n_rects - 1)]
    margin_before = random.randint(MIN_MARGIN, MAX_MARGIN)
    margin_after = random.randint(MIN_MARGIN, MAX_MARGIN)
    total = margin_before + sum(lengths) + sum(gaps) + margin_after
    if not (GRID_MIN <= total <= GRID_MAX):
        return None

    intervals: list[tuple[int, int]] = []
    pos = margin_before
    for i, length in enumerate(lengths):
        intervals.append((pos, pos + length - 1))
        pos += length
        if i < len(gaps):
            pos += gaps[i]
    return intervals, total


def _corner_colors(branch: str) -> tuple[int, int, int, int]:
    if branch == "mono":
        color = random.choice(INPUT_COLORS)
        return color, color, color, color
    if branch == "diag":
        a, b = random.sample(INPUT_COLORS, 2)
        return a, b, b, a
    if branch == "side":
        a, b = random.sample(INPUT_COLORS, 2)
        return a, b, a, b

    a, b, c = random.sample(INPUT_COLORS, 3)
    patterns = [
        (a, b, c, a),
        (a, b, c, b),
        (a, b, c, c),
    ]
    return random.choice(patterns)


def _build_attempt() -> np.ndarray | None:
    n_rects = random.choice([2, 3, 4, 4])
    row_axis = _sample_axis(n_rects)
    col_axis = _sample_axis(n_rects)
    if row_axis is None or col_axis is None:
        return None
    row_intervals, height = row_axis
    col_intervals, width = col_axis

    grid = np.zeros((height, width), dtype=int)
    col_order = random.sample(range(n_rects), n_rects)
    branches = random.sample(BRANCH_TYPES, k=n_rects)

    for rect_idx, branch in enumerate(branches):
        top, bottom = row_intervals[rect_idx]
        left, right = col_intervals[col_order[rect_idx]]
        tl, tr, bl, br = _corner_colors(branch)
        grid[top, left] = tl
        grid[top, right] = tr
        grid[bottom, left] = bl
        grid[bottom, right] = br

    try:
        recovered = _recover_rectangles(grid)
        if len(recovered) != n_rects:
            return None
        _ = main(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
