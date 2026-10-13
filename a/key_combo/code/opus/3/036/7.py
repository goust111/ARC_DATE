from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components


GREY_RED = {Color.GREY, Color.RED}
RECT_COLORS = (Color.BLUE, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.MAROON)
GEN_ATTEMPTS = 500


def _plus_cells(shape: tuple[int, int], r: int, c: int, radius: int) -> tuple[tuple[int, int], ...]:
    h, w = map(int, shape)
    cells: list[tuple[int, int]] = []
    for dc in range(-radius, radius + 1):
        cc = c + dc
        if 0 <= cc < w:
            cells.append((r, cc))
    for dr in range(-radius, radius + 1):
        rr = r + dr
        if 0 <= rr < h and rr != r:
            cells.append((rr, c))
    return tuple(cells)


def _red_spans_in_grey_runs(line: np.ndarray) -> list[int]:
    spans: list[int] = []
    i = 0
    n = int(line.shape[0])
    while i < n:
        if int(line[i]) not in GREY_RED:
            i += 1
            continue
        reds: list[int] = []
        j = i
        while j < n and int(line[j]) in GREY_RED:
            if int(line[j]) == Color.RED:
                reds.append(j)
            j += 1
        if len(reds) >= 2:
            span = reds[-1] - reds[0] + 1
            if span >= 3 and span % 2 == 1:
                spans.append(int(span))
        i = j
    return spans


def _radius_candidates(grid: np.ndarray) -> list[int]:
    spans: set[int] = set()
    for r in range(grid.shape[0]):
        spans.update((span - 1) // 2 for span in _red_spans_in_grey_runs(grid[r, :]))
    for c in range(grid.shape[1]):
        spans.update((span - 1) // 2 for span in _red_spans_in_grey_runs(grid[:, c]))

    max_radius = max(1, min(grid.shape) // 2)
    ordered = sorted((radius for radius in spans if radius >= 1), reverse=True)
    for radius in range(max_radius, 0, -1):
        if radius not in spans:
            ordered.append(radius)
    return ordered


def _candidate_crosses(grid: np.ndarray, radius: int) -> list[dict]:
    h, w = map(int, grid.shape)
    candidates: list[dict] = []
    for r in range(h):
        for c in range(w):
            cells = _plus_cells(grid.shape, r, c, radius)
            if any(int(grid[rr, cc]) not in GREY_RED for rr, cc in cells):
                continue

            row_has_red = any(
                int(grid[r, cc]) == Color.RED
                for cc in range(max(0, c - radius), min(w, c + radius + 1))
            )
            col_has_red = any(
                int(grid[rr, c]) == Color.RED
                for rr in range(max(0, r - radius), min(h, r + radius + 1))
            )
            if not (row_has_red and col_has_red):
                continue

            red_cells = tuple((rr, cc) for rr, cc in cells if int(grid[rr, cc]) == Color.RED)
            if len(red_cells) < 3:
                continue

            candidates.append(
                {
                    "center": (int(r), int(c)),
                    "cells": cells,
                    "red_cells": red_cells,
                }
            )
    return candidates


def _select_min_cover(candidates: list[dict], all_red: set[tuple[int, int]]) -> list[dict] | None:
    reds = sorted(all_red)
    red_index = {coord: idx for idx, coord in enumerate(reds)}
    for cand in candidates:
        mask = 0
        for coord in cand["red_cells"]:
            mask |= 1 << red_index[coord]
        cand["mask"] = mask

    if not reds:
        return []

    full_mask = (1 << len(reds)) - 1
    by_red: list[list[int]] = [[] for _ in reds]
    for idx, cand in enumerate(candidates):
        mask = int(cand["mask"])
        for bit in range(len(reds)):
            if (mask >> bit) & 1:
                by_red[bit].append(idx)

    best_indices: list[int] | None = None
    best_centers: tuple[tuple[int, int], ...] | None = None
    seen: dict[int, int] = {}

    def search(covered: int, chosen: list[int]) -> None:
        nonlocal best_indices, best_centers
        if covered == full_mask:
            centers = tuple(sorted(candidates[idx]["center"] for idx in chosen))
            if (
                best_indices is None
                or len(chosen) < len(best_indices)
                or (len(chosen) == len(best_indices) and centers < best_centers)
            ):
                best_indices = chosen.copy()
                best_centers = centers
            return

        if best_indices is not None and len(chosen) >= len(best_indices):
            return
        prev = seen.get(covered)
        if prev is not None and prev <= len(chosen):
            return
        seen[covered] = len(chosen)

        uncovered = [bit for bit in range(len(reds)) if not ((covered >> bit) & 1)]
        target = min(
            uncovered,
            key=lambda bit: sum(1 for idx in by_red[bit] if int(candidates[idx]["mask"]) & ~covered),
        )
        options = [idx for idx in by_red[target] if int(candidates[idx]["mask"]) & ~covered]
        options.sort(
            key=lambda idx: (
                -int(int(candidates[idx]["mask"]) & ~covered).bit_count(),
                candidates[idx]["center"],
            )
        )
        for idx in options:
            if idx in chosen:
                continue
            search(covered | int(candidates[idx]["mask"]), chosen + [idx])

    search(0, [])
    if best_indices is None:
        return None
    return [candidates[idx] for idx in best_indices]


def _solve_crosses(grid: np.ndarray) -> tuple[int, list[dict]]:
    all_red = set(map(tuple, np.argwhere(grid == Color.RED)))
    assert all_red, "expected red plus markers"

    best_radius = None
    best_selected: list[dict] | None = None
    best_key: tuple[int, int, tuple[tuple[int, int], ...]] | None = None

    for radius in _radius_candidates(grid):
        candidates = _candidate_crosses(grid, radius)
        if not candidates:
            continue
        selected = _select_min_cover(candidates, all_red)
        if selected is None:
            continue
        centers = tuple(sorted(cand["center"] for cand in selected))
        key = (len(selected), -int(radius), centers)
        if best_key is None or key < best_key:
            best_key = key
            best_radius = int(radius)
            best_selected = selected

    assert best_radius is not None and best_selected is not None, "failed to identify the plus set"
    return best_radius, best_selected


def _solid_rectangles(grid: np.ndarray) -> list[dict]:
    colors = sorted(set(map(int, grid.ravel())) - {Color.BLACK, Color.GREY, Color.RED, Color.TEAL})
    rects: list[dict] = []
    if not colors:
        return rects
    for comp in find_components(
        grid,
        bg=Color.BLACK,
        colors=colors,
        connectivity=4,
        monochromatic=True,
    ):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        color = int(comp["color"])
        if int(comp["area"]) != h * w:
            continue
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
            continue
        rects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "height": h,
                "width": w,
                "color": color,
            }
        )
    return rects


def _classify_pair(a: dict, b: dict) -> dict | None:
    if (a["height"], a["width"]) != (b["height"], b["width"]):
        return None

    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]

    if (ar0, ar1) == (br0, br1):
        left, right = (a, b) if ac0 < bc0 else (b, a)
        _lr0, _lc0, _lr1, lc1 = left["bbox"]
        _rr0, rc0, _rr1, _rc1 = right["bbox"]
        gap = rc0 - lc1 - 1
        if gap >= 1:
            return {
                "orientation": "h",
                "first": left,
                "second": right,
                "row_span": (ar0, ar1),
                "gap_span": (lc1 + 1, rc0 - 1),
            }

    if (ac0, ac1) == (bc0, bc1):
        top, bottom = (a, b) if ar0 < br0 else (b, a)
        _tr0, _tc0, tr1, _tc1 = top["bbox"]
        br0, _bc0, _br1, _bc1 = bottom["bbox"]
        gap = br0 - tr1 - 1
        if gap >= 1:
            return {
                "orientation": "v",
                "first": top,
                "second": bottom,
                "col_span": (ac0, ac1),
                "gap_span": (tr1 + 1, br0 - 1),
            }

    return None


def _pair_rectangles(rects: list[dict]) -> list[dict]:
    n = len(rects)
    if n == 0:
        return []
    assert n % 2 == 0, "expected an even number of rectangles"

    options: dict[int, list[tuple[int, dict]]] = {idx: [] for idx in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            pair = _classify_pair(rects[i], rects[j])
            if pair is None:
                continue
            options[i].append((j, pair))
            options[j].append((i, pair))

    best: list[dict] | None = None
    best_key: tuple | None = None

    def search(remaining: tuple[int, ...], chosen: list[dict]) -> None:
        nonlocal best, best_key
        if not remaining:
            key = tuple(
                (
                    pair["orientation"],
                    pair["first"]["bbox"],
                    pair["second"]["bbox"],
                )
                for pair in chosen
            )
            if best is None or key < best_key:
                best = chosen.copy()
                best_key = key
            return

        first = remaining[0]
        tail = set(remaining[1:])
        for other, pair in sorted(
            options.get(first, []),
            key=lambda item: (item[1]["orientation"], item[1]["first"]["bbox"], item[1]["second"]["bbox"]),
        ):
            if other not in tail:
                continue
            rest = tuple(sorted(tail - {other}))
            search(rest, chosen + [pair])

    search(tuple(range(n)), [])
    assert best is not None, "failed to pair aligned rectangles"
    return best


def _fill_pair_gap(out: np.ndarray, pair: dict) -> None:
    if pair["orientation"] == "h":
        r0, r1 = pair["row_span"]
        c0, c1 = pair["gap_span"]
        width = c1 - c0 + 1
        span = width // 2
        out[r0 : r1 + 1, c0 : c0 + span] = int(pair["first"]["color"])
        out[r0 : r1 + 1, c1 - span + 1 : c1 + 1] = int(pair["second"]["color"])
    else:
        c0, c1 = pair["col_span"]
        r0, r1 = pair["gap_span"]
        height = r1 - r0 + 1
        span = height // 2
        out[r0 : r0 + span, c0 : c1 + 1] = int(pair["first"]["color"])
        out[r1 - span + 1 : r1 + 1, c0 : c1 + 1] = int(pair["second"]["color"])


def _apply_crosses(grid: np.ndarray, selected: list[dict]) -> tuple[np.ndarray, set[tuple[int, int]], list[tuple[int, int]]]:
    out = grid.copy()
    cross_cells: set[tuple[int, int]] = set()
    centers: list[tuple[int, int]] = []
    for cand in selected:
        centers.append(tuple(map(int, cand["center"])))
        for rr, cc in cand["cells"]:
            cross_cells.add((int(rr), int(cc)))
            if int(out[rr, cc]) == Color.GREY:
                out[rr, cc] = Color.TEAL
    return out, cross_cells, centers


def _render_expected(
    grid: np.ndarray,
    true_plus_cells: set[tuple[int, int]],
    centers: list[tuple[int, int]],
    pairs: list[dict],
) -> np.ndarray:
    cross_out = grid.copy()
    for rr, cc in true_plus_cells:
        if int(cross_out[rr, cc]) == Color.GREY:
            cross_out[rr, cc] = Color.TEAL

    out = cross_out.copy()
    for pair in pairs:
        if pair["orientation"] == "h":
            r0, r1 = pair["row_span"]
            matches = [center for center in centers if r0 <= center[0] <= r1]
            assert len(matches) == 1
        else:
            c0, c1 = pair["col_span"]
            matches = [center for center in centers if c0 <= center[1] <= c1]
            assert len(matches) == 1
        _fill_pair_gap(out, pair)

    for rr, cc in true_plus_cells:
        out[rr, cc] = cross_out[rr, cc]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the grey plus-sign masks indicated by red markers, then bisect the aligned rectangle gaps hit by those completed arm rows or columns.

    Input structure
    - A black-and-grey canvas contains several red marker pixels embedded in the grey substrate.
    - The red markers belong to 2 or 3 incomplete equal-arm plus signs that all share one radius.
    - Elsewhere, 2 or 3 pairs of same-sized solid rectangles are aligned either side-by-side or top-to-bottom, with a grey or black gap between them.

    Rule
    - Infer the shared plus radius from the widest odd red span inside a contiguous grey-or-red row or column run.
    - Among all radius-sized plus masks whose cells stay inside the grey/red substrate and whose row and column both contain red evidence, choose the minimum set that covers every red marker.
    - Recolor the grey cells of those selected plus masks teal.
    - For every aligned rectangle pair whose row span contains one selected plus center row, or whose column span contains one selected plus center column, bisect its gap as in the aligned-rectangle task: the near half takes the first rectangle's color and the far half takes the second rectangle's color, leaving an odd middle line unchanged.
    - Finally restore the selected plus-mask cells so the plus pixels are never overwritten by the gap fill.

    Key insight
    - The plus completion is also the control signal: only the rows and columns of the minimum red-explaining pluses activate the rectangle-gap bisection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    _radius, selected = _solve_crosses(g)
    cross_out, cross_cells, centers = _apply_crosses(g, selected)

    out = cross_out.copy()
    active_rows = {r for r, _c in centers}
    active_cols = {c for _r, c in centers}
    for pair in _pair_rectangles(_solid_rectangles(g)):
        if pair["orientation"] == "h":
            r0, r1 = pair["row_span"]
            if any(r0 <= r <= r1 for r in active_rows):
                _fill_pair_gap(out, pair)
        else:
            c0, c1 = pair["col_span"]
            if any(c0 <= c <= c1 for c in active_cols):
                _fill_pair_gap(out, pair)

    for rr, cc in cross_cells:
        out[rr, cc] = cross_out[rr, cc]
    return out


def _gen_place_centers(h: int, w: int, radius: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [
        (r, c)
        for r in range(radius + 1, h - radius - 1)
        for c in range(radius + 1, w - radius - 1)
    ]
    if not candidates:
        return None

    for _ in range(120):
        random.shuffle(candidates)
        centers: list[tuple[int, int]] = []
        occupied: set[tuple[int, int]] = set()
        for r, c in candidates:
            cells = set(_plus_cells((h, w), r, c, radius))
            if any(abs(r - rr) < 4 or abs(c - cc) < 4 for rr, cc in centers):
                continue
            if cells & occupied:
                continue
            if any(
                (rr + dr, cc + dc) in occupied
                for rr, cc in cells
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ):
                continue
            centers.append((r, c))
            occupied.update(cells)
            if len(centers) == count:
                return centers
    return None


def _gen_red_cells(shape: tuple[int, int], center: tuple[int, int], radius: int) -> set[tuple[int, int]]:
    h, w = shape
    r, c = center
    row_cells = [(r, cc) for cc in range(c - radius, c + radius + 1) if 0 <= cc < w]
    col_cells = [(rr, c) for rr in range(r - radius, r + radius + 1) if 0 <= rr < h]

    if random.choice((False, True)):
        chosen = set(row_cells)
        other = [cell for cell in col_cells if cell != (r, c)]
    else:
        chosen = set(col_cells)
        other = [cell for cell in row_cells if cell != (r, c)]

    extra_count = random.randint(1, min(2, len(other)))
    chosen.update(random.sample(other, extra_count))
    return chosen


def _gen_same_true_arm(
    a: tuple[int, int],
    b: tuple[int, int],
    centers: list[tuple[int, int]],
    owner: dict[tuple[int, int], int],
    *,
    axis: str,
) -> bool:
    ia = owner.get(a)
    ib = owner.get(b)
    if ia is None or ia != ib:
        return False
    cr, cc = centers[ia]
    if axis == "row":
        return a[0] == b[0] == cr
    return a[1] == b[1] == cc


def _gen_break_line_red_runs(
    grid: np.ndarray,
    centers: list[tuple[int, int]],
    owner: dict[tuple[int, int], int],
    axis: str,
) -> bool:
    line_count = grid.shape[0] if axis == "row" else grid.shape[1]
    for index in range(line_count):
        if axis == "row":
            reds = [c for c in range(grid.shape[1]) if int(grid[index, c]) == Color.RED]
        else:
            reds = [r for r in range(grid.shape[0]) if int(grid[r, index]) == Color.RED]

        for a, b in zip(reds, reds[1:]):
            if axis == "row":
                first = (index, a)
                second = (index, b)
                has_black = any(int(grid[index, c]) == Color.BLACK for c in range(a + 1, b))
                choices = [(index, c) for c in range(a + 1, b) if (index, c) not in owner]
            else:
                first = (a, index)
                second = (b, index)
                has_black = any(int(grid[r, index]) == Color.BLACK for r in range(a + 1, b))
                choices = [(r, index) for r in range(a + 1, b) if (r, index) not in owner]
            if _gen_same_true_arm(first, second, centers, owner, axis=axis):
                continue
            if has_black:
                continue
            if not choices:
                return False
            rr, cc = random.choice(choices)
            grid[rr, cc] = Color.BLACK
    return True


def _gen_break_accidental_red_runs(
    grid: np.ndarray,
    centers: list[tuple[int, int]],
    owner: dict[tuple[int, int], int],
) -> bool:
    return _gen_break_line_red_runs(grid, centers, owner, "row") and _gen_break_line_red_runs(
        grid, centers, owner, "col"
    )


def _gen_block_false_candidates(
    grid: np.ndarray,
    centers: list[tuple[int, int]],
    radius: int,
    true_plus_cells: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    true_centers = set(centers)
    candidate_centers = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(candidate_centers)

    for r, c in candidate_centers:
        if (r, c) in true_centers:
            continue
        cells = _plus_cells((h, w), r, c, radius)
        if any(int(grid[rr, cc]) not in GREY_RED for rr, cc in cells):
            continue
        if sum(int(grid[rr, cc]) == Color.RED for rr, cc in cells) < 3:
            continue
        row_has_red = any(int(grid[r, cc]) == Color.RED for cc in range(max(0, c - radius), min(w, c + radius + 1)))
        col_has_red = any(int(grid[rr, c]) == Color.RED for rr in range(max(0, r - radius), min(h, r + radius + 1)))
        if not (row_has_red and col_has_red):
            continue

        blockers = [(rr, cc) for rr, cc in cells if (rr, cc) not in true_plus_cells]
        if not blockers:
            return False
        rr, cc = random.choice(blockers)
        grid[rr, cc] = Color.BLACK
    return True


def _gen_add_background_black(grid: np.ndarray, true_plus_cells: set[tuple[int, int]]) -> bool:
    h, w = grid.shape
    total = h * w
    target = random.randint(int(0.18 * total), int(0.32 * total))
    current = int(np.sum(grid == Color.BLACK))
    free = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r, c) not in true_plus_cells and int(grid[r, c]) != Color.BLACK
    ]
    if current > target:
        return True
    if current + len(free) < target:
        return False
    random.shuffle(free)
    for r, c in free[: target - current]:
        grid[r, c] = Color.BLACK
    return True


def _gen_render_plus_input(
    h: int,
    w: int,
    radius: int,
    centers: list[tuple[int, int]],
) -> tuple[np.ndarray, dict[tuple[int, int], int], set[tuple[int, int]]]:
    grid = np.full((h, w), Color.GREY, dtype=int)
    owner: dict[tuple[int, int], int] = {}
    true_plus_cells: set[tuple[int, int]] = set()
    for idx, center in enumerate(centers):
        for cell in _plus_cells((h, w), center[0], center[1], radius):
            owner[cell] = idx
            true_plus_cells.add(cell)
        for rr, cc in _gen_red_cells((h, w), center, radius):
            grid[rr, cc] = Color.RED
    return grid, owner, true_plus_cells


def _sample_orientations(count: int) -> list[str]:
    if count == 2:
        out = ["h", "v"]
    else:
        out = ["h", "v", random.choice(("h", "v"))]
    random.shuffle(out)
    return out


def _sample_gap_widths(count: int) -> list[int]:
    values = [random.choice((3, 5, 7)), random.choice((4, 6))]
    if count == 3:
        values.append(random.choice((3, 4, 5, 6, 7)))
    random.shuffle(values)
    return values


def _sample_gap_colors(count: int) -> list[int]:
    colors = [Color.GREY, Color.BLACK]
    if count == 3:
        colors.append(random.choice((Color.GREY, Color.BLACK)))
    random.shuffle(colors)
    return colors


def _bbox_is_free(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int = 1) -> bool:
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(blocked.shape[0], r0 + h + padding)
    cc1 = min(blocked.shape[1], c0 + w + padding)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _mark_bbox(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int = 1) -> None:
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(blocked.shape[0], r0 + h + padding)
    cc1 = min(blocked.shape[1], c0 + w + padding)
    blocked[rr0:rr1, cc0:cc1] = True


def _gen_place_pairs(
    grid: np.ndarray,
    centers: list[tuple[int, int]],
    true_plus_cells: set[tuple[int, int]],
) -> list[dict] | None:
    count = len(centers)
    pair_colors = random.sample(list(RECT_COLORS), 2 * count)
    orientations = _sample_orientations(count)
    gaps = _sample_gap_widths(count)
    gap_colors = _sample_gap_colors(count)
    dims = [(random.randint(2, 3), random.randint(2, 3)) for _ in range(count)]

    blocked = np.zeros_like(grid, dtype=bool)
    for rr, cc in true_plus_cells:
        blocked[max(0, rr - 1) : min(grid.shape[0], rr + 2), max(0, cc - 1) : min(grid.shape[1], cc + 2)] = True

    infos: list[dict] = []
    for idx, center in enumerate(centers):
        orientation = orientations[idx]
        gap = gaps[idx]
        gap_color = int(gap_colors[idx])
        color_a = int(pair_colors[2 * idx])
        color_b = int(pair_colors[2 * idx + 1])
        rect_h, rect_w = dims[idx]

        if orientation == "h":
            patch_h = rect_h
            patch_w = 2 * rect_w + gap
            top_options = [
                top
                for top in range(center[0] - rect_h + 1, center[0] + 1)
                if 1 <= top and top + patch_h <= grid.shape[0] - 1
            ]
            candidates: list[tuple[int, int]] = []
            for top in top_options:
                for left in range(1, grid.shape[1] - patch_w):
                    if _bbox_is_free(blocked, top, left, patch_h, patch_w):
                        candidates.append((top, left))
            if not candidates:
                return None
            top, left = random.choice(candidates)
            grid[top : top + rect_h, left : left + rect_w] = color_a
            grid[top : top + rect_h, left + rect_w : left + rect_w + gap] = gap_color
            grid[top : top + rect_h, left + rect_w + gap : left + 2 * rect_w + gap] = color_b
            _mark_bbox(blocked, top, left, patch_h, patch_w)
            infos.append(
                {
                    "orientation": "h",
                    "first": {
                        "bbox": (top, left, top + rect_h - 1, left + rect_w - 1),
                        "height": rect_h,
                        "width": rect_w,
                        "color": color_a,
                    },
                    "second": {
                        "bbox": (top, left + rect_w + gap, top + rect_h - 1, left + 2 * rect_w + gap - 1),
                        "height": rect_h,
                        "width": rect_w,
                        "color": color_b,
                    },
                    "row_span": (top, top + rect_h - 1),
                    "gap_span": (left + rect_w, left + rect_w + gap - 1),
                }
            )
        else:
            patch_h = 2 * rect_h + gap
            patch_w = rect_w
            left_options = [
                left
                for left in range(center[1] - rect_w + 1, center[1] + 1)
                if 1 <= left and left + patch_w <= grid.shape[1] - 1
            ]
            candidates = []
            for left in left_options:
                for top in range(1, grid.shape[0] - patch_h):
                    if _bbox_is_free(blocked, top, left, patch_h, patch_w):
                        candidates.append((top, left))
            if not candidates:
                return None
            top, left = random.choice(candidates)
            grid[top : top + rect_h, left : left + rect_w] = color_a
            grid[top + rect_h : top + rect_h + gap, left : left + rect_w] = gap_color
            grid[top + rect_h + gap : top + 2 * rect_h + gap, left : left + rect_w] = color_b
            _mark_bbox(blocked, top, left, patch_h, patch_w)
            infos.append(
                {
                    "orientation": "v",
                    "first": {
                        "bbox": (top, left, top + rect_h - 1, left + rect_w - 1),
                        "height": rect_h,
                        "width": rect_w,
                        "color": color_a,
                    },
                    "second": {
                        "bbox": (top + rect_h + gap, left, top + 2 * rect_h + gap - 1, left + rect_w - 1),
                        "height": rect_h,
                        "width": rect_w,
                        "color": color_b,
                    },
                    "col_span": (left, left + rect_w - 1),
                    "gap_span": (top + rect_h, top + rect_h + gap - 1),
                }
            )

    return infos


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 28)
    w = random.randint(20, 28)
    radius = random.choice((2, 3))
    count = random.choice((2, 3))

    centers = _gen_place_centers(h, w, radius, count)
    if centers is None:
        return None

    inp, owner, true_plus_cells = _gen_render_plus_input(h, w, radius, centers)
    if not _gen_break_accidental_red_runs(inp, centers, owner):
        return None
    if not _gen_block_false_candidates(inp, centers, radius, true_plus_cells):
        return None
    if not _gen_add_background_black(inp, true_plus_cells):
        return None

    pair_infos = _gen_place_pairs(inp, centers, true_plus_cells)
    if pair_infos is None:
        return None

    expected = _render_expected(inp, true_plus_cells, centers, pair_infos)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cross-gap-bisection ARC input")
