import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

AXIS_ORDER = (RED, BLUE, GREEN)
AXIS_COLORS = set(AXIS_ORDER)
PRIORITY_LOW_TO_HIGH = (BLUE, RED, AZURE, GREEN, YELLOW, GREY, MAGENTA, ORANGE, MAROON)
PRIORITY = {color: rank for rank, color in enumerate(PRIORITY_LOW_TO_HIGH)}
CONTENT_COLORS = PRIORITY_LOW_TO_HIGH

GRID_MIN = 12
GRID_MAX = 18
GEN_RETRIES = 400

SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)


def _crop_non_bg(grid: np.ndarray) -> np.ndarray:
    pts = np.argwhere(grid != BG)
    if pts.size == 0:
        return np.zeros((1, 1), dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_axis_row(row: np.ndarray, color: int) -> bool:
    return np.all(np.isin(row, tuple(AXIS_COLORS))) and int(np.count_nonzero(row == color)) >= row.size - 2


def _is_axis_col(col: np.ndarray, color: int) -> bool:
    return np.all(np.isin(col, tuple(AXIS_COLORS))) and int(np.count_nonzero(col == color)) >= col.size - 2


def _find_horizontal_marker(grid: np.ndarray, row: int, color: int) -> Tuple[str, List[Tuple[int, int]]]:
    h, w = grid.shape
    matches: List[Tuple[str, List[Tuple[int, int]]]] = []

    if row > 0 and grid[row - 1, 0] == color and grid[row - 1, 1] == color:
        matches.append(("up", [(row - 1, 0), (row - 1, 1)]))
    if row > 0 and grid[row - 1, w - 1] == color and grid[row - 1, w - 2] == color:
        matches.append(("up", [(row - 1, w - 1), (row - 1, w - 2)]))
    if row + 1 < h and grid[row + 1, 0] == color and grid[row + 1, 1] == color:
        matches.append(("down", [(row + 1, 0), (row + 1, 1)]))
    if row + 1 < h and grid[row + 1, w - 1] == color and grid[row + 1, w - 2] == color:
        matches.append(("down", [(row + 1, w - 1), (row + 1, w - 2)]))

    assert len(matches) == 1, f"expected one horizontal marker for color {color}"
    return matches[0]


def _find_vertical_marker(grid: np.ndarray, col: int, color: int) -> Tuple[str, List[Tuple[int, int]]]:
    h, w = grid.shape
    matches: List[Tuple[str, List[Tuple[int, int]]]] = []

    if col > 0 and grid[0, col - 1] == color and grid[1, col - 1] == color:
        matches.append(("left", [(0, col - 1), (1, col - 1)]))
    if col > 0 and grid[h - 1, col - 1] == color and grid[h - 2, col - 1] == color:
        matches.append(("left", [(h - 1, col - 1), (h - 2, col - 1)]))
    if col + 1 < w and grid[0, col + 1] == color and grid[1, col + 1] == color:
        matches.append(("right", [(0, col + 1), (1, col + 1)]))
    if col + 1 < w and grid[h - 1, col + 1] == color and grid[h - 2, col + 1] == color:
        matches.append(("right", [(h - 1, col + 1), (h - 2, col + 1)]))

    assert len(matches) == 1, f"expected one vertical marker for color {color}"
    return matches[0]


def _detect_axes(grid: np.ndarray) -> List[Dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    axes: List[Dict[str, object]] = []

    for color in AXIS_ORDER:
        row_candidates = [r for r in range(h) if _is_axis_row(g[r, :], color)]
        col_candidates = [c for c in range(w) if _is_axis_col(g[:, c], color)]

        if not row_candidates and not col_candidates:
            continue
        assert not (row_candidates and col_candidates), f"color {color} cannot define both a row and a col axis"

        if row_candidates:
            assert len(row_candidates) == 1
            row = int(row_candidates[0])
            target, marker_cells = _find_horizontal_marker(g, row, color)
            axes.append(
                {
                    "color": color,
                    "orientation": "h",
                    "pos": row,
                    "target": target,
                    "marker_cells": marker_cells,
                }
            )
        else:
            assert len(col_candidates) == 1
            col = int(col_candidates[0])
            target, marker_cells = _find_vertical_marker(g, col, color)
            axes.append(
                {
                    "color": color,
                    "orientation": "v",
                    "pos": col,
                    "target": target,
                    "marker_cells": marker_cells,
                }
            )

    return axes


def _instruction_mask(shape: Tuple[int, int], axes: Sequence[Dict[str, object]]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for axis in axes:
        if axis["orientation"] == "h":
            mask[int(axis["pos"]), :] = True
        else:
            mask[:, int(axis["pos"])] = True
        for r, c in axis["marker_cells"]:
            mask[int(r), int(c)] = True
    return mask


def _should_replace(existing: int, incoming: int) -> bool:
    if existing == BG:
        return True
    return PRIORITY[int(incoming)] > PRIORITY[int(existing)]


def _apply_fold(content: np.ndarray, axis: Dict[str, object]) -> np.ndarray:
    g = np.asarray(content, dtype=int)
    h, w = g.shape
    out = g.copy()

    if axis["orientation"] == "h":
        row = int(axis["pos"])
        if axis["target"] == "up":
            src = np.argwhere(g[row + 1 :, :] != BG)
            src[:, 0] += row + 1
            out[row + 1 :, :] = BG
            for sr, sc in src:
                tr = 2 * row - int(sr)
                if 0 <= tr < h and _should_replace(int(out[tr, sc]), int(g[sr, sc])):
                    out[tr, sc] = int(g[sr, sc])
        else:
            src = np.argwhere(g[:row, :] != BG)
            out[:row, :] = BG
            for sr, sc in src:
                tr = 2 * row - int(sr)
                if 0 <= tr < h and _should_replace(int(out[tr, sc]), int(g[sr, sc])):
                    out[tr, sc] = int(g[sr, sc])
    else:
        col = int(axis["pos"])
        if axis["target"] == "left":
            src = np.argwhere(g[:, col + 1 :] != BG)
            src[:, 1] += col + 1
            out[:, col + 1 :] = BG
            for sr, sc in src:
                tc = 2 * col - int(sc)
                if 0 <= tc < w and _should_replace(int(out[sr, tc]), int(g[sr, sc])):
                    out[sr, tc] = int(g[sr, sc])
        else:
            src = np.argwhere(g[:, :col] != BG)
            out[:, :col] = BG
            for sr, sc in src:
                tc = 2 * col - int(sc)
                if 0 <= tc < w and _should_replace(int(out[sr, tc]), int(g[sr, sc])):
                    out[sr, tc] = int(g[sr, sc])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat full red/blue/green lines as fold axes and use their endpoint elbow markers
    to decide which side receives the mirrored content from the opposite side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    axes = _detect_axes(g)
    assert axes, "expected at least one fold axis"

    content = g.copy()
    content[_instruction_mask(g.shape, axes)] = BG

    for color in AXIS_ORDER:
        for axis in axes:
            if int(axis["color"]) == color:
                content = _apply_fold(content, axis)

    return _crop_non_bg(content)


def _shape_variants(shape: Sequence[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    variants = set()
    out: List[List[Tuple[int, int]]] = []
    pts0 = [(int(r), int(c)) for r, c in shape]
    for rot in range(4):
        rotated = pts0
        for _ in range(rot):
            rotated = [(c, -r) for r, c in rotated]
        for flip in (False, True):
            pts = [(r, -c) if flip else (r, c) for r, c in rotated]
            r0 = min(r for r, _ in pts)
            c0 = min(c for _, c in pts)
            norm = tuple(sorted((r - r0, c - c0) for r, c in pts))
            if norm not in variants:
                variants.add(norm)
                out.append(list(norm))
    return out


SHAPE_VARIANTS = [variant for shape in SHAPES for variant in _shape_variants(shape)]


def _line_cells(axis: Dict[str, object], shape: Tuple[int, int]) -> set[Tuple[int, int]]:
    h, w = shape
    pos = int(axis["pos"])
    if axis["orientation"] == "h":
        return {(pos, c) for c in range(w)}
    return {(r, pos) for r in range(h)}


def _marker_cells(axis: Dict[str, object], shape: Tuple[int, int], end: str) -> List[Tuple[int, int]]:
    h, w = shape
    pos = int(axis["pos"])
    target = str(axis["target"])
    if axis["orientation"] == "h":
        rr = pos - 1 if target == "up" else pos + 1
        if end == "left":
            return [(rr, 0), (rr, 1)]
        return [(rr, w - 1), (rr, w - 2)]
    cc = pos - 1 if target == "left" else pos + 1
    if end == "top":
        return [(0, cc), (1, cc)]
    return [(h - 1, cc), (h - 2, cc)]


def _sample_axes(rng: random.Random, h: int, w: int) -> Optional[List[Dict[str, object]]]:
    n_axes = rng.randint(1, 3)
    colors = sorted(rng.sample(list(AXIS_ORDER), n_axes), key=AXIS_ORDER.index)

    axes: List[Dict[str, object]] = []
    used_rows: List[int] = []
    used_cols: List[int] = []

    for color in colors:
        orientation = rng.choice(("h", "v"))
        if orientation == "h":
            target = rng.choice(("up", "down"))
            if target == "up":
                row_choices = [r for r in range(2, h - 2) if r >= (h - 1 + 1) // 2]
            else:
                row_choices = [r for r in range(2, h - 2) if r <= (h - 1) // 2]
            row_choices = [r for r in row_choices if all(abs(r - prev) >= 2 for prev in used_rows)]
            if not row_choices:
                return None
            pos = rng.choice(row_choices)
            used_rows.append(pos)
        else:
            target = rng.choice(("left", "right"))
            if target == "left":
                col_choices = [c for c in range(2, w - 2) if c >= (w - 1 + 1) // 2]
            else:
                col_choices = [c for c in range(2, w - 2) if c <= (w - 1) // 2]
            col_choices = [c for c in col_choices if all(abs(c - prev) >= 2 for prev in used_cols)]
            if not col_choices:
                return None
            pos = rng.choice(col_choices)
            used_cols.append(pos)

        axes.append({"color": color, "orientation": orientation, "pos": pos, "target": target})

    base_line_cells = set()
    for axis in axes:
        base_line_cells |= _line_cells(axis, (h, w))

    used_marker_cells: set[Tuple[int, int]] = set()
    for axis in axes:
        ends = ("left", "right") if axis["orientation"] == "h" else ("top", "bottom")
        rng.shuffle(list(ends))
        options = list(ends)
        rng.shuffle(options)
        chosen = None
        for end in options:
            cells = _marker_cells(axis, (h, w), end)
            if any(cell in base_line_cells for cell in cells):
                continue
            if any(cell in used_marker_cells for cell in cells):
                continue
            chosen = cells
            break
        if chosen is None:
            return None
        axis["marker_cells"] = chosen
        used_marker_cells.update(chosen)

    return axes


def _bounds_for_axes(shape: Tuple[int, int], axes: Sequence[Dict[str, object]]) -> Optional[Tuple[int, int, int, int, int, int, int, int]]:
    h, w = shape
    final_r0, final_r1 = 0, h - 1
    final_c0, final_c1 = 0, w - 1
    core_r0, core_r1 = 0, h - 1
    core_c0, core_c1 = 0, w - 1

    for axis in axes:
        pos = int(axis["pos"])
        if axis["orientation"] == "h":
            if axis["target"] == "up":
                src_depth = h - 1 - pos
                final_r1 = min(final_r1, pos - 1)
                core_r0 = max(core_r0, pos - src_depth)
                core_r1 = min(core_r1, pos - 1)
            else:
                src_depth = pos
                final_r0 = max(final_r0, pos + 1)
                core_r0 = max(core_r0, pos + 1)
                core_r1 = min(core_r1, pos + src_depth)
        else:
            if axis["target"] == "left":
                src_depth = w - 1 - pos
                final_c1 = min(final_c1, pos - 1)
                core_c0 = max(core_c0, pos - src_depth)
                core_c1 = min(core_c1, pos - 1)
            else:
                src_depth = pos
                final_c0 = max(final_c0, pos + 1)
                core_c0 = max(core_c0, pos + 1)
                core_c1 = min(core_c1, pos + src_depth)

    if final_r0 > final_r1 or final_c0 > final_c1:
        return None
    if core_r0 > core_r1 or core_c0 > core_c1:
        return None
    if core_r1 - core_r0 + 1 < 3 or core_c1 - core_c0 + 1 < 3:
        return None
    return final_r0, final_r1, final_c0, final_c1, core_r0, core_r1, core_c0, core_c1


def _place_final_pattern(
    rng: random.Random,
    shape: Tuple[int, int],
    core_bounds: Tuple[int, int, int, int],
    instruction_mask: np.ndarray,
) -> Optional[np.ndarray]:
    h, w = shape
    core_r0, core_r1, core_c0, core_c1 = core_bounds
    max_h = min(6, core_r1 - core_r0 + 1)
    max_w = min(6, core_c1 - core_c0 + 1)
    if max_h < 3 or max_w < 3:
        return None

    for _ in range(80):
        box_h = rng.randint(3, max_h)
        box_w = rng.randint(3, max_w)
        top = rng.randint(core_r0, core_r1 - box_h + 1)
        left = rng.randint(core_c0, core_c1 - box_w + 1)

        out = np.zeros((h, w), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        colors_used = set()
        total_cells = 0
        n_components = rng.randint(2, 4)

        ok = True
        for _comp in range(n_components):
            placed = False
            for _ in range(60):
                shape_cells = rng.choice(SHAPE_VARIANTS)
                sh = max(r for r, _ in shape_cells) + 1
                sw = max(c for _, c in shape_cells) + 1
                if sh > box_h or sw > box_w:
                    continue
                r0 = rng.randint(top, top + box_h - sh)
                c0 = rng.randint(left, left + box_w - sw)
                cells = [(r0 + dr, c0 + dc) for dr, dc in shape_cells]
                if any(instruction_mask[r, c] for r, c in cells):
                    continue
                if any(occupied[r, c] for r, c in cells):
                    continue
                if any(
                    0 <= nr < h and 0 <= nc < w and occupied[nr, nc]
                    for r, c in cells
                    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                    if (nr, nc) not in cells
                ):
                    continue
                color = rng.choice(CONTENT_COLORS)
                for r, c in cells:
                    out[r, c] = color
                    occupied[r, c] = True
                colors_used.add(color)
                total_cells += len(cells)
                placed = True
                break
            if not placed:
                ok = False
                break

        if ok and total_cells >= 5 and len(colors_used) >= 2:
            return out

    return None


def _mirror_to_source(
    axis: Dict[str, object], cell: Tuple[int, int], shape: Tuple[int, int]
) -> Optional[Tuple[int, int]]:
    h, w = shape
    r, c = cell
    pos = int(axis["pos"])
    if axis["orientation"] == "h":
        if axis["target"] == "up":
            if not (0 <= r < pos):
                return None
            sr = 2 * pos - r
            if sr <= pos or sr >= h:
                return None
            return sr, c
        if not (pos < r < h):
            return None
        sr = 2 * pos - r
        if sr >= pos or sr < 0:
            return None
        return sr, c

    if axis["target"] == "left":
        if not (0 <= c < pos):
            return None
        sc = 2 * pos - c
        if sc <= pos or sc >= w:
            return None
        return r, sc
    if not (pos < c < w):
        return None
    sc = 2 * pos - c
    if sc >= pos or sc < 0:
        return None
    return r, sc


def _lower_or_equal_colors(color: int) -> List[int]:
    rank = PRIORITY[int(color)]
    return [candidate for candidate in CONTENT_COLORS if PRIORITY[int(candidate)] <= rank]


def _reverse_unfold(
    rng: random.Random, final_pattern: np.ndarray, axes: Sequence[Dict[str, object]], instruction_mask: np.ndarray
) -> Optional[np.ndarray]:
    pattern = final_pattern.copy()
    h, w = pattern.shape

    for axis in reversed(axes):
        current_cells = [(int(r), int(c)) for r, c in np.argwhere(pattern != BG)]
        candidates: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        for cell in current_cells:
            src = _mirror_to_source(axis, cell, (h, w))
            if src is None:
                continue
            if instruction_mask[src]:
                continue
            if pattern[src] != BG:
                continue
            candidates.append((cell, src))

        if not candidates:
            return None

        rng.shuffle(candidates)
        max_take = min(len(candidates), 2 + rng.randint(0, 3))
        take = rng.randint(1, max_take)
        used_sources = set()

        placed = 0
        for tgt, src in candidates:
            if src in used_sources:
                continue
            tr, tc = tgt
            sr, sc = src
            base_color = int(pattern[tr, tc])
            options = _lower_or_equal_colors(base_color)
            if len(options) > 1 and rng.random() < 0.7:
                non_equal = [color for color in options if color != base_color]
                color = rng.choice(non_equal) if non_equal else base_color
            else:
                color = base_color
            pattern[sr, sc] = int(color)
            used_sources.add(src)
            placed += 1
            if placed >= take:
                break

        if placed == 0:
            return None

    return pattern


def _draw_axes(pattern: np.ndarray, axes: Sequence[Dict[str, object]]) -> np.ndarray:
    out = pattern.copy()
    h, w = out.shape
    for axis in axes:
        color = int(axis["color"])
        pos = int(axis["pos"])
        if axis["orientation"] == "h":
            out[pos, :] = color
        else:
            out[:, pos] = color
    for axis in axes:
        color = int(axis["color"])
        for r, c in axis["marker_cells"]:
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = color
    return out


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)

    axes = _sample_axes(rng, h, w)
    if not axes:
        return None

    bounds = _bounds_for_axes((h, w), axes)
    if bounds is None:
        return None
    _fr0, _fr1, _fc0, _fc1, cr0, cr1, cc0, cc1 = bounds

    instruction_mask = _instruction_mask((h, w), axes)
    final_pattern = _place_final_pattern(rng, (h, w), (cr0, cr1, cc0, cc1), instruction_mask)
    if final_pattern is None:
        return None

    unfolded = _reverse_unfold(rng, final_pattern, axes, instruction_mask)
    if unfolded is None:
        return None

    grid = _draw_axes(unfolded, axes)
    expected = _crop_non_bg(final_pattern)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fold instruction puzzle")
