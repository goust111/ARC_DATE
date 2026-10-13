import random
from collections import Counter

import numpy as np


BG = 0
DIVIDER_COLOR = 5
MARKER_COLOR = 9
SHAPE_COLORS = (1, 2, 3, 4, 6, 7, 8)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

BASE_MASKS = (
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
)

GEN_ATTEMPTS = 400
PLACEMENT_TRIALS = 500


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _find_divider_column(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    candidates = []
    for c in range(1, w - 1):
        if c != w - c - 1:
            continue
        if int(grid[0, c]) == bg:
            continue
        if np.all(grid[:, c] == grid[0, c]):
            candidates.append(c)
    assert len(candidates) == 1, "expected exactly one full-height divider column"
    return int(candidates[0])


def _infer_marker_color(
    grid: np.ndarray, components: list[list[tuple[int, int]]], priority_colors: list[int]
) -> int:
    candidate = None
    for comp in components:
        counts = Counter(int(grid[r, c]) for r, c in comp)
        singles = {color for color, cnt in counts.items() if cnt == 1}
        assert singles, "every left-side component must contain a singleton marker pixel"
        if candidate is None:
            candidate = singles
        else:
            candidate &= singles
    assert candidate, "could not infer a shared marker color"
    candidate -= set(priority_colors)
    assert len(candidate) == 1, "marker color must be unique after excluding priority colors"
    return int(next(iter(candidate)))


def _project_coords(
    shape_coords: list[tuple[int, int]], divider: int, marker_side: str
) -> list[tuple[int, int]]:
    _, min_c, _, max_c = _bbox(shape_coords)
    base_shift_c = 2 * divider - min_c - max_c
    extra_r, extra_c = ((0, 1) if marker_side == "left" else (1, 0))
    return [(r + extra_r, c + base_shift_c + extra_c) for r, c in shape_coords]


def _extract_shapes(grid: np.ndarray) -> tuple[int, list[dict], list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    divider = _find_divider_column(g, bg)

    left_mask = np.zeros((h, w), dtype=bool)
    left_mask[:, :divider] = g[:, :divider] != bg
    comps = _components(left_mask)
    assert comps, "expected at least one left-side shape component"

    priority_colors = [int(g[h - 1, c]) for c in range(divider + 1, w) if int(g[h - 1, c]) != bg]
    assert priority_colors, "priority row on the right half must be non-empty"

    marker_color = _infer_marker_color(g, comps, priority_colors)
    shapes = []
    for comp in comps:
        counts = Counter(int(g[r, c]) for r, c in comp)
        marker_pts = [(r, c) for r, c in comp if int(g[r, c]) == marker_color]
        assert len(marker_pts) == 1, "each component must have exactly one marker pixel"
        shape_colors = [color for color in counts if color != marker_color]
        assert len(shape_colors) == 1, "each component must contain exactly one solid shape color"

        color = int(shape_colors[0])
        shape_coords = [(r, c) for r, c in comp if int(g[r, c]) == color]
        assert len(shape_coords) > 1, "shape area must be larger than the 1-pixel marker"
        min_r, min_c, max_r, max_c = _bbox(shape_coords)
        shape_set = set(shape_coords)
        mr, mc = marker_pts[0]

        marker_side = None
        if mc == min_c - 1 and min_r <= mr <= max_r and (mr, mc + 1) in shape_set:
            marker_side = "left"
        if mr == min_r - 1 and min_c <= mc <= max_c and (mr + 1, mc) in shape_set:
            assert marker_side is None, "marker position is ambiguous"
            marker_side = "top"
        assert marker_side is not None, "marker must sit just outside the left or top edge"

        shapes.append(
            {
                "color": color,
                "coords": shape_coords,
                "marker_side": marker_side,
                "projected": _project_coords(shape_coords, divider, marker_side),
            }
        )

    return divider, shapes, priority_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror-project the left shapes into the right half, then stack them using the
    color order encoded by the bottom row of the right half.

    Each left-side object is a solid monochrome shape with a single adjacent
    marker pixel. The marker side adds a small extra offset after the horizontal
    mirror placement: left-side marker => shift right by 1, top-side marker =>
    shift down by 1. Shapes are then drawn from lowest to highest priority using
    the right-half bottom-row color sequence from left to right.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    divider, shapes, priority_colors = _extract_shapes(g)
    by_color = {shape["color"]: shape for shape in shapes}
    assert set(priority_colors) == set(by_color), "priority row colors must match the left-side shapes"

    out = g.copy()
    for color in priority_colors:
        for r, c in by_color[color]["projected"]:
            if 0 <= r < h and divider < c < w:
                out[r, c] = int(color)
    return out


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _random_mask() -> np.ndarray:
    mask = random.choice(BASE_MASKS).copy()
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    mask = np.rot90(mask, random.randrange(4))
    return _trim(mask)


def _mask_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if bool(mask[r, c]):
                cells.append((top + r, left + c))
    return cells


def _marker_options(mask: np.ndarray, side: str) -> list[int]:
    if side == "left":
        return [r for r in range(mask.shape[0]) if bool(mask[r, 0])]
    return [c for c in range(mask.shape[1]) if bool(mask[0, c])]


def _object_cells(shape_cells: list[tuple[int, int]], marker: tuple[int, int]) -> list[tuple[int, int]]:
    return shape_cells + [marker]


def _expand8(cells: list[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _all_in_projected_bounds(cells: list[tuple[int, int]], h: int, w: int, divider: int) -> bool:
    return all(0 <= r < h and divider < c < w for r, c in cells)


def _place_object(
    mask: np.ndarray,
    side: str,
    h: int,
    left_w: int,
    blocked_left: set[tuple[int, int]],
    projected_overlap_with: set[tuple[int, int]] | None = None,
    near_bbox: tuple[int, int, int, int] | None = None,
) -> dict | None:
    mh, mw = mask.shape
    top_min = 1 if side == "top" else 0
    top_max = h - mh - (1 if side == "top" else 0)
    left_min = 1 if side == "left" else 0
    left_max = left_w - mw
    if top_min > top_max or left_min > left_max:
        return None

    marker_indices = _marker_options(mask, side)
    if not marker_indices:
        return None

    full_rows = list(range(top_min, top_max + 1))
    full_cols = list(range(left_min, left_max + 1))
    if near_bbox is not None:
        nr0, nc0, nr1, nc1 = near_bbox
        row_lo = max(top_min, nr0 - mh - 2)
        row_hi = min(top_max, nr1 + 2)
        col_lo = max(left_min, nc0 - mw - 2)
        col_hi = min(left_max, nc1 + 2)
        focus_rows = list(range(row_lo, row_hi + 1)) if row_lo <= row_hi else []
        focus_cols = list(range(col_lo, col_hi + 1)) if col_lo <= col_hi else []
    else:
        focus_rows = []
        focus_cols = []

    row_choices = focus_rows if focus_rows else full_rows
    col_choices = focus_cols if focus_cols else full_cols

    for _ in range(PLACEMENT_TRIALS):
        top = random.choice(row_choices)
        left = random.choice(col_choices)
        marker_index = random.choice(marker_indices)
        shape_cells = _mask_cells(mask, top, left)
        if side == "left":
            marker = (top + marker_index, left - 1)
        else:
            marker = (top - 1, left + marker_index)
        object_cells = _object_cells(shape_cells, marker)

        if any((r, c) in blocked_left for r, c in object_cells):
            continue

        projected = _project_coords(shape_cells, left_w, side)
        if not _all_in_projected_bounds(projected, h, 2 * left_w + 1, left_w):
            continue
        if projected_overlap_with is not None and set(projected).isdisjoint(projected_overlap_with):
            continue

        return {
            "cells": shape_cells,
            "marker": marker,
            "side": side,
            "projected": projected,
            "bbox": _bbox(shape_cells),
        }

    return None


def _priority_positions(left_w: int, count: int) -> list[int]:
    if count == 1:
        return [left_w + random.randint(1, left_w)]

    gaps = [random.choice((1, 2)) for _ in range(count - 1)]
    span = 1 + sum(gaps)
    max_start = 2 * left_w - span + 1
    start = random.randint(left_w + 1, max_start)
    out = [start]
    for gap in gaps:
        out.append(out[-1] + gap)
    return out


def _render_expected(grid: np.ndarray, objects: list[dict], priority: list[int]) -> np.ndarray:
    out = grid.copy()
    by_color = {obj["color"]: obj for obj in objects}
    for color in priority:
        for r, c in by_color[color]["projected"]:
            out[r, c] = int(color)
    return out


def _build_input() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(10, 15)
    left_w = random.randint(8, 12)
    total_w = 2 * left_w + 1
    count = random.randint(2, 4)

    colors = random.sample(SHAPE_COLORS, count)
    masks = [_random_mask() for _ in range(count)]

    sides = ["left", "top"]
    if count > 2:
        sides.extend(random.choice(("left", "top")) for _ in range(count - 2))
    random.shuffle(sides)

    order = list(range(count))
    random.shuffle(order)
    colors = [colors[i] for i in order]
    masks = [masks[i] for i in order]
    sides = [sides[i] for i in order]

    objects: list[dict] = []
    blocked_left: set[tuple[int, int]] = set()

    first = _place_object(masks[0], sides[0], h, left_w, blocked_left)
    if first is None:
        return None
    first["color"] = colors[0]
    objects.append(first)
    blocked_left |= _expand8(_object_cells(first["cells"], first["marker"]), h, left_w)

    second = _place_object(
        masks[1],
        sides[1],
        h,
        left_w,
        blocked_left,
        projected_overlap_with=set(first["projected"]),
        near_bbox=first["bbox"],
    )
    if second is None:
        return None
    second["color"] = colors[1]
    objects.append(second)
    blocked_left |= _expand8(_object_cells(second["cells"], second["marker"]), h, left_w)

    for i in range(2, count):
        overlap_target = None
        near_bbox = None
        if random.random() < 0.5:
            overlap_target = set().union(*(set(obj["projected"]) for obj in objects))
            near_bbox = _bbox([cell for obj in objects for cell in obj["cells"]])
        obj = _place_object(
            masks[i],
            sides[i],
            h,
            left_w,
            blocked_left,
            projected_overlap_with=overlap_target,
            near_bbox=near_bbox,
        )
        if obj is None:
            return None
        obj["color"] = colors[i]
        objects.append(obj)
        blocked_left |= _expand8(_object_cells(obj["cells"], obj["marker"]), h, left_w)

    projected_sets = [set(obj["projected"]) for obj in objects]
    overlap_count = 0
    for i in range(len(projected_sets)):
        for j in range(i + 1, len(projected_sets)):
            overlap_count += len(projected_sets[i] & projected_sets[j])
    if overlap_count == 0:
        return None

    grid = np.full((h, total_w), BG, dtype=int)
    grid[:, left_w] = DIVIDER_COLOR
    for obj in objects:
        color = int(obj["color"])
        for r, c in obj["cells"]:
            grid[r, c] = color
        mr, mc = obj["marker"]
        grid[mr, mc] = MARKER_COLOR

    priority = colors.copy()
    random.shuffle(priority)
    positions = _priority_positions(left_w, count)
    for color, col in zip(priority, positions, strict=False):
        grid[h - 1, col] = int(color)

    expected = _render_expected(grid, objects, priority)
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_input()
        if built is not None:
            grid, _expected = built
            return grid
    raise RuntimeError("failed to generate a valid input for task 4")
