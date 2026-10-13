from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BLEND_COLOR = 1
REGION_COLORS = (2, 3, 4, 6, 7, 8, 9)
MIN_SIZE = 8
MAX_SIZE = 18
MIN_RECT_SIDE = 2
MAX_REGIONS = 8
GEN_RETRIES = 400


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            assert area == len(coords), "each color component must be a solid rectangle"
            parts.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "height": r1 - r0 + 1,
                    "width": c1 - c0 + 1,
                }
            )
    return parts


def _adjacent_pairs(rects: list[dict]) -> list[tuple[int, int, str, int, int, int]]:
    pairs: list[tuple[int, int, str, int, int, int]] = []
    for i in range(len(rects)):
        a = rects[i]
        ar0, ac0, ar1, ac1 = a["bbox"]
        for j in range(i + 1, len(rects)):
            b = rects[j]
            br0, bc0, br1, bc1 = b["bbox"]

            if ac1 + 1 == bc0 or bc1 + 1 == ac0:
                lo = max(ar0, br0)
                hi = min(ar1, br1)
                if lo <= hi:
                    pairs.append((i, j, "vertical", lo, hi, hi - lo + 1))

            if ar1 + 1 == br0 or br1 + 1 == ar0:
                lo = max(ac0, bc0)
                hi = min(ac1, bc1)
                if lo <= hi:
                    pairs.append((i, j, "horizontal", lo, hi, hi - lo + 1))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Selectively erase shared borders between solid rectangular regions.

    The input is fully tiled by monochrome rectangles. For each directly adjacent
    pair, compare the shared-boundary length to the full side length of each
    rectangle in that direction. If the shared boundary spans one rectangle's
    entire side, keep the border. Otherwise recolor both sides of that partial
    boundary to a fixed blend color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    rects = _components(g)
    assert len(rects) >= 2, "expected multiple rectangular regions"

    out = g.copy()
    changed = 0
    for i, j, axis, lo, hi, shared_len in _adjacent_pairs(rects):
        a = rects[i]
        b = rects[j]
        if axis == "vertical":
            full_a = a["height"]
            full_b = b["height"]
            if shared_len in (full_a, full_b):
                continue

            _, ac0, _, ac1 = a["bbox"]
            _, bc0, _, bc1 = b["bbox"]
            if ac1 + 1 == bc0:
                left_col, right_col = ac1, bc0
            else:
                left_col, right_col = bc1, ac0
            out[lo : hi + 1, left_col] = BLEND_COLOR
            out[lo : hi + 1, right_col] = BLEND_COLOR
            changed += 2 * shared_len
        else:
            full_a = a["width"]
            full_b = b["width"]
            if shared_len in (full_a, full_b):
                continue

            ar0, _, ar1, _ = a["bbox"]
            br0, _, br1, _ = b["bbox"]
            if ar1 + 1 == br0:
                top_row, bottom_row = ar1, br0
            else:
                top_row, bottom_row = br1, ar0
            out[top_row, lo : hi + 1] = BLEND_COLOR
            out[bottom_row, lo : hi + 1] = BLEND_COLOR
            changed += 2 * shared_len

    assert changed > 0, "expected at least one partial shared boundary"
    return out


def _can_split(rect: tuple[int, int, int, int], axis: str) -> bool:
    r0, c0, r1, c1 = rect
    if axis == "vertical":
        return c1 - c0 + 1 >= 2 * MIN_RECT_SIDE
    return r1 - r0 + 1 >= 2 * MIN_RECT_SIDE


def _split_rect(rect: tuple[int, int, int, int], axis: str) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    r0, c0, r1, c1 = rect
    if axis == "vertical":
        cut = random.randint(c0 + MIN_RECT_SIDE - 1, c1 - MIN_RECT_SIDE)
        return (r0, c0, r1, cut), (r0, cut + 1, r1, c1)
    cut = random.randint(r0 + MIN_RECT_SIDE - 1, r1 - MIN_RECT_SIDE)
    return (r0, c0, cut, c1), (cut + 1, c0, r1, c1)


def _random_split(rects: list[tuple[int, int, int, int]]) -> bool:
    options: list[tuple[int, str]] = []
    for idx, rect in enumerate(rects):
        if _can_split(rect, "vertical"):
            options.append((idx, "vertical"))
        if _can_split(rect, "horizontal"):
            options.append((idx, "horizontal"))
    if not options:
        return False

    idx, axis = random.choice(options)
    first, second = _split_rect(rects[idx], axis)
    rects[idx : idx + 1] = [first, second]
    return True


def _make_partition(height: int, width: int) -> list[tuple[int, int, int, int]] | None:
    root = (0, 0, height - 1, width - 1)
    first_axes = [axis for axis in ("vertical", "horizontal") if _can_split(root, axis)]
    if not first_axes:
        return None
    first_axis = random.choice(first_axes)
    left, right = _split_rect(root, first_axis)
    rects = [left, right]

    second_axis = "horizontal" if first_axis == "vertical" else "vertical"
    candidates = [idx for idx, rect in enumerate(rects) if _can_split(rect, second_axis)]
    if not candidates:
        return None
    idx = random.choice(candidates)
    first, second = _split_rect(rects[idx], second_axis)
    rects[idx : idx + 1] = [first, second]

    target = random.randint(4, MAX_REGIONS)
    while len(rects) < target and random.random() < 0.85:
        if not _random_split(rects):
            break
    return rects


def _rect_adjacency(rects: list[tuple[int, int, int, int]]) -> dict[int, set[int]]:
    meta = [
        {
            "bbox": rect,
            "height": rect[2] - rect[0] + 1,
            "width": rect[3] - rect[1] + 1,
        }
        for rect in rects
    ]
    graph: dict[int, set[int]] = defaultdict(set)
    for i, j, _axis, _lo, _hi, _length in _adjacent_pairs(meta):
        graph[i].add(j)
        graph[j].add(i)
    return graph


def _assign_colors(rects: list[tuple[int, int, int, int]]) -> list[int] | None:
    graph = _rect_adjacency(rects)
    order = sorted(range(len(rects)), key=lambda idx: (-len(graph[idx]), rects[idx]))
    color_of: dict[int, int] = {}
    for idx in order:
        random.shuffle(list(REGION_COLORS))
        used = {color_of[nbr] for nbr in graph[idx] if nbr in color_of}
        choices = [color for color in REGION_COLORS if color not in used]
        if not choices:
            return None
        color_of[idx] = random.choice(choices)
    return [color_of[i] for i in range(len(rects))]


def _render(rects: list[tuple[int, int, int, int]], colors: list[int], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, -1, dtype=int)
    for (r0, c0, r1, c1), color in zip(rects, colors):
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    assert np.all(grid >= 0)
    return grid


def _has_both_boundary_types(grid: np.ndarray) -> bool:
    rects = _components(grid)
    kept = False
    erased = False
    for i, j, axis, _lo, _hi, shared_len in _adjacent_pairs(rects):
        a = rects[i]
        b = rects[j]
        fulls = (a["height"], b["height"]) if axis == "vertical" else (a["width"], b["width"])
        if shared_len in fulls:
            kept = True
        else:
            erased = True
    return kept and erased


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_SIZE, MAX_SIZE)
    width = random.randint(MIN_SIZE, MAX_SIZE)
    rects = _make_partition(height, width)
    if rects is None:
        return None

    colors = _assign_colors(rects)
    if colors is None:
        return None

    grid = _render(rects, colors, (height, width))
    if not _has_both_boundary_types(grid):
        return None

    out = main(grid)
    if np.array_equal(grid, out):
        return None
    if BLEND_COLOR not in np.unique(out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rectangular tiling")
