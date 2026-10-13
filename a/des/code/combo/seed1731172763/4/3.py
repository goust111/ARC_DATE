from __future__ import annotations

from collections import deque
import math
import random

import numpy as np


BG = 8
PALETTE = (0, 1, 2, 3, 4, 5, 6, 7, 9)
GAP = 1
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) == bg:
                continue

            color = int(grid[r0, c0])
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []

            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for rr, cc in _neighbors4(r, c):
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc]:
                        if int(grid[rr, cc]) == color:
                            seen[rr, cc] = True
                            q.append((rr, cc))

            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            r1, r2 = min(rows), max(rows)
            c1, c2 = min(cols), max(cols)
            mask = np.zeros((r2 - r1 + 1, c2 - c1 + 1), dtype=bool)
            for r, c in coords:
                mask[r - r1, c - c1] = True

            objects.append(
                {
                    "color": color,
                    "bbox": (r1, c1, r2, c2),
                    "mask": mask,
                    "h": int(mask.shape[0]),
                    "w": int(mask.shape[1]),
                    "area": int(mask.sum()),
                }
            )

    return objects


def _has_enclosed_background(mask: np.ndarray) -> bool:
    bg_mask = ~mask
    if not bool(bg_mask.any()):
        return False

    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bg_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and bg_mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))

    return bool(np.any(bg_mask & ~seen))


def _sort_key(obj: dict[str, object]) -> tuple[int, int, int]:
    r0, c0, _r1, _c1 = obj["bbox"]
    return (int(obj["area"]), int(r0), int(c0))


def _sequence_width(objects: list[dict[str, object]]) -> int:
    if not objects:
        return 0
    return sum(int(obj["w"]) for obj in objects) + GAP * (len(objects) - 1)


def _paint_sprite(out: np.ndarray, obj: dict[str, object], top: int, left: int) -> None:
    mask = obj["mask"]
    color = int(obj["color"])
    h = int(obj["h"])
    w = int(obj["w"])
    window = out[top : top + h, left : left + w]
    window[mask] = color


def _pack_bands(
    convex: list[dict[str, object]],
    non_convex: list[dict[str, object]],
    bg: int,
) -> np.ndarray:
    top_h = max((int(obj["h"]) for obj in convex), default=0)
    bottom_h = max((int(obj["h"]) for obj in non_convex), default=0)
    out_h = top_h + bottom_h
    out_w = max(_sequence_width(convex), _sequence_width(non_convex))
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for obj in convex:
        row = top_h - int(obj["h"])
        _paint_sprite(out, obj, row, col)
        col += int(obj["w"]) + GAP

    col = 0
    for obj in non_convex:
        _paint_sprite(out, obj, top_h, col)
        col += int(obj["w"]) + GAP

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move convex objects to the top band and non-convex objects to the bottom band,
    sorting each band by increasing area.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG if np.any(g == BG) else _dominant_color(g)

    objects = _extract_objects(g, bg)
    assert objects, "expected at least one foreground object"

    convex: list[dict[str, object]] = []
    non_convex: list[dict[str, object]] = []
    for obj in objects:
        if _has_enclosed_background(obj["mask"]):
            non_convex.append(obj)
        else:
            convex.append(obj)

    convex.sort(key=_sort_key)
    non_convex.sort(key=_sort_key)
    return _pack_bands(convex, non_convex, bg)


def _rectangle_mask() -> np.ndarray:
    height = random.randint(1, 4)
    width = random.randint(1, 4)
    return np.ones((height, width), dtype=bool)


def _one_hole_mask() -> np.ndarray:
    outer_h = random.randint(4, 7)
    outer_w = random.randint(4, 7)
    hole_h = random.randint(1, outer_h - 2)
    hole_w = random.randint(1, outer_w - 2)
    hole_r = random.randint(1, outer_h - hole_h - 1)
    hole_c = random.randint(1, outer_w - hole_w - 1)

    mask = np.ones((outer_h, outer_w), dtype=bool)
    mask[hole_r : hole_r + hole_h, hole_c : hole_c + hole_w] = False
    return mask


def _two_hole_mask() -> np.ndarray:
    for _ in range(100):
        outer_h = random.randint(5, 8)
        outer_w = random.randint(5, 8)
        mask = np.ones((outer_h, outer_w), dtype=bool)

        if random.random() < 0.5:
            hole_h = random.randint(1, outer_h - 2)
            hole_r = random.randint(1, outer_h - hole_h - 1)
            pieces: list[tuple[int, int]] = []
            for _idx in range(2):
                hole_w = random.randint(1, outer_w - 3)
                hole_c = random.randint(1, outer_w - hole_w - 1)
                pieces.append((hole_c, hole_c + hole_w))
            pieces.sort()
            if pieces[0][1] + 1 > pieces[1][0]:
                continue
            mask[hole_r : hole_r + hole_h, pieces[0][0] : pieces[0][1]] = False
            mask[hole_r : hole_r + hole_h, pieces[1][0] : pieces[1][1]] = False
        else:
            hole_w = random.randint(1, outer_w - 2)
            hole_c = random.randint(1, outer_w - hole_w - 1)
            pieces = []
            for _idx in range(2):
                hole_h = random.randint(1, outer_h - 3)
                hole_r = random.randint(1, outer_h - hole_h - 1)
                pieces.append((hole_r, hole_r + hole_h))
            pieces.sort()
            if pieces[0][1] + 1 > pieces[1][0]:
                continue
            mask[pieces[0][0] : pieces[0][1], hole_c : hole_c + hole_w] = False
            mask[pieces[1][0] : pieces[1][1], hole_c : hole_c + hole_w] = False

        if _has_enclosed_background(mask):
            return mask

    raise RuntimeError("failed to build a two-hole shape")


def _make_shape(non_convex: bool) -> np.ndarray:
    if not non_convex:
        return _rectangle_mask()
    if random.random() < 0.5:
        return _one_hole_mask()
    return _two_hole_mask()


def _build_specs() -> list[dict[str, object]] | None:
    n_convex = random.randint(1, 3)
    n_non_convex = random.randint(1, 3)
    total = n_convex + n_non_convex
    colors = random.sample(PALETTE, total)

    specs: list[dict[str, object]] = []
    used_convex_areas: set[int] = set()
    used_non_convex_areas: set[int] = set()

    for idx in range(total):
        non_convex = idx >= n_convex
        used_areas = used_non_convex_areas if non_convex else used_convex_areas

        for _ in range(120):
            mask = _make_shape(non_convex)
            area = int(mask.sum())
            if area in used_areas:
                continue
            if _has_enclosed_background(mask) != non_convex:
                continue

            used_areas.add(area)
            specs.append(
                {
                    "color": colors[idx],
                    "bbox": (0, 0, int(mask.shape[0]) - 1, int(mask.shape[1]) - 1),
                    "mask": mask,
                    "h": int(mask.shape[0]),
                    "w": int(mask.shape[1]),
                    "area": area,
                    "non_convex": non_convex,
                }
            )
            break
        else:
            return None

    random.shuffle(specs)
    return specs


def _canvas_shape(specs: list[dict[str, object]]) -> tuple[int, int] | None:
    total_bbox_area = sum(int(obj["h"]) * int(obj["w"]) for obj in specs)
    max_h = max(int(obj["h"]) for obj in specs)
    max_w = max(int(obj["w"]) for obj in specs)
    side = int(math.ceil(math.sqrt(total_bbox_area * 3.2))) + 2
    rows_min = max(max_h + 2, side - 2)
    cols_min = max(max_w + 2, side - 2)
    if rows_min > 30 or cols_min > 30:
        return None
    rows = random.randint(rows_min, min(30, rows_min + 5))
    cols = random.randint(cols_min, min(30, cols_min + 7))
    return rows, cols


def _place_specs(specs: list[dict[str, object]], rows: int, cols: int) -> np.ndarray | None:
    grid = np.full((rows, cols), BG, dtype=int)
    blocked = np.zeros((rows, cols), dtype=bool)

    order = sorted(specs, key=lambda obj: int(obj["h"]) * int(obj["w"]), reverse=True)
    for obj in order:
        h = int(obj["h"])
        w = int(obj["w"])
        positions = [(r, c) for r in range(rows - h + 1) for c in range(cols - w + 1)]
        random.shuffle(positions)

        placed = False
        for r0, c0 in positions:
            r1 = min(rows, r0 + h + 1)
            c1 = min(cols, c0 + w + 1)
            if blocked[max(0, r0 - 1) : r1, max(0, c0 - 1) : c1].any():
                continue

            window = grid[r0 : r0 + h, c0 : c0 + w]
            window[obj["mask"]] = int(obj["color"])
            blocked[max(0, r0 - 1) : r1, max(0, c0 - 1) : c1] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _expected_output(specs: list[dict[str, object]]) -> np.ndarray:
    convex = sorted((obj for obj in specs if not bool(obj["non_convex"])), key=lambda obj: int(obj["area"]))
    non_convex = sorted((obj for obj in specs if bool(obj["non_convex"])), key=lambda obj: int(obj["area"]))
    return _pack_bands(list(convex), list(non_convex), BG)


def _gen_attempt() -> np.ndarray | None:
    specs = _build_specs()
    if specs is None:
        return None

    canvas = _canvas_shape(specs)
    if canvas is None:
        return None
    rows, cols = canvas
    grid = _place_specs(specs, rows, cols)
    if grid is None:
        return None

    bg_count = int(np.count_nonzero(grid == BG))
    if bg_count <= sum(int(obj["area"]) for obj in specs):
        return None

    objects = _extract_objects(grid, BG)
    if len(objects) != len(specs):
        return None

    if sum(_has_enclosed_background(obj["mask"]) for obj in objects) < 1:
        return None
    if sum(not _has_enclosed_background(obj["mask"]) for obj in objects) < 1:
        return None

    expected = _expected_output(specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
