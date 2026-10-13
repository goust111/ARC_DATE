from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
SOURCE_POOL = (1, 2, 3, 4)
FILL_POOL = (6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
OBJECT_COUNT_RANGE = (1, 4)
PALETTE_ROW_RANGE = (1, 4)
RECT_H_RANGE = (4, 9)
RECT_W_RANGE = (4, 9)
PLACEMENT_RETRIES = 120
GENERATION_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _find_palette(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 2

    top = h
    while top > 0 and int(g[top - 1, w - 2]) != BLACK and int(g[top - 1, w - 1]) != BLACK:
        top -= 1
    assert top < h, "expected a bottom-right 2-column palette"

    palette_mask = np.zeros((h, w), dtype=bool)
    palette_mask[top:h, w - 2 : w] = True
    assert bool(np.all(g[palette_mask] != BLACK))

    mapping: dict[int, int] = {}
    for r in range(top, h):
        src = int(g[r, w - 2])
        dst = int(g[r, w - 1])
        mapping[src] = dst
    assert mapping, "palette must contain at least one row"
    return mapping, palette_mask


def _side_complete(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int], side: str) -> bool:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return bool(np.all(grid[r0, c0 : c1 + 1] == color))
    if side == "bottom":
        return bool(np.all(grid[r1, c0 : c1 + 1] == color))
    if side == "left":
        return bool(np.all(grid[r0 : r1 + 1, c0] == color))
    if side == "right":
        return bool(np.all(grid[r0 : r1 + 1, c1] == color))
    raise AssertionError("unknown side")


def _draw_full_rectangle(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _draw_partial_rectangle(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, missing_side: str) -> None:
    r0, c0, r1, c1 = bbox
    if missing_side != "top":
        grid[r0, c0 : c1 + 1] = color
    if missing_side != "bottom":
        grid[r1, c0 : c1 + 1] = color
    if missing_side != "left":
        grid[r0 : r1 + 1, c0] = color
    if missing_side != "right":
        grid[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each same-colored rectangle implied by paired L-polylines and fill its interior
    with the palette-mapped complementary color.

    The solver reads the bottom-right 2-column palette, then treats each foreground component
    outside the palette as a one-cell-thick three-sided rectangle border produced by two
    same-colored L-polylines sharing one side. The missing side is restored and black interior
    cells are recolored using the palette mapping.
    """
    g = np.asarray(grid, dtype=int)
    mapping, palette_mask = _find_palette(g)

    out = g.copy()
    work_mask = (g != BLACK) & (~palette_mask)

    for coords in _connected_components(work_mask):
        color = int(g[tuple(coords[0])])
        if color not in mapping:
            continue

        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        sides = {side: _side_complete(g, color, (r0, c0, r1, c1), side) for side in SIDES}
        missing = [side for side, full in sides.items() if not full]
        if len(missing) != 1:
            continue

        _draw_full_rectangle(out, (r0, c0, r1, c1), color)
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BLACK] = int(mapping[color])

    return out


def _render_expected(grid: np.ndarray, objects: list[dict[str, object]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for obj in objects:
        color = int(obj["border"])
        fill = int(obj["fill"])
        bbox = obj["bbox"]
        assert isinstance(bbox, tuple)
        _draw_full_rectangle(out, bbox, color)
        r0, c0, r1, c1 = bbox
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BLACK] = fill
    return out


def _sample_palette_rows(obj_count: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    max_rows = min(len(SOURCE_POOL), len(FILL_POOL))
    palette_rows = random.randint(max(obj_count, PALETTE_ROW_RANGE[0]), min(max_rows, PALETTE_ROW_RANGE[1]))
    srcs = random.sample(list(SOURCE_POOL), palette_rows)
    fills = random.sample(list(FILL_POOL), palette_rows)
    pairs = list(zip(srcs, fills))
    random.shuffle(pairs)
    used = random.sample(pairs, obj_count)
    return pairs, used


def _place_objects(h: int, w: int, palette_rows: int, used_pairs: list[tuple[int, int]]) -> list[dict[str, object]] | None:
    blocked = np.zeros((h, w), dtype=bool)
    blocked[h - palette_rows : h, max(0, w - 3) : w] = True
    objects: list[dict[str, object]] = []

    max_h = min(RECT_H_RANGE[1], h - 2)
    max_w = min(RECT_W_RANGE[1], w - 5)
    if max_h < RECT_H_RANGE[0] or max_w < RECT_W_RANGE[0]:
        return None

    for border, fill in used_pairs:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            rect_h = random.randint(RECT_H_RANGE[0], max_h)
            rect_w = random.randint(RECT_W_RANGE[0], max_w)

            r_lo = 1
            r_hi = h - rect_h - 1
            c_lo = 1
            c_hi = w - rect_w - 4
            if r_hi < r_lo or c_hi < c_lo:
                return None

            r0 = random.randint(r_lo, r_hi)
            c0 = random.randint(c_lo, c_hi)
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1

            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h - 1, r1 + 1)
            cc1 = min(w - 1, c1 + 1)
            if bool(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                continue

            blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            objects.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "border": int(border),
                    "fill": int(fill),
                    "missing": random.choice(SIDES),
                }
            )
            placed = True
            break
        if not placed:
            return None

    return objects


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    obj_count = random.randint(*OBJECT_COUNT_RANGE)
    palette_pairs, used_pairs = _sample_palette_rows(obj_count)
    palette_rows = len(palette_pairs)

    objects = _place_objects(h, w, palette_rows, used_pairs)
    if objects is None:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    for row_idx, (src, dst) in enumerate(palette_pairs):
        r = h - palette_rows + row_idx
        grid[r, w - 2] = int(src)
        grid[r, w - 1] = int(dst)

    for obj in objects:
        bbox = obj["bbox"]
        assert isinstance(bbox, tuple)
        _draw_partial_rectangle(grid, bbox, int(obj["border"]), str(obj["missing"]))

    expected = _render_expected(grid, objects)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
