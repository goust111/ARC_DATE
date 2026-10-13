from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")
GEN_ATTEMPTS = 300
MIN_SIZE = 5
ROW_GAP_RANGE = (2, 4)
COL_GAP_RANGE = (2, 4)
OUTER_PAD_RANGE = (1, 2)
MAX_GRID_H = 28
MAX_GRID_W = 38


@dataclass
class Frame:
    color: int
    bbox: tuple[int, int, int, int]
    coords: tuple[tuple[int, int], ...]
    h: int
    w: int
    side: str
    patch: np.ndarray


@dataclass
class RectSpec:
    kind: str
    color: int
    side: str
    h: int
    w: int
    gap_start: int
    top: int = 0
    left: int = 0


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(grid: np.ndarray, bg: int = BG) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append((color, coords))
    return comps


def _consecutive_pair(values: np.ndarray) -> bool:
    if len(values) != 2:
        return False
    lo, hi = sorted(int(v) for v in values.tolist())
    return hi == lo + 1


def _gap_side(missing: np.ndarray, h: int, w: int) -> str:
    if missing.shape != (2, 2):
        raise ValueError("frame must have exactly one gap of width 2")
    rows = missing[:, 0]
    cols = missing[:, 1]
    if np.all(rows == 0) and _consecutive_pair(cols):
        return "top"
    if np.all(rows == h - 1) and _consecutive_pair(cols):
        return "bottom"
    if np.all(cols == 0) and _consecutive_pair(rows):
        return "left"
    if np.all(cols == w - 1) and _consecutive_pair(rows):
        return "right"
    raise ValueError("gap must lie on exactly one side")


def _analyze_component(color: int, coords: list[tuple[int, int]]) -> Frame:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 4 or w < 4:
        raise ValueError("rectangle outline is too small")

    mask = np.zeros((h, w), dtype=bool)
    mask[arr[:, 0] - r0, arr[:, 1] - c0] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(mask & ~border):
        raise ValueError("component is not a hollow rectangle outline")

    missing = np.argwhere(border & ~mask)
    side = _gap_side(missing, h, w)
    if int(mask.sum()) != int(border.sum()) - 2:
        raise ValueError("outline must be a full rectangle border with a width-2 gap")

    patch = np.zeros((h, w), dtype=int)
    patch[mask] = int(color)
    return Frame(
        color=int(color),
        bbox=(int(r0), int(c0), int(r1), int(c1)),
        coords=tuple((int(r), int(c)) for r, c in coords),
        h=h,
        w=w,
        side=side,
        patch=patch,
    )


def _extract_pairs(grid: np.ndarray) -> list[tuple[Frame, Frame]]:
    by_color: dict[int, list[Frame]] = defaultdict(list)
    for color, coords in _components(grid, BG):
        by_color[int(color)].append(_analyze_component(int(color), coords))

    pairs: list[tuple[Frame, Frame]] = []
    for color in sorted(by_color):
        items = by_color[color]
        if len(items) != 2:
            raise ValueError("each color must appear in exactly two outlines")
        small, big = sorted(items, key=lambda item: (item.h * item.w, item.h + item.w))
        if small.side != big.side:
            raise ValueError("paired outlines must have gaps on the same side")
        if not (small.h < big.h - 2 and small.w < big.w - 2):
            raise ValueError("template must fit strictly inside the container interior")
        pairs.append((big, small))
    return pairs


def _fill_stripes(out: np.ndarray, frame: Frame) -> None:
    r0, c0, r1, c1 = frame.bbox
    out[r0 + 1 : r1, c0 + 1 : c1] = BG
    if frame.side in ("top", "bottom"):
        for idx, r in enumerate(range(r0 + 1, r1)):
            if idx % 2 == 0:
                out[r, c0 + 1 : c1] = frame.color
    else:
        for idx, c in enumerate(range(c0 + 1, c1)):
            if idx % 2 == 0:
                out[r0 + 1 : r1, c] = frame.color


def _embed_template(out: np.ndarray, container: Frame, template: Frame) -> None:
    inner_h = container.h - 2
    inner_w = container.w - 2
    top = container.bbox[0] + 1 + (inner_h - template.h) // 2
    left = container.bbox[1] + 1 + (inner_w - template.w) // 2
    out[top : top + template.h, left : left + template.w] = template.patch


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    pairs = _extract_pairs(g)

    for _container, template in pairs:
        for r, c in template.coords:
            out[r, c] = BG

    for container, template in pairs:
        _fill_stripes(out, container)
        _embed_template(out, container, template)

    return out


def _random_gap_start(rng: random.Random, side: str, h: int, w: int) -> int:
    length = w if side in ("top", "bottom") else h
    return rng.randint(1, length - 3)


def _make_pair_specs(rng: random.Random, color: int, side: str, pair_count: int) -> tuple[RectSpec, RectSpec]:
    if pair_count == 2:
        small_h = rng.randint(5, 8)
        small_w = rng.randint(5, 8)
        big_h = rng.randint(small_h + 3, min(12, small_h + 5))
        big_w = rng.randint(small_w + 3, min(12, small_w + 5))
    else:
        small_h = rng.randint(5, 7)
        small_w = rng.randint(5, 7)
        big_h = rng.randint(small_h + 3, min(10, small_h + 4))
        big_w = rng.randint(small_w + 3, min(10, small_w + 4))

    container = RectSpec(
        kind="container",
        color=color,
        side=side,
        h=big_h,
        w=big_w,
        gap_start=_random_gap_start(rng, side, big_h, big_w),
    )
    template = RectSpec(
        kind="template",
        color=color,
        side=side,
        h=small_h,
        w=small_w,
        gap_start=_random_gap_start(rng, side, small_h, small_w),
    )
    return container, template


def _choose_rows(rng: random.Random, objects: list[RectSpec]) -> list[list[RectSpec]]:
    ranked = objects[:]
    rng.shuffle(ranked)
    ranked.sort(key=lambda item: (item.w, item.h), reverse=True)

    rows = [[], []]
    row_loads = [0, 0]
    for obj in ranked:
        idx = 0 if row_loads[0] <= row_loads[1] else 1
        rows[idx].append(obj)
        row_loads[idx] += obj.w

    for row in rows:
        rng.shuffle(row)
    return rows


def _layout_objects(rng: random.Random, objects: list[RectSpec]) -> tuple[int, int] | None:
    rows = _choose_rows(rng, objects)
    if any(not row for row in rows):
        return None

    row_widths: list[int] = []
    row_heights: list[int] = []
    row_gaps: list[list[int]] = []
    for row in rows:
        gaps = [rng.randint(*COL_GAP_RANGE) for _ in range(len(row) - 1)]
        row_gaps.append(gaps)
        row_widths.append(sum(obj.w for obj in row) + sum(gaps))
        row_heights.append(max(obj.h for obj in row))

    left_pad = rng.randint(*OUTER_PAD_RANGE)
    right_pad = rng.randint(*OUTER_PAD_RANGE)
    top_pad = rng.randint(*OUTER_PAD_RANGE)
    bottom_pad = rng.randint(*OUTER_PAD_RANGE)
    row_gap = rng.randint(*ROW_GAP_RANGE)
    extra_width = rng.randint(0, 3)

    inner_w = max(row_widths) + extra_width
    grid_w = left_pad + inner_w + right_pad
    grid_h = top_pad + row_heights[0] + row_gap + row_heights[1] + bottom_pad
    if grid_h > MAX_GRID_H or grid_w > MAX_GRID_W:
        return None

    y = top_pad
    for row_idx, row in enumerate(rows):
        row_w = row_widths[row_idx]
        x = left_pad + rng.randint(0, inner_w - row_w)
        cursor = x
        for idx, obj in enumerate(row):
            obj.top = y + rng.randint(0, row_heights[row_idx] - obj.h)
            obj.left = cursor
            cursor += obj.w
            if idx < len(row) - 1:
                cursor += row_gaps[row_idx][idx]
        y += row_heights[row_idx]
        if row_idx == 0:
            y += row_gap

    return grid_h, grid_w


def _draw_rect_outline(grid: np.ndarray, spec: RectSpec) -> None:
    r0, c0 = spec.top, spec.left
    r1, c1 = r0 + spec.h - 1, c0 + spec.w - 1

    grid[r0, c0 : c1 + 1] = spec.color
    grid[r1, c0 : c1 + 1] = spec.color
    grid[r0 : r1 + 1, c0] = spec.color
    grid[r0 : r1 + 1, c1] = spec.color

    if spec.side == "top":
        start = c0 + spec.gap_start
        grid[r0, start : start + 2] = BG
    elif spec.side == "bottom":
        start = c0 + spec.gap_start
        grid[r1, start : start + 2] = BG
    elif spec.side == "left":
        start = r0 + spec.gap_start
        grid[start : start + 2, c0] = BG
    else:
        start = r0 + spec.gap_start
        grid[start : start + 2, c1] = BG


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(2, 3)
    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    colors = rng.sample(palette, pair_count)
    sides = rng.sample(list(SIDES), pair_count)

    objects: list[RectSpec] = []
    for color, side in zip(colors, sides):
        container, template = _make_pair_specs(rng, color, side, pair_count)
        objects.extend([container, template])

    grid_shape = _layout_objects(rng, objects)
    if grid_shape is None:
        return None
    grid = np.zeros(grid_shape, dtype=int)
    for obj in objects:
        _draw_rect_outline(grid, obj)

    try:
        pairs = _extract_pairs(grid)
        if len(pairs) != pair_count:
            return None
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
