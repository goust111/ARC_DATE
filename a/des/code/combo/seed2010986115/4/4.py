from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    comps.sort(key=lambda comp: comp["bbox"])
    return comps


def _is_hollow_frame(grid: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False
    if len(comp["cells"]) != 2 * (height + width) - 4:
        return False
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(sub[0, :] == color):
        return False
    if not np.all(sub[-1, :] == color):
        return False
    if not np.all(sub[:, 0] == color):
        return False
    if not np.all(sub[:, -1] == color):
        return False
    if height > 2 and width > 2 and np.any(sub[1:-1, 1:-1] == color):
        return False
    return True


def _segment_info(comp: dict) -> tuple[str, int]:
    r0, c0, r1, c1 = comp["bbox"]
    cells = comp["cells"]
    if r0 == r1:
        length = c1 - c0 + 1
        if len(cells) != length:
            raise AssertionError("horizontal segment must be continuous")
        return "h", length
    if c0 == c1:
        length = r1 - r0 + 1
        if len(cells) != length:
            raise AssertionError("vertical segment must be continuous")
        return "v", length
    raise AssertionError("component must be a straight line segment")


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble each color's four scattered strips into a hollow rectangle, then
    nest the rectangles from largest to smallest at the grey reference position.

    The output keeps the input canvas size. The largest rectangle's top-left
    corner matches the grey marker's top-left corner, and each smaller
    rectangle is centered inside the previous one.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _components(g)
    ref_candidates = [comp for comp in comps if int(comp["color"]) == GREY and _is_hollow_frame(g, comp)]
    assert len(ref_candidates) == 1, "expected one grey reference frame"
    ref = ref_candidates[0]
    start_r, start_c, _, _ = ref["bbox"]

    groups: dict[int, list[dict]] = {}
    for comp in comps:
        if comp is ref:
            continue
        color = int(comp["color"])
        groups.setdefault(color, []).append(comp)

    rects: list[dict] = []
    for color, color_comps in groups.items():
        assert len(color_comps) == 4, "each strip color must have exactly four segments"
        infos = [(_segment_info(comp), comp) for comp in color_comps]
        horiz = [item for item in infos if item[0][0] == "h"]
        vert = [item for item in infos if item[0][0] == "v"]
        assert len(horiz) == 2 and len(vert) == 2, "each color needs two horizontal and two vertical strips"

        widths = {item[0][1] for item in horiz}
        inner_heights = {item[0][1] for item in vert}
        assert len(widths) == 1 and len(inner_heights) == 1, "paired strips must match in length"

        width = next(iter(widths))
        height = next(iter(inner_heights)) + 2
        appearance = min((comp["bbox"][0], comp["bbox"][1]) for comp in color_comps)
        rects.append(
            {
                "color": color,
                "width": width,
                "height": height,
                "area": width * height,
                "appearance": appearance,
            }
        )

    assert rects, "expected at least one colored strip group"
    rects.sort(key=lambda item: (-int(item["area"]), item["appearance"][0], item["appearance"][1]))

    out = np.zeros_like(g)
    top = int(start_r)
    left = int(start_c)
    prev_height = None
    prev_width = None
    for rect in rects:
        height = int(rect["height"])
        width = int(rect["width"])
        if prev_height is not None and prev_width is not None:
            assert prev_height - height >= 2 and prev_width - width >= 2, "rectangles must fit nested"
            top += (prev_height - height) // 2
            left += (prev_width - width) // 2
        assert 0 <= top and 0 <= left
        assert top + height <= g.shape[0] and left + width <= g.shape[1], "nested output must stay in bounds"
        _draw_frame(out, top, left, height, width, int(rect["color"]))
        prev_height = height
        prev_width = width

    return out


def _block_bbox(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_segment(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    orientation: str,
    length: int,
) -> bool:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    if orientation == "h":
        for r in range(h):
            for c in range(w - length + 1):
                if not np.any(blocked[r, c : c + length]):
                    candidates.append((r, c))
        if not candidates:
            return False
        r, c = random.choice(candidates)
        grid[r, c : c + length] = color
        _block_bbox(blocked, r, c, r, c + length - 1, pad=1)
        return True

    for r in range(h - length + 1):
        for c in range(w):
            if not np.any(blocked[r : r + length, c]):
                candidates.append((r, c))
    if not candidates:
        return False
    r, c = random.choice(candidates)
    grid[r : r + length, c] = color
    _block_bbox(blocked, r, c, r + length - 1, c, pad=1)
    return True


def _random_sizes(count: int) -> list[tuple[int, int]] | None:
    for _ in range(100):
        height = random.randint(4, 7)
        width = random.randint(4, 8)
        sizes = [(height, width)]
        for _ in range(count - 1):
            height += random.choice((2, 4))
            width += random.choice((2, 4))
            sizes.append((height, width))
        largest_h, largest_w = sizes[-1]
        if largest_h <= 13 and largest_w <= 13:
            return list(reversed(sizes))
    return None


def _draw_reference(grid: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    _draw_frame(grid, top, left, height, width, GREY)


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(2, 4)
    sizes = _random_sizes(count)
    if sizes is None:
        return None

    largest_h, largest_w = sizes[0]
    min_h = max(22, largest_h + 10)
    min_w = max(22, largest_w + 10)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)
    anchor_r = random.randint(max(0, grid_h - largest_h - 2), grid_h - largest_h)
    anchor_c = random.randint(max(0, grid_w - largest_w - 2), grid_w - largest_w)

    marker_h = random.randint(3, min(4, largest_h - 1))
    marker_w = random.randint(3, min(4, largest_w - 1))

    grid = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    _block_bbox(blocked, anchor_r, anchor_c, anchor_r + largest_h - 1, anchor_c + largest_w - 1, pad=1)
    _draw_reference(grid, anchor_r, anchor_c, marker_h, marker_w)

    colors = random.sample(COLOR_POOL, count)
    random.shuffle(colors)

    specs: list[tuple[int, str, int]] = []
    for color, (height, width) in zip(colors, sizes):
        specs.extend(
            [
                (color, "h", width),
                (color, "h", width),
                (color, "v", height - 2),
                (color, "v", height - 2),
            ]
        )
    random.shuffle(specs)

    for color, orientation, length in specs:
        if not _place_segment(grid, blocked, color, orientation, length):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.any(out == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
