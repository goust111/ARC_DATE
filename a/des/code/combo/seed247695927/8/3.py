from __future__ import annotations

import random

import numpy as np


MAROON = 9
PALETTE = [0, 1, 2, 3, 4, 5, 6, 7, 8]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.asarray(cells, dtype=int))
    return parts


def _infer_floor_start(g: np.ndarray, floor: int) -> int:
    rows = np.all(g == floor, axis=1)
    assert rows[-1], "bottom row must be the floor color"
    start = g.shape[0] - 1
    while start > 0 and rows[start - 1]:
        start -= 1
    return start


def _extract_object(g: np.ndarray) -> tuple[int, int, np.ndarray, str, int]:
    bg = int(g[0, 0])
    assert np.all(g[0, :] == bg), "top row must be uniform background"

    floor = int(g[-1, 0])
    assert floor != bg, "background and floor must differ"
    floor_start = _infer_floor_start(g, floor)

    obj_mask = (g != bg) & (g != floor)
    comps = _components(obj_mask)
    assert len(comps) == 1, "expected exactly one foreground object"
    coords = comps[0]

    colors = {int(v) for v in g[obj_mask]}
    assert len(colors) == 1, "foreground object must be monochrome"
    color = colors.pop()

    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    assert r1 == floor_start - 1, "object bottom must rest directly above the first floor row"

    left_touch = c0 == 0
    right_touch = c1 == g.shape[1] - 1
    assert left_touch ^ right_touch, "object must touch exactly one vertical edge"
    side = "left" if left_touch else "right"

    return bg, floor, coords, side, color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the side-attached foreground object to the opposite wall, shear lower rows inward, and draw a floor path.

    Input structure:
    - The top row is the upper background color and the bottom row is the floor color.
    - A single non-background non-floor object touches exactly one vertical grid edge, and its bottom sits immediately above the first solid floor row.

    Rule:
    - Clear the source object to the upper background color.
    - Redraw the object against the opposite wall: the first three relative rows stay wall-aligned, and each lower row shifts one extra cell inward.
    - On the first floor row, paint the spanning segment from the source side toward the moved object in maroon, stopping just before the moved object's outer bottom support pixel.

    Key insight:
    - The touched side gives the horizontal direction, and each object's row index contributes one extra unit of diagonal shear below the third row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg, floor, coords, side, color = _extract_object(g)
    floor_start = _infer_floor_start(g, floor)

    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    obj_h = int(coords[:, 0].max()) - r0 + 1
    obj_w = c1 - c0 + 1
    assert obj_w + max(0, obj_h - 3) <= w, "sheared destination must fit"

    out = g.copy()
    out[coords[:, 0], coords[:, 1]] = bg

    bottom_outer = None
    for rr in range(obj_h):
        row_cols = coords[coords[:, 0] == r0 + rr, 1] - c0
        inward = max(0, rr - 2)
        if side == "left":
            base = w - obj_w - inward
            new_cols = row_cols + base
        else:
            base = inward
            new_cols = row_cols + base
        out[r0 + rr, new_cols] = color
        if rr == obj_h - 1:
            bottom_outer = int(new_cols.max()) if side == "left" else int(new_cols.min())

    assert bottom_outer is not None
    if side == "left":
        if bottom_outer > 0:
            out[floor_start, :bottom_outer] = MAROON
    else:
        if bottom_outer < w - 1:
            out[floor_start, bottom_outer + 1 :] = MAROON

    return out


def _sample_rows(height: int, width: int) -> list[int]:
    rows = []
    current = random.randint(max(1, width // 2), width)
    for _ in range(height):
        delta = random.choice([-1, 0, 1])
        current = min(width, max(1, current + delta))
        rows.append(current)
    if len(set(rows)) == 1 and width >= 2:
        idx = random.randrange(height)
        rows[idx] = max(1, min(width, rows[idx] - 1 if rows[idx] == width else rows[idx] + 1))
    return rows


def _build_input(
    grid_h: int,
    grid_w: int,
    floor_h: int,
    side: str,
    obj_color: int,
    bg: int,
    floor: int,
    row_lengths: list[int],
) -> np.ndarray:
    floor_start = grid_h - floor_h
    obj_h = len(row_lengths)
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    grid[floor_start:, :] = floor
    top = floor_start - obj_h

    for rr, length in enumerate(row_lengths):
        if side == "left":
            grid[top + rr, :length] = obj_color
        else:
            grid[top + rr, grid_w - length :] = obj_color
    return grid


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(8, 16)
    grid_w = random.randint(8, 16)
    floor_h = random.randint(1, 4)
    floor_start = grid_h - floor_h
    max_obj_h = floor_start - 1
    if max_obj_h < 4:
        return None

    obj_h = random.randint(4, min(8, max_obj_h))
    max_obj_w = grid_w - max(0, obj_h - 3)
    if max_obj_w < 2:
        return None
    obj_w = random.randint(2, max_obj_w)

    bg, floor, obj_color = random.sample(PALETTE, 3)
    side = random.choice(["left", "right"])
    row_lengths = _sample_rows(obj_h, obj_w)
    if side == "left" and max(row_lengths) != obj_w:
        row_lengths[random.randrange(obj_h)] = obj_w
    if side == "right" and max(row_lengths) != obj_w:
        row_lengths[random.randrange(obj_h)] = obj_w

    inp = _build_input(grid_h, grid_w, floor_h, side, obj_color, bg, floor, row_lengths)

    try:
        out = main(inp)
    except AssertionError:
        return None

    floor_row = floor_start
    if side == "left":
        if not np.any(out[floor_row] == MAROON):
            return None
    else:
        if not np.any(out[floor_row] == MAROON):
            return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
