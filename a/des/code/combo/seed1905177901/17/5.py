from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
SIDE_RANGE = (3, 5)
FG_COUNT_RANGE = (3, 9)
GRID_EXTRA_RANGE = (0, 8)
GEN_RETRIES = 300
MASK_RETRIES = 200


def _extract_template(grid: np.ndarray) -> tuple[np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fg_coords = np.argwhere(g != BG_COLOR)
    assert fg_coords.size > 0, "expected a foreground pattern"

    (r0, c0), (r1, c1) = fg_coords.min(axis=0), fg_coords.max(axis=0)
    template = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()

    colors = [int(v) for v in np.unique(template) if int(v) != BG_COLOR]
    assert len(colors) == 1, "expected a monochrome foreground template"
    return template, colors[0]


def _self_embed(template: np.ndarray) -> np.ndarray:
    h, w = map(int, template.shape)
    out = np.full((h * h, w * w), BG_COLOR, dtype=int)
    mask = template != BG_COLOR

    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]):
                out[r * h : (r + 1) * h, c * w : (c + 1) * w] = template
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each foreground template pixel with a full copy of the template.
    """
    template, _fg = _extract_template(grid)
    return _self_embed(template)


def _touches_every_side(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _sample_mask(side: int) -> np.ndarray | None:
    cells = [(r, c) for r in range(side) for c in range(side)]
    max_fg = min(FG_COUNT_RANGE[1], side * side - 1)
    if max_fg < FG_COUNT_RANGE[0]:
        return None

    for _ in range(MASK_RETRIES):
        required = {
            (0, random.randrange(side)),
            (side - 1, random.randrange(side)),
            (random.randrange(side), 0),
            (random.randrange(side), side - 1),
        }
        fg_count = random.randint(max(FG_COUNT_RANGE[0], len(required)), max_fg)
        extras = random.sample([cell for cell in cells if cell not in required], fg_count - len(required))

        mask = np.zeros((side, side), dtype=bool)
        for r, c in required.union(extras):
            mask[int(r), int(c)] = True

        if not _touches_every_side(mask):
            continue
        if int(mask.sum()) >= side * side:
            continue
        return mask
    return None


def _gen_attempt() -> np.ndarray | None:
    side = random.randint(*SIDE_RANGE)
    mask = _sample_mask(side)
    if mask is None:
        return None

    fg_color = random.randint(1, 9)
    template = np.where(mask, fg_color, BG_COLOR).astype(int)

    height = side + random.randint(*GRID_EXTRA_RANGE)
    width = side + random.randint(*GRID_EXTRA_RANGE)
    r0 = random.randint(0, height - side)
    c0 = random.randint(0, width - side)

    grid = np.full((height, width), BG_COLOR, dtype=int)
    region = grid[r0 : r0 + side, c0 : c0 + side]
    region[mask] = fg_color

    expected = _self_embed(template)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
