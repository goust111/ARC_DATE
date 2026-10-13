from __future__ import annotations

import random

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _template_masks(template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    colors = np.unique(template)
    assert colors.size == 2, "template row must contain exactly two colors"

    keep_color = int(template[0])
    erase_color = int(template[1])
    assert keep_color != erase_color, "template colors must differ"

    cols = np.arange(template.size)
    keep_mask = (cols % 2 == 0)
    erase_mask = ~keep_mask
    assert np.all(template[keep_mask] == keep_color), "template keep columns must alternate from the first cell"
    assert np.all(template[erase_mask] == erase_color), "template erase columns must alternate with the second color"
    return keep_mask, erase_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the alternating top template row as a column mask.

    - The first template color marks keep columns.
    - The second template color marks erase columns.
    - The template row itself is cleared to background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 2 and g.shape[1] >= 2, "expected a template row plus data rows"

    bg = _background_color(g)
    template = g[0]
    _keep_mask, erase_mask = _template_masks(template)

    out = g.copy()
    out[0, :] = bg
    out[1:, erase_mask] = bg
    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (6, 14)
GRID_WIDTH_RANGE = (6, 18)
OBJECT_COUNT_RANGE = (4, 10)
MAX_GEN_ATTEMPTS = 400


def _random_colors() -> tuple[int, int, int, int]:
    bg = random.randint(0, 9)
    remaining = [c for c in range(10) if c != bg]
    keep_color, erase_color = random.sample(remaining, 2)
    foreground = random.sample([c for c in remaining if c not in {keep_color, erase_color}], 4)
    return bg, keep_color, erase_color, foreground[0]


def _shape_cells(kind: str, r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    if kind == "pixel":
        return [(r, c)]
    if kind == "hline":
        length = random.randint(2, min(5, w - c))
        return [(r, cc) for cc in range(c, c + length)]
    if kind == "vline":
        length = random.randint(2, min(4, h - r))
        return [(rr, c) for rr in range(r, r + length)]
    if kind == "rect":
        hh = random.randint(2, min(4, h - r))
        ww = random.randint(2, min(5, w - c))
        return [(rr, cc) for rr in range(r, r + hh) for cc in range(c, c + ww)]
    raise AssertionError("unknown shape kind")


def _draw_object(grid: np.ndarray, occupied: np.ndarray, color: int) -> bool:
    h, w = grid.shape
    data_h = h - 1
    kinds = ["pixel", "hline", "vline", "rect"]
    kind = random.choice(kinds)

    if kind == "pixel":
        r = random.randint(1, h - 1)
        c = random.randint(0, w - 1)
    elif kind == "hline":
        r = random.randint(1, h - 1)
        c = random.randint(0, w - 2)
    elif kind == "vline":
        r = random.randint(1, h - min(2, data_h))
        c = random.randint(0, w - 1)
    else:
        r = random.randint(1, h - min(2, data_h))
        c = random.randint(0, w - 2)

    cells = _shape_cells(kind, r, c, h, w)
    if any(occupied[rr, cc] for rr, cc in cells):
        return False

    for rr, cc in cells:
        grid[rr, cc] = color
        occupied[rr, cc] = True
    return True


def _build_template(width: int, keep_color: int, erase_color: int, start_with_keep: bool) -> np.ndarray:
    colors = [keep_color, erase_color] if start_with_keep else [erase_color, keep_color]
    return np.array([colors[c % 2] for c in range(width)], dtype=int)


def _expected_output(inp: np.ndarray, bg: int) -> np.ndarray:
    out = inp.copy()
    template = inp[0]
    keep_mask = template == template[0]
    erase_mask = ~keep_mask
    assert np.all(template[keep_mask] == template[0])
    assert np.all(template[erase_mask] == template[1])
    out[0, :] = bg
    out[1:, erase_mask] = bg
    return out


def _valid_sample(inp: np.ndarray, bg: int) -> bool:
    template = inp[0]
    keep_cols = np.flatnonzero(template == template[0])
    erase_cols = np.flatnonzero(template != template[0])
    data = inp[1:]
    return (
        np.any(data[:, keep_cols] != bg)
        and np.any(data[:, erase_cols] != bg)
        and np.any(data[:, keep_cols] == bg)
        and np.any(data[:, erase_cols] == bg)
    )


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    bg, keep_color, erase_color, first_fg = _random_colors()
    start_with_keep = random.choice([True, False])

    grid = np.full((h, w), bg, dtype=int)
    grid[0] = _build_template(w, keep_color, erase_color, start_with_keep)

    occupied = np.zeros((h, w), dtype=bool)
    occupied[0, :] = True

    palette = [c for c in range(10) if c not in {bg, keep_color, erase_color}]
    random.shuffle(palette)
    palette = palette[: random.randint(2, min(6, len(palette)))]
    if first_fg not in palette:
        palette[0] = first_fg

    target_objects = random.randint(*OBJECT_COUNT_RANGE)
    placed = 0
    failures = 0
    while placed < target_objects and failures < 80:
        if _draw_object(grid, occupied, random.choice(palette)):
            placed += 1
            failures = 0
        else:
            failures += 1

    if placed < 3 or not _valid_sample(grid, bg):
        return None

    expected = _expected_output(grid, bg)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
