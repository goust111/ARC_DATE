from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")


def _load_arc_utils():
    if not _ARC_UTILS_PATH.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_arc_utils_024_1", _ARC_UTILS_PATH)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


_ARC = _load_arc_utils()


if _ARC is not None:
    Color = _ARC.Color
    bbox = _ARC.bbox
    in_bounds = _ARC.in_bounds
else:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        return 0 <= r < shape[0] and 0 <= c < shape[1]

    def bbox(mask: np.ndarray):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)


UPPER_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK)
GEN_RETRIES = 400


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == Color.GREY)]
    assert len(rows) == 1, f"expected one full-width separator row, got {rows}"
    return int(rows[0])


def _find_short_bar(grid: np.ndarray, separator_r: int) -> tuple[int, int, int]:
    grey_mask = grid == Color.GREY
    grey_mask[separator_r, :] = False
    bar_box = bbox(grey_mask)
    assert bar_box is not None, "missing short gray bar"
    r0, c0, r1, c1 = bar_box
    assert r0 == r1, "short bar must be one row tall"
    assert np.all(grid[r0, c0 : c1 + 1] == Color.GREY), "bar must be contiguous"
    return int(r0), int(c0), int(c1)


def _find_seed(grid: np.ndarray, separator_r: int) -> tuple[int, int]:
    seed_cells = np.argwhere(grid[separator_r + 1 :] == Color.ORANGE)
    assert len(seed_cells) == 1, f"expected one orange seed, got {len(seed_cells)}"
    seed_r = int(seed_cells[0, 0]) + separator_r + 1
    seed_c = int(seed_cells[0, 1])
    return seed_r, seed_c


def _upper_pixels(grid: np.ndarray, separator_r: int) -> list[tuple[int, int, int]]:
    mask = np.isin(grid[:separator_r], UPPER_COLORS)
    return [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(mask)]


def _settled_palette(grid: np.ndarray, bar_r: int, separator_r: int) -> list[int]:
    pixels = _upper_pixels(grid, separator_r)
    support = {(int(r), int(c)) for r, c in np.argwhere(grid == Color.GREY)}
    settled: dict[tuple[int, int], int] = {}

    for r, c, color in sorted(pixels, key=lambda item: (bar_r - item[0], item[1])):
        nr = r
        while nr + 1 < grid.shape[0] and (nr + 1, c) not in support:
            nr += 1
        settled[(nr, c)] = color
        support.add((nr, c))

    palette: list[int] = []
    for r in range(bar_r - 1, -1, -1):
        for c in range(grid.shape[1]):
            color = settled.get((r, c), Color.BLACK)
            if color != Color.BLACK:
                palette.append(int(color))
    assert palette, "palette must contain at least one color"
    return palette


def _paint_rings(grid: np.ndarray, separator_r: int, seed: tuple[int, int], palette: list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    seed_r, seed_c = seed
    rows, cols = np.indices(out.shape)
    distance = np.maximum(np.abs(rows - seed_r), np.abs(cols - seed_c))
    lower_mask = rows > separator_r
    for dist, color in enumerate(palette, start=1):
        mask = lower_mask & (distance == dist) & (out == Color.BLACK)
        out[mask] = int(color)
    out[seed_r, seed_c] = Color.ORANGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the upper colored pixels into gray supports, read the short-bar pile as a palette, and paint square rings around the lower seed.

    Input structure:
    - The grid has a black background, one short horizontal gray bar in the upper half, and one full-width gray separator row below it.
    - Several distinct single-pixel colored markers sit above the short bar in separate columns.
    - One orange seed pixel sits below the separator in the lower half.

    Rule:
    - Process the upper colored pixels by increasing distance to the short bar, breaking ties by smaller column, and let each one fall straight down until the next cell in that column is occupied by gray support or by an already settled colored pixel.
    - Read only the hidden settled pixels that lie above the short gray bar: scan the row just above the bar from left to right, then the next row up, and so on, skipping blanks, to form a palette.
    - Keep the upper half unchanged. In the lower half, keep the seed orange and paint each background cell at Chebyshev distance d from the seed with the d-th palette color, clipping at the grid boundary and stopping once the palette is exhausted.

    Key insight:
    - The falling pixels are an intermediate code, not the visible answer; their readout order becomes a one-dimensional palette that controls a distant concentric expansion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_r = _find_separator_row(g)
    bar_r, _bar_c0, _bar_c1 = _find_short_bar(g, separator_r)
    seed = _find_seed(g, separator_r)
    palette = _settled_palette(g, bar_r, separator_r)
    return _paint_rings(g, separator_r, seed, palette)


def _render_output_from_palette(
    grid: np.ndarray, separator_r: int, seed: tuple[int, int], palette: list[int]
) -> np.ndarray:
    out = grid.copy()
    seed_r, seed_c = seed
    for r in range(separator_r + 1, grid.shape[0]):
        for c in range(grid.shape[1]):
            if (r, c) == (seed_r, seed_c) or out[r, c] != Color.BLACK:
                continue
            dist = max(abs(r - seed_r), abs(c - seed_c))
            if 1 <= dist <= len(palette):
                out[r, c] = palette[dist - 1]
    out[seed_r, seed_c] = Color.ORANGE
    return out


def _wrong_palette_outputs(
    grid: np.ndarray,
    separator_r: int,
    seed: tuple[int, int],
    pixels: list[dict[str, int | bool]],
    bar_c0: int,
    bar_c1: int,
) -> list[np.ndarray]:
    inside = [p for p in pixels if p["inside"]]
    wrong_palettes: list[list[int]] = []

    row_major_all = [int(p["color"]) for p in sorted(pixels, key=lambda p: (int(p["row"]), int(p["col"])))]
    if row_major_all:
        wrong_palettes.append(row_major_all)

    ltr_all = [int(p["color"]) for p in sorted(pixels, key=lambda p: int(p["col"]))]
    if ltr_all:
        wrong_palettes.append(ltr_all)

    row_major_inside = [
        int(p["color"])
        for p in sorted(inside, key=lambda p: (int(p["row"]), int(p["col"])))
        if bar_c0 <= int(p["col"]) <= bar_c1
    ]
    if row_major_inside:
        wrong_palettes.append(row_major_inside)

    outputs = []
    seen = set()
    for palette in wrong_palettes:
        key = tuple(palette)
        if key in seen:
            continue
        seen.add(key)
        outputs.append(_render_output_from_palette(grid, separator_r, seed, palette))
    return outputs


def _sample_seed(h: int, w: int, separator_r: int, palette_len: int) -> tuple[int, int] | None:
    lower_top = separator_r + 2
    lower_bottom = h - 2
    if lower_top > lower_bottom:
        return None

    clipped = random.random() < 0.4
    if clipped:
        mode = random.choice(("left", "right", "top", "bottom"))
        if mode == "left":
            seed_c = random.randint(1, min(2, w - 2))
            seed_r = random.randint(lower_top, lower_bottom)
        elif mode == "right":
            seed_c = random.randint(max(1, w - 3), w - 2)
            seed_r = random.randint(lower_top, lower_bottom)
        elif mode == "top":
            seed_r = random.randint(lower_top, min(lower_top + 1, lower_bottom))
            seed_c = random.randint(1, w - 2)
        else:
            seed_r = random.randint(max(lower_top, h - 3), lower_bottom)
            seed_c = random.randint(1, w - 2)
        return seed_r, seed_c

    margin = min(2, palette_len)
    r0 = max(lower_top, separator_r + 1 + margin)
    r1 = min(lower_bottom, h - 1 - margin)
    c0 = 1 + margin
    c1 = w - 2 - margin
    if r0 <= r1 and c0 <= c1:
        return random.randint(r0, r1), random.randint(c0, c1)
    return random.randint(lower_top, lower_bottom), random.randint(1, w - 2)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 22)
    w = random.randint(16, 22)
    bar_r_max = min(h // 2 - 2, h - 8)
    if bar_r_max < 4:
        return None
    bar_r = random.randint(4, bar_r_max)
    separator_r = bar_r + 3

    bar_w = random.randint(7, min(9, w - 4))
    center = (w - bar_w) // 2
    c0_min = max(1, center - 2)
    c0_max = min(w - bar_w - 1, center + 2)
    if c0_min > c0_max:
        return None
    bar_c0 = random.randint(c0_min, c0_max)
    bar_c1 = bar_c0 + bar_w - 1

    n_objects = random.choice((3, 4, 4, 5))
    decoy_count = 0
    if n_objects >= 4 and random.random() < 0.7:
        decoy_count = random.randint(1, min(2, n_objects - 3))
    palette_count = n_objects - decoy_count

    colors = random.sample(list(UPPER_COLORS), n_objects)
    inside_cols = random.sample(list(range(bar_c0, bar_c1 + 1)), palette_count)
    outside_candidates = list(range(0, bar_c0)) + list(range(bar_c1 + 1, w))
    if len(outside_candidates) < decoy_count:
        return None
    outside_cols = random.sample(outside_candidates, decoy_count)

    require_tie = n_objects >= 4 and random.random() < 0.5
    rows = None
    for _ in range(60):
        candidate = [bar_r - random.randint(1, bar_r) for _ in range(n_objects)]
        if len(set(candidate)) < 2:
            continue
        if require_tie and len(set(candidate)) == n_objects:
            continue
        rows = candidate
        break
    if rows is None:
        return None

    pixels: list[dict[str, int | bool]] = []
    for color, col, row in zip(colors[:palette_count], inside_cols, rows[:palette_count]):
        pixels.append({"row": row, "col": col, "color": color, "inside": True})
    for color, col, row in zip(colors[palette_count:], outside_cols, rows[palette_count:]):
        pixels.append({"row": row, "col": col, "color": color, "inside": False})
    random.shuffle(pixels)

    seed = _sample_seed(h, w, separator_r, palette_count)
    if seed is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[bar_r, bar_c0 : bar_c1 + 1] = Color.GREY
    grid[separator_r, :] = Color.GREY
    seed_r, seed_c = seed
    grid[seed_r, seed_c] = Color.ORANGE

    occupied_cols = set()
    for pixel in pixels:
        row = int(pixel["row"])
        col = int(pixel["col"])
        color = int(pixel["color"])
        if not (0 <= row < bar_r):
            return None
        if col in occupied_cols:
            return None
        occupied_cols.add(col)
        grid[row, col] = color

    palette = [int(p["color"]) for p in sorted(pixels, key=lambda p: (not bool(p["inside"]), int(p["col"]))) if bool(p["inside"])]
    if len(palette) != palette_count:
        return None

    expected = _render_output_from_palette(grid, separator_r, seed, palette)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    wrong_outputs = _wrong_palette_outputs(grid, separator_r, seed, pixels, bar_c0, bar_c1)
    distinct_wrong = sum(not np.array_equal(expected, wrong) for wrong in wrong_outputs)
    required_wrong = 2 if decoy_count > 0 else 1
    if distinct_wrong < required_wrong:
        return None

    max_radius = max(
        seed_r - (separator_r + 1),
        grid.shape[0] - 1 - seed_r,
        seed_c,
        grid.shape[1] - 1 - seed_c,
    )
    if max_radius < 2:
        return None

    assert np.array_equal(grid[: separator_r + 1], actual[: separator_r + 1])
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stacked-palette ARC input")
