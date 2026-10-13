import random

import numpy as np


_GEN_RETRIES = 400
_GRID_H_RANGE = (14, 20)
_GRID_W_RANGE = (22, 30)
_BAND_COUNT_RANGE = (3, 5)
_BAND_MIN_LEN = 2
_INPUT_SIZE_RANGE = (2, 6)
_ORIENTATIONS = ("bl", "br", "tl", "tr")


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _compress_runs(row: np.ndarray, bg: int) -> list[int]:
    colors: list[int] = []
    prev = None
    for value in map(int, row):
        if value == prev:
            continue
        prev = value
        if value != bg:
            colors.append(value)
    return colors


def _base_triangle(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for r in range(size):
        mask[r, : r + 1] = True
    return mask


def _triangle_mask(size: int, orientation: str) -> np.ndarray:
    mask = _base_triangle(size)
    if orientation == "bl":
        return mask
    if orientation == "br":
        return np.fliplr(mask)
    if orientation == "tl":
        return np.flipud(mask)
    if orientation == "tr":
        return np.flipud(np.fliplr(mask))
    raise ValueError(f"unknown orientation: {orientation}")


def _draw_output_triangle(grid: np.ndarray, left: int, size: int, color: int) -> None:
    top = grid.shape[0] - size
    for r in range(size):
        grid[top + r, left : left + r + 1] = color


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the bottom-row color strip, then rebuild one left-to-right sequence of
    lower-left right triangles whose sizes are 1..N in strip order.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    band_colors = _compress_runs(g[-1], bg)
    triangle_colors = {int(color) for color in np.unique(g[:-1]) if int(color) != bg}
    ordered_colors = [color for color in band_colors if color in triangle_colors]
    if not ordered_colors:
        ordered_colors = band_colors

    n = len(ordered_colors)
    width = n * (n + 1) // 2 + max(0, n - 1)
    out = np.full((n, width), bg, dtype=int)

    left = 0
    for size, color in enumerate(ordered_colors, start=1):
        _draw_output_triangle(out, left, size, color)
        left += size + 1

    return out


def _random_lengths(total: int, parts: int, minimum: int, rng: random.Random) -> list[int]:
    lengths = [minimum] * parts
    for _ in range(total - parts * minimum):
        lengths[rng.randrange(parts)] += 1
    return lengths


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = color


def _valid_positions(occupied: np.ndarray, mask: np.ndarray, max_bottom: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    mh, mw = mask.shape
    positions: list[tuple[int, int]] = []
    max_r0 = max_bottom - mh + 1
    if max_r0 < 0:
        return positions
    for r0 in range(max_r0 + 1):
        for c0 in range(w - mw + 1):
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + mh + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + mw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            positions.append((r0, c0))
    return positions


def _place_triangle(
    grid: np.ndarray,
    occupied: np.ndarray,
    mask: np.ndarray,
    color: int,
    max_bottom: int,
    rng: random.Random,
) -> bool:
    positions = _valid_positions(occupied, mask, max_bottom)
    if not positions:
        return False
    r0, c0 = rng.choice(positions)
    _paint_mask(grid, mask, r0, c0, color)
    mh, mw = mask.shape
    occupied[r0 : r0 + mh, c0 : c0 + mw] |= mask
    return True


def _triangle_specs(colors: list[int], h: int, w: int, rng: random.Random) -> list[tuple[int, np.ndarray]]:
    max_size = min(_INPUT_SIZE_RANGE[1], h - 4, 2 + w // (len(colors) + 2))
    min_size = min(_INPUT_SIZE_RANGE[0], max_size)
    if max_size < _INPUT_SIZE_RANGE[0]:
        return []
    specs: list[tuple[int, np.ndarray]] = []
    for color in colors:
        size = rng.randint(min_size, max_size)
        orientation = rng.choice(_ORIENTATIONS)
        specs.append((color, _triangle_mask(size, orientation)))
    specs.sort(key=lambda item: int(item[1].sum()), reverse=True)
    return specs


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*_GRID_H_RANGE)
    w = rng.randint(*_GRID_W_RANGE)
    band_count = rng.randint(*_BAND_COUNT_RANGE)

    palette = list(range(10))
    bg = rng.choice(palette)
    palette.remove(bg)
    band_colors = rng.sample(palette, band_count)
    band_lengths = _random_lengths(w, band_count, _BAND_MIN_LEN, rng)

    grid = np.full((h, w), bg, dtype=int)
    start = 0
    for color, length in zip(band_colors, band_lengths):
        grid[-1, start : start + length] = color
        start += length

    specs = _triangle_specs(band_colors, h, w, rng)
    if len(specs) != band_count:
        return None

    occupied = np.zeros((h, w), dtype=bool)
    max_bottom = h - 3
    for color, mask in specs:
        if not _place_triangle(grid, occupied, mask, color, max_bottom, rng):
            return None

    upper = grid[:-1]
    if _get_bg(grid) != bg:
        return None
    if _compress_runs(grid[-1], bg) != band_colors:
        return None
    if {int(color) for color in np.unique(upper) if int(color) != bg} != set(band_colors):
        return None
    for color in band_colors:
        if len(_components(upper == color)) != 1:
            return None

    out = main(grid)
    if out.shape[0] != band_count:
        return None
    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
