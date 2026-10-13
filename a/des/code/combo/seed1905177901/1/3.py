from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
QUADRANTS = ("ul", "ur", "ll", "lr")

GRID_SIZES = [11, 13, 15, 17, 19]
MAJORITY_COUNTS = [1, 2, 3, 4, 5, 6]
TIE_COUNTS = [2, 3, 4, 6]
GEN_ATTEMPTS = 400


def _quadrant_slices(mid: int) -> dict[str, tuple[slice, slice]]:
    return {
        "ul": (slice(0, mid), slice(0, mid)),
        "ur": (slice(0, mid), slice(mid + 1, 2 * mid + 1)),
        "ll": (slice(mid + 1, 2 * mid + 1), slice(0, mid)),
        "lr": (slice(mid + 1, 2 * mid + 1), slice(mid + 1, 2 * mid + 1)),
    }


def _distance_key(r: int, c: int, mid: int) -> tuple[int, int, int, int, int]:
    dr = abs(r - mid)
    dc = abs(c - mid)
    return dr + dc, dr, dc, r, c


def _quadrant_pixels(grid: np.ndarray, name: str) -> list[tuple[int, int, int]]:
    side = int(grid.shape[0])
    mid = side // 2
    rows, cols = _quadrant_slices(mid)[name]
    pixels: list[tuple[int, int, int]] = []
    for r in range(rows.start, rows.stop):
        for c in range(cols.start, cols.stop):
            color = int(grid[r, c])
            if color not in (BLACK, GREY):
                pixels.append((r, c, color))
    return pixels


def _winner_color(pixels: list[tuple[int, int, int]], mid: int) -> int:
    counts: dict[int, int] = {}
    for _r, _c, color in pixels:
        counts[color] = counts.get(color, 0) + 1

    top_count = max(counts.values())
    contenders = {color for color, count in counts.items() if count == top_count}
    if len(contenders) == 1:
        return next(iter(contenders))

    best = None
    for idx, (r, c, color) in enumerate(pixels):
        if color not in contenders:
            continue
        candidate = (_distance_key(r, c, mid), idx, color)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return int(best[2])


def _summarize_quadrants(grid: np.ndarray) -> dict[str, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    side = int(g.shape[0])
    assert g.ndim == 2 and side == int(g.shape[1]) and side % 2 == 1
    mid = side // 2
    assert np.all(g[mid, :] == GREY)
    assert np.all(g[:, mid] == GREY)

    summary: dict[str, dict[str, int]] = {}
    for name in QUADRANTS:
        pixels = _quadrant_pixels(g, name)
        assert pixels, f"quadrant {name} must contain foreground"
        summary[name] = {
            "count": len(pixels),
            "color": _winner_color(pixels, mid),
        }
    return summary


def _paint_square(out: np.ndarray, r0: int, c0: int, side: int, color: int) -> None:
    out[r0 : r0 + side, c0 : c0 + side] = int(color)


def _render_output(summary: dict[str, dict[str, int]]) -> np.ndarray:
    ul_n = int(summary["ul"]["count"])
    ur_n = int(summary["ur"]["count"])
    ll_n = int(summary["ll"]["count"])
    lr_n = int(summary["lr"]["count"])

    top_h = max(ul_n, ur_n)
    bottom_h = max(ll_n, lr_n)
    left_w = max(ul_n, ll_n)
    right_w = max(ur_n, lr_n)

    out = np.full((top_h + bottom_h, left_w + right_w), BLACK, dtype=int)
    _paint_square(out, top_h - ul_n, left_w - ul_n, ul_n, int(summary["ul"]["color"]))
    _paint_square(out, top_h - ur_n, left_w, ur_n, int(summary["ur"]["color"]))
    _paint_square(out, top_h, left_w - ll_n, ll_n, int(summary["ll"]["color"]))
    _paint_square(out, top_h, left_w, lr_n, int(summary["lr"]["color"]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered foreground pixels by quadrant.

    Each quadrant contributes:
    - side length = number of non-black, non-grey pixels in that quadrant
    - fill color = majority foreground color in that quadrant
    - count ties = color owning the closest pixel to the center cross

    The four solid squares keep their quadrant order and meet at a shared center
    corner; the output is the tight bounding box of that arrangement.
    """
    return _render_output(_summarize_quadrants(grid))


def _quadrant_cells(side: int, name: str) -> list[tuple[int, int]]:
    mid = side // 2
    rows, cols = _quadrant_slices(mid)[name]
    cells = [(r, c) for r in range(rows.start, rows.stop) for c in range(cols.start, cols.stop)]
    cells.sort(key=lambda rc: _distance_key(rc[0], rc[1], mid))
    return cells


def _majority_colors(count: int) -> list[int]:
    color_count = 1 if count == 1 else random.randint(1, min(3, count - 1))
    colors = random.sample(PALETTE, color_count)
    if color_count == 1:
        return [colors[0]] * count

    while True:
        parts = [1] * color_count
        for _ in range(count - color_count):
            parts[random.randrange(color_count)] += 1
        top = max(parts)
        if parts.count(top) == 1:
            break

    pixels: list[int] = []
    for color, part in zip(colors, parts):
        pixels.extend([color] * part)
    random.shuffle(pixels)
    return pixels


def _tie_colors(count: int) -> tuple[list[int], int]:
    divisors = [d for d in (2, 3) if count % d == 0]
    tied_color_count = random.choice(divisors)
    per_color = count // tied_color_count
    colors = random.sample(PALETTE, tied_color_count)
    winner = random.choice(colors)

    pixels: list[int] = []
    for color in colors:
        pixels.extend([color] * per_color)
    return pixels, winner


def _place_quadrant_pixels(grid: np.ndarray, name: str, count: int, mode: str) -> None:
    cells = _quadrant_cells(int(grid.shape[0]), name)
    if mode == "tie":
        colors, winner = _tie_colors(count)
        anchor = cells[0]
        rest_cells = random.sample(cells[1:], count - 1)
        remaining_colors = list(colors)
        remaining_colors.remove(winner)
        random.shuffle(remaining_colors)
        grid[anchor] = winner
        for (r, c), color in zip(rest_cells, remaining_colors):
            grid[r, c] = int(color)
        return

    chosen_cells = random.sample(cells, count)
    colors = _majority_colors(count)
    for (r, c), color in zip(chosen_cells, colors):
        grid[r, c] = int(color)


def _generate_candidate() -> np.ndarray:
    side = random.choice(GRID_SIZES)
    grid = np.full((side, side), BLACK, dtype=int)
    mid = side // 2
    grid[mid, :] = GREY
    grid[:, mid] = GREY

    modes = ["tie" if random.random() < 0.5 else "majority" for _ in QUADRANTS]
    counts = [
        random.choice(TIE_COUNTS if mode == "tie" else MAJORITY_COUNTS)
        for mode in modes
    ]
    if sum(counts) > 25:
        return None

    for name, count, mode in zip(QUADRANTS, counts, modes):
        _place_quadrant_pixels(grid, name, count, mode)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_candidate()
        if grid is None:
            continue
        out = main(grid)
        assert isinstance(out, np.ndarray)
        assert out.ndim == 2
        assert not np.any(out == GREY)
        return grid
    raise RuntimeError(f"failed to generate a valid sample in {GEN_ATTEMPTS} attempts")
