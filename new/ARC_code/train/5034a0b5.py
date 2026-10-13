from __future__ import annotations

import numpy as np

from arc_utils import get_bg
import random


def _solid_color(cells: np.ndarray) -> int:
    vals = np.unique(cells)
    assert len(vals) == 1
    return int(vals[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift each interior pixel whose color matches a frame side one cell toward that matching side.

    Input structure
    The grid has a one-cell-thick rectangular frame: the four corners are the
    background color, and each side between corners is a solid, distinct side
    color. The interior is mostly background with sparse foreground pixels.

    Rule
    Interior pixels with the top-side color move one row up, bottom-side color
    move one row down, left-side color move one column left, and right-side color
    move one column right. A pixel already adjacent to its matching side stays in
    place; colors that do not match a frame side and the frame itself are unchanged.

    Key insight
    The colored frame sides act as direction labels for matching interior pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape
    assert H >= 3 and W >= 3

    bg = int(g[0, 0])
    assert int(get_bg(g)) == bg
    assert {int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])} == {bg}

    top = _solid_color(g[0, 1:-1])
    bottom = _solid_color(g[-1, 1:-1])
    left = _solid_color(g[1:-1, 0])
    right = _solid_color(g[1:-1, -1])
    assert len({bg, top, bottom, left, right}) == 5

    directions = {
        top: (-1, 0),
        bottom: (1, 0),
        left: (0, -1),
        right: (0, 1),
    }

    moves: list[tuple[int, int, int, int, int]] = []
    moving_sources: set[tuple[int, int]] = set()
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            color = int(g[r, c])
            if color not in directions:
                continue
            dr, dc = directions[color]
            nr, nc = r + dr, c + dc
            if not (1 <= nr < H - 1 and 1 <= nc < W - 1):
                assert int(g[nr, nc]) == color
                continue
            moves.append((r, c, nr, nc, color))
            moving_sources.add((r, c))

    destinations = [(nr, nc) for _r, _c, nr, nc, _color in moves]
    assert len(destinations) == len(set(destinations))
    for _r, _c, nr, nc, _color in moves:
        assert int(g[nr, nc]) == bg or (nr, nc) in moving_sources

    out = g.copy()
    for r, c, _nr, _nc, _color in moves:
        out[r, c] = bg
    for _r, _c, nr, nc, color in moves:
        out[nr, nc] = color
    return out

# --------------- generator (att1, Claude) ---------------

# Canvas and palette bounds.
CANVAS_SIZE_MIN = 8
CANVAS_SIZE_MAX = 13
COLOR_COUNT = 10
FRAME_COLOR_COUNT = 5
ALL_COLORS = list(range(COLOR_COUNT))

# Interior density and object counts.
MOVING_PIXELS_MIN = 8
MOVING_PIXELS_MAX = 19
STATIC_PIXELS_MIN = 0
STATIC_PIXELS_MAX = 9
DENSITY_MIN = 0.12
DENSITY_MAX = 0.27
STATIC_COLOR_MAX = 3

# Per-side moving pixel allocation.
SIDE_COUNT_BINS = 4
SIDE_COUNT_MIN = 0
SIDE_COUNT_MAX = 6

# Retry budget for rejection sampling.
GENERATION_ATTEMPTS = 300


def _make_framed_output(size, bg, top_c, bot_c, left_c, right_c):
    out = np.full((size, size), bg, dtype=int)
    out[0, 1:-1] = top_c
    out[-1, 1:-1] = bot_c
    out[1:-1, 0] = left_c
    out[1:-1, -1] = right_c
    return out


def _adjust_pixel_counts(total_interior, num_moving, num_static):
    total_pixels = num_moving + num_static
    min_pixels = max(1, int(np.ceil(DENSITY_MIN * total_interior)))
    max_pixels = int(np.floor(DENSITY_MAX * total_interior))

    if total_pixels < min_pixels or total_pixels > max_pixels:
        total_pixels = max(min_pixels, min(max_pixels, total_pixels))
        num_moving = min(num_moving, total_pixels)
        num_static = total_pixels - num_moving
        if num_moving < MOVING_PIXELS_MIN:
            return None

    if total_pixels > total_interior:
        return None
    return num_moving, num_static, total_pixels


def _choose_static_palette(rng, bg, side_colors, num_moving, num_static, total_pixels):
    available_static_colors = [c for c in ALL_COLORS if c != bg and c not in side_colors]
    num_static_colors = rng.randint(0, min(STATIC_COLOR_MAX, len(available_static_colors)))

    if num_static > 0 and num_static_colors == 0:
        num_static = 0
        num_moving = total_pixels
        if num_moving > MOVING_PIXELS_MAX:
            num_moving = MOVING_PIXELS_MAX

    static_palette = (
        rng.sample(available_static_colors, num_static_colors)
        if num_static_colors > 0
        else []
    )
    if num_static > 0 and not static_palette:
        num_static = 0
    return num_moving, num_static, static_palette


def _random_partition(rng, total, n_bins, lo, hi):
    """Partition total into bounded bins using the original biased sampler."""
    if total < n_bins * lo or total > n_bins * hi:
        return None

    result = [lo] * n_bins
    remaining = total - n_bins * lo
    for i in range(n_bins):
        max_add = min(hi - lo, remaining)
        add = rng.randint(0, max_add)
        result[i] += add
        remaining -= add

    while remaining > 0:
        indices = [i for i in range(n_bins) if result[i] < hi]
        if not indices:
            return None
        idx = rng.choice(indices)
        add = min(remaining, hi - result[idx])
        result[idx] += add
        remaining -= add
    return result


def _place_moving_pixels(rng, positions, size, side_colors, partition, forward, reverse):
    """Choose output/source cells for side-colored pixels without source collisions."""
    occupied_out = set()
    occupied_inp = set()
    moving_pixels = []
    last = size - 2

    rng.shuffle(positions)
    for color, count in zip(side_colors, partition):
        dr_fwd, dc_fwd = forward[color]
        dr_rev, dc_rev = reverse[color]
        candidates = []
        for row, col in positions:
            if (row, col) in occupied_out:
                continue

            nr_fwd, nc_fwd = row + dr_fwd, col + dc_fwd
            if not (1 <= nr_fwd <= last and 1 <= nc_fwd <= last):
                inp_r, inp_c = row, col
            else:
                inp_r, inp_c = row + dr_rev, col + dc_rev
                if not (1 <= inp_r <= last and 1 <= inp_c <= last):
                    continue

            if (inp_r, inp_c) in occupied_inp:
                continue
            candidates.append((row, col, inp_r, inp_c))

        if len(candidates) < count:
            return None

        rng.shuffle(candidates)
        for out_r, out_c, inp_r, inp_c in candidates[:count]:
            occupied_out.add((out_r, out_c))
            occupied_inp.add((inp_r, inp_c))
            moving_pixels.append((out_r, out_c, inp_r, inp_c, color))

    return moving_pixels, occupied_out, occupied_inp


def _place_static_pixels(rng, positions, num_static, static_palette, occupied_out, occupied_inp):
    static_placed = []
    for _ in range(num_static):
        found = False
        for row, col in positions:
            if (row, col) not in occupied_out and (row, col) not in occupied_inp:
                color = rng.choice(static_palette)
                occupied_out.add((row, col))
                occupied_inp.add((row, col))
                static_placed.append((row, col, color))
                found = True
                break
        if not found:
            break
    return static_placed


def _draw_pixels(out, moving_pixels, static_placed):
    for out_r, out_c, _, _, color in moving_pixels:
        out[out_r, out_c] = color
    for row, col, color in static_placed:
        out[row, col] = color


def _derive_input(out, bg, moving_pixels):
    inp = out.copy()
    for out_r, out_c, _inp_r, _inp_c, _color in moving_pixels:
        inp[out_r, out_c] = bg
    for _out_r, _out_c, inp_r, inp_c, color in moving_pixels:
        inp[inp_r, inp_c] = color
    return inp


def _is_valid_generated_pair(inp, out, bg):
    if int(np.bincount(inp.flatten()).argmax()) != bg:
        return False
    try:
        result = main(inp)
    except Exception:
        return False
    return np.array_equal(result, out)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    size = rng.randint(CANVAS_SIZE_MIN, CANVAS_SIZE_MAX)
    bg, top_c, bot_c, left_c, right_c = rng.sample(ALL_COLORS, FRAME_COLOR_COUNT)
    out = _make_framed_output(size, bg, top_c, bot_c, left_c, right_c)

    total_interior = (size - 2) * (size - 2)
    num_moving = rng.randint(MOVING_PIXELS_MIN, MOVING_PIXELS_MAX)
    num_static = rng.randint(STATIC_PIXELS_MIN, STATIC_PIXELS_MAX)

    counts = _adjust_pixel_counts(total_interior, num_moving, num_static)
    if counts is None:
        return None
    num_moving, num_static, total_pixels = counts

    forward = {top_c: (-1, 0), bot_c: (1, 0), left_c: (0, -1), right_c: (0, 1)}
    reverse = {top_c: (1, 0), bot_c: (-1, 0), left_c: (0, 1), right_c: (0, -1)}
    side_colors = [top_c, bot_c, left_c, right_c]
    num_moving, num_static, static_palette = _choose_static_palette(
        rng, bg, side_colors, num_moving, num_static, total_pixels
    )

    all_positions = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    rng.shuffle(all_positions)

    partition = _random_partition(
        rng, num_moving, SIDE_COUNT_BINS, SIDE_COUNT_MIN, SIDE_COUNT_MAX
    )
    if partition is None:
        return None

    moving_result = _place_moving_pixels(
        rng, all_positions, size, side_colors, partition, forward, reverse
    )
    if moving_result is None:
        return None
    moving_pixels, occupied_out, occupied_inp = moving_result

    static_placed = _place_static_pixels(
        rng, all_positions, num_static, static_palette, occupied_out, occupied_inp
    )
    _draw_pixels(out, moving_pixels, static_placed)
    inp = _derive_input(out, bg, moving_pixels)

    if _is_valid_generated_pair(inp, out, bg):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
