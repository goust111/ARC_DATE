from __future__ import annotations

import random

import numpy as np


COLOR_VALUES = tuple(range(10))
MAX_STRUCTURE_SIDE = 15
MAX_CANVAS_SIDE = 30
RING_COUNT_RANGE = (2, 5)
THICKNESS_OPTIONS = (1, 2, 3)
CANVAS_EXTRA_RANGE = (2, 12)
CENTER_JITTER = 2
GEN_RETRY_LIMIT = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _frame_mask(shape: tuple[int, int], thickness: int) -> np.ndarray:
    h, w = map(int, shape)
    rows, cols = np.indices((h, w))
    return (
        (rows < int(thickness))
        | (rows >= int(h - thickness))
        | (cols < int(thickness))
        | (cols >= int(w - thickness))
    )


def _render_ring(h: int, w: int, thickness: int, color: int, bg: int) -> np.ndarray:
    ring = np.full((int(h), int(w)), int(bg), dtype=int)
    ring[_frame_mask((int(h), int(w)), int(thickness))] = int(color)
    return ring


def _paint_ring(
    grid: np.ndarray,
    top: int,
    left: int,
    h: int,
    w: int,
    thickness: int,
    color: int,
) -> None:
    view = grid[int(top) : int(top + h), int(left) : int(left + w)]
    view[_frame_mask((int(h), int(w)), int(thickness))] = int(color)


def _extract_rings(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    coords = np.argwhere(g != bg)
    assert coords.size > 0, "expected a non-background structure"

    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    rings: list[tuple[int, int, int, int]] = []

    while True:
        sub = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        h, w = map(int, sub.shape)
        color = int(sub[0, 0])
        assert color != bg, "outermost ring must occupy the structure bbox corners"

        thickness = 0
        for t in range(1, int(min(h, w) // 2) + 1):
            if np.all(sub[_frame_mask((h, w), t)] == color):
                thickness = int(t)
            else:
                break
        assert thickness > 0, "failed to detect a positive ring thickness"

        rings.append((int(h), int(w), int(thickness), int(color)))

        inner_h = int(h - 2 * thickness)
        inner_w = int(w - 2 * thickness)
        if inner_h <= 0 or inner_w <= 0:
            break

        inner = sub[int(thickness) : int(h - thickness), int(thickness) : int(w - thickness)]
        if not np.any(inner != bg):
            break

        # The next nested ring must be flush with this inset box because layers touch directly.
        inner_coords = np.argwhere(inner != bg)
        ir0, ic0 = map(int, inner_coords.min(axis=0))
        ir1, ic1 = map(int, inner_coords.max(axis=0))
        assert (ir0, ic0, ir1, ic1) == (0, 0, inner_h - 1, inner_w - 1)

        r0 += int(thickness)
        c0 += int(thickness)
        r1 -= int(thickness)
        c1 -= int(thickness)

    return int(bg), rings


def _build_output(rings: list[tuple[int, int, int, int]], bg: int) -> np.ndarray:
    height = max(int(h + thickness - 1) for h, _w, thickness, _color in rings)
    width = sum(int(w) for _h, w, _thickness, _color in rings) + len(rings) - 1
    out = np.full((int(height), int(width)), int(bg), dtype=int)

    col = 0
    for h, w, thickness, color in rings:
        top = int(thickness - 1)
        out[int(top) : int(top + h), int(col) : int(col + w)] = _render_ring(
            int(h), int(w), int(thickness), int(color), int(bg)
        )
        col += int(w) + 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Peel nested rectangular rings and lay them out left-to-right.

    The input contains one nested concentric structure on a dominant background.
    Each layer is a closed rectangular ring with its own color and thickness.
    Extract layers from outermost to innermost, keep each ring's shape, place
    them left-to-right with one background column between neighbors, and offset
    each ring downward by thickness - 1 rows.
    """
    bg, rings = _extract_rings(np.asarray(grid, dtype=int))
    return _build_output(rings, bg)


def _sample_thicknesses() -> list[int] | None:
    ring_count = random.randint(*RING_COUNT_RANGE)
    for _ in range(200):
        thicknesses = [random.choice(THICKNESS_OPTIONS) for _ in range(ring_count)]
        if sum(thicknesses) <= 7:
            return thicknesses
    return None


def _rings_from_params(
    thicknesses: list[int],
    inner_h: int,
    inner_w: int,
    colors: list[int],
) -> list[tuple[int, int, int, int]]:
    h = int(inner_h + 2 * sum(thicknesses))
    w = int(inner_w + 2 * sum(thicknesses))
    rings: list[tuple[int, int, int, int]] = []
    for thickness, color in zip(thicknesses, colors, strict=False):
        rings.append((int(h), int(w), int(thickness), int(color)))
        h -= 2 * int(thickness)
        w -= 2 * int(thickness)
    assert h == int(inner_h) and w == int(inner_w)
    return rings


def _sample_canvas(
    outer_h: int,
    outer_w: int,
    structure_area: int,
) -> tuple[int, int, int, int] | None:
    min_h = int(outer_h + 2)
    min_w = int(outer_w + 2)
    max_h = int(min(MAX_CANVAS_SIDE, outer_h + CANVAS_EXTRA_RANGE[1]))
    max_w = int(min(MAX_CANVAS_SIDE, outer_w + CANVAS_EXTRA_RANGE[1]))

    for _ in range(100):
        if min_h > max_h or min_w > max_w:
            return None
        canvas_h = random.randint(min_h, max_h)
        canvas_w = random.randint(min_w, max_w)
        bg_area = int(canvas_h * canvas_w - structure_area)
        if bg_area <= int(structure_area):
            continue

        max_top = int(canvas_h - outer_h - 1)
        max_left = int(canvas_w - outer_w - 1)
        if max_top < 1 or max_left < 1:
            continue

        center_top = int((canvas_h - outer_h) // 2)
        center_left = int((canvas_w - outer_w) // 2)
        top_lo = max(1, center_top - CENTER_JITTER)
        top_hi = min(max_top, center_top + CENTER_JITTER)
        left_lo = max(1, center_left - CENTER_JITTER)
        left_hi = min(max_left, center_left + CENTER_JITTER)
        if top_lo > top_hi or left_lo > left_hi:
            continue

        top = random.randint(top_lo, top_hi)
        left = random.randint(left_lo, left_hi)
        return int(canvas_h), int(canvas_w), int(top), int(left)

    return None


def _render_input(
    bg: int,
    rings: list[tuple[int, int, int, int]],
    canvas_h: int,
    canvas_w: int,
    top: int,
    left: int,
) -> np.ndarray:
    grid = np.full((int(canvas_h), int(canvas_w)), int(bg), dtype=int)
    cur_top = int(top)
    cur_left = int(left)
    for h, w, thickness, color in rings:
        _paint_ring(grid, cur_top, cur_left, int(h), int(w), int(thickness), int(color))
        cur_top += int(thickness)
        cur_left += int(thickness)
    return grid


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(COLOR_VALUES)
    thicknesses = _sample_thicknesses()
    if thicknesses is None:
        return None

    total_thickness = int(sum(thicknesses))
    max_inner_h = int(MAX_STRUCTURE_SIDE - 2 * total_thickness)
    max_inner_w = int(MAX_STRUCTURE_SIDE - 2 * total_thickness)
    if max_inner_h < 1 or max_inner_w < 1:
        return None

    inner_h = random.randint(1, max_inner_h)
    inner_w = random.randint(1, max_inner_w)
    outer_h = int(inner_h + 2 * total_thickness)
    outer_w = int(inner_w + 2 * total_thickness)

    palette = [color for color in COLOR_VALUES if color != bg]
    if len(palette) < len(thicknesses):
        return None
    colors = random.sample(palette, len(thicknesses))

    rings = _rings_from_params(thicknesses, inner_h, inner_w, colors)
    structure_area = int(outer_h * outer_w - inner_h * inner_w)
    canvas = _sample_canvas(outer_h, outer_w, structure_area)
    if canvas is None:
        return None
    canvas_h, canvas_w, top, left = canvas

    inp = _render_input(bg, rings, canvas_h, canvas_w, top, left)
    parsed_bg, parsed_rings = _extract_rings(inp)
    if parsed_bg != bg or parsed_rings != rings:
        return None

    expected = _build_output(rings, bg)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRY_LIMIT} attempts")
