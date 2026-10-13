import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
SIDES = ("top", "bottom", "left", "right")

# Generator bounds.
GRID_MIN_SIZE = 14
GRID_MAX_SIZE = 30
RING_SIDE_MIN = 4
RING_SIDE_MAX = 11
MAX_OBJECTS = 4
PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    banned = set() if exclude is None else {int(x) for x in exclude}
    return [int(color) for color in np.unique(grid) if int(color) not in banned]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _pick_gap_side(side_gaps: dict[str, np.ndarray]) -> tuple[str, np.ndarray]:
    nonempty = [(side, idx) for side, idx in side_gaps.items() if idx.size > 0]
    assert nonempty, "each ring must contain one missing border segment"
    if len(nonempty) == 1:
        return nonempty[0]

    max_len = max(idx.size for _side, idx in nonempty)
    winners = [(side, idx) for side, idx in nonempty if idx.size == max_len]
    assert len(winners) == 1, "gap side is ambiguous"
    return winners[0]


def _find_gap(mask: np.ndarray) -> tuple[tuple[int, int, int, int], str, int, int]:
    r0, c0, r1, c1 = _bbox(mask)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height >= 3 and width >= 3

    side_gaps = {
        "top": np.flatnonzero(~mask[r0, c0 + 1 : c1]),
        "bottom": np.flatnonzero(~mask[r1, c0 + 1 : c1]),
        "left": np.flatnonzero(~mask[r0 + 1 : r1, c0]),
        "right": np.flatnonzero(~mask[r0 + 1 : r1, c1]),
    }
    side, idx = _pick_gap_side(side_gaps)
    assert idx[-1] - idx[0] + 1 == idx.size, "gap must be one contiguous run"

    if side in ("top", "bottom"):
        start = c0 + 1 + int(idx[0])
        end = c0 + 1 + int(idx[-1])
    else:
        start = r0 + 1 + int(idx[0])
        end = r0 + 1 + int(idx[-1])
    return (r0, c0, r1, c1), side, start, end


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect each broken rectangular ring and grow an equal-length line outside the opposite side.

    Input structure:
    - The background is black.
    - The grid contains several non-overlapping one-pixel-thick hollow rectangles, each in a unique color.
    - Every rectangle has exactly one contiguous gap on one side, while the other border pixels stay axis-aligned.

    Rule:
    - For each colored ring, find which side contains the gap and measure the gap length L.
    - Draw a same-color straight segment of length L immediately outside the opposite side of the ring.
    - Keep the new segment parallel to that opposite side, and align its center with the center of the gap.
    - Leave the original missing border cells unchanged.

    Key insight:
    - The missing run simultaneously encodes direction and length: its side chooses the mirrored outside side, and its span is copied directly as the new segment.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for color in _colors_present(g, exclude={BG}):
        mask = g == color
        (r0, c0, r1, c1), side, start, end = _find_gap(mask)

        if side == "top":
            assert r1 + 1 < g.shape[0]
            target = out[r1 + 1, start : end + 1]
            assert np.all((target == BG) | (target == color))
            out[r1 + 1, start : end + 1] = color
        elif side == "bottom":
            assert r0 - 1 >= 0
            target = out[r0 - 1, start : end + 1]
            assert np.all((target == BG) | (target == color))
            out[r0 - 1, start : end + 1] = color
        elif side == "left":
            assert c1 + 1 < g.shape[1]
            target = out[start : end + 1, c1 + 1]
            assert np.all((target == BG) | (target == color))
            out[start : end + 1, c1 + 1] = color
        else:
            assert c0 - 1 >= 0
            target = out[start : end + 1, c0 - 1]
            assert np.all((target == BG) | (target == color))
            out[start : end + 1, c0 - 1] = color

    return out


def _build_ring(height: int, width: int, side: str, gap_start: int, gap_len: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    ring = np.zeros((height, width), dtype=bool)
    ring[0, :] = True
    ring[height - 1, :] = True
    ring[:, 0] = True
    ring[:, width - 1] = True

    if side == "top":
        ring[0, gap_start : gap_start + gap_len] = False
        line = [(height, c) for c in range(gap_start, gap_start + gap_len)]
    elif side == "bottom":
        ring[height - 1, gap_start : gap_start + gap_len] = False
        line = [(-1, c) for c in range(gap_start, gap_start + gap_len)]
    elif side == "left":
        ring[gap_start : gap_start + gap_len, 0] = False
        line = [(r, width) for r in range(gap_start, gap_start + gap_len)]
    else:
        ring[gap_start : gap_start + gap_len, width - 1] = False
        line = [(r, -1) for r in range(gap_start, gap_start + gap_len)]

    return ring, line


def _used_bbox(height: int, width: int, line: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [0, height - 1] + [r for r, _c in line]
    cols = [0, width - 1] + [c for _r, c in line]
    return min(rows), min(cols), max(rows), max(cols)


def _paint_ring(grid: np.ndarray, r0: int, c0: int, ring: np.ndarray, color: int) -> None:
    view = grid[r0 : r0 + ring.shape[0], c0 : c0 + ring.shape[1]]
    view[ring] = int(color)


def _paint_line(grid: np.ndarray, line: list[tuple[int, int]], color: int) -> None:
    for rr, cc in line:
        assert grid[rr, cc] == BG
        grid[rr, cc] = int(color)


def _max_objects(rows: int, cols: int) -> int:
    area = rows * cols
    if area >= 520:
        return 4
    if area >= 320:
        return 3
    return 2


def _sample_gap(height: int, width: int, side: str) -> tuple[int, int]:
    span = width - 2 if side in ("top", "bottom") else height - 2
    gap_len = random.randint(1, span)
    gap_start = random.randint(1, span - gap_len + 1)
    return gap_start, gap_len


def _place_object(
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
) -> tuple[int, list[tuple[int, int]]] | None:
    rows, cols = grid.shape
    max_height = min(RING_SIDE_MAX, rows - 4)
    max_width = min(RING_SIDE_MAX, cols - 4)
    if max_height < RING_SIDE_MIN or max_width < RING_SIDE_MIN:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        side = random.choice(SIDES)
        height = random.randint(RING_SIDE_MIN, max_height)
        width = random.randint(RING_SIDE_MIN, max_width)
        gap_start, gap_len = _sample_gap(height, width, side)
        ring, line = _build_ring(height, width, side, gap_start, gap_len)

        used_r0, used_c0, used_r1, used_c1 = _used_bbox(height, width, line)
        place_r_lo = 1 - used_r0
        place_r_hi = rows - used_r1 - 2
        place_c_lo = 1 - used_c0
        place_c_hi = cols - used_c1 - 2
        if place_r_lo > place_r_hi or place_c_lo > place_c_hi:
            continue

        r0 = random.randint(place_r_lo, place_r_hi)
        c0 = random.randint(place_c_lo, place_c_hi)

        reserve_r0 = r0 + used_r0 - 1
        reserve_c0 = c0 + used_c0 - 1
        reserve_r1 = r0 + used_r1 + 1
        reserve_c1 = c0 + used_c1 + 1
        if reserved[reserve_r0 : reserve_r1 + 1, reserve_c0 : reserve_c1 + 1].any():
            continue

        _paint_ring(grid, r0, c0, ring, color)
        reserved[reserve_r0 : reserve_r1 + 1, reserve_c0 : reserve_c1 + 1] = True
        return color, [(r0 + dr, c0 + dc) for dr, dc in line]

    return None


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    cols = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    grid = np.full((rows, cols), BG, dtype=int)
    reserved = np.zeros((rows, cols), dtype=bool)

    n_objects = random.randint(1, _max_objects(rows, cols))
    colors = random.sample(PALETTE, n_objects)

    line_specs: list[tuple[int, list[tuple[int, int]]]] = []
    for color in colors:
        placed = _place_object(grid, reserved, color)
        if placed is None:
            return None
        line_specs.append(placed)

    expected = grid.copy()
    for color, line_coords in line_specs:
        _paint_line(expected, line_coords, color)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
