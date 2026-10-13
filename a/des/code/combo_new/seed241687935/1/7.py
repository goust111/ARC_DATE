from __future__ import annotations

import random

import numpy as np


BG = 0
RING = [2, 7, 4, 3, 8, 1, 6]  # red -> orange -> yellow -> green -> azure -> blue -> magenta
RING_INDEX = {color: idx for idx, color in enumerate(RING)}
GEN_ATTEMPTS = 400


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _turn(direction: tuple[int, int], spin: int) -> tuple[int, int]:
    if spin == 1:
        return _turn_right(direction)
    if spin == -1:
        return _turn_left(direction)
    raise AssertionError(f"invalid spin: {spin}")


def _sequence_spin(colors: list[int]) -> int | None:
    if len(colors) < 2:
        return None
    if any(color not in RING_INDEX for color in colors):
        return None

    first = colors[0]
    second = colors[1]
    if RING[(RING_INDEX[first] + 1) % len(RING)] == second:
        spin = 1
    elif RING[(RING_INDEX[first] - 1) % len(RING)] == second:
        spin = -1
    else:
        return None

    for a, b in zip(colors, colors[1:]):
        if RING[(RING_INDEX[a] + spin) % len(RING)] != b:
            return None
    return spin


def _parse_seed(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[int], int, tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid != BG)]
    assert 3 <= len(coords) <= 5, f"expected a 3-5 pixel seed line, got {len(coords)}"

    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    assert (len(rows) == 1) ^ (len(cols) == 1), "seed must be horizontal or vertical"

    if len(rows) == 1:
        ordered = sorted(coords, key=lambda rc: rc[1])
        cols_sorted = [c for _, c in ordered]
        assert cols_sorted == list(range(cols_sorted[0], cols_sorted[-1] + 1)), "horizontal seed must be contiguous"
    else:
        ordered = sorted(coords, key=lambda rc: rc[0])
        rows_sorted = [r for r, _ in ordered]
        assert rows_sorted == list(range(rows_sorted[0], rows_sorted[-1] + 1)), "vertical seed must be contiguous"

    colors = [int(grid[r, c]) for r, c in ordered]
    spin = _sequence_spin(colors)
    assert spin is not None, "seed colors must move by single ring steps in reading order"

    progression = ordered
    assert len(set(colors)) == len(colors), "seed colors must be distinct"
    head = progression[-2]
    tail = progression[-1]
    direction = (tail[0] - head[0], tail[1] - head[1])
    assert direction in {(-1, 0), (1, 0), (0, -1), (0, 1)}
    return progression, colors, int(spin), direction


def _spiral_positions(
    seed: list[tuple[int, int]],
    direction: tuple[int, int],
    spin: int,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    cur = seed[-1]
    segment_index = 0
    out: list[tuple[int, int]] = []
    while True:
        direction = _turn(direction, spin)
        seg_len = len(seed) + (segment_index + 1) // 2
        for _ in range(seg_len):
            nr = cur[0] + direction[0]
            nc = cur[1] + direction[1]
            if not (0 <= nr < h and 0 <= nc < w):
                return out
            cur = (nr, nc)
            out.append(cur)
            if nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1:
                return out
        segment_index += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a central seed line into a color-cycling rectangular spiral until it reaches the grid edge.

    Input structure:
    - The background is black.
    - Near the middle of the grid, there is exactly one horizontal or vertical seed line of 3 to 5 non-black pixels.
    - The seed colors come from the fixed visible ring red -> orange -> yellow -> green -> azure -> blue -> magenta -> red, and adjacent seed pixels are adjacent on that ring.

    Rule:
    - Read the seed line in the only direction whose colors advance by one step around the visible color ring; that determines both the tail pixel and whether the ring is traversed clockwise or counterclockwise.
    - Treat the seed line as the first segment of a rectangular spiral. Starting from the tail pixel, immediately turn with the same clockwise/counterclockwise sense as the ring traversal, then keep adding spiral segments with lengths L, L+1, L+1, L+2, L+2, ... where L is the seed length.
    - Every new spiral pixel takes the next color on the same ring in the same traversal direction. Keep the seed unchanged and stop when the spiral first reaches the grid boundary.

    Key insight:
    - The seed encodes two linked pieces of state at once: its endpoint order reveals the direction around the color ring, and that same direction decides whether the outward spiral turns right or left.
    """
    g = np.asarray(grid, dtype=int)
    seed, colors, spin, direction = _parse_seed(g)
    out = g.copy()

    color_idx = RING_INDEX[colors[-1]]
    for r, c in _spiral_positions(seed, direction, spin, g.shape):
        color_idx = (color_idx + spin) % len(RING)
        out[r, c] = RING[color_idx]
    return out


def _seed_colors(length: int, spin: int) -> list[int]:
    start = random.randrange(len(RING))
    return [RING[(start + spin * i) % len(RING)] for i in range(length)]


def _paint_seed(
    grid: np.ndarray,
    orientation: str,
    start: tuple[int, int],
    colors: list[int],
) -> None:
    r0, c0 = start
    for i, color in enumerate(colors):
        if orientation == "h":
            grid[r0, c0 + i] = color
        else:
            grid[r0 + i, c0] = color


def _build_candidate(
    h: int,
    w: int,
    orientation: str,
    length: int,
    spin: int,
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    colors = _seed_colors(length, spin)

    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0
    jitter = [-2, -1, 0, 1, 2]

    if orientation == "h":
        row = min(max(1, int(round(center_r + random.choice(jitter)))), h - 2)
        base_c = int(round(center_c - (length - 1) / 2.0))
        c0 = min(max(1, base_c + random.choice(jitter)), w - length - 1)
        _paint_seed(grid, orientation, (row, c0), colors)
    else:
        col = min(max(1, int(round(center_c + random.choice(jitter)))), w - 2)
        base_r = int(round(center_r - (length - 1) / 2.0))
        r0 = min(max(1, base_r + random.choice(jitter)), h - length - 1)
        _paint_seed(grid, orientation, (r0, col), colors)

    return grid


def _is_valid_example(inp: np.ndarray, out: np.ndarray) -> bool:
    seed_pixels = int(np.count_nonzero(inp))
    out_pixels = int(np.count_nonzero(out))
    if out_pixels - seed_pixels < 6:
        return False
    if not (
        np.any(out[0, :] != BG)
        or np.any(out[-1, :] != BG)
        or np.any(out[:, 0] != BG)
        or np.any(out[:, -1] != BG)
    ):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        orientation = random.choice(["h", "v"])
        spin = random.choice([-1, 1])
        length = random.randint(3, 5)
        h = random.randint(12, 24)
        w = random.randint(12, 24)
        if orientation == "h" and w < length + 2:
            continue
        if orientation == "v" and h < length + 2:
            continue

        inp = _build_candidate(h, w, orientation, length, spin)
        try:
            out = main(inp)
        except AssertionError:
            continue
        if _is_valid_example(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid spiral-seed ARC input")
