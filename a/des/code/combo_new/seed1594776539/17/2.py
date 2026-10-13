from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
RING_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
OUTPUT_GAP = 2
MAX_GENERATION_ATTEMPTS = 300


def _border_values(square: np.ndarray) -> np.ndarray:
    side = int(square.shape[0])
    if side == 1:
        return square.reshape(-1)
    if side == 2:
        return square.reshape(-1)
    return np.concatenate(
        [
            square[0, :],
            square[-1, :],
            square[1:-1, 0],
            square[1:-1, -1],
        ]
    )


def _border_coords(side: int) -> list[tuple[int, int]]:
    if side == 1:
        return [(0, 0)]
    if side == 2:
        return [(0, 0), (0, 1), (1, 0), (1, 1)]

    coords: list[tuple[int, int]] = []
    for c in range(side):
        coords.append((0, c))
    for c in range(side):
        coords.append((side - 1, c))
    for r in range(1, side - 1):
        coords.append((r, 0))
    for r in range(1, side - 1):
        coords.append((r, side - 1))
    return coords


def _paint_full_ring(canvas: np.ndarray, top: int, left: int, side: int, color: int) -> None:
    canvas[top, left : left + side] = int(color)
    canvas[top + side - 1, left : left + side] = int(color)
    canvas[top : top + side, left] = int(color)
    canvas[top : top + side, left + side - 1] = int(color)


def _extract_rings(grid: np.ndarray) -> list[dict[str, int]]:
    coords = np.argwhere(grid != BG)
    assert coords.size > 0, "expected at least one ring"

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    assert height == width, "nested structure must have a square outer bbox"

    crop = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    side = int(crop.shape[0])

    rings: list[dict[str, int]] = []
    for offset in range((side + 1) // 2):
        inner_side = int(side - 2 * offset)
        if inner_side <= 0:
            break

        square = crop[offset : offset + inner_side, offset : offset + inner_side]
        border = _border_values(square)
        if not bool(np.any(border != BG)):
            continue

        visible_colors = {int(v) for v in np.unique(border) if int(v) != BG}
        ring_colors = visible_colors - {YELLOW}
        assert len(ring_colors) == 1, f"ambiguous ring colors on offset {offset}: {visible_colors}"

        rings.append(
            {
                "size": int(inner_side),
                "color": int(next(iter(ring_colors))),
                "marks": int(np.count_nonzero(border == YELLOW)),
            }
        )

    assert rings, "failed to extract any ring"
    return rings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split concentric square frames into separate rings ordered by their yellow marker counts.

    Input structure:
    - Background is black.
    - One nested target made of concentric one-pixel-thick square rings sits on the grid, with one black row/column gap between neighboring rings.
    - Each ring has its own non-yellow color, and some border cells on that ring are replaced by yellow marker pixels.

    Rule:
    - Detect every ring from the shared outer square bbox and count how many yellow pixels lie on each ring.
    - Sort the rings from fewest yellow markers to most yellow markers.
    - Redraw the sorted rings as separate complete square frames, restoring every yellow marker cell to the ring's own color; place the frames left-to-right with two black columns between them and top-align them in an output canvas whose height equals the largest ring side length.

    Key insight:
    - The yellow pixels are not part of the final drawing; they are per-ring count signals that determine the horizontal order after the nested structure is decomposed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rings = sorted(_extract_rings(g), key=lambda item: item["marks"])
    out_h = max(int(item["size"]) for item in rings)
    out_w = sum(int(item["size"]) for item in rings) + OUTPUT_GAP * (len(rings) - 1)
    out = np.full((out_h, out_w), BG, dtype=int)

    cursor = 0
    for idx, ring in enumerate(rings):
        side = int(ring["size"])
        _paint_full_ring(out, 0, cursor, side, int(ring["color"]))
        cursor += side
        if idx + 1 < len(rings):
            cursor += OUTPUT_GAP

    return out


def _sample_geometry() -> tuple[int, list[int]]:
    ring_count = random.choice((2, 3))
    if ring_count == 2:
        outer_side = random.randint(7, 16)
    else:
        outer_side = random.randint(11, 12)
    return ring_count, [int(outer_side - 4 * idx) for idx in range(ring_count)]


def _draw_marked_ring(
    canvas: np.ndarray,
    top: int,
    left: int,
    side: int,
    color: int,
    mark_count: int,
) -> None:
    _paint_full_ring(canvas, top, left, side, color)
    border = _border_coords(side)
    for rr, cc in random.sample(border, mark_count):
        canvas[top + rr, left + cc] = YELLOW


def _gen_attempt() -> np.ndarray | None:
    ring_count, sizes = _sample_geometry()
    outer_side = int(sizes[0])
    output_width = sum(sizes) + OUTPUT_GAP * (ring_count - 1)
    assert outer_side <= 30 and output_width <= 30

    height = random.randint(outer_side, 30)
    width = random.randint(outer_side, 30)
    top = random.randint(0, height - outer_side)
    left = random.randint(0, width - outer_side)

    ring_colors = random.sample(RING_COLORS, ring_count)
    mark_counts = random.sample((1, 2, 3, 4, 5), ring_count)

    grid = np.full((height, width), BG, dtype=int)
    for idx, side in enumerate(sizes):
        offset = 2 * idx
        _draw_marked_ring(
            grid,
            top + offset,
            left + offset,
            int(side),
            int(ring_colors[idx]),
            int(mark_counts[idx]),
        )

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape[0] > 30 or out.shape[1] > 30:
        return None
    if int(out.shape[0]) != max(sizes):
        return None
    if int(out.shape[1]) != output_width:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-ring input")
