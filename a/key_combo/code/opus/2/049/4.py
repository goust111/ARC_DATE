from __future__ import annotations

import random
import numpy as np

BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

MARKER_COLORS = [RED, GREEN, YELLOW, ORANGE]
MARKER_TO_ROT = {
    RED: 0,
    GREEN: 3,   # clockwise 90° -> rot90(k=3)
    YELLOW: 2,  # 180°
    ORANGE: 1,  # counter-clockwise 90° (rot90 k=1)
}


def _rotate_square_keep_shape(quad: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(quad, k)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Hollow solid rectangles to reveal markers, decode rotation angles, then rotate each quadrant.

    Input structure:
    - A (2H+1)x(2W+1) grid with four blue solid rectangles arranged in a 2x2 layout separated by one blank row/column.
    - Each rectangle contains one marker color (red/green/yellow/orange) at its geometric center.
    - Marker colors are unique across rectangles.

    Rule:
    - Identify each marker and its enclosing rectangle's quadrant.
    - Rotate each quadrant by the angle mapped from its marker color (2->0°,3->CW90,4->180,7->CCW90).
    - Rotation is done on the quadrant region itself (including separator half-lines around it).
    - Erase all markers after rotation.

    Key insight:
    - Marker colors are instructions, so the rectangles must first be treated as a control encoding stage before transformation.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    assert H == W and H >= 7 and H % 2 == 1
    mid_r, mid_c = H // 2, W // 2

    out = g.copy()
    marker_positions: dict[int, tuple[int, int]] = {}
    for color in MARKER_COLORS:
        locs = np.argwhere(g == color)
        if locs.size != 2:
            raise AssertionError("each marker color must appear exactly once")
        marker_positions[color] = (int(locs[0, 0]), int(locs[0, 1]))

    # Quadrant slices include separator row/column halves as defined in the task.
    quadrants = {
        "tl": (slice(0, mid_r + 1), slice(0, mid_c + 1)),
        "tr": (slice(0, mid_r + 1), slice(mid_c, W)),
        "bl": (slice(mid_r, H), slice(0, mid_c + 1)),
        "br": (slice(mid_r, H), slice(mid_c, W)),
    }

    marker_to_quadrant: dict[int, str] = {}
    for color, (r, c) in marker_positions.items():
        if r <= mid_r and c <= mid_c:
            q = "tl"
        elif r <= mid_r and c > mid_c:
            q = "tr"
        elif r > mid_r and c <= mid_c:
            q = "bl"
        else:
            q = "br"
        marker_to_quadrant[color] = q

    for color, q in marker_to_quadrant.items():
        rs, cs = quadrants[q]
        quad = out[rs, cs]
        assert quad.shape[0] == quad.shape[1]
        k = MARKER_TO_ROT[color]
        out[rs, cs] = _rotate_square_keep_shape(quad, k)

    for color in MARKER_COLORS:
        out[out == color] = BG
    return out


def generate_input() -> np.ndarray:
    for _ in range(800):
        n = random.randint(3, 7)
        H = 2 * n + 1
        g = np.full((H, H), BG, dtype=int)
        Hs = n
        Ws = n
        mid = n

        rect_slices = [
            (slice(0, Hs), slice(0, Ws)),
            (slice(0, Hs), slice(Hs + 1, 2 * Hs + 1)),
            (slice(Hs + 1, 2 * Hs + 1), slice(0, Ws)),
            (slice(Hs + 1, 2 * Hs + 1), slice(Hs + 1, 2 * Hs + 1)),
        ]
        marker_colors = MARKER_COLORS.copy()
        random.shuffle(marker_colors)

        for idx, sl in enumerate(rect_slices):
            rs, cs = sl
            for r in range(rs.start, rs.stop):
                for c in range(cs.start, cs.stop):
                    g[r, c] = BLUE

            mr = (rs.start + rs.stop) // 2
            mc = (cs.start + cs.stop) // 2
            g[mr, mc] = marker_colors[idx]

        try:
            _ = main(g)
            return g
        except Exception:
            continue

    raise RuntimeError("failed to generate valid puzzle 4 input")

