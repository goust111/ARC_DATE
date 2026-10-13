from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _rotate_clockwise(arr: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(arr, -int(quarter_turns) % 4)


def _find_cross(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    assert h == w and h % 2 == 1, "grid must be a square of side 2*M+1"
    mid = h // 2
    divider = int(grid[mid, mid])
    assert np.all(grid[mid, :] == divider), "middle row must be a uniform divider"
    assert np.all(grid[:, mid] == divider), "middle col must be a uniform divider"
    return mid, mid, divider


def _opposite_side(side: str) -> str:
    return {"up": "down", "right": "left", "down": "up", "left": "right"}[side]


def _pointer_to_turns(side: str) -> int:
    return {"up": 0, "right": 1, "down": 2, "left": 3}[side]


def _decode_clock(quad: np.ndarray, bg: int) -> tuple[int, int]:
    non_bg = quad[quad != bg]
    assert non_bg.size > 0, "target quadrant must contain a clock"
    counts = Counter(int(v) for v in non_bg)
    clock_color, _ = counts.most_common(1)[0]

    clock_pts = np.argwhere(quad == clock_color)
    r0, c0 = clock_pts.min(axis=0)
    r1, c1 = clock_pts.max(axis=0)
    assert (int(r1 - r0), int(c1 - c0)) == (2, 2), "clock bbox must be 3x3"
    center = (int(r0 + 1), int(c0 + 1))
    assert int(quad[center]) == clock_color, "clock center must be present"

    pointer_side = None

    marker_pts = np.argwhere((quad != bg) & (quad != clock_color))
    for mr, mc in marker_pts:
        mr = int(mr)
        mc = int(mc)
        if mr == int(r0 - 1) and int(c0) <= mc <= int(c1):
            pointer_side = "down"
            marker_color = int(quad[mr, mc])
            break
        if mr == int(r1 + 1) and int(c0) <= mc <= int(c1):
            pointer_side = "up"
            marker_color = int(quad[mr, mc])
            break
        if mc == int(c0 - 1) and int(r0) <= mr <= int(r1):
            pointer_side = "right"
            marker_color = int(quad[mr, mc])
            break
        if mc == int(c1 + 1) and int(r0) <= mr <= int(r1):
            pointer_side = "left"
            marker_color = int(quad[mr, mc])
            break
    else:
        marker_color = None

    if pointer_side is None:
        side_checks = {
            "up": (int(r0), int(c0 + 1)),
            "right": (int(r0 + 1), int(c1)),
            "down": (int(r1), int(c0 + 1)),
            "left": (int(r0 + 1), int(c0)),
        }
        sides = [
            side
            for side, (rr, cc) in side_checks.items()
            if int(quad[rr, cc]) == clock_color
        ]
        assert len(sides) == 1, "clock direction must be uniquely decodable"
        pointer_side = sides[0]

        extra_colors = [int(v) for v in non_bg if int(v) != clock_color]
        assert extra_colors, "target quadrant must contain a color marker"
        marker_color = extra_colors[0]

    assert marker_color is not None
    return _pointer_to_turns(pointer_side), marker_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode per-quadrant clock markers, rotate the top-left pattern, and recolor it.

    Input structure:
    - Background is black.
    - Grey cross lines split the grid into four MxM quadrants.
    - The top-left quadrant contains the original monochrome sprite.
    - Each other quadrant contains a 3x3 maroon clock marker plus a single-pixel
      color marker on the opposite side of the clock hand.

    Rule:
    - Read the clock hand direction to obtain a clockwise rotation.
    - Rotate the full top-left MxM pattern canvas around the quadrant center.
    - Recolor the rotated sprite with the target quadrant's color marker.
    - Remove the clock and color marker in the output.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    split_r, split_c, divider = _find_cross(g)
    m = split_r

    tl = g[:m, :m].copy()
    template_mask = tl != bg

    out = np.full_like(g, bg)
    out[split_r, :] = divider
    out[:, split_c] = divider
    out[:m, :m] = tl

    quadrants = [
        (slice(0, m), slice(split_c + 1, g.shape[1])),
        (slice(split_r + 1, g.shape[0]), slice(0, m)),
        (slice(split_r + 1, g.shape[0]), slice(split_c + 1, g.shape[1])),
    ]

    for rs, cs in quadrants:
        quad = g[rs, cs]
        turns, fill_color = _decode_clock(quad, bg)
        rotated_mask = _rotate_clockwise(template_mask, turns)
        quad_out = np.full((m, m), bg, dtype=int)
        quad_out[rotated_mask] = fill_color
        out[rs, cs] = quad_out

    return out


BG = 0
DIVIDER = 5
CLOCK = 9
PALETTE = [1, 2, 3, 4, 6, 7, 8]
M_CHOICES = [5, 6, 7, 8, 9]
MAX_ATTEMPTS = 400

_BASE_CLOCK = {
    "corners": [(0, 0), (0, 2), (2, 0), (2, 2)],
    "center": (1, 1),
    "pointer": {
        0: (0, 1),  # up
        1: (1, 2),  # right
        2: (2, 1),  # down
        3: (1, 0),  # left
    },
    "marker": {
        0: (3, 1),   # pointer up -> marker below
        1: (1, -1),  # pointer right -> marker left
        2: (-1, 1),  # pointer down -> marker above
        3: (1, 3),   # pointer left -> marker right
    },
}


def _trim(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _random_connected_shape(rng: random.Random, h: int, w: int, cells: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (rng.randrange(h), rng.randrange(w))
    mask[start] = True

    while int(mask.sum()) < cells:
        frontier = []
        for r, c in np.argwhere(mask):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        nr, nc = frontier[rng.randrange(len(frontier))]
        mask[nr, nc] = True

    return _trim(mask)


def _is_good_sprite(mask: np.ndarray) -> bool:
    if int(mask.sum()) < 3:
        return False
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        return False
    for turns in (1, 2, 3):
        rotated = _rotate_clockwise(mask, turns)
        if rotated.shape == mask.shape and np.array_equal(rotated, mask):
            return False
    return True


def _generate_sprite_canvas(rng: random.Random, m: int) -> np.ndarray:
    max_box = min(m - 2, 5)
    for _ in range(200):
        h = rng.randint(2, max_box)
        w = rng.randint(2, max_box)
        max_cells = min(h * w - 1, 7)
        min_cells = min(max_cells, 4)
        if max_cells < 3:
            continue
        cells = rng.randint(3, max_cells)
        shape = _random_connected_shape(rng, h, w, cells)
        if not _is_good_sprite(shape):
            continue

        sh, sw = shape.shape
        row_choices = list(range(1, m - sh))
        col_choices = list(range(1, m - sw))
        if not row_choices or not col_choices:
            continue

        r0 = rng.choice(row_choices)
        c0 = rng.choice(col_choices)
        canvas = np.zeros((m, m), dtype=bool)
        canvas[r0 : r0 + sh, c0 : c0 + sw] = shape
        return canvas
    raise RuntimeError("failed to generate a non-symmetric sprite")


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = min(h, h - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(w, w - dc)
            dst_r0 = max(0, dr)
            dst_r1 = min(h, h + dr)
            dst_c0 = max(0, dc)
            dst_c1 = min(w, w + dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _clock_cells(top: int, left: int, turns: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    cells = [(top + r, left + c) for r, c in _BASE_CLOCK["corners"]]
    cr, cc = _BASE_CLOCK["center"]
    cells.append((top + cr, left + cc))
    pr, pc = _BASE_CLOCK["pointer"][turns]
    cells.append((top + pr, left + pc))
    mr, mc = _BASE_CLOCK["marker"][turns]
    marker = (top + mr, left + mc)
    return cells, marker


def _find_clock_placement(
    rng: random.Random, occupied: np.ndarray, turns: int
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    m = occupied.shape[0]
    padded = _dilate(occupied)
    for strict in (True, False):
        candidates = []
        for top in range(m - 2):
            for left in range(m - 2):
                clock_cells, marker = _clock_cells(top, left, turns)
                all_cells = clock_cells + [marker]
                if not all(0 <= r < m and 0 <= c < m for r, c in all_cells):
                    continue
                mask_to_check = padded if strict else occupied
                if any(mask_to_check[r, c] for r, c in all_cells):
                    continue
                candidates.append((clock_cells, marker))
        if candidates:
            return candidates[rng.randrange(len(candidates))]
    return None


def _embed_quadrants(quadrants: list[np.ndarray], divider: int) -> np.ndarray:
    m = quadrants[0].shape[0]
    grid = np.full((2 * m + 1, 2 * m + 1), BG, dtype=int)
    grid[m, :] = divider
    grid[:, m] = divider
    grid[:m, :m] = quadrants[0]
    grid[:m, m + 1 :] = quadrants[1]
    grid[m + 1 :, :m] = quadrants[2]
    grid[m + 1 :, m + 1 :] = quadrants[3]
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    m = rng.choice(M_CHOICES)
    sprite_color, *marker_colors = rng.sample(PALETTE, 4)
    turns = [rng.randrange(4) for _ in range(3)]

    template_mask = _generate_sprite_canvas(rng, m)
    tl_quad = np.full((m, m), BG, dtype=int)
    tl_quad[template_mask] = sprite_color

    output_quads = [tl_quad]
    input_quads = [tl_quad.copy()]
    target_turns = list(turns)

    for turns_k, marker_color in zip(target_turns, marker_colors):
        rotated_mask = _rotate_clockwise(template_mask, turns_k)
        out_quad = np.full((m, m), BG, dtype=int)
        out_quad[rotated_mask] = marker_color
        placement = _find_clock_placement(rng, rotated_mask, turns_k)
        if placement is None:
            return None
        clock_cells, marker = placement

        in_quad = np.full((m, m), BG, dtype=int)
        for r, c in clock_cells:
            in_quad[r, c] = CLOCK
        in_quad[marker] = marker_color

        output_quads.append(out_quad)
        input_quads.append(in_quad)

    inp = _embed_quadrants(input_quads, DIVIDER)
    expected = _embed_quadrants(output_quads, DIVIDER)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
