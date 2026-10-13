from collections import defaultdict

import numpy as np

from arc_utils import colors_present, get_bg, in_bounds
import random


def _find_anchor(mask: np.ndarray) -> tuple[int, int, int, int]:
    h, w = mask.shape
    anchors = []
    for r in range(h - 2):
        for c in range(w - 2):
            if mask[r : r + 3, c : c + 3].all():
                anchors.append((r, c, r + 2, c + 2))
    assert len(anchors) == 1
    return anchors[0]


def _ray_terminal(
    coord: tuple[int, int],
    anchor: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    r, c = coord
    r0, c0, r1, c1 = anchor
    dr = 1 if r < r0 else -1 if r > r1 else 0
    dc = 1 if c < c0 else -1 if c > c1 else 0
    assert (dr, dc) != (0, 0)

    cr, cc = r, c
    for _ in range(max(shape) * 2):
        nr, nc = cr + dr, cc + dc
        if r0 <= nr <= r1 and c0 <= nc <= c1:
            return (cr, cc), (dr, dc)
        assert in_bounds(shape, nr, nc)
        cr, cc = nr, nc
    raise AssertionError("ray does not reach anchor")


def main(grid: np.ndarray) -> np.ndarray:
    """Compress each colored ray pixel inward until it touches that color's 3x3 anchor block.
    Input structure
    - A uniform background grid with non-background colors.
    - Each foreground color contains one solid 3x3 anchor block; all other pixels of that color lie on straight 8-neighbor rays aimed at that block.
    Rule
    - Keep anchor blocks fixed, erase loose foreground pixels, and for each ray pack its pixels into consecutive cells adjacent to the anchor along the same ray.
    Key insight
    - The scattered pixels are displaced pixels on visible row, column, or diagonal sightlines back to their same-colored 3x3 block.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = np.full_like(g, bg)
    occupied: dict[tuple[int, int], int] = {}

    fg_colors = colors_present(g, exclude={bg})
    assert fg_colors

    for color in fg_colors:
        mask = g == color
        anchor = _find_anchor(mask)
        r0, c0, r1, c1 = anchor

        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                assert (r, c) not in occupied
                occupied[(r, c)] = color
                out[r, c] = color

        rays: dict[
            tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int, int]]
        ] = defaultdict(list)
        for raw_r, raw_c in np.argwhere(mask):
            r, c = int(raw_r), int(raw_c)
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            terminal, step = _ray_terminal((r, c), anchor, g.shape)
            dist = max(abs(r - terminal[0]), abs(c - terminal[1]))
            rays[(terminal, step)].append((dist, r, c))

        for (terminal, (dr, dc)), cells in rays.items():
            distances = [dist for dist, _, _ in cells]
            assert len(distances) == len(set(distances))
            tr, tc = terminal
            for i, _ in enumerate(sorted(cells)):
                dest = (tr - i * dr, tc - i * dc)
                assert in_bounds(g.shape, *dest)
                assert dest not in occupied
                occupied[dest] = color
                out[dest] = color

        assert int((out == color).sum()) == int(mask.sum())

    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, and retries.
_GEN_BG = 6
_GEN_SHAPE = (9, 9)
_GEN_FOREGROUND_COLORS = [0, 1, 2, 3, 4, 5, 7, 8, 9]
_GEN_MAX_ATTEMPTS = 300

# Fixed anchor layout.
_GEN_TOP_LEFT_ANCHOR = (0, 0, 2, 2)
_GEN_BOTTOM_RIGHT_ANCHOR = (6, 6, 8, 8)
_GEN_ANCHORS = (_GEN_TOP_LEFT_ANCHOR, _GEN_BOTTOM_RIGHT_ANCHOR)

# Ray sampling.
_GEN_RAYS_MIN = 3
_GEN_RAYS_MAX = 6
_GEN_MAX_LOOSE_DISTANCE = 5


def _gen_anchor_cells(anchor):
    r0, c0, r1, c1 = anchor
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _gen_terminals(anchor):
    r0, c0, r1, c1 = anchor
    if (r0, c0) == (0, 0):
        return (
            [(r, c1 + 1, 0, 1) for r in range(r0, r1 + 1)]
            + [(r1 + 1, c, 1, 0) for c in range(c0, c1 + 1)]
            + [(r1 + 1, c1 + 1, 1, 1)]
        )
    return (
        [(r, c0 - 1, 0, -1) for r in range(r0, r1 + 1)]
        + [(r0 - 1, c, -1, 0) for c in range(c0, c1 + 1)]
        + [(r0 - 1, c0 - 1, -1, -1)]
    )


def _gen_seed_anchors(inp, out, colors):
    """Paint both fixed anchors and return occupied input cells."""
    occupied_in = set()

    for color, anchor in zip(colors, _GEN_ANCHORS):
        for r, c in _gen_anchor_cells(anchor):
            inp[r, c] = color
            out[r, c] = color
            occupied_in.add((r, c))

    return occupied_in


def _gen_place_loose_pixels(inp, out, color, anchor, occupied_in):
    """Sample terminals for one anchor and place their loose input pixels."""
    terminals = _gen_terminals(anchor)
    chosen = random.sample(terminals, random.randint(_GEN_RAYS_MIN, _GEN_RAYS_MAX))
    for tr, tc, dr, dc in chosen:
        out[tr, tc] = color
        max_dist = 0
        rr, cc = tr + dr, tc + dc
        while 0 <= rr < _GEN_SHAPE[0] and 0 <= cc < _GEN_SHAPE[1]:
            max_dist += 1
            rr += dr
            cc += dc
        dist = random.randint(1, min(_GEN_MAX_LOOSE_DISTANCE, max_dist))
        loose = (tr + dr * dist, tc + dc * dist)
        if loose in occupied_in:
            return False
        occupied_in.add(loose)
        inp[loose] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    colors = random.sample(_GEN_FOREGROUND_COLORS, 2)
    random.shuffle(colors)

    inp = np.full(_GEN_SHAPE, _GEN_BG, dtype=int)
    out = np.full(_GEN_SHAPE, _GEN_BG, dtype=int)
    occupied_in = _gen_seed_anchors(inp, out, colors)

    for color, anchor in zip(colors, _GEN_ANCHORS):
        if not _gen_place_loose_pixels(inp, out, color, anchor, occupied_in):
            return None

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid f28a3cbb input")
