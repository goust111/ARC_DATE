from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


# A same-color singleton pair has no visible intrinsic ordering, so the solver
# uses row-major order to make "first" and "second" deterministic.
ARROW_OFFSETS = {
    (-1, 0): [(-1, -1), (-1, 1)],
    (1, 0): [(1, -1), (1, 1)],
    (0, -1): [(-1, -1), (1, -1)],
    (0, 1): [(-1, 1), (1, 1)],
    (-1, -1): [(-1, 0), (0, -1)],
    (-1, 1): [(-1, 0), (0, 1)],
    (1, -1): [(1, 0), (0, -1)],
    (1, 1): [(1, 0), (0, 1)],
}

GENERATABLE_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
PERPENDICULAR_STEPS = {
    (0, 1): (1, 0),
    (1, 0): (0, 1),
    (1, 1): (1, -1),
    (1, -1): (1, 1),
}

GRID_MIN = 20
GRID_MAX = 26
MAX_GENERATION_ATTEMPTS = 400


def _sorted_pair(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    pts = sorted((int(r), int(c)) for r, c in coords)
    assert len(pts) == 2 and pts[0] != pts[1], "each marker color must appear exactly twice"
    return pts[0], pts[1]


def _sign(x: int) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _primary_direction(first: tuple[int, int], second: tuple[int, int]) -> tuple[int, int]:
    dr = int(second[0] - first[0])
    dc = int(second[1] - first[1])
    assert dr != 0 or dc != 0, "a pair needs two distinct cells"
    sr = _sign(dr)
    sc = _sign(dc)
    if abs(dr) > abs(dc):
        return (sr, 0)
    if abs(dc) > abs(dr):
        return (0, sc)
    return (sr, sc)


def _round_half_up(total: int, n: int) -> int:
    return int(np.floor(total / n + 0.5))


def _draw_arrow(
    out: np.ndarray,
    center: tuple[int, int],
    direction: tuple[int, int],
    color: int,
) -> None:
    h, w = out.shape
    cr, cc = center
    out[cr, cc] = color
    for dr, dc in ARROW_OFFSETS[direction]:
        rr = int(cr + dr)
        cc2 = int(cc + dc)
        if 0 <= rr < h and 0 <= cc2 < w:
            out[rr, cc2] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group same-direction marker pairs and replace them with 3-pixel arrows.

    Input structure:
    - One dominant background color.
    - Several singleton marker colors; every non-background color appears exactly twice.

    Rule:
    - For each color pair, sort the two cells in row-major order to make the pair
      direction deterministic.
    - Compute the primary direction from the first cell to the second cell:
      vertical if |dr| > |dc|, horizontal if |dc| > |dr|, diagonal on ties.
    - Group all pairs that share the same primary direction.
    - For each direction group, average all marker coordinates in the group and
      draw a 3-pixel arrow at that center using the color of the lexicographically
      first pair in the group.
    - Erase all original singleton markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    counts = Counter(map(int, g.ravel()))
    bg = int(counts.most_common(1)[0][0])

    groups: dict[tuple[int, int], list[tuple[int, tuple[int, int], tuple[int, int]]]] = defaultdict(list)
    for color, count in counts.items():
        color = int(color)
        if color == bg:
            continue
        assert count == 2, "each foreground color must appear exactly twice"
        first, second = _sorted_pair(np.argwhere(g == color))
        direction = _primary_direction(first, second)
        groups[direction].append((color, first, second))

    out = np.full_like(g, bg)
    for direction, pairs in groups.items():
        total_r = sum(first[0] + second[0] for _color, first, second in pairs)
        total_c = sum(first[1] + second[1] for _color, first, second in pairs)
        n_points = 2 * len(pairs)
        center = (_round_half_up(total_r, n_points), _round_half_up(total_c, n_points))
        color, _first, _second = min(pairs, key=lambda item: (item[1][0], item[1][1], item[2][0], item[2][1]))
        _draw_arrow(out, center, direction, int(color))

    return out


def _partition_blocks(h: int, w: int) -> list[tuple[int, int, int, int]]:
    r_mid = h // 2
    c_mid = w // 2
    return [
        (0, r_mid - 1, 0, c_mid - 1),
        (0, r_mid - 1, c_mid, w - 1),
        (r_mid, h - 1, 0, c_mid - 1),
        (r_mid, h - 1, c_mid, w - 1),
    ]


def _pair_offsets(n_pairs: int) -> list[int]:
    if n_pairs == 1:
        return [0]
    if n_pairs == 2:
        return [-1, 1]
    if n_pairs == 3:
        return [-1, 0, 1]
    raise ValueError(f"unsupported pair count: {n_pairs}")


def _sample_group_pair_counts(rng: np.random.Generator, n_groups: int) -> list[int]:
    counts = [1] * n_groups
    extra_capacity = min(8 - n_groups, 2 * n_groups)
    extra_pairs = int(rng.integers(1, extra_capacity + 1))
    while extra_pairs > 0:
        idx = int(rng.integers(0, n_groups))
        if counts[idx] >= 3:
            continue
        counts[idx] += 1
        extra_pairs -= 1
    return counts


def _pair_cells(
    center: tuple[int, int],
    direction: tuple[int, int],
    span: int,
    offset_index: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    cr, cc = center
    dr = int(direction[0] * span)
    dc = int(direction[1] * span)
    pr, pc = PERPENDICULAR_STEPS[direction]
    mr = int(cr + offset_index * pr)
    mc = int(cc + offset_index * pc)
    return (mr - dr, mc - dc), (mr + dr, mc + dc)


def _center_range(block: tuple[int, int, int, int], span: int) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, r1, c0, c1 = block
    margin = span + 1
    return (r0 + margin, r1 - margin), (c0 + margin, c1 - margin)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    bg = int(rng.integers(0, 10))
    n_groups = int(rng.integers(2, 5))

    blocks = _partition_blocks(h, w)
    chosen_blocks = list(rng.permutation(len(blocks))[:n_groups])
    chosen_directions = [GENERATABLE_DIRECTIONS[i] for i in rng.permutation(len(GENERATABLE_DIRECTIONS))[:n_groups]]
    pair_counts = _sample_group_pair_counts(rng, n_groups)
    total_pairs = sum(pair_counts)

    usable_colors = [color for color in range(10) if color != bg]
    if total_pairs > len(usable_colors):
        return None
    pair_colors = list(map(int, rng.choice(usable_colors, size=total_pairs, replace=False)))

    grid = np.full((h, w), bg, dtype=int)
    occupied: set[tuple[int, int]] = set()

    color_idx = 0
    for block_idx, direction, n_pairs in zip(chosen_blocks, chosen_directions, pair_counts):
        block = blocks[int(block_idx)]
        span = int(rng.integers(2, 4))
        (r_lo, r_hi), (c_lo, c_hi) = _center_range(block, span)
        if r_lo > r_hi or c_lo > c_hi:
            return None
        center = (int(rng.integers(r_lo, r_hi + 1)), int(rng.integers(c_lo, c_hi + 1)))
        offsets = _pair_offsets(n_pairs)

        cells_for_group: list[tuple[int, int]] = []
        pair_specs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
        for offset_index in offsets:
            a, b = _pair_cells(center, direction, span, offset_index)
            for cell in (a, b):
                r, c = cell
                if not (0 <= r < h and 0 <= c < w):
                    return None
                if cell in occupied:
                    return None
                cells_for_group.append(cell)
            pair_specs.append((pair_colors[color_idx], a, b))
            color_idx += 1

        for color, a, b in pair_specs:
            grid[a] = int(color)
            grid[b] = int(color)
            occupied.add(a)
            occupied.add(b)

        center_r, center_c = center
        arrow_box = {
            (center_r + dr, center_c + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if 0 <= center_r + dr < h and 0 <= center_c + dc < w
        }
        if occupied & arrow_box:
            return None

    if np.count_nonzero(grid != bg) != 2 * total_pairs:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.count_nonzero(out != bg) != 3 * n_groups:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
