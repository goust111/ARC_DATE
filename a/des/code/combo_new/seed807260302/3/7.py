from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
STAR_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MIN_SIZE = 12
MAX_SIZE = 20
GEN_RETRIES = 800


Coord = tuple[int, int]


def _legend_pairs(g: np.ndarray) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    seen: set[frozenset[int]] = set()
    row = g[-1]
    c = 0
    while c < g.shape[1] - 1:
        a = int(row[c])
        b = int(row[c + 1])
        if a != BLACK and b != BLACK and a != b:
            key = frozenset((a, b))
            if key not in seen:
                pairs.append((a, b))
                seen.add(key)
            c += 2
        else:
            c += 1
    return pairs


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _pair_stars(g: np.ndarray, pairs: list[tuple[int, int]]) -> list[tuple[Coord, Coord]]:
    matches: list[tuple[Coord, Coord]] = []
    used: set[Coord] = set()

    for a, b in pairs:
        pts: list[Coord] = []
        for r, c in np.argwhere((g[:-1] == a) | (g[:-1] == b)):
            pos = (int(r), int(c))
            if pos not in used:
                pts.append(pos)
        pts.sort()

        for p in pts:
            if p in used:
                continue
            p_col = int(g[p])
            other = b if p_col == a else a
            candidates = [q for q in pts if q not in used and int(g[q]) == other]
            if not candidates:
                continue
            q = min(candidates, key=lambda x: (_manhattan(p, x), x[0], x[1]))
            used.add(p)
            used.add(q)
            matches.append((p, q))

    return matches


def _l_path(start: Coord, end: Coord) -> list[Coord]:
    r0, c0 = start
    r1, c1 = end
    cells: list[Coord] = []

    dc = 1 if c1 >= c0 else -1
    for c in range(c0, c1 + dc, dc):
        cells.append((r0, c))

    if r1 != r0:
        dr = 1 if r1 > r0 else -1
        for r in range(r0 + dr, r1 + dr, dr):
            cells.append((r, c1))

    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode bottom-edge color-pair legends, greedily pair matching stars, and draw grey L-shaped connectors.

    Input structure:
    - The background is black. Interior cells contain isolated single-pixel colored stars.
    - The bottom row contains one or more legend blocks: two horizontally adjacent, different-colored pixels, with black gaps between blocks.
    - A legend block names the two star colors that should be connected; colors not named by a block remain unchanged.

    Rule:
    - Read the bottom row from left to right to recover the unordered color pairs.
    - For each legend pair, scan all still-unpaired stars of those two colors from top to bottom, then left to right.
    - Pair each available star with the nearest available star of the other legend color by Manhattan distance; ties choose the smaller row, then the smaller column.
    - Draw each connection in grey as a one-pixel-wide orthogonal path that first moves horizontally to the target column, then vertically to the target row; clear the legend row to black and keep all stars in their original colors.

    Key insight: The edge legend supplies the color relationships, and the actual endpoints are resolved by visible nearest-neighbor matching in scan order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h > 0 and w > 0

    out = g.copy()
    out[-1, :] = BLACK

    pairs = _legend_pairs(g)
    matches = _pair_stars(g, pairs)
    stars = {(int(r), int(c)): int(g[int(r), int(c)]) for r, c in np.argwhere(g[:-1] != BLACK)}

    for start, end in matches:
        for r, c in _l_path(start, end):
            if (r, c) not in stars:
                out[r, c] = GREY

    for (r, c), color in stars.items():
        out[r, c] = color
    return out


def _paint_legend(g: np.ndarray, pairs: list[tuple[int, int]]) -> bool:
    w = g.shape[1]
    gaps = [random.randint(1, 3) for _ in range(len(pairs) - 1)]
    needed = 2 * len(pairs) + sum(gaps)
    if needed > w:
        return False

    c = random.randint(0, w - needed)
    for i, pair in enumerate(pairs):
        a, b = pair
        if random.random() < 0.5:
            a, b = b, a
        g[-1, c] = a
        g[-1, c + 1] = b
        c += 2
        if i < len(gaps):
            c += gaps[i]
    return True


def _cell_clear(pos: Coord, occupied: set[Coord], h: int, w: int) -> bool:
    r, c = pos
    if not (1 <= r <= h - 3 and 1 <= c <= w - 2):
        return False
    for rr, cc in occupied:
        if max(abs(rr - r), abs(cc - c)) <= 1:
            return False
    return True


def _place_cells(
    g: np.ndarray,
    colored_cells: list[tuple[Coord, int]],
    occupied: set[Coord],
) -> bool:
    h, w = g.shape
    coords = [pos for pos, _color in colored_cells]
    if len(set(coords)) != len(coords):
        return False
    for pos in coords:
        if not _cell_clear(pos, occupied, h, w):
            return False
    for pos, color in colored_cells:
        g[pos] = color
        occupied.add(pos)
    return True


def _place_random_duo(
    g: np.ndarray,
    color_a: int,
    color_b: int,
    occupied: set[Coord],
) -> tuple[Coord, Coord] | None:
    h, w = g.shape
    for _ in range(250):
        mode = random.choice(["horizontal", "vertical", "corner", "corner"])
        if mode == "horizontal":
            dr = 0
            dc = random.choice([-4, -3, -2, 2, 3, 4])
        elif mode == "vertical":
            dr = random.choice([-4, -3, -2, 2, 3, 4])
            dc = 0
        else:
            dr = random.choice([-4, -3, -2, 2, 3, 4])
            dc = random.choice([-4, -3, -2, 2, 3, 4])

        r = random.randint(1, h - 3)
        c = random.randint(1, w - 2)
        p = (r, c)
        q = (r + dr, c + dc)
        if random.random() < 0.5:
            first = (p, color_a)
            second = (q, color_b)
        else:
            first = (p, color_b)
            second = (q, color_a)
        if _place_cells(g, [first, second], occupied):
            return first[0], second[0]
    return None


def _place_tie_duos(
    g: np.ndarray,
    color_a: int,
    color_b: int,
    occupied: set[Coord],
) -> list[tuple[Coord, Coord]] | None:
    h, w = g.shape
    for _ in range(120):
        d = random.choice([3, 4])
        r = random.randint(1, h - d - 3)
        c = random.randint(1, w - d - 2)
        a1 = (r, c)
        b1 = (r, c + d)
        b2 = (r + d, c)
        a2 = (r + d, c + 2)
        cells = [(a1, color_a), (b1, color_b), (b2, color_b), (a2, color_a)]
        if _place_cells(g, cells, occupied):
            return [(a1, b1), (b2, a2)]
    return None


def _place_distractor_stars(g: np.ndarray, color: int, count: int, occupied: set[Coord]) -> bool:
    h, w = g.shape
    for _ in range(count):
        placed = False
        for _attempt in range(200):
            pos = (random.randint(1, h - 3), random.randint(1, w - 2))
            if _place_cells(g, [(pos, color)], occupied):
                placed = True
                break
        if not placed:
            return False
    return True


def _path_is_clear(g: np.ndarray, matches: list[tuple[Coord, Coord]]) -> bool:
    star_set = {tuple(map(int, p)) for p in np.argwhere(g[:-1] != BLACK)}
    used_path: set[Coord] = set()
    for start, end in matches:
        endpoints = {start, end}
        for pos in _l_path(start, end):
            if pos in endpoints:
                continue
            if pos in star_set or pos in used_path:
                return False
            used_path.add(pos)
    return True


def _validate_generated(g: np.ndarray, intended: list[tuple[Coord, Coord]], expected_pair_count: int) -> bool:
    pairs = _legend_pairs(g)
    if len(pairs) != expected_pair_count:
        return False

    matches = _pair_stars(g, pairs)
    intended_keys = {frozenset((a, b)) for a, b in intended}
    match_keys = {frozenset((a, b)) for a, b in matches}
    if match_keys != intended_keys:
        return False
    if len(matches) != len(intended):
        return False
    if not _path_is_clear(g, matches):
        return False

    out = main(g)
    if out.shape != g.shape or max(out.shape) > 30:
        return False
    if not np.any(out == GREY):
        return False
    if np.any(out[-1] != BLACK):
        return False

    for r, c in np.argwhere(g[:-1] != BLACK):
        if int(out[int(r), int(c)]) != int(g[int(r), int(c)]):
            return False
    return True


def _choose_scene() -> tuple[list[int], list[tuple[int, int]]]:
    color_count = random.choice([3, 4])
    colors = random.sample(STAR_COLORS, color_count)

    if color_count == 4 and random.random() < 0.6:
        shuffled = colors[:]
        random.shuffle(shuffled)
        pairs = [(shuffled[0], shuffled[1]), (shuffled[2], shuffled[3])]
    else:
        a, b = random.sample(colors, 2)
        pairs = [(a, b)]
    return colors, pairs


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_SIZE, MAX_SIZE)
    w = random.randint(MIN_SIZE, MAX_SIZE)
    g = np.full((h, w), BLACK, dtype=int)

    colors, pairs = _choose_scene()
    random.shuffle(pairs)
    if not _paint_legend(g, pairs):
        return None

    occupied: set[Coord] = set()
    intended: list[tuple[Coord, Coord]] = []
    paired_colors = {color for pair in pairs for color in pair}

    for idx, (a, b) in enumerate(pairs):
        max_count = 4 if len(pairs) == 1 or min(h, w) >= 16 else 3
        count = random.randint(2, max_count)
        placed_count = 0

        use_tie = idx == 0 and len(pairs) == 1 and count >= 2 and random.random() < 0.3
        if use_tie:
            tie_pairs = _place_tie_duos(g, a, b, occupied)
            if tie_pairs is not None:
                intended.extend(tie_pairs)
                placed_count = 2

        while placed_count < count:
            duo = _place_random_duo(g, a, b, occupied)
            if duo is None:
                return None
            intended.append(duo)
            placed_count += 1

    for color in colors:
        if color in paired_colors:
            continue
        if not _place_distractor_stars(g, color, random.randint(2, 4), occupied):
            return None

    if not _validate_generated(g, intended, len(pairs)):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
