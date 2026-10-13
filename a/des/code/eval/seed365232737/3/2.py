from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# All generator sprites fit in a 3x3 halo around their chosen visual center.
SPRITE_SHAPES = (
    ((0, -1), (0, 0), (0, 1)),
    ((-1, 0), (0, 0), (1, 0)),
    ((0, 0), (1, 0), (0, 1)),
    ((0, 0), (-1, 0), (0, 1)),
    ((0, 0), (1, 0), (0, -1)),
    ((0, 0), (-1, 0), (0, -1)),
    ((-1, 0), (0, -1), (0, 0), (0, 1)),
    ((1, 0), (0, -1), (0, 0), (0, 1)),
    ((-1, 0), (0, 0), (1, 0), (0, 1)),
    ((-1, 0), (0, 0), (1, 0), (0, -1)),
)

GEN_HEIGHT_RANGE = (18, 26)
GEN_WIDTH_RANGE = (18, 28)
GEN_PAIR_COUNT_RANGE = (3, 5)
GEN_NON_RED_POOL = (1, 3, 5, 6, 8, 9)
GEN_FORCED_PAIR_TRIES = 200
GEN_EXTRA_PAIR_TRIES = 400
GEN_SHAPE_TRIES = 80
GEN_ATTEMPTS = 500
GEN_MIN_MARGIN = 2
GEN_MIN_HORIZONTAL = 4
GEN_MIN_VERTICAL = 3


Coord = tuple[int, int]
PairSpec = tuple[int, Coord, Coord]


def _connected_components(mask: np.ndarray) -> list[list[Coord]]:
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[list[Coord]] = []
    height, width = mask.shape

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue: deque[Coord] = deque([(r, c)])
            comp: list[Coord] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for dr, dc in CARDINALS:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _visual_center(cells: list[Coord]) -> Coord:
    # Use the Manhattan medoid of the connected blob as its visual center.
    best_score: tuple[int, int, int] | None = None
    for r, c in cells:
        score = sum(abs(r - rr) + abs(c - cc) for rr, cc in cells)
        candidate = (score, r, c)
        if best_score is None or candidate < best_score:
            best_score = candidate
    assert best_score is not None
    return best_score[1], best_score[2]


def _normalize_pair(color: int, a: Coord, b: Coord) -> PairSpec:
    if (a[1], a[0]) <= (b[1], b[0]):
        return int(color), a, b
    return int(color), b, a


def _path_cells(left: Coord, right: Coord) -> set[Coord]:
    r0, c0 = left
    r1, c1 = right
    h_step = 1 if c1 >= c0 else -1
    v_step = 1 if r1 >= r0 else -1
    cells = {(r0, c) for c in range(c0, c1 + h_step, h_step)}
    cells.update((r, c1) for r in range(r0, r1 + v_step, v_step))
    return cells


def _extract_pairs(grid: np.ndarray) -> list[PairSpec]:
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if int(color) != BLACK]
    assert 3 <= len(colors) <= 5, "expected 3-5 non-background colors"

    pairs: list[PairSpec] = []
    for color in colors:
        comps = _connected_components(g == color)
        assert len(comps) == 2, "each non-background color must form exactly two sprites"
        for comp in comps:
            assert 2 <= len(comp) <= 4, "each sprite must contain 2-4 pixels"
        centers = [_visual_center(comp) for comp in comps]
        pairs.append(_normalize_pair(color, centers[0], centers[1]))

    pairs.sort()
    return pairs


def _paint_paths(grid: np.ndarray, pairs: list[PairSpec]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    claims: dict[Coord, set[int]] = {}

    for color, left, right in pairs:
        for cell in _path_cells(left, right):
            r, c = cell
            if int(g[r, c]) != BLACK:
                continue
            claims.setdefault(cell, set()).add(int(color))

    out = g.copy()
    for (r, c), colors in claims.items():
        if len(colors) == 1:
            out[r, c] = next(iter(colors))
        elif RED in colors:
            out[r, c] = YELLOW
        else:
            out[r, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == BLACK), "expected black background"
    pairs = _extract_pairs(g)
    return _paint_paths(g, pairs)


def _anchor_halo(anchor: Coord, height: int, width: int) -> set[Coord]:
    r, c = anchor
    return {
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if 0 <= r + dr < height and 0 <= c + dc < width
    }


def _shape_cells(anchor: Coord, offsets: tuple[Coord, ...]) -> set[Coord]:
    ar, ac = anchor
    return {(ar + dr, ac + dc) for dr, dc in offsets}


def _sprite_sets_are_separate(sprite_sets: list[set[Coord]]) -> bool:
    for i, cells_a in enumerate(sprite_sets):
        for j in range(i + 1, len(sprite_sets)):
            cells_b = sprite_sets[j]
            if cells_a & cells_b:
                return False
            for r, c in cells_a:
                for dr, dc in CARDINALS:
                    if (r + dr, c + dc) in cells_b:
                        return False
    return True


def _pair_paths(pairs: list[PairSpec]) -> list[set[Coord]]:
    return [_path_cells(left, right) for _color, left, right in pairs]


def _path_hits_foreign_halo(path: set[Coord], foreign_anchor: Coord, height: int, width: int) -> bool:
    return bool(path & _anchor_halo(foreign_anchor, height, width))


def _pair_is_compatible(candidate: PairSpec, existing: list[PairSpec], height: int, width: int) -> bool:
    cand_color, left, right = candidate
    del cand_color
    cand_path = _path_cells(left, right)
    cand_anchors = (left, right)
    existing_paths = _pair_paths(existing)
    existing_anchors = [anchor for _color, a, b in existing for anchor in (a, b)]

    for path in existing_paths:
        if cand_path & path:
            return False
    for anchor in existing_anchors:
        if _path_hits_foreign_halo(cand_path, anchor, height, width):
            return False
    for path in existing_paths:
        for anchor in cand_anchors:
            if _path_hits_foreign_halo(path, anchor, height, width):
                return False
    for anchor in cand_anchors:
        halo = _anchor_halo(anchor, height, width)
        for other in existing_anchors:
            if other in halo or anchor in _anchor_halo(other, height, width):
                return False
    return True


def _choose_colors(pair_count: int, red_crossing: bool) -> list[int]:
    if red_crossing:
        others = random.sample(GEN_NON_RED_POOL, pair_count - 1)
        colors = [RED, *others]
    else:
        colors = list(random.sample(GEN_NON_RED_POOL, pair_count))
    random.shuffle(colors)
    return colors


def _sample_anchor(height: int, width: int) -> Coord:
    return (
        random.randint(GEN_MIN_MARGIN, height - GEN_MIN_MARGIN - 1),
        random.randint(GEN_MIN_MARGIN, width - GEN_MIN_MARGIN - 1),
    )


def _build_forced_crossing_pairs(colors: list[int], height: int, width: int, red_crossing: bool) -> list[PairSpec] | None:
    crossing_colors = list(colors[:2])
    if red_crossing and RED not in crossing_colors:
        red_idx = colors.index(RED)
        colors[1], colors[red_idx] = colors[red_idx], colors[1]
        crossing_colors = list(colors[:2])

    for _ in range(GEN_FORCED_PAIR_TRIES):
        r_cross = random.randint(
            GEN_MIN_MARGIN + GEN_MIN_VERTICAL,
            height - GEN_MIN_MARGIN - GEN_MIN_VERTICAL - 1,
        )
        c_cross = random.randint(6, width - 7)

        a_left = (r_cross, random.randint(GEN_MIN_MARGIN, c_cross - GEN_MIN_HORIZONTAL))
        if random.random() < 0.5:
            a_right_row = random.randint(GEN_MIN_MARGIN, r_cross - GEN_MIN_VERTICAL)
        else:
            a_right_row = random.randint(r_cross + GEN_MIN_VERTICAL, height - GEN_MIN_MARGIN - 1)
        a_right = (a_right_row, random.randint(c_cross + GEN_MIN_HORIZONTAL, width - GEN_MIN_MARGIN - 1))

        if random.random() < 0.5:
            b_left_row = random.randint(r_cross + GEN_MIN_VERTICAL, height - GEN_MIN_MARGIN - 1)
            b_right_row = random.randint(GEN_MIN_MARGIN, r_cross - GEN_MIN_VERTICAL)
        else:
            b_left_row = random.randint(GEN_MIN_MARGIN, r_cross - GEN_MIN_VERTICAL)
            b_right_row = random.randint(r_cross + GEN_MIN_VERTICAL, height - GEN_MIN_MARGIN - 1)
        b_left = (b_left_row, random.randint(GEN_MIN_MARGIN, c_cross - GEN_MIN_HORIZONTAL))
        b_right = (b_right_row, c_cross)

        pair_a = _normalize_pair(crossing_colors[0], a_left, a_right)
        pair_b = _normalize_pair(crossing_colors[1], b_left, b_right)
        pairs = [pair_a, pair_b]

        anchors = [anchor for _color, x, y in pairs for anchor in (x, y)]
        if len(set(anchors)) != 4:
            continue

        paths = _pair_paths(pairs)
        if paths[0] & paths[1] != {(r_cross, c_cross)}:
            continue

        if any(_path_hits_foreign_halo(paths[0], anchor, height, width) for anchor in (b_left, b_right)):
            continue
        if any(_path_hits_foreign_halo(paths[1], anchor, height, width) for anchor in (a_left, a_right)):
            continue
        if any(
            other in _anchor_halo(anchor, height, width) or anchor in _anchor_halo(other, height, width)
            for i, anchor in enumerate(anchors)
            for other in anchors[i + 1 :]
        ):
            continue
        return pairs

    return None


def _sample_extra_pair(color: int, existing: list[PairSpec], height: int, width: int) -> PairSpec | None:
    for _ in range(GEN_EXTRA_PAIR_TRIES):
        left = _sample_anchor(height, width)
        right = _sample_anchor(height, width)
        if left == right:
            continue
        if right[1] <= left[1]:
            left, right = right, left
        if right[1] - left[1] < GEN_MIN_HORIZONTAL:
            continue
        if abs(right[0] - left[0]) < GEN_MIN_VERTICAL:
            continue
        pair = _normalize_pair(color, left, right)
        if _pair_is_compatible(pair, existing, height, width):
            return pair
    return None


def _assign_shapes(pairs: list[PairSpec], height: int, width: int) -> dict[Coord, tuple[Coord, ...]] | None:
    anchors = [anchor for _color, left, right in pairs for anchor in (left, right)]
    for _ in range(GEN_SHAPE_TRIES):
        shape_by_anchor = {anchor: random.choice(SPRITE_SHAPES) for anchor in anchors}
        sprite_sets = [_shape_cells(anchor, shape_by_anchor[anchor]) for anchor in anchors]
        if not _sprite_sets_are_separate(sprite_sets):
            continue
        if not all(_visual_center(sorted(cells)) == anchor for anchor, cells in zip(anchors, sprite_sets)):
            continue
        return shape_by_anchor
    return None


def _render_input(pairs: list[PairSpec], shape_by_anchor: dict[Coord, tuple[Coord, ...]], height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    for color, left, right in pairs:
        for anchor in (left, right):
            for r, c in _shape_cells(anchor, shape_by_anchor[anchor]):
                grid[r, c] = int(color)
    return grid


def _valid_input_layout(grid: np.ndarray, pairs: list[PairSpec]) -> bool:
    extracted = _extract_pairs(grid)
    return extracted == sorted(pairs)


def _crossing_requirements_met(inp: np.ndarray, out: np.ndarray, red_crossing: bool) -> bool:
    painted = (inp == BLACK) & (out != BLACK)
    if red_crossing:
        return bool(np.any(out[painted] == YELLOW))
    return bool(np.any(out[painted] == RED))


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    pair_count = random.randint(*GEN_PAIR_COUNT_RANGE)
    red_crossing = random.random() < 0.5
    colors = _choose_colors(pair_count, red_crossing)

    pairs = _build_forced_crossing_pairs(colors, height, width, red_crossing)
    if pairs is None:
        return None

    for color in colors[2:]:
        pair = _sample_extra_pair(color, pairs, height, width)
        if pair is None:
            return None
        pairs.append(pair)
    pairs.sort()

    shape_by_anchor = _assign_shapes(pairs, height, width)
    if shape_by_anchor is None:
        return None

    inp = _render_input(pairs, shape_by_anchor, height, width)
    if not _valid_input_layout(inp, pairs):
        return None

    expected = _paint_paths(inp, pairs)
    if not _crossing_requirements_met(inp, expected, red_crossing):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
