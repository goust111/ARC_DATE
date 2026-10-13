from __future__ import annotations

from collections import Counter, defaultdict, deque
from itertools import combinations
import random

import numpy as np


Coord = tuple[int, int]


def _label4(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label True cells by 4-connectivity using the common (row, column) convention."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    next_label = 0
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if labels[sr, sc] != 0:
            continue
        next_label += 1
        labels[sr, sc] = next_label
        queue = deque([(sr, sc)])
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and labels[rr, cc] == 0:
                    labels[rr, cc] = next_label
                    queue.append((rr, cc))
    return labels, next_label


def _strict_most_common(values: np.ndarray, context: str) -> int:
    counts = Counter(map(int, values.ravel()))
    ordered = counts.most_common()
    assert ordered, f"{context} must be nonempty"
    assert len(ordered) == 1 or ordered[0][1] > ordered[1][1], f"{context} must have a unique majority color"
    return int(ordered[0][0])


def _analyze_partition(g: np.ndarray, bg: int, sprite_colors: set[int]) -> dict[str, object]:
    """Test one visible two-color object as the sprite and derive its signed region graph."""
    all_foreground = set(map(int, g.ravel())) - {bg}
    barrier_colors = all_foreground - sprite_colors
    assert len(sprite_colors) == len(barrier_colors) == 2
    sprite_mask = np.isin(g, list(sprite_colors))
    barrier_mask = np.isin(g, list(barrier_colors))
    _sprite_labels, n_sprite_parts = _label4(sprite_mask)
    assert n_sprite_parts == 1, "sprite colors must form one 4-connected object"
    assert int(np.sum(barrier_mask)) > int(np.sum(sprite_mask)), "separators must be larger than the sprite"

    barrier_values, sprite_values = g[barrier_mask], g[sprite_mask]
    wall = _strict_most_common(barrier_values, "separator network")
    door = int(next(iter(barrier_colors - {wall})))
    color_a = _strict_most_common(sprite_values, "sprite")
    color_b = int(next(iter(sprite_colors - {color_a})))
    region_labels, n_regions = _label4(~barrier_mask)
    assert n_regions >= 2, "separators must create multiple passable regions"
    seed_ids = set(map(int, region_labels[sprite_mask]))
    assert len(seed_ids) == 1 and 0 not in seed_ids, "the whole sprite must lie in one region"
    seed = int(next(iter(seed_ids)))

    # Each separator pixel can witness an edge across its left/right sides or
    # across its top/bottom sides. Repeated witnesses must agree on the edge type.
    edge_relation: dict[tuple[int, int], int] = {}
    h, w = g.shape
    for r, c in np.argwhere(barrier_mask):
        r, c = int(r), int(c)
        relation = 1 if int(g[r, c]) == wall else 0
        opposite_pairs = (
            ((r, c - 1), (r, c + 1)),
            ((r - 1, c), (r + 1, c)),
        )
        for (r1, c1), (r2, c2) in opposite_pairs:
            if not (0 <= r1 < h and 0 <= c1 < w and 0 <= r2 < h and 0 <= c2 < w):
                continue
            a, b = int(region_labels[r1, c1]), int(region_labels[r2, c2])
            if a <= 0 or b <= 0 or a == b:
                continue
            edge = (a, b) if a < b else (b, a)
            if edge in edge_relation:
                assert edge_relation[edge] == relation, "one region boundary cannot be both wall and door"
            else:
                edge_relation[edge] = relation

    graph: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (a, b), relation in edge_relation.items():
        graph[a].append((b, relation))
        graph[b].append((a, relation))
    assert edge_relation, "separator network must induce region adjacencies"

    parity = {seed: 0}
    queue = deque([seed])
    while queue:
        region = queue.popleft()
        for neighbor, relation in graph[region]:
            wanted = parity[region] ^ relation
            if neighbor in parity:
                assert parity[neighbor] == wanted, "wall/door constraints must be consistent"
            else:
                parity[neighbor] = wanted
                queue.append(neighbor)
    assert set(parity) == set(range(1, n_regions + 1)), "all regions must be connected in the quotient graph"
    assert 0 in edge_relation.values() and 1 in edge_relation.values(), "both doors and walls must constrain regions"

    return {
        "grid": g,
        "bg": bg,
        "barrier_mask": barrier_mask,
        "sprite_mask": sprite_mask,
        "wall": wall,
        "door": door,
        "color_a": color_a,
        "color_b": color_b,
        "region_labels": region_labels,
        "n_regions": n_regions,
        "seed": seed,
        "edge_relation": edge_relation,
        "parity": parity,
    }


def _analyze(grid: np.ndarray) -> dict[str, object]:
    """Uniquely identify the sprite palette, then recover the signed region graph."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    bg = _strict_most_common(g, "grid")
    foreground_colors = set(map(int, g.ravel())) - {bg}
    assert len(foreground_colors) == 4, "expected wall, door, and two sprite colors"

    candidates: list[dict[str, object]] = []
    for pair in combinations(sorted(foreground_colors), 2):
        try:
            candidates.append(_analyze_partition(g, bg, set(pair)))
        except (AssertionError, ValueError, IndexError, StopIteration):
            continue
    assert len(candidates) == 1, "visible structure must uniquely identify the sprite and separator palettes"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor wall-separated regions by seeded wall/door parity, using the sprite's two colors.

    Input structure:
    - The unique dominant color is background. One-cell-thick separator lines
      contain a majority wall color and shorter minority-colored door segments.
    - A smaller, detached, 4-connected two-color sprite lies wholly inside one region
      without touching a separator; its majority color is A and minority color is B.

    Rule:
    - Treat passable regions as graph vertices. Crossing a wall toggles parity, while
      crossing a door preserves parity. Seed the sprite's region with parity zero.
    - Recolor every parity-zero region A and every parity-one region B, covering and
      erasing the sprite, while leaving every wall and door pixel unchanged.

    Key insight:
    - Doors contract equal-parity neighbors and only walls flip the bipartite class;
      the detached sprite uniquely supplies both the seed vertex and the A/B palette.
    """
    info = _analyze(grid)
    g = info["grid"]
    assert isinstance(g, np.ndarray)
    out = g.copy()
    labels = info["region_labels"]
    parity = info["parity"]
    assert isinstance(labels, np.ndarray) and isinstance(parity, dict)
    color_a, color_b = int(info["color_a"]), int(info["color_b"])
    for region, value in parity.items():
        out[labels == int(region)] = color_a if int(value) == 0 else color_b
    return out


# ------------------------------ generator ------------------------------

_REGION_COUNT_CHOICES = (3, 4)
_REGION_SIDE_RANGE = (5, 6)
_MAX_ATTEMPTS = 400

_SPRITES: tuple[tuple[Coord, ...], ...] = (
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
)


def _spans(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    spans: list[tuple[int, int]] = []
    dividers: list[int] = []
    cursor = 0
    for i, size in enumerate(sizes):
        spans.append((cursor, cursor + size - 1))
        cursor += size
        if i + 1 < len(sizes):
            dividers.append(cursor)
            cursor += 1
    return spans, dividers, cursor


def _normalize_shape(coords: tuple[Coord, ...]) -> tuple[Coord, ...]:
    r0 = min(r for r, _c in coords)
    c0 = min(c for _r, c in coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _random_sprite_shape() -> tuple[Coord, ...]:
    coords = random.choice(_SPRITES)
    for _ in range(random.randrange(4)):
        coords = tuple((c, -r) for r, c in coords)
    if random.random() < 0.5:
        coords = tuple((r, -c) for r, c in coords)
    return _normalize_shape(coords)


def _sample_parity(nr: int, nc: int, seed: tuple[int, int]) -> np.ndarray | None:
    """Sample a nondegenerate signed coloring with both edge types in both orientations."""
    corners = ((0, 0), (0, nc - 1), (nr - 1, 0), (nr - 1, nc - 1))
    for _ in range(300):
        parity = np.array([[random.randrange(2) for _ in range(nc)] for _ in range(nr)], dtype=int)
        parity ^= int(parity[seed])
        if any(int(parity[pos]) != 1 for pos in corners):
            continue
        zeros = int(np.sum(parity == 0))
        ones = int(np.sum(parity == 1))
        if min(zeros, ones) < 3:
            continue
        vertical = [int(parity[i, j] != parity[i, j + 1]) for i in range(nr) for j in range(nc - 1)]
        horizontal = [int(parity[i, j] != parity[i + 1, j]) for i in range(nr - 1) for j in range(nc)]
        if set(vertical) == {0, 1} and set(horizontal) == {0, 1}:
            return parity
    return None


def _render_barriers(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    row_dividers: list[int],
    col_spans: list[tuple[int, int]],
    col_dividers: list[int],
    parity: np.ndarray,
    wall: int,
    door: int,
) -> None:
    # A complete atomic segment between two junctions has one unambiguous type.
    for i, (r0, r1) in enumerate(row_spans):
        for j, divider_c in enumerate(col_dividers):
            color = wall if int(parity[i, j]) != int(parity[i, j + 1]) else door
            grid[r0 : r1 + 1, divider_c] = color
    for i, divider_r in enumerate(row_dividers):
        for j, (c0, c1) in enumerate(col_spans):
            color = wall if int(parity[i, j]) != int(parity[i + 1, j]) else door
            grid[divider_r, c0 : c1 + 1] = color
    # Junctions belong to the base wall and connect the entire separator network.
    for r in row_dividers:
        for c in col_dividers:
            grid[r, c] = wall


def _place_sprite(
    grid: np.ndarray,
    span_r: tuple[int, int],
    span_c: tuple[int, int],
    color_a: int,
    color_b: int,
) -> set[Coord] | None:
    shape = _random_sprite_shape()
    sh = max(r for r, _c in shape) + 1
    sw = max(c for _r, c in shape) + 1
    # One background cell of clearance makes non-contact with every separator visible.
    top_choices = range(span_r[0] + 1, span_r[1] - sh + 2)
    left_choices = range(span_c[0] + 1, span_c[1] - sw + 2)
    tops, lefts = list(top_choices), list(left_choices)
    if not tops or not lefts:
        return None
    top, left = random.choice(tops), random.choice(lefts)

    cells = [(top + r, left + c) for r, c in shape]
    minority_count = random.randint(1, min(2, (len(cells) - 1) // 2))
    minority = set(random.sample(cells, minority_count))
    for r, c in cells:
        grid[r, c] = color_b if (r, c) in minority else color_a
    return set(cells)


def _semantic_check(
    inp: np.ndarray,
    expected: np.ndarray,
    parity: np.ndarray,
    seed: tuple[int, int],
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    colors: tuple[int, int, int, int, int],
    sprite_cells: set[Coord],
) -> bool:
    """Check the stated semantics against construction metadata, not only solver closure."""
    bg, wall, door, color_a, color_b = colors
    if inp.ndim != 2 or max(inp.shape) > 30 or inp.shape != expected.shape:
        return False
    if _strict_most_common(inp, "generated grid") != bg:
        return False
    if int(np.sum(inp == wall)) <= int(np.sum(inp == door)):
        return False
    sprite_rows, sprite_cols = zip(*sprite_cells)
    sprite_values = inp[np.array(sprite_rows), np.array(sprite_cols)]
    if int(np.sum(sprite_values == color_a)) <= int(np.sum(sprite_values == color_b)):
        return False
    if np.array_equal(inp, expected):
        return False
    if not np.array_equal(expected[(inp == wall) | (inp == door)], inp[(inp == wall) | (inp == door)]):
        return False
    if any(int(expected[r, c]) != color_a for r, c in sprite_cells):
        return False

    # The seed is not replaceable by a fixed-corner convention: every corner region is B.
    nr, nc = parity.shape
    if int(parity[seed]) != 0 or any(int(parity[pos]) != 1 for pos in ((0, 0), (0, nc - 1), (nr - 1, 0), (nr - 1, nc - 1))):
        return False

    vertical_relations: set[int] = set()
    horizontal_relations: set[int] = set()
    for i in range(nr):
        for j in range(nc - 1):
            relation = int(parity[i, j] != parity[i, j + 1])
            vertical_relations.add(relation)
            left = int(expected[row_spans[i][0], col_spans[j][0]])
            right = int(expected[row_spans[i][0], col_spans[j + 1][0]])
            if (left != right) != bool(relation):
                return False
    for i in range(nr - 1):
        for j in range(nc):
            relation = int(parity[i, j] != parity[i + 1, j])
            horizontal_relations.add(relation)
            upper = int(expected[row_spans[i][0], col_spans[j][0]])
            lower = int(expected[row_spans[i + 1][0], col_spans[j][0]])
            if (upper != lower) != bool(relation):
                return False
    if vertical_relations != {0, 1} or horizontal_relations != {0, 1}:
        return False

    try:
        info = _analyze(inp)
        if (int(info["wall"]), int(info["door"]), int(info["color_a"]), int(info["color_b"])) != (
            wall,
            door,
            color_a,
            color_b,
        ):
            return False
        solved = main(inp)
    except (AssertionError, ValueError, IndexError):
        return False
    return bool(np.array_equal(solved, expected))


def _generate_attempt() -> np.ndarray | None:
    nr = random.choice(_REGION_COUNT_CHOICES)
    nc = random.choice(_REGION_COUNT_CHOICES)
    row_sizes = [random.randint(*_REGION_SIDE_RANGE) for _ in range(nr)]
    col_sizes = [random.randint(*_REGION_SIDE_RANGE) for _ in range(nc)]
    row_spans, row_dividers, h = _spans(row_sizes)
    col_spans, col_dividers, w = _spans(col_sizes)
    if max(h, w) > 30:
        return None

    noncorners = [
        (i, j)
        for i in range(nr)
        for j in range(nc)
        if not (i in (0, nr - 1) and j in (0, nc - 1))
    ]
    seed = random.choice(noncorners)
    parity = _sample_parity(nr, nc, seed)
    if parity is None:
        return None

    bg, wall, door, color_a, color_b = random.sample(range(10), 5)
    inp = np.full((h, w), bg, dtype=int)
    _render_barriers(inp, row_spans, row_dividers, col_spans, col_dividers, parity, wall, door)
    if int(np.sum(inp == wall)) <= int(np.sum(inp == door)):
        return None

    sprite_cells = _place_sprite(inp, row_spans[seed[0]], col_spans[seed[1]], color_a, color_b)
    if sprite_cells is None:
        return None

    expected = inp.copy()
    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            expected[r0 : r1 + 1, c0 : c1 + 1] = color_a if int(parity[i, j]) == 0 else color_b

    colors = (bg, wall, door, color_a, color_b)
    if _semantic_check(inp, expected, parity, seed, row_spans, col_spans, colors, sprite_cells):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        candidate = _generate_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid wall/door parity puzzle")
