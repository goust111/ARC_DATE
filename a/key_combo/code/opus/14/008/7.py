from __future__ import annotations

from collections import Counter
from itertools import combinations, permutations
import random

import numpy as np


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _quadrant_slices(n: int) -> tuple[tuple[slice, slice], ...]:
    return (
        (slice(0, n), slice(0, n)),
        (slice(0, n), slice(n, 2 * n)),
        (slice(n, 2 * n), slice(0, n)),
        (slice(n, 2 * n), slice(n, 2 * n)),
    )


def _coords_from_mask(mask: np.ndarray) -> frozenset[Coord]:
    return frozenset((int(r), int(c)) for r, c in np.argwhere(mask))


def _rotate_coords(coords: frozenset[Coord], n: int, turns: int) -> frozenset[Coord]:
    turns %= 4
    out = set(coords)
    for _ in range(turns):
        out = {(c, n - 1 - r) for r, c in out}
    return frozenset(out)


def _reflect_lr(coords: frozenset[Coord], n: int) -> frozenset[Coord]:
    return frozenset((r, n - 1 - c) for r, c in coords)


def _majority_non_bg_color(grid: np.ndarray, bg: int) -> int:
    vals = [int(v) for v in grid[grid != bg]]
    assert vals, "expected at least one foreground pixel"
    return int(Counter(vals).most_common(1)[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each quadrant fragment into the uniquely matching rotation of the full template quadrant.

    Input structure:
    - The grid is square with even side length and a black background, split into four equal square quadrants.
    - Exactly one quadrant contains the complete monochrome template shape and therefore has the most foreground pixels.
    - Each other quadrant contains a same-color fragment that is a subset of one non-identity quarter-turn rotation of that template.

    Rule:
    - Identify the template quadrant as the unique quadrant with the largest number of non-background cells.
    - Extract its foreground mask, build the 90-degree clockwise, 180-degree, and 270-degree clockwise rotations inside the quadrant canvas, and compare each fragment quadrant against those three masks.
    - Every fragment matches exactly one rotation by subset inclusion; fill in the missing cells of that rotation in the fragment's quadrant, leaving the template quadrant unchanged.

    Key insight:
    - A fragment's screen position does not determine its orientation; only subset-matching against all three rotated template masks reveals which completion to use.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w and h % 2 == 0, "expected an even-sided square grid"

    bg = _get_bg(g)
    n = h // 2
    slices = _quadrant_slices(n)
    quadrants = [g[rs, cs] for rs, cs in slices]
    fg_counts = [int(np.count_nonzero(q != bg)) for q in quadrants]

    template_area = max(fg_counts)
    template_indices = [i for i, count in enumerate(fg_counts) if count == template_area]
    assert len(template_indices) == 1, "template quadrant must be unique"
    template_idx = template_indices[0]

    template_quad = quadrants[template_idx]
    template_mask = template_quad != bg
    template_coords = _coords_from_mask(template_mask)
    assert template_coords, "template quadrant must be non-empty"

    rotations = {turn: _rotate_coords(template_coords, n, turn) for turn in (1, 2, 3)}
    out = g.copy()

    for idx, quad in enumerate(quadrants):
        if idx == template_idx:
            continue

        fragment = _coords_from_mask(quad != bg)
        assert fragment, "fragment quadrants must be non-empty"
        matches = [turn for turn in (1, 2, 3) if fragment.issubset(rotations[turn])]
        assert len(matches) == 1, "each fragment must determine a unique rotation"

        fill_color = _majority_non_bg_color(quad, bg)
        rs, cs = slices[idx]
        for r, c in rotations[matches[0]]:
            out[rs.start + r, cs.start + c] = fill_color

    return out


GEN_ATTEMPTS = 400
GEN_N_RANGE = (5, 7)
GEN_COLOR_CHOICES = tuple(range(1, 10))
GEN_FRAGMENT_SIZES = (3, 4, 5)
GEN_CANONICAL_ASSIGNMENT = (1, 3, 2)  # UR, LL, LR


def _neighbors4(r: int, c: int, n: int) -> list[Coord]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < n:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < n:
        out.append((r, c + 1))
    return out


def _bbox(coords: frozenset[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_connected(coords: frozenset[Coord]) -> bool:
    start = next(iter(coords))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _template_area_bounds(n: int) -> tuple[int, int]:
    lo = max(8, int(np.ceil(0.30 * n * n)))
    hi = min(15, int(np.floor(0.60 * n * n)))
    assert lo <= hi
    return lo, hi


def _all_dihedral_variants(coords: frozenset[Coord], n: int) -> set[frozenset[Coord]]:
    variants = set()
    base = coords
    for reflected in (base, _reflect_lr(base, n)):
        for turn in range(4):
            variants.add(_rotate_coords(reflected, n, turn))
    return variants


def _template_valid(coords: frozenset[Coord], n: int) -> bool:
    if len(coords) < 8:
        return False
    if not _is_connected(coords):
        return False

    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return False

    box_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    density = len(coords) / box_area
    if not (0.35 <= density <= 0.80):
        return False

    rotations = {_rotate_coords(coords, n, turn) for turn in range(4)}
    if len(rotations) != 4:
        return False
    if len(_all_dihedral_variants(coords, n)) != 8:
        return False
    return True


def _grow_template(n: int, area: int) -> frozenset[Coord] | None:
    cells = {(random.randrange(n), random.randrange(n))}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            frontier.update(_neighbors4(r, c, n))
        frontier.difference_update(cells)
        if not frontier:
            return None
        frontier_list = sorted(frontier)
        weights = []
        for r, c in frontier_list:
            center_bias = (n - abs(2 * r - n + 1)) + (n - abs(2 * c - n + 1))
            touch = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c, n))
            weights.append(max(1, center_bias + 2 * touch))
        cells.add(random.choices(frontier_list, weights=weights, k=1)[0])
    coords = frozenset(cells)
    return coords if _template_valid(coords, n) else None


def _fragment_visual_ok(fragment: frozenset[Coord]) -> bool:
    r0, c0, r1, c1 = _bbox(fragment)
    if r1 == r0 or c1 == c0:
        return False
    return any((r + 1, c) in fragment or (r, c + 1) in fragment for r, c in fragment)


def _choose_unique_fragment(rotations: dict[int, frozenset[Coord]], turn: int) -> frozenset[Coord] | None:
    target = tuple(sorted(rotations[turn]))
    candidates: list[frozenset[Coord]] = []
    for size in GEN_FRAGMENT_SIZES:
        if size >= len(target):
            continue
        for combo in combinations(target, size):
            fragment = frozenset(combo)
            if not _fragment_visual_ok(fragment):
                continue
            match_count = sum(fragment.issubset(rotations[other]) for other in (1, 2, 3))
            if match_count == 1:
                candidates.append(fragment)
    if not candidates:
        return None
    return random.choice(candidates)


def _rotation_assignment() -> tuple[int, int, int]:
    perms = list(permutations((1, 2, 3)))
    noncanonical = [perm for perm in perms if perm != GEN_CANONICAL_ASSIGNMENT]
    if random.random() < 0.8:
        return random.choice(noncanonical)
    return GEN_CANONICAL_ASSIGNMENT


def _paint_coords(grid: np.ndarray, coords: frozenset[Coord], rs: slice, cs: slice, color: int) -> None:
    for r, c in coords:
        grid[rs.start + r, cs.start + c] = color


def _build_expected_output(
    n: int,
    color: int,
    template: frozenset[Coord],
    assignment: tuple[int, int, int],
) -> np.ndarray:
    out = np.zeros((2 * n, 2 * n), dtype=int)
    slices = _quadrant_slices(n)
    rotations = {turn: _rotate_coords(template, n, turn) for turn in range(4)}
    _paint_coords(out, rotations[0], *slices[0], color)
    for quad_idx, turn in zip((1, 2, 3), assignment):
        _paint_coords(out, rotations[turn], *slices[quad_idx], color)
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        n = random.randint(*GEN_N_RANGE)
        lo, hi = _template_area_bounds(n)
        area = random.randint(lo, hi)
        color = random.choice(GEN_COLOR_CHOICES)

        template = _grow_template(n, area)
        if template is None:
            continue

        rotations = {turn: _rotate_coords(template, n, turn) for turn in range(4)}
        fragments = {turn: _choose_unique_fragment(rotations, turn) for turn in (1, 2, 3)}
        if any(fragment is None for fragment in fragments.values()):
            continue

        assignment = _rotation_assignment()
        inp = np.zeros((2 * n, 2 * n), dtype=int)
        slices = _quadrant_slices(n)
        _paint_coords(inp, template, *slices[0], color)
        for quad_idx, turn in zip((1, 2, 3), assignment):
            _paint_coords(inp, fragments[turn], *slices[quad_idx], color)

        expected = _build_expected_output(n, color, template, assignment)
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp

    raise RuntimeError("failed to generate a valid input")
