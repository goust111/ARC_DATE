from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


Coord = tuple[int, int]
Diamond = tuple[int, int, int]  # center row, center column, radius


def _diamond_outline(shape: tuple[int, int], cr: int, cc: int, radius: int) -> frozenset[Coord]:
    """Return the complete in-bounds Manhattan-diamond contour."""
    h, w = shape
    points: set[Coord] = set()
    for dr in range(-radius, radius + 1):
        dc = radius - abs(dr)
        for c in {cc - dc, cc + dc}:
            r = cr + dr
            if 0 <= r < h and 0 <= c < w:
                points.add((r, c))
    return frozenset(points)


def _sampled_outline(
    shape: tuple[int, int],
    cr: int,
    cc: int,
    radius: int,
    line_rows: frozenset[int],
    line_cols: frozenset[int],
) -> frozenset[Coord]:
    """Sample a contour in the input's own row/column lattice reference frame."""
    return frozenset(
        (r, c)
        for r, c in _diamond_outline(shape, cr, cc, radius)
        if r in line_rows or c in line_cols
    )


def _line_geometry(grid: np.ndarray, bg: int) -> tuple[frozenset[int], frozenset[int], int]:
    """Recover full non-background lattice rows/columns and their common color."""
    non_bg = grid != bg
    line_rows = frozenset(int(r) for r in range(grid.shape[0]) if bool(non_bg[r, :].all()))
    line_cols = frozenset(int(c) for c in range(grid.shape[1]) if bool(non_bg[:, c].all()))
    if not line_rows or not line_cols:
        raise ValueError("the grid must contain both horizontal and vertical lattice lines")

    lattice = np.zeros(grid.shape, dtype=bool)
    lattice[list(line_rows), :] = True
    lattice[:, list(line_cols)] = True
    values = [int(v) for v in grid[lattice] if int(v) != bg]
    if not values:
        raise ValueError("lattice has no visible line color")
    line_color = int(Counter(values).most_common(1)[0][0])
    return line_rows, line_cols, line_color


def _candidate_diamonds(
    points: frozenset[Coord],
    shape: tuple[int, int],
    line_rows: frozenset[int],
    line_cols: frozenset[int],
) -> list[Diamond]:
    """Find complete diamonds whose entire lattice sample is exactly ``points``."""
    if not points:
        return []
    h, w = shape
    candidates: list[Diamond] = []
    max_radius = min((h - 1) // 2, (w - 1) // 2)
    for radius in range(1, max_radius + 1):
        for cr in range(radius, h - radius):
            for cc in range(radius, w - radius):
                # The equality test is cheap and rejects almost every center before
                # constructing the complete set of lattice intersections.
                if any(abs(r - cr) + abs(c - cc) != radius for r, c in points):
                    continue
                if _sampled_outline(shape, cr, cc, radius, line_rows, line_cols) == points:
                    candidates.append((cr, cc, radius))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """Reconstruct erased, equal-radius Manhattan-diamond outlines from their colored lattice residues.

    Input structure:
    - A dominant background contains full-width and full-height lines of one common color.
    - Every other color occurs only on those lines; each such color is the complete set of
      intersections between one erased, fully in-bounds diamond outline and the lattice.

    Rule:
    - Group residue pixels by color and enumerate diamonds whose exact lattice intersections
      equal each group. A group with one candidate reveals the radius shared by all diamonds.
    - Filter every underdetermined group by that radius, require its center to become unique,
      then draw each complete outline in its residue color while retaining the lattice.

    Key insight:
    - Exact intersection sampling determines one anchor diamond; its radius propagates to
      sparse one- or two-pixel groups that cannot determine their diamonds independently.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be two-dimensional")
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    line_rows, line_cols, line_color = _line_geometry(g, bg)

    residue_colors = [
        int(color)
        for color in np.unique(g)
        if int(color) not in {bg, line_color}
    ]
    if len(residue_colors) < 2:
        raise ValueError("at least two colored residue groups are required")

    candidates_by_color: dict[int, list[Diamond]] = {}
    for color in residue_colors:
        points = frozenset((int(r), int(c)) for r, c in np.argwhere(g == color))
        candidates = _candidate_diamonds(points, g.shape, line_rows, line_cols)
        if not candidates:
            raise ValueError(f"residue color {color} has no exact diamond interpretation")
        candidates_by_color[color] = candidates

    anchors = [candidates[0] for candidates in candidates_by_color.values() if len(candidates) == 1]
    if not anchors:
        raise ValueError("no independently determined anchor diamond")
    shared_radii = {radius for _cr, _cc, radius in anchors}
    if len(shared_radii) != 1:
        raise ValueError("anchor diamonds disagree on the shared radius")
    shared_radius = shared_radii.pop()

    solved: dict[int, Diamond] = {}
    for color, candidates in candidates_by_color.items():
        compatible = [candidate for candidate in candidates if candidate[2] == shared_radius]
        if len(compatible) != 1:
            raise ValueError(f"shared radius does not uniquely determine residue color {color}")
        solved[color] = compatible[0]

    out = g.copy()
    for color, (cr, cc, radius) in solved.items():
        for r, c in _diamond_outline(g.shape, cr, cc, radius):
            out[r, c] = color
    return out


# Generator configuration. The successful-call phase cycles through one-point,
# two-point, and mixed sparse branches, so every three successive examples cover
# both underdetermined cases while all geometric choices remain randomized.
_GENERATION_INDEX = 0
_MAX_GENERATION_ATTEMPTS = 350


def _spaced_positions(rng: random.Random, count: int, limit: int) -> list[int] | None:
    """Choose interior line positions with at least two background rows/cols between."""
    choices = list(range(2, limit - 2))
    for _ in range(100):
        positions = sorted(rng.sample(choices, count))
        if all(b - a >= 3 for a, b in zip(positions, positions[1:])):
            return positions
    return None


def _candidate_catalog(
    shape: tuple[int, int],
    line_rows: frozenset[int],
    line_cols: frozenset[int],
) -> dict[frozenset[Coord], list[Diamond]]:
    """Index every visible nonempty sample by all complete diamonds producing it."""
    h, w = shape
    catalog: dict[frozenset[Coord], list[Diamond]] = defaultdict(list)
    max_radius = min((h - 1) // 2, (w - 1) // 2)
    for radius in range(1, max_radius + 1):
        for cr in range(radius, h - radius):
            for cc in range(radius, w - radius):
                sample = _sampled_outline(shape, cr, cc, radius, line_rows, line_cols)
                if sample:
                    catalog[sample].append((cr, cc, radius))
    return dict(catalog)


def _compatible(candidate: Diamond, selected: list[Diamond]) -> bool:
    """Keep the filled diamonds separated by at least two cells in Manhattan space."""
    cr, cc, radius = candidate
    return all(
        abs(cr - sr) + abs(cc - sc) > radius + other_radius + 2
        for sr, sc, other_radius in selected
    )


def _choose_objects(
    rng: random.Random,
    catalog: dict[frozenset[Coord], list[Diamond]],
    shared_radius: int,
    phase: int,
) -> list[tuple[frozenset[Coord], Diamond, bool]] | None:
    """Choose one anchor followed by genuinely radius-dependent sparse objects."""
    anchor_pool: list[tuple[frozenset[Coord], Diamond]] = []
    sparse: dict[int, list[tuple[frozenset[Coord], Diamond]]] = {1: [], 2: []}

    for sample, all_candidates in catalog.items():
        at_radius = [candidate for candidate in all_candidates if candidate[2] == shared_radius]
        if len(at_radius) != 1:
            continue
        chosen = at_radius[0]
        if len(all_candidates) == 1 and 3 <= len(sample) <= 10:
            anchor_pool.append((sample, chosen))
        elif len(all_candidates) > 1 and len(sample) in sparse:
            # len(at_radius)==1 plus multiple global candidates means the shared
            # radius eliminates at least one genuinely different-radius solution.
            sparse[len(sample)].append((sample, chosen))

    if not anchor_pool or not sparse[1] or not sparse[2]:
        return None

    object_count = rng.choice([2, 2, 3, 3, 4])
    if phase == 0:
        requirements = [1]
    elif phase == 1:
        requirements = [2]
    else:
        object_count = max(3, object_count)
        requirements = [1, 2]
    while len(requirements) < object_count - 1:
        requirements.append(rng.choice([1, 2]))

    for _ in range(250):
        rng.shuffle(anchor_pool)
        selected_sparse: list[tuple[frozenset[Coord], Diamond]] = []
        selected_diamonds: list[Diamond] = []
        ok = True
        for sample_size in requirements:
            pool = sparse[sample_size][:]
            rng.shuffle(pool)
            option = next(
                (
                    item
                    for item in pool
                    if item[0] not in {sample for sample, _candidate in selected_sparse}
                    and _compatible(item[1], selected_diamonds)
                ),
                None,
            )
            if option is None:
                ok = False
                break
            selected_sparse.append(option)
            selected_diamonds.append(option[1])
        if not ok:
            continue

        anchor = next(
            (item for item in anchor_pool if _compatible(item[1], selected_diamonds)),
            None,
        )
        if anchor is None:
            continue
        # Anchor is first so validation can explicitly verify information flow.
        return [(anchor[0], anchor[1], True)] + [
            (sample, candidate, False) for sample, candidate in selected_sparse
        ]
    return None


def _direct_expected(
    inp: np.ndarray,
    objects: list[tuple[frozenset[Coord], Diamond, bool]],
    colors: list[int],
) -> np.ndarray:
    """Construct the expected result directly from latent geometry, not via main()."""
    expected = inp.copy()
    for (_sample, (cr, cc, radius), _is_anchor), color in zip(objects, colors):
        for r, c in _diamond_outline(inp.shape, cr, cc, radius):
            expected[r, c] = color
    return expected


def _semantic_validation(
    inp: np.ndarray,
    expected: np.ndarray,
    bg: int,
    line_color: int,
    line_rows: frozenset[int],
    line_cols: frozenset[int],
    objects: list[tuple[frozenset[Coord], Diamond, bool]],
    colors: list[int],
    shared_radius: int,
    phase: int,
) -> bool:
    """Validate rule semantics, uniqueness, dependency, and non-degeneracy."""
    if inp.ndim != 2 or max(inp.shape) > 30 or expected.shape != inp.shape:
        return False
    if int(Counter(map(int, inp.ravel())).most_common(1)[0][0]) != bg:
        return False
    try:
        found_rows, found_cols, found_line_color = _line_geometry(inp, bg)
    except ValueError:
        return False
    if (found_rows, found_cols, found_line_color) != (line_rows, line_cols, line_color):
        return False

    lattice = np.zeros(inp.shape, dtype=bool)
    lattice[list(line_rows), :] = True
    lattice[:, list(line_cols)] = True
    if bool(np.any((inp != bg) & ~lattice)):
        return False

    anchor_count = 0
    sparse_sizes: set[int] = set()
    combined_points: set[Coord] = set()
    for (sample, candidate, is_anchor), color in zip(objects, colors):
        if frozenset((int(r), int(c)) for r, c in np.argwhere(inp == color)) != sample:
            return False
        candidates = _candidate_diamonds(sample, inp.shape, line_rows, line_cols)
        if is_anchor:
            anchor_count += 1
            if candidates != [candidate] or candidate[2] != shared_radius:
                return False
        else:
            sparse_sizes.add(len(sample))
            if len(candidates) <= 1:
                return False
            compatible = [item for item in candidates if item[2] == shared_radius]
            if compatible != [candidate]:
                return False
            if not any(item[2] != shared_radius for item in candidates):
                return False
        combined_points.update(sample)

        outline = _diamond_outline(inp.shape, *candidate)
        if len([point for point in outline if point not in sample]) < 4:
            return False

    if anchor_count != 1:
        return False
    if phase == 0 and 1 not in sparse_sizes:
        return False
    if phase == 1 and 2 not in sparse_sizes:
        return False
    if phase == 2 and not {1, 2}.issubset(sparse_sizes):
        return False

    # Grouping by color is indispensable: pooling residues cannot describe one diamond.
    if _candidate_diamonds(frozenset(combined_points), inp.shape, line_rows, line_cols):
        return False
    if np.array_equal(inp, expected):
        return False
    try:
        solved = main(inp)
    except (AssertionError, ValueError):
        return False
    return np.array_equal(solved, expected)


def _generate_attempt(rng: random.Random, phase: int) -> np.ndarray | None:
    h = rng.randint(23, 30)
    w = rng.randint(23, 30)
    row_count = rng.randint(2, 4)
    col_count = rng.randint(2, 4)
    rows = _spaced_positions(rng, row_count, h)
    cols = _spaced_positions(rng, col_count, w)
    if rows is None or cols is None:
        return None
    line_rows = frozenset(rows)
    line_cols = frozenset(cols)

    shared_radius = rng.randint(3, 5)
    catalog = _candidate_catalog((h, w), line_rows, line_cols)
    objects = _choose_objects(rng, catalog, shared_radius, phase)
    if objects is None:
        return None

    palette = list(range(10))
    rng.shuffle(palette)
    bg, line_color = palette[:2]
    colors = palette[2 : 2 + len(objects)]

    inp = np.full((h, w), bg, dtype=int)
    inp[list(line_rows), :] = line_color
    inp[:, list(line_cols)] = line_color
    for (sample, _candidate, _is_anchor), color in zip(objects, colors):
        for r, c in sample:
            inp[r, c] = color

    expected = _direct_expected(inp, objects, colors)
    if not _semantic_validation(
        inp,
        expected,
        bg,
        line_color,
        line_rows,
        line_cols,
        objects,
        colors,
        shared_radius,
        phase,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, semantically checked residue-reconstruction puzzle."""
    global _GENERATION_INDEX
    phase = _GENERATION_INDEX % 3
    rng = random.Random()
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng, phase)
        if grid is not None:
            _GENERATION_INDEX += 1
            return grid
    raise RuntimeError(f"failed to generate a valid diamond-residue input for phase {phase}")
