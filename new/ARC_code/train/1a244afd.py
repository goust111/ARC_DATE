from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, coords_where, get_bg, in_bounds
import random

Coord = tuple[int, int]


def _coords(mask: np.ndarray) -> list[Coord]:
    return [(int(r), int(c)) for r, c in coords_where(mask)]


def _unique_row_col_pairs(sources: list[Coord], pivots: list[Coord]) -> list[tuple[Coord, Coord]]:
    assert len(sources) == len(pivots) > 0, "expected equal nonzero source and pivot counts"
    candidates = {
        s: [p for p in pivots if p[0] == s[0] or p[1] == s[1]]
        for s in sources
    }
    assert all(candidates[s] for s in sources), "each source must align with a pivot"

    ordered_sources = sorted(sources, key=lambda s: (len(candidates[s]), s[0], s[1]))
    solutions: list[dict[Coord, Coord]] = []

    def search(i: int, used: set[Coord], pairs: dict[Coord, Coord]) -> None:
        if len(solutions) > 1:
            return
        if i == len(ordered_sources):
            solutions.append(pairs.copy())
            return

        s = ordered_sources[i]
        for p in candidates[s]:
            if p in used:
                continue
            used.add(p)
            pairs[s] = p
            search(i + 1, used, pairs)
            del pairs[s]
            used.remove(p)

    search(0, set(), {})
    assert len(solutions) == 1, "expected a unique one-to-one row/column matching"
    return [(s, solutions[0][s]) for s in sources]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each magenta source pixel 90 degrees counterclockwise around its matched blue pivot, recoloring the new pixel orange.

    Input structure:
    - The background is the dominant color.
    - The only foreground colors are blue pivot pixels and magenta source pixels.
    - Each magenta source can be paired one-to-one with a blue pivot in the same
      row or column.

    Rule:
    - Remove all magenta source pixels.
    - For each matched source/pivot pair, preserve the source distance from the
      pivot but rotate its offset 90 degrees counterclockwise.
    - Paint the rotated in-bounds target orange; if the target is outside the
      grid, the source simply disappears.

    Key insight:
    The blue pixels are rotation centers, and row/column alignment gives a
    unique global matching from each magenta source to its pivot.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert set(colors_present(g)) == {bg, Color.BLUE, Color.PINK}, "unexpected color set"

    sources = _coords(g == Color.PINK)
    pivots = _coords(g == Color.BLUE)
    pairs = _unique_row_col_pairs(sources, pivots)

    out = g.copy()
    out[g == Color.PINK] = bg

    for (sr, sc), (pr, pc) in pairs:
        dr = sr - pr
        dc = sc - pc
        tr = pr - dc
        tc = pc + dr
        if in_bounds(g.shape, tr, tc):
            assert out[tr, tc] == bg, "rotated targets must land on background"
            out[tr, tc] = Color.ORANGE

    return out

# --------------- generator (o5x) ---------------

Triple = tuple[Coord, Coord, Coord | None, bool]

# Canvas, colors, and retry budget.
GRID_SIZE_RANGE = (8, 16)
BACKGROUND_COLOR = Color.TEAL
GENERATION_RETRIES = 300

# Pair counts and geometric offsets.
PAIR_COUNT_RANGE = (3, 7)
SOURCE_DISTANCE_RANGE = (1, 4)
SOURCE_OFFSETS = tuple(
    offset for d in range(SOURCE_DISTANCE_RANGE[0], SOURCE_DISTANCE_RANGE[1] + 1) for offset in ((0, -d), (0, d), (-d, 0), (d, 0))
)

# Layout mix.
MAX_OOB_TARGETS = 2
SAME_ROW_PROBABILITY = 0.3
MIN_SAME_ROW_PAIRS = 1


def _matching_count(sources: list[Coord], pivots: list[Coord]) -> int:
    candidates = {
        s: [p for p in pivots if p[0] == s[0] or p[1] == s[1]]
        for s in sources
    }
    if any(not ps for ps in candidates.values()):
        return 0

    ordered = sorted(sources, key=lambda s: (len(candidates[s]), s[0], s[1]))
    count = 0

    def search(i: int, used: set[Coord]) -> None:
        nonlocal count
        if count > 1:
            return
        if i == len(ordered):
            count += 1
            return
        for pivot in candidates[ordered[i]]:
            if pivot in used:
                continue
            used.add(pivot)
            search(i + 1, used)
            used.remove(pivot)

    search(0, set())
    return count


def _all_triples(n: int, want_oob: bool, want_same_row: bool) -> list[Triple]:
    triples: list[Triple] = []
    for pr in range(n):
        for pc in range(n):
            for dr, dc in SOURCE_OFFSETS:
                same_row = dr == 0
                if same_row != want_same_row:
                    continue
                sr, sc = pr + dr, pc + dc
                if not in_bounds((n, n), sr, sc):
                    continue
                tr, tc = pr - dc, pc + dr
                target: Coord | None = (tr, tc) if in_bounds((n, n), tr, tc) else None
                if (target is None) != want_oob:
                    continue
                triples.append(((pr, pc), (sr, sc), target, same_row))
    random.shuffle(triples)
    return triples


def _compatible(
    triple: Triple,
    pivots: list[Coord],
    sources: list[Coord],
    targets: set[Coord],
) -> bool:
    pivot, source, target, _ = triple
    input_cells = set(pivots) | set(sources)
    if pivot in input_cells or source in input_cells or pivot == source:
        return False

    for old_source in sources:
        if old_source[0] == pivot[0] or old_source[1] == pivot[1]:
            return False
    for old_pivot in pivots:
        if source[0] == old_pivot[0] or source[1] == old_pivot[1]:
            return False

    if target is not None and (target in targets or target in input_cells or target in (pivot, source)):
        return False
    return True


def _sample_requirements(k: int) -> list[tuple[bool, bool]]:
    """Return shuffled per-pair requirements: (target_is_oob, source_same_row)."""
    oob_goal = random.randint(0, min(MAX_OOB_TARGETS, k - 1))
    same_row_count = sum(1 for _ in range(k) if random.random() < SAME_ROW_PROBABILITY)
    same_row_count = max(MIN_SAME_ROW_PAIRS, min(k - 1, same_row_count))
    oob_flags = [True] * oob_goal + [False] * (k - oob_goal)
    same_row_flags = [True] * same_row_count + [False] * (k - same_row_count)
    random.shuffle(oob_flags)
    random.shuffle(same_row_flags)
    return list(zip(oob_flags, same_row_flags))


def _place_pairs(
    n: int,
    requirements: list[tuple[bool, bool]],
) -> tuple[list[Coord], list[Coord], set[Coord]] | None:
    """Build non-ambiguous pivot/source/target sets that satisfy requirements."""
    pivots: list[Coord] = []
    sources: list[Coord] = []
    targets: set[Coord] = set()
    triples_by_req: dict[tuple[bool, bool], list[Triple]] = {}

    for want_oob, want_same_row in requirements:
        key = (want_oob, want_same_row)
        if key not in triples_by_req:
            triples_by_req[key] = _all_triples(n, want_oob, want_same_row)
        choices = [
            triple
            for triple in triples_by_req[key]
            if _compatible(triple, pivots, sources, targets)
        ]
        if not choices:
            return None
        pivot, source, target, _ = random.choice(choices)
        pivots.append(pivot)
        sources.append(source)
        if target is not None:
            targets.add(target)

    if _matching_count(sources, pivots) != 1:
        return None
    return pivots, sources, targets


def _build_grids(n: int, pivots: list[Coord], sources: list[Coord], targets: set[Coord]):
    """Render the input grid and expected output grid for a valid layout."""
    inp = np.full((n, n), BACKGROUND_COLOR, dtype=int)
    out = np.full((n, n), BACKGROUND_COLOR, dtype=int)
    for r, c in pivots:
        inp[r, c] = Color.BLUE
        out[r, c] = Color.BLUE
    for r, c in sources:
        inp[r, c] = Color.PINK
    for r, c in targets:
        out[r, c] = Color.ORANGE
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_RANGE[0], GRID_SIZE_RANGE[1])
    k = random.randint(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1])
    placed = _place_pairs(n, _sample_requirements(k))
    if placed is None:
        return None
    inp, expected = _build_grids(n, *placed)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 1a244afd input")
