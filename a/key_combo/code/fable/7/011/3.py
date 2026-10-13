"""ARC task: edge beacons independently constrain a sprite's hole row and column."""

from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]


def _components4(mask: np.ndarray) -> list[list[Coord]]:
    """Return 4-connected True components using the shared ARC (row, col) convention."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[list[Coord]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue = deque([(sr, sc)])
        component: list[Coord] = []
        while queue:
            r, c = queue.popleft()
            component.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(component)
    return result


def _single_hole(grid: np.ndarray, component: list[Coord], bg: int) -> Coord:
    """Find the unique one-cell enclosed background region inside a component's bbox."""
    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    enclosed: list[list[Coord]] = []
    for region in _components4(patch == bg):
        if any(r == 0 or c == 0 or r == patch.shape[0] - 1 or c == patch.shape[1] - 1
               for r, c in region):
            continue
        enclosed.append(region)
    assert len(enclosed) == 1 and len(enclosed[0]) == 1, (
        "each internal sprite must enclose exactly one one-cell background hole"
    )
    hr, hc = enclosed[0][0]
    return r0 + hr, c0 + hc


def _parse(grid: np.ndarray):
    """Parse background, edge constraints, and monochromatic internal sprites."""
    bg = int(Counter(map(int, grid.ravel())).most_common(1)[0][0])
    top: dict[int, int] = {}
    left: dict[int, int] = {}
    for c in range(1, grid.shape[1]):
        color = int(grid[0, c])
        if color != bg:
            assert color not in top, "a color may occur at most once on the top edge"
            top[color] = c
    for r in range(1, grid.shape[0]):
        color = int(grid[r, 0])
        if color != bg:
            assert color not in left, "a color may occur at most once on the left edge"
            left[color] = r

    sprites = []
    internal_colors = set(map(int, np.unique(grid[1:, 1:]))) - {bg}
    for color in internal_colors:
        mask = grid == color
        mask[0, :] = False
        mask[:, 0] = False
        for component in _components4(mask):
            sprites.append((color, component, _single_hole(grid, component, bg)))
    assert sprites, "expected at least one internal sprite"
    return bg, top, left, sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each monochromatic sprite so its hole satisfies its same-color edge beacons.

    Input structure:
    - The dominant color is background; isolated colored beacons lie on the top row
      and left column, at most once per color on each edge.
    - One or more monochromatic 4-connected sprites lie strictly inside the grid,
      and each encloses exactly one single background-colored hole.

    Rule:
    - A top-edge beacon fixes the column of every same-color sprite's hole, while a
      left-edge beacon fixes its row. Translate the entire sprite by the conjunction
      of its zero, one, or two available axis constraints, preserving shape and color.
    - An unconstrained axis keeps its original hole coordinate. Clear old sprite
      pixels to background and preserve all edge beacons.

    Key insight:
    - The two edges provide independent one-dimensional constraints; the invisible
      row-column intersection is recovered only by aligning the sprite's unique hole.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, top, left, sprites = _parse(g)
    h, w = g.shape

    out = g.copy()
    for _, component, _ in sprites:
        for r, c in component:
            out[r, c] = bg

    occupied: set[Coord] = set()
    placements: list[tuple[int, list[Coord]]] = []
    for color, component, (hole_r, hole_c) in sprites:
        dr = left.get(color, hole_r) - hole_r
        dc = top.get(color, hole_c) - hole_c
        shifted = [(r + dr, c + dc) for r, c in component]
        assert all(1 <= r < h and 1 <= c < w for r, c in shifted), (
            "a constrained sprite must remain strictly inside the grid"
        )
        assert not occupied.intersection(shifted), "translated sprites must not overlap"
        assert all(out[r, c] == bg for r, c in shifted), (
            "translated sprites must not cover retained non-background cells"
        )
        occupied.update(shifted)
        placements.append((color, shifted))

    for color, shifted in placements:
        for r, c in shifted:
            out[r, c] = color
    return out


_BASE_SHAPES = (
    ((-2, -1),),
    ((-1, 2), (0, 2)),
    ((1, -2), (0, -2)),
    ((2, 0), (2, 1)),
    ((-2, 0), (0, 2)),
)


def _random_shape() -> set[Coord]:
    """Make an asymmetric connected sprite around a one-cell hole at (0, 0)."""
    shape = {(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr, dc) != (0, 0)}
    shape.update(random.choice(_BASE_SHAPES))
    turns = random.randrange(4)
    for _ in range(turns):
        shape = {(c, -r) for r, c in shape}
    if random.random() < 0.5:
        shape = {(r, -c) for r, c in shape}
    return shape


def _separated(groups: list[set[Coord]]) -> bool:
    """Require a visible one-cell background moat between different sprites."""
    for i, first in enumerate(groups):
        expanded = {(r + dr, c + dc) for r, c in first
                    for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        if any(expanded.intersection(other) for other in groups[i + 1 :]):
            return False
    return True


def _different_coordinate(value: int, low: int, high: int) -> int:
    choices = [x for x in range(low, high + 1) if abs(x - value) >= 3]
    return random.choice(choices)


def _independent_expected(
    inp: np.ndarray,
    records: list[tuple[int, set[Coord], Coord, Coord, str]],
) -> np.ndarray:
    """Construct the expected answer from generation metadata, independently of parsing."""
    expected = inp.copy()
    for _, shape, initial, _, _ in records:
        ir, ic = initial
        for dr, dc in shape:
            expected[ir + dr, ic + dc] = 0
    for color, shape, _, target, _ in records:
        tr, tc = target
        for dr, dc in shape:
            expected[tr + dr, tc + dc] = color
    return expected


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    records: list[tuple[int, set[Coord], Coord, Coord, str]],
) -> bool:
    """Check branch semantics and show that every painted beacon is result-critical."""
    if not np.array_equal(main(inp), expected):
        return False
    branch_names = {record[4] for record in records}
    if branch_names != {"both", "column", "row", "none"}:
        return False
    for color, shape, (ir, ic), (tr, tc), branch in records:
        if branch == "both" and not (ir != tr and ic != tc):
            return False
        if branch == "column" and not (ir == tr and ic != tc):
            return False
        if branch == "row" and not (ir != tr and ic == tc):
            return False
        if branch == "none" and not (ir == tr and ic == tc):
            return False
        # The unique missing center is still the hole after translation.
        if expected[tr, tc] != 0 or any(expected[tr + dr, tc + dc] != color for dr, dc in shape):
            return False

        # Removing a real beacon must change that sprite's result on its axis.
        if branch in {"both", "column"}:
            counter = inp.copy()
            counter[0, tc] = 0
            if np.array_equal(main(counter), expected):
                return False
        if branch in {"both", "row"}:
            counter = inp.copy()
            counter[tr, 0] = 0
            if np.array_equal(main(counter), expected):
                return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(20, 28)
    w = random.randint(20, 28)
    sprite_count = random.randint(4, 5)
    colors = random.sample(range(1, 10), sprite_count)
    branches = ["both", "column", "row", "none"]
    branches.extend(random.choice(("both", "column", "row", "none"))
                    for _ in range(sprite_count - 4))
    random.shuffle(branches)
    shapes = [_random_shape() for _ in branches]

    for _ in range(500):
        records: list[tuple[int, set[Coord], Coord, Coord, str]] = []
        used_top_cols: set[int] = set()
        used_left_rows: set[int] = set()
        valid = True
        for color, shape, branch in zip(colors, shapes, branches):
            ir = random.randint(4, h - 5)
            ic = random.randint(4, w - 5)
            tr, tc = ir, ic
            if branch in {"both", "row"}:
                tr = _different_coordinate(ir, 4, h - 5)
            if branch in {"both", "column"}:
                tc = _different_coordinate(ic, 4, w - 5)
            if branch in {"both", "column"} and tc in used_top_cols:
                valid = False
                break
            if branch in {"both", "row"} and tr in used_left_rows:
                valid = False
                break
            if branch in {"both", "column"}:
                used_top_cols.add(tc)
            if branch in {"both", "row"}:
                used_left_rows.add(tr)
            records.append((color, shape, (ir, ic), (tr, tc), branch))
        if not valid:
            continue

        starts = [{(ir + dr, ic + dc) for dr, dc in shape}
                  for _, shape, (ir, ic), _, _ in records]
        finishes = [{(tr + dr, tc + dc) for dr, dc in shape}
                    for _, shape, _, (tr, tc), _ in records]
        if not _separated(starts) or not _separated(finishes):
            continue

        inp = np.zeros((h, w), dtype=int)
        for color, shape, (ir, ic), (tr, tc), branch in records:
            for dr, dc in shape:
                inp[ir + dr, ic + dc] = color
            if branch in {"both", "column"}:
                inp[0, tc] = color
            if branch in {"both", "row"}:
                inp[tr, 0] = color
        expected = _independent_expected(inp, records)
        try:
            if _semantic_checks(inp, expected, records):
                return inp
        except (AssertionError, ValueError):
            pass
    return None


def generate_input() -> np.ndarray:
    """Generate a diverse, nondegenerate example covering all four constraint branches."""
    for _ in range(100):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid edge-beacon puzzle")
