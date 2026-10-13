"""ARC task: tile a gray mask with a scattered, translation-only inventory."""

from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GRAY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
Coord = tuple[int, int]


def _components(mask: np.ndarray) -> list[set[Coord]]:
    """Return 4-connected components using arc_utils' (row, column) convention."""
    seen = np.zeros(mask.shape, dtype=bool)
    answer: list[set[Coord]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q = deque([(sr, sc)])
        comp: set[Coord] = set()
        while q:
            r, c = q.popleft()
            comp.add((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]
                        and mask[rr, cc] and not seen[rr, cc]):
                    seen[rr, cc] = True
                    q.append((rr, cc))
        answer.append(comp)
    return answer


def _normalize(cells: set[Coord]) -> tuple[Coord, ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _inventory(grid: np.ndarray) -> tuple[set[Coord], list[tuple[int, tuple[Coord, ...]]]]:
    mask_components = _components(grid == GRAY)
    assert len(mask_components) == 1, "expected one connected gray mask"
    mask = mask_components[0]

    pieces: list[tuple[int, tuple[Coord, ...]]] = []
    colors = [int(x) for x in np.unique(grid) if int(x) not in (BG, GRAY)]
    assert len(colors) >= 2, "expected several colored inventory pieces"
    for color in colors:
        comps = _components(grid == color)
        assert len(comps) == 1, "each color must identify exactly one connected piece"
        pieces.append((color, _normalize(comps[0])))
    assert sum(len(shape) for _, shape in pieces) == len(mask), "inventory area must equal mask area"
    return mask, pieces


def _placements(mask: set[Coord], shape: tuple[Coord, ...]) -> list[frozenset[Coord]]:
    """Enumerate translations only: the normalized shape is never rotated or reflected."""
    anchor = shape[0]
    result: set[frozenset[Coord]] = set()
    for mr, mc in mask:
        dr, dc = mr - anchor[0], mc - anchor[1]
        placed = frozenset((r + dr, c + dc) for r, c in shape)
        if placed <= mask:
            result.add(placed)
    return sorted(result, key=lambda p: sorted(p))


def _exact_covers(mask: set[Coord], pieces: list[tuple[int, tuple[Coord, ...]]], limit: int = 2):
    options = [_placements(mask, shape) for _, shape in pieces]
    assert all(options), "every inventory piece must fit inside the mask"
    solutions: list[list[frozenset[Coord]]] = []

    def search(unused: tuple[int, ...], occupied: frozenset[Coord], chosen: dict[int, frozenset[Coord]]) -> None:
        if len(solutions) >= limit:
            return
        if not unused:
            if len(occupied) == len(mask):
                solutions.append([chosen[i] for i in range(len(pieces))])
            return

        # Dynamic minimum-remaining-values ordering keeps irregular puzzles quick.
        best_i = -1
        best_valid: list[frozenset[Coord]] | None = None
        for i in unused:
            valid = [place for place in options[i] if place.isdisjoint(occupied)]
            if not valid:
                return
            if best_valid is None or len(valid) < len(best_valid):
                best_i, best_valid = i, valid
        assert best_valid is not None
        rest = tuple(i for i in unused if i != best_i)
        for place in best_valid:
            chosen[best_i] = place
            search(rest, occupied | place, chosen)
            chosen.pop(best_i, None)
            if len(solutions) >= limit:
                return

    search(tuple(range(len(pieces))), frozenset(), {})
    return solutions, options


def main(grid: np.ndarray) -> np.ndarray:
    """Use every scattered colored piece once to recolor an exactly tiled gray mask.

    Input structure:
    - Black is the background; one solid 4-connected gray region is the target mask.
    - Elsewhere are several mutually separated, single-color 4-connected pieces.
      Every piece has its own color and their combined area equals the gray mask area.

    Rule:
    - Translate every colored piece without rotating or reflecting it, using each exactly
      once, so the pieces cover the gray mask with neither overlap nor gaps.
    - Color each mask cell by the piece covering it, erase every piece at its original
      position to black, and otherwise preserve the grid.  The exact cover is unique.

    Key insight:
    - The scattered shapes are an exhaustible jigsaw inventory: their unchanged shapes
      and their joint, gap-free exact cover determine both placement and mask colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size and max(g.shape) <= 30
    assert np.all((0 <= g) & (g <= 9)), "expected ARC palette values"
    mask, pieces = _inventory(g)
    solutions, _ = _exact_covers(mask, pieces, limit=2)
    assert len(solutions) == 1, "the translation-only exact cover must be unique"

    out = g.copy()
    for color, _shape in pieces:
        out[out == color] = BG
    for (color, _shape), placement in zip(pieces, solutions[0]):
        for r, c in placement:
            out[r, c] = color
    return out


def _random_partition(h: int, w: int, count: int) -> list[set[Coord]] | None:
    """Grow connected regions across a rectangle, then reject tiny inventories."""
    cells = [(r, c) for r in range(h) for c in range(w)]
    seeds = random.sample(cells, count)
    owner: dict[Coord, int] = {cell: i for i, cell in enumerate(seeds)}
    frontiers = [set() for _ in range(count)]

    def refresh(i: int) -> None:
        frontiers[i].clear()
        for r, c in (p for p, who in owner.items() if who == i):
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= q[0] < h and 0 <= q[1] < w and q not in owner:
                    frontiers[i].add(q)

    for i in range(count):
        refresh(i)
    while len(owner) < h * w:
        active = [i for i in range(count) if frontiers[i]]
        if not active:
            return None
        # Favor small regions, while random choice makes boundaries non-template-like.
        sizes = [sum(v == i for v in owner.values()) for i in active]
        m = min(sizes)
        candidates = [i for i, size in zip(active, sizes) if size <= m + 2]
        i = random.choice(candidates)
        q = random.choice(tuple(frontiers[i]))
        owner[q] = i
        for j in range(count):
            refresh(j)
    parts = [{cell for cell, i in owner.items() if i == j} for j in range(count)]
    if min(map(len, parts)) < 3:
        return None
    return parts


def _is_rectangle(cells: set[Coord]) -> bool:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return len(cells) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)


def _dilated(cells: set[Coord], shape: tuple[int, int]) -> set[Coord]:
    h, w = shape
    return {(rr, cc) for r, c in cells for rr in range(max(0, r - 1), min(h, r + 2))
            for cc in range(max(0, c - 1), min(w, c + 2))}


def _scatter(canvas: np.ndarray, shapes: list[tuple[Coord, ...]], colors: list[int], forbidden: set[Coord]) -> bool:
    occupied = set(forbidden)
    for shape, color in zip(shapes, colors):
        ph = max(r for r, _ in shape) + 1
        pw = max(c for _, c in shape) + 1
        choices = [(r0, c0) for r0 in range(1, canvas.shape[0] - ph)
                   for c0 in range(1, canvas.shape[1] - pw)]
        random.shuffle(choices)
        placed_ok = None
        blocked = _dilated(occupied, canvas.shape)
        for r0, c0 in choices:
            placed = {(r0 + r, c0 + c) for r, c in shape}
            if placed.isdisjoint(blocked):
                placed_ok = placed
                break
        if placed_ok is None:
            return False
        for r, c in placed_ok:
            canvas[r, c] = color
        occupied |= placed_ok
    return True


def _attempt() -> np.ndarray | None:
    h = random.randint(4, 7)
    w = random.randint(4, 7)
    count = random.randint(3, min(5, h * w // 4))
    parts = _random_partition(h, w, count)
    if parts is None or sum(not _is_rectangle(p) for p in parts) < 2:
        return None

    # Randomly transpose the entire target and shapes, expanding orientation coverage.
    if random.random() < 0.5:
        parts = [{(c, r) for r, c in p} for p in parts]
        h, w = w, h
    shapes = [_normalize(part) for part in parts]
    if len(set(shapes)) != count:
        return None
    dummy_pieces = [(i + 1, shape) for i, shape in enumerate(shapes)]
    local_mask = {(r, c) for r in range(h) for c in range(w)}
    solutions, options = _exact_covers(local_mask, dummy_pieces, limit=2)
    # Multiple individual fits rule out bbox/one-piece-at-a-time shortcuts; only the
    # joint inventory constraints may select the unique arrangement.
    if len(solutions) != 1 or sum(len(x) >= 2 for x in options) < 2:
        return None

    H = random.randint(max(18, h + 10), 25)
    W = random.randint(max(18, w + 10), 27)
    mr = random.randint(1, H - h - 2)
    mc = random.randint(1, W - w - 2)
    mask_global = {(mr + r, mc + c) for r, c in local_mask}
    canvas = np.zeros((H, W), dtype=int)
    for r, c in mask_global:
        canvas[r, c] = GRAY

    colors = random.sample(PALETTE, count)
    order = list(range(count))
    random.shuffle(order)
    scattered_shapes = [shapes[i] for i in order]
    scattered_colors = [colors[i] for i in order]
    if not _scatter(canvas, scattered_shapes, scattered_colors, mask_global):
        return None

    # Independent planted oracle: color the known partition, not the solver's cover.
    expected = canvas.copy()
    for color in colors:
        expected[expected == color] = BG
    for color, part in zip(colors, parts):
        for r, c in part:
            expected[mr + r, mc + c] = color
    try:
        actual = main(canvas)
        parsed_mask, parsed_pieces = _inventory(canvas)
        parsed_solutions, parsed_options = _exact_covers(parsed_mask, parsed_pieces, limit=2)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(actual, expected) or len(parsed_solutions) != 1:
        return None
    if len(parsed_mask) != sum(len(shape) for _, shape in parsed_pieces):
        return None
    if sum(len(x) >= 2 for x in parsed_options) < 2:
        return None
    # Every inventory item moves and every colored output is confined to the old mask.
    for color in colors:
        if set(map(tuple, np.argwhere(canvas == color))) & set(map(tuple, np.argwhere(actual == color))):
            return None
    return canvas


def generate_input() -> np.ndarray:
    """Generate varied, uniquely solvable exact-cover inventories."""
    for _ in range(1200):
        result = _attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a unique inventory-tiling puzzle")
