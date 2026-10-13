"""ARC task: uniquely dissect a fused island and rotate its pieces into slots."""

from __future__ import annotations

import random
from collections import deque

import numpy as np


Coord = tuple[int, int]


def _components(mask: np.ndarray) -> list[set[Coord]]:
    """Return 4-connected components using the common ARC (row, column) convention."""
    unseen = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result: list[set[Coord]] = []
    while unseen:
        seed = unseen.pop()
        comp = {seed}
        queue = deque([seed])
        while queue:
            r, c = queue.popleft()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in unseen:
                    unseen.remove(q)
                    comp.add(q)
                    queue.append(q)
        result.append(comp)
    return result


def _normalize(cells: set[Coord]) -> frozenset[Coord]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _rotate(shape: frozenset[Coord]) -> frozenset[Coord]:
    return _normalize({(c, -r) for r, c in shape})


def _orientations(shape: frozenset[Coord]) -> list[frozenset[Coord]]:
    result: list[frozenset[Coord]] = []
    cur = shape
    for _ in range(4):
        if cur not in result:
            result.append(cur)
        cur = _rotate(cur)
    return result


def _canonical(shape: frozenset[Coord]) -> tuple[Coord, ...]:
    return min(tuple(sorted(s)) for s in _orientations(shape))


def _candidate_placements(
    slot_shape: frozenset[Coord], island: set[Coord]
) -> list[tuple[frozenset[Coord], frozenset[Coord]]]:
    """All rotated translations of one slot that lie wholly in the island."""
    candidates: list[tuple[frozenset[Coord], frozenset[Coord]]] = []
    seen: set[frozenset[Coord]] = set()
    for oriented in _orientations(slot_shape):
        anchor = min(oriented)
        for ir, ic in island:
            dr, dc = ir - anchor[0], ic - anchor[1]
            placed = frozenset((r + dr, c + dc) for r, c in oriented)
            if placed <= island and placed not in seen:
                seen.add(placed)
                candidates.append((placed, oriented))
    return candidates


def _unique_cover(
    slot_shapes: list[frozenset[Coord]], island: set[Coord], limit: int = 2
) -> list[list[tuple[frozenset[Coord], frozenset[Coord]]]]:
    """Find at most ``limit`` exact covers, one rotated connected piece per slot."""
    candidates = [_candidate_placements(shape, island) for shape in slot_shapes]
    if any(not options for options in candidates):
        return []
    solutions: list[list[tuple[frozenset[Coord], frozenset[Coord]]]] = []

    def search(remaining: tuple[int, ...], used: frozenset[Coord], chosen: list):
        if len(solutions) >= limit:
            return
        if not remaining:
            if set(used) == island:
                solutions.append(chosen.copy())
            return
        viable = {
            i: [candidate for candidate in candidates[i] if candidate[0].isdisjoint(used)]
            for i in remaining
        }
        i = min(remaining, key=lambda j: len(viable[j]))
        if not viable[i]:
            return
        rest = tuple(j for j in remaining if j != i)
        for candidate in viable[i]:
            search(rest, used | candidate[0], chosen + [(i, candidate)])

    search(tuple(range(len(slot_shapes))), frozenset(), [])
    return solutions


def _parse(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, _ = g.shape
    bg = int(g[0, 0])
    assert np.all(g[0] == bg), "the top border must expose the background"
    platform = int(g[-1, 0])
    assert platform != bg and np.all(g[-1] == platform)

    platform_rows = np.argwhere(g == platform)[:, 0]
    top = int(platform_rows.min())
    assert 1 < top < height - 1
    band = g[top:, :]
    holes = [comp for comp in _components(band == bg) if any(r == 0 for r, _ in comp)]
    holes.sort(key=lambda comp: min(c for _, c in comp))
    assert len(holes) >= 2
    assert all(not any(r == band.shape[0] - 1 for r, _ in comp) for comp in holes)

    other_colors = set(map(int, np.unique(g))) - {bg, platform}
    assert len(other_colors) == 1
    island_color = other_colors.pop()
    island_components = _components(g == island_color)
    assert len(island_components) == 1
    island = island_components[0]
    assert max(r for r, _ in island) < top - 1
    slot_shapes = [_normalize(comp) for comp in holes]
    assert sum(map(len, slot_shapes)) == len(island)
    return g, bg, platform, island_color, top, holes, slot_shapes, island


def main(grid: np.ndarray) -> np.ndarray:
    """Dissect the fused floating island and rotate its pieces to fill every platform slot.

    Input structure:
    - A background canvas has a single-color connected island above a solid-color
      bottom platform. Several separated, connected jagged background slots open
      through the platform's top edge, and their total area equals the island area.

    Rule:
    - Find the unique exact partition of the island into one connected piece per
      slot, where pieces may be rotated by 0, 90, 180, or 270 degrees. Erase the
      source island and fill every slot with the island color; preserve the platform.

    Key insight:
    - The invisible cut lines are forced by an exact cover of the fused island by
      rotated copies of the negative-space slot contours, not by separate objects.
    """
    g, bg, _platform, color, top, holes, shapes, island = _parse(grid)
    solutions = _unique_cover(shapes, island)
    assert len(solutions) == 1, "the island must have exactly one valid dissection"

    out = g.copy()
    for r, c in island:
        out[r, c] = bg
    for hole in holes:
        for r, c in hole:
            out[top + r, c] = color
    return out


# Twelve free pentomino-like contours. Every selected slot has area five, so
# area alone cannot pair pieces to slots; their visible jagged contours must.
_SHAPES = tuple(
    _normalize(set(shape))
    for shape in (
        ((0, 1), (0, 2), (1, 0), (1, 1), (2, 1)),  # F
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),  # P
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),  # T
        ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),  # U
        ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),  # V
        ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),  # W
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),  # X
        ((0, 1), (1, 0), (1, 1), (1, 2), (1, 3)),  # Y
        ((0, 0), (0, 1), (0, 2), (1, 2), (1, 3)),  # Z
        ((0, 0), (0, 1), (1, 1), (1, 2), (1, 3)),  # N
        ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),  # L
        ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),  # I
    )
)


def _fuse(pieces: list[frozenset[Coord]], rng: random.Random) -> set[Coord] | None:
    """Attach rotated pieces edge-to-edge while retaining their invisible identities."""
    union = set(pieces[0])
    for piece in pieces[1:]:
        options: list[set[Coord]] = []
        for pr, pc in piece:
            for ur, uc in union:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    shift_r, shift_c = ur + dr - pr, uc + dc - pc
                    placed = {(r + shift_r, c + shift_c) for r, c in piece}
                    if placed.isdisjoint(union):
                        merged = union | placed
                        nr = max(r for r, _ in merged) - min(r for r, _ in merged) + 1
                        nc = max(c for _, c in merged) - min(c for _, c in merged) + 1
                        if nr <= 10 and nc <= 20:
                            options.append(placed)
        if not options:
            return None
        union |= rng.choice(options)
    normalized = _normalize(union)
    return set(normalized)


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.choice((2, 3, 4))
    targets = rng.sample(_SHAPES[:-1], count)
    if len({_canonical(s) for s in targets}) != count:
        return None

    # Independently orient negative-space slots and their hidden source pieces.
    slots = [rng.choice(_orientations(shape)) for shape in targets]
    source_rots = [rng.choice(_orientations(shape)) for shape in targets]
    order = list(range(count))
    rng.shuffle(order)
    fused = _fuse([source_rots[i] for i in order], rng)
    if fused is None or len(_components(_mask_from_cells(fused))) != 1:
        return None

    covers = _unique_cover(slots, fused)
    if len(covers) != 1:
        return None
    chosen = covers[0]
    # Rotation must materially matter in each accepted puzzle batch instance.
    if not any(oriented != slots[i] for i, (_placed, oriented) in chosen):
        return None

    slot_widths = [max(c for _, c in s) + 1 for s in slots]
    slot_heights = [max(r for r, _ in s) + 1 for s in slots]
    separators = [rng.randint(1, 2) for _ in range(count + 1)]
    width = sum(slot_widths) + sum(separators)
    extra_width = rng.randint(0, min(4, 30 - width))
    width += extra_width
    band_h = max(slot_heights) + rng.randint(1, 3)
    island_h = max(r for r, _ in fused) + 1
    island_w = max(c for _, c in fused) + 1
    top_margin = rng.randint(1, 3)
    gap = rng.randint(2, 3)
    height = top_margin + island_h + gap + band_h
    if height > 30 or island_w + 2 > width:
        return None

    palette = list(range(1, 10))
    platform = rng.choice(palette)
    island_color = rng.choice([c for c in palette if c != platform])
    grid = np.zeros((height, width), dtype=int)
    platform_top = height - band_h
    grid[platform_top:, :] = platform

    c = separators[0]
    for i, shape in enumerate(slots):
        for r, dc in shape:
            grid[platform_top + r, c + dc] = 0
        c += slot_widths[i] + separators[i + 1]

    island_r = top_margin
    island_c = rng.randint(1, width - island_w - 1)
    for r, c in fused:
        grid[island_r + r, island_c + c] = island_color

    try:
        parsed = _parse(grid)
        if len(_unique_cover(parsed[-2], parsed[-1])) != 1:
            return None
        out = main(grid)
    except (AssertionError, ValueError):
        return None
    # Independent render oracle: do not accept merely because main closes its own loop.
    expected = grid.copy()
    for r, c0 in fused:
        expected[island_r + r, island_c + c0] = 0
    c0 = separators[0]
    for i, shape in enumerate(slots):
        for r, dc in shape:
            expected[platform_top + r, c0 + dc] = island_color
        c0 += slot_widths[i] + separators[i + 1]
    if not np.array_equal(out, expected):
        return None
    # Semantic checks: source erased, every negative-space slot filled, platform kept.
    if np.count_nonzero(out == island_color) != sum(map(len, slots)):
        return None
    if np.any(out[island_r : island_r + island_h, island_c : island_c + island_w] == island_color):
        return None
    return grid


def _mask_from_cells(cells: set[Coord]) -> np.ndarray:
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def generate_input() -> np.ndarray:
    """Generate varied, oracle-checked instances with two to four equal-area slots."""
    rng = random.Random()
    for _ in range(1200):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a uniquely dissectable fused-island puzzle")
