"""ARC task: nearest-anchor relay with accumulating quarter turns."""

from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


GRAY = 5
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components of a boolean mask in (row, column) coordinates."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    answer: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue = deque([(r, c)])
        component: list[tuple[int, int]] = []
        while queue:
            rr, cc = queue.popleft()
            component.append((rr, cc))
            for dr, dc in DIR4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        answer.append(component)
    return answer


def _rotate_offset(dr: int, dc: int, quarter_turns: int) -> tuple[int, int]:
    """Rotate a row/column offset clockwise around the origin."""
    for _ in range(quarter_turns % 4):
        dr, dc = dc, -dr
    return dr, dc


def _relay(start: tuple[int, int], targets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Greedily visit the uniquely nearest unused target using squared Euclidean distance."""
    current = start
    unused = set(targets)
    order: list[tuple[int, int]] = []
    while unused:
        distances = sorted(
            ((r - current[0]) ** 2 + (c - current[1]) ** 2, (r, c))
            for r, c in unused
        )
        assert len(distances) == 1 or distances[0][0] < distances[1][0], (
            "nearest unvisited gray anchor must be unique"
        )
        current = distances[0][1]
        order.append(current)
        unused.remove(current)
    return order


def _parse(grid: np.ndarray):
    bg = _background(grid)
    assert bg != GRAY, "gray is reserved for anchors, not the background"
    components = _components(grid != bg)

    sources: list[list[tuple[int, int]]] = []
    isolated_gray: list[tuple[int, int]] = []
    for component in components:
        values = [int(grid[r, c]) for r, c in component]
        if GRAY not in values:
            continue
        if len(component) == 1 and values[0] == GRAY:
            isolated_gray.append(component[0])
        elif any(value != GRAY for value in values):
            sources.append(component)

    assert len(sources) == 1, "expected one multicolor source sprite containing a gray anchor"
    source = sources[0]
    source_anchors = [(r, c) for r, c in source if int(grid[r, c]) == GRAY]
    assert len(source_anchors) == 1, "source sprite must contain exactly one gray anchor"
    assert isolated_gray, "expected at least one isolated gray destination anchor"
    return bg, source, source_anchors[0], isolated_gray


def main(grid: np.ndarray) -> np.ndarray:
    """Copy a multicolor sprite along a nearest-gray-anchor relay, adding one clockwise quarter-turn at every station.

    Input structure:
    - The dominant background contains one 4-connected multicolor source sprite with exactly one gray anchor.
    - Two or more other gray pixels are isolated one-cell components and serve as destination anchors.

    Rule:
    - Starting at the source anchor, repeatedly select the Euclidean-nearest unvisited isolated gray anchor; every nearest choice is unique.
    - At relay station k (counting the first destination as 1), rotate the complete source sprite clockwise by 90*k degrees about its own anchor, then translate it so that anchor coincides with the station.
    - Draw each copy over background, retain every anchor in gray, and leave the source sprite unchanged.

    Key insight:
    - The gray destinations form a greedy relay path rather than an unordered set: orientation accumulates with relay order, which can differ from distance order measured only from the source.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional ARC grid"
    bg, source, anchor, targets = _parse(g)
    order = _relay(anchor, targets)

    sprite = [(r - anchor[0], c - anchor[1], int(g[r, c])) for r, c in source]
    out = g.copy()
    h, w = g.shape
    for station, (tr, tc) in enumerate(order, start=1):
        for dr, dc, color in sprite:
            rr_off, cc_off = _rotate_offset(dr, dc, station)
            rr, cc = tr + rr_off, tc + cc_off
            assert 0 <= rr < h and 0 <= cc < w, "rotated copy must remain in bounds"
            assert int(g[rr, cc]) in (bg, GRAY), "copies must be placed over background anchors"
            out[rr, cc] = color
    return out


def _greedy_unique(start: tuple[int, int], targets: list[tuple[int, int]]):
    current = start
    unused = set(targets)
    order = []
    while unused:
        ranked = sorted(
            ((r - current[0]) ** 2 + (c - current[1]) ** 2, (r, c))
            for r, c in unused
        )
        if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
            return None
        current = ranked[0][1]
        order.append(current)
        unused.remove(current)
    return order


def _stamp(
    base: np.ndarray,
    sprite: list[tuple[int, int, int]],
    anchor: tuple[int, int],
    turns: int,
) -> set[tuple[int, int]] | None:
    h, w = base.shape
    cells: set[tuple[int, int]] = set()
    for dr, dc, color in sprite:
        dr2, dc2 = _rotate_offset(dr, dc, turns)
        r, c = anchor[0] + dr2, anchor[1] + dc2
        if not (0 <= r < h and 0 <= c < w):
            return None
        base[r, c] = color
        cells.add((r, c))
    return cells


def _render_semantic(
    shape: tuple[int, int],
    bg: int,
    sprite: list[tuple[int, int, int]],
    source_anchor: tuple[int, int],
    order: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    assert _stamp(out, sprite, source_anchor, 0) is not None
    for k, anchor in enumerate(order, 1):
        assert _stamp(out, sprite, anchor, k) is not None
    return out


def _random_sprite(colors: list[int]) -> list[tuple[int, int, int]] | None:
    """Make a connected, asymmetric sprite whose anchor-relative rotations are visible."""
    cells = {(0, 0)}
    while len(cells) < random.randint(7, 10):
        r, c = random.choice(tuple(cells))
        dr, dc = random.choice(DIR4)
        candidate = (r + dr, c + dc)
        if max(abs(candidate[0]), abs(candidate[1])) <= 2:
            cells.add(candidate)
    if len(cells) < 7:
        return None

    ordered = sorted(cells)
    sprite = []
    for i, (dr, dc) in enumerate(ordered):
        color = GRAY if (dr, dc) == (0, 0) else colors[i % len(colors)]
        sprite.append((dr, dc, color))

    signatures = []
    for turns in range(4):
        signatures.append(
            frozenset((*_rotate_offset(dr, dc, turns), color) for dr, dc, color in sprite)
        )
    return sprite if len(set(signatures)) == 4 else None


def _disjoint_stamp_sets(
    shape: tuple[int, int],
    sprite: list[tuple[int, int, int]],
    source: tuple[int, int],
    order: list[tuple[int, int]],
) -> bool:
    occupied: set[tuple[int, int]] = set()
    dummy = np.zeros(shape, dtype=int)
    for turns, anchor in [(0, source), *[(k, p) for k, p in enumerate(order, 1)]]:
        cells = _stamp(dummy, sprite, anchor, turns)
        if cells is None or occupied & cells:
            return False
        occupied |= cells
    return True


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    bg: int,
    sprite: list[tuple[int, int, int]],
    source: tuple[int, int],
    targets: list[tuple[int, int]],
    order: list[tuple[int, int]],
) -> bool:
    """Reject ambiguity and the two principal shortcuts independently of main()."""
    radial = sorted(targets, key=lambda p: (p[0] - source[0]) ** 2 + (p[1] - source[1]) ** 2)
    if order == radial:
        return False
    if len(order) < 4 or not _disjoint_stamp_sets(inp.shape, sprite, source, order):
        return False

    # Wrong rule 1: every target receives an unrotated source.
    unrotated = np.full(inp.shape, bg, dtype=int)
    _stamp(unrotated, sprite, source, 0)
    for target in targets:
        _stamp(unrotated, sprite, target, 0)

    # Wrong rule 2: rotation index is assigned by distance from the source.
    radial_output = _render_semantic(inp.shape, bg, sprite, source, radial)
    if np.array_equal(expected, unrotated) or np.array_equal(expected, radial_output):
        return False

    # Input anchors are genuinely isolated and output retains all anchors gray.
    for r, c in targets:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr, dc) != (0, 0) and int(inp[r + dr, c + dc]) != bg:
                    return False
        if int(expected[r, c]) != GRAY:
            return False
    return np.array_equal(main(inp), expected) and not np.array_equal(inp, expected)


def _generate_attempt() -> np.ndarray | None:
    h, w = random.randint(22, 30), random.randint(22, 30)
    bg = random.choice([0, 1, 2, 3, 4, 6, 7, 8, 9])
    palette = [c for c in range(10) if c not in (bg, GRAY)]
    sprite_colors = random.sample(palette, random.randint(2, 3))
    sprite = _random_sprite(sprite_colors)
    if sprite is None:
        return None

    margin = 3
    station_count = random.randint(4, 6)
    anchors: list[tuple[int, int]] = []
    candidates = [(r, c) for r in range(margin, h - margin) for c in range(margin, w - margin)]
    random.shuffle(candidates)
    for candidate in candidates:
        # Wide separation keeps both input markers and output sprites visually distinct.
        if all(max(abs(candidate[0] - r), abs(candidate[1] - c)) >= 6 for r, c in anchors):
            anchors.append(candidate)
            if len(anchors) == station_count + 1:
                break
    if len(anchors) != station_count + 1:
        return None

    source, targets = anchors[0], anchors[1:]
    order = _greedy_unique(source, targets)
    if order is None:
        return None
    radial_distances = [(r - source[0]) ** 2 + (c - source[1]) ** 2 for r, c in targets]
    if len(radial_distances) != len(set(radial_distances)):
        return None
    if not _disjoint_stamp_sets((h, w), sprite, source, order):
        return None

    expected = _render_semantic((h, w), bg, sprite, source, order)
    inp = np.full((h, w), bg, dtype=int)
    if _stamp(inp, sprite, source, 0) is None:
        return None
    for r, c in targets:
        inp[r, c] = GRAY

    counts = Counter(map(int, inp.ravel()))
    if counts[bg] <= max(v for k, v in counts.items() if k != bg):
        return None
    try:
        if not _semantic_checks(inp, expected, bg, sprite, source, targets, order):
            return None
    except (AssertionError, IndexError):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate diverse, non-degenerate relay instances with explicit shortcut counterexamples."""
    for _ in range(1000):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nearest-anchor relay input")
