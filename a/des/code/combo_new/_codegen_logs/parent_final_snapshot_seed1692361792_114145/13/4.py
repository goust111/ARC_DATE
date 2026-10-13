from __future__ import annotations

from collections import Counter, deque

import numpy as np
from numpy.random import Generator, default_rng


BACKGROUND = 0
PALETTE = tuple(range(10))
GENERATE_ATTEMPTS = 400
GROUP_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 120
SIDE_NAMES = ("top", "bottom", "left", "right")
CORNER_NAMES = ("tl", "tr", "bl", "br")


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(cells)
    return parts


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _erode4(mask: np.ndarray, steps: int = 1) -> np.ndarray:
    cur = mask.copy()
    for _ in range(steps):
        nxt = np.zeros_like(cur, dtype=bool)
        if cur.shape[0] < 3 or cur.shape[1] < 3:
            return nxt
        nxt[1:-1, 1:-1] = (
            cur[1:-1, 1:-1]
            & cur[:-2, 1:-1]
            & cur[2:, 1:-1]
            & cur[1:-1, :-2]
            & cur[1:-1, 2:]
        )
        cur = nxt
    return cur


def _boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~_erode4(mask, 1)


def _interior_of_ring(cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(cells)
    local_h = r1 - r0 + 3
    local_w = c1 - c0 + 3
    blocked = np.zeros((local_h, local_w), dtype=bool)
    for r, c in cells:
        blocked[r - r0 + 1, c - c0 + 1] = True

    reached = np.zeros_like(blocked, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(local_h):
        for c in range(local_w):
            if r not in (0, local_h - 1) and c not in (0, local_w - 1):
                continue
            if not blocked[r, c] and not reached[r, c]:
                reached[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, local_h, local_w):
            if blocked[nr, nc] or reached[nr, nc]:
                continue
            reached[nr, nc] = True
            q.append((nr, nc))

    interior: set[tuple[int, int]] = set()
    for r in range(1, local_h - 1):
        for c in range(1, local_w - 1):
            if not blocked[r, c] and not reached[r, c]:
                interior.add((r0 + r - 1, c0 + c - 1))
    return interior


def main(grid: np.ndarray) -> np.ndarray:
    """
    Peel nested monochrome ring contours from outside to inside and recolor each ring with its own marker color.

    Input structure:
    - The background is black.
    - One shared non-black color draws several closed 4-connected ring contours;
      separate nesting stacks may appear in the same grid.
    - Every ring has exactly one single-pixel marker somewhere in its own
      interior zone, while deeper rings may sit further inside that zone.

    Rule:
    - Identify the common ring color and split it into connected contour layers.
    - For each marker, find the smallest enclosed contour whose interior contains
      that marker; recolor that whole contour to the marker color.
    - Erase every marker pixel to black and leave the remaining background alone.

    Key insight:
    - All layers share the same contour color, so layer ownership comes from
      containment depth: a marker belongs to the smallest ring that surrounds it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    counts = Counter(map(int, g.flatten()))
    fg_counts = {color: count for color, count in counts.items() if color != BACKGROUND}
    assert fg_counts, "expected foreground content"

    max_count = max(fg_counts.values())
    ring_candidates = [color for color, count in fg_counts.items() if count == max_count]
    assert len(ring_candidates) == 1, "expected a unique ring color"
    ring_color = ring_candidates[0]

    ring_components = _components(g == ring_color)
    assert ring_components, "expected at least one ring contour"

    ring_interiors: list[set[tuple[int, int]]] = []
    for cells in ring_components:
        interior = _interior_of_ring(cells)
        assert interior, "each ring contour must enclose an interior"
        ring_interiors.append(interior)

    marker_cells = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere((g != BACKGROUND) & (g != ring_color))]
    assert marker_cells, "expected marker pixels"

    assignments: dict[int, int] = {}
    for r, c, color in marker_cells:
        containing = [idx for idx, interior in enumerate(ring_interiors) if (r, c) in interior]
        assert containing, "every marker must be inside at least one ring"
        owner = min(containing, key=lambda idx: len(ring_interiors[idx]))
        assert owner not in assignments, "each ring must have exactly one marker"
        assignments[owner] = color

    assert len(assignments) == len(ring_components), "every ring must receive one marker color"

    out = g.copy()
    for r, c, _ in marker_cells:
        out[r, c] = BACKGROUND
    for idx, cells in enumerate(ring_components):
        color = assignments[idx]
        for r, c in cells:
            out[r, c] = color
    return out


def _is_connected(mask: np.ndarray) -> bool:
    cells = int(mask.sum())
    return cells > 0 and len(_components(mask)) == 1


def _carve_side_bite(mask: np.ndarray, rng: Generator, side: str) -> None:
    h, w = mask.shape
    if side in ("top", "bottom"):
        if h < 8 or w < 6:
            return
        depth = int(rng.integers(1, 3))
        width = int(rng.integers(2, max(3, w - 1)))
        if width >= w:
            width = w - 1
        start = int(rng.integers(1, w - width + 1))
        rows = slice(0, depth) if side == "top" else slice(h - depth, h)
        mask[rows, start : start + width] = False
    else:
        if h < 6 or w < 8:
            return
        depth = int(rng.integers(1, 3))
        height = int(rng.integers(2, max(3, h - 1)))
        if height >= h:
            height = h - 1
        start = int(rng.integers(1, h - height + 1))
        cols = slice(0, depth) if side == "left" else slice(w - depth, w)
        mask[start : start + height, cols] = False


def _carve_corner(mask: np.ndarray, rng: Generator, corner: str) -> None:
    h, w = mask.shape
    if h < 6 or w < 6:
        return
    ch = int(rng.integers(1, min(3, h - 2) + 1))
    cw = int(rng.integers(1, min(3, w - 2) + 1))
    rows = slice(0, ch) if "t" in corner else slice(h - ch, h)
    cols = slice(0, cw) if "l" in corner else slice(w - cw, w)
    mask[rows, cols] = False


def _make_outer_fill(rng: Generator, layers: int) -> np.ndarray | None:
    min_dim = 4 * layers + 3
    max_dim = min(4 * layers + 7, 23)
    h = int(rng.integers(min_dim, max_dim + 1))
    w = int(rng.integers(min_dim, max_dim + 1))
    mask = np.ones((h, w), dtype=bool)

    bite_count = int(rng.integers(0, 5))
    for _ in range(bite_count):
        if rng.random() < 0.7:
            _carve_side_bite(mask, rng, str(rng.choice(SIDE_NAMES)))
        else:
            _carve_corner(mask, rng, str(rng.choice(CORNER_NAMES)))

    if not _is_connected(mask):
        return None
    return mask


def _build_group(rng: Generator, ring_color: int, marker_colors: list[int]) -> np.ndarray | None:
    layers = len(marker_colors)
    for _ in range(GROUP_ATTEMPTS):
        fills: list[np.ndarray] = []
        outer = _make_outer_fill(rng, layers)
        if outer is None:
            continue
        fills.append(outer)
        ok = True
        for _ in range(layers - 1):
            inner = _erode4(fills[-1], 2)
            if not _is_connected(inner):
                ok = False
                break
            fills.append(inner)
        if not ok:
            continue

        rings = [_boundary(fill_mask) for fill_mask in fills]
        if any(not _is_connected(ring) for ring in rings):
            continue

        zones: list[np.ndarray] = []
        for idx, fill_mask in enumerate(fills):
            zone = fill_mask & ~rings[idx]
            if idx + 1 < layers:
                zone &= ~fills[idx + 1]
            if not zone.any():
                ok = False
                break
            zones.append(zone)
        if not ok:
            continue

        group = np.zeros_like(outer, dtype=int)
        for ring in rings:
            group[ring] = ring_color
        for color, zone in zip(marker_colors, zones):
            choices = np.argwhere(zone)
            pick = choices[int(rng.integers(len(choices)))]
            group[int(pick[0]), int(pick[1])] = int(color)
        return group
    return None


def _choose_layer_counts(rng: Generator, group_count: int) -> list[int]:
    if group_count == 1:
        return [int(rng.choice([2, 3, 4]))]
    pair = rng.choice([(2, 2), (2, 3), (3, 2), (3, 3)])
    return [int(pair[0]), int(pair[1])]


def _place_groups(groups: list[np.ndarray], h: int, w: int, rng: Generator) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    order = list(range(len(groups)))
    rng.shuffle(order)
    for idx in order:
        group = groups[idx]
        gh, gw = group.shape
        candidates: list[tuple[int, int]] = []
        for r in range(1, h - gh):
            for c in range(1, w - gw):
                r0 = r - 1
                c0 = c - 1
                r1 = r + gh + 1
                c1 = c + gw + 1
                if reserved[r0:r1, c0:c1].any():
                    continue
                candidates.append((r, c))
        if not candidates:
            return None

        r, c = candidates[int(rng.integers(len(candidates)))]
        patch = grid[r : r + gh, c : c + gw]
        patch[group != BACKGROUND] = group[group != BACKGROUND]
        reserved[r : r + gh, c : c + gw] = True
    return grid


def _gen_attempt(rng: Generator) -> np.ndarray | None:
    group_count = int(rng.choice([1, 2]))
    layer_counts = _choose_layer_counts(rng, group_count)
    total_layers = sum(layer_counts)

    ring_color = int(rng.choice([color for color in PALETTE if color != BACKGROUND]))
    marker_pool = [color for color in PALETTE if color not in (BACKGROUND, ring_color)]
    if total_layers > len(marker_pool):
        return None
    marker_colors = list(map(int, rng.choice(marker_pool, size=total_layers, replace=False)))

    groups: list[np.ndarray] = []
    cursor = 0
    for layers in layer_counts:
        group = _build_group(rng, ring_color, marker_colors[cursor : cursor + layers])
        if group is None:
            return None
        cursor += layers
        groups.append(group)

    min_side = 18 if group_count == 1 else 22
    max_h = min(30, max(min_side, max(group.shape[0] for group in groups) + 8))
    max_w = min(30, max(min_side, max(group.shape[1] for group in groups) + 8))
    base_h = max(max(group.shape[0] for group in groups) + 2, min_side)
    base_w = max(max(group.shape[1] for group in groups) + 2, min_side)
    if group_count == 2:
        base_h = max(base_h, min(30, sum(group.shape[0] for group in groups) // 2 + 8))
        base_w = max(base_w, min(30, sum(group.shape[1] for group in groups) // 2 + 8))
    if base_h > 30 or base_w > 30:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        h = int(rng.integers(base_h, max_h + 1))
        w = int(rng.integers(base_w, max_w + 1))
        grid = _place_groups(groups, h, w, rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = default_rng()
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATE_ATTEMPTS} attempts")
