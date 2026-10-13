import random
from collections import Counter

import numpy as np


_RING = (
    (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 1), (1, 4),
    (2, 1), (2, 4),
    (3, 1), (3, 4),
    (4, 1), (4, 4),
    (5, 1), (5, 2), (5, 3), (5, 4),
)
_RING_SET = set(_RING)
_RING_H = 6
_RING_W = 6

_GEN_ATTEMPTS = 500
_GEN_RING_COUNT = (3, 5)
_GEN_TOP_H = (7, 10)
_GEN_WIDTH = (14, 18)
_GEN_VISIBLE = (5, 13)
_GEN_NOISE = (1, 3)
_PALETTE = tuple(range(1, 10))


def _ring_cells(offset, shape):
    r0, c0 = offset
    h, w = shape
    cells = set()
    for dr, dc in _RING:
        rr = r0 + dr
        cc = c0 + dc
        if 0 <= rr < h and 0 <= cc < w:
            cells.add((rr, cc))
    return cells


def _candidate_offsets(coords, shape):
    h, w = shape
    coords = {(int(r), int(c)) for r, c in coords}
    matches = []
    for r0 in range(-_RING_H + 1, h):
        for c0 in range(-_RING_W + 1, w):
            if coords.issubset(_ring_cells((r0, c0), shape)):
                matches.append((r0, c0))
    return matches


def _find_separator_row(grid):
    h = grid.shape[0]
    candidates = []
    for r in range(1, h - 1):
        if np.all(grid[r] == 0) and np.any(grid[:r] != 0) and np.any(grid[r + 1 :] != 0):
            candidates.append(r)
    assert len(candidates) == 1, "expected a unique all-zero separator row"
    return candidates[0]


def _recover_fragments(scene):
    colors = [int(c) for c in np.unique(scene) if int(c) != 0]
    fragments = {}
    for color in colors:
        coords = np.argwhere(scene == color)
        matches = _candidate_offsets(coords, scene.shape)
        assert len(matches) == 1, f"ring color {color} must have a unique template offset"
        fragments[color] = {"offset": matches[0], "coords": coords}
    return fragments


def _unique_topological_order(colors, edges):
    indeg = {color: 0 for color in colors}
    adj = {color: set() for color in colors}
    for src, dst in edges:
        if dst not in adj[src]:
            adj[src].add(dst)
            indeg[dst] += 1

    order = []
    ready = sorted([color for color in colors if indeg[color] == 0])
    while ready:
        assert len(ready) == 1, "occlusion relations must define a unique order"
        color = ready.pop()
        order.append(color)
        for nxt in sorted(adj[color]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
                ready.sort()

    assert len(order) == len(colors), "expected an acyclic occlusion graph"
    return order


def _recover_layer_order(scene):
    fragments = _recover_fragments(scene)
    edges = set()
    for color, info in fragments.items():
        for rr, cc in _ring_cells(info["offset"], scene.shape):
            seen = int(scene[rr, cc])
            if seen != 0 and seen != color:
                edges.add((seen, color))
    return _unique_topological_order(sorted(fragments), edges)


def _stripe_majority(row):
    counts = Counter(map(int, row))
    color, freq = counts.most_common(1)[0]
    tied = [c for c, n in counts.items() if n == freq]
    assert len(tied) == 1, "stripe row must have a unique dominant color"
    return int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the occlusion order of overlapping ring sprites and use it to reorder denoised horizontal stripes.

    Input structure:
    - A single all-black separator row splits the grid into an upper ring scene and a lower stripe scene.
    - Above the separator, each non-black color is one partially occluded 6x6 hollow rectangle ring fragment.
    - Below the separator, each stripe row spans the full width, is mostly one ring color, and contains a few noise pixels.

    Rule:
    - Recover each ring color's unique 6x6 template placement from its visible fragment.
    - Whenever one color occupies another ring's expected template cell, the visible color is above the hidden color; topologically sort these relations from front to back.
    - Denoise every stripe row by repainting it entirely with its unique row-majority color.
    - Reorder the clean stripe rows to follow the recovered ring layer order and return only that reordered stripe block.

    Key insight:
    - The upper half provides only the permutation through geometric occlusion, while the lower half provides only the stripe contents through color-majority denoising.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _find_separator_row(g)
    top = g[:sep]
    bottom = g[sep + 1 :]

    order = _recover_layer_order(top)

    stripe_rows = [r for r in range(bottom.shape[0]) if int(np.count_nonzero(bottom[r])) > bottom.shape[1] // 2]
    assert len(stripe_rows) == len(order), "stripe count must equal ring count"

    cleaned = {}
    for r in stripe_rows:
        color = _stripe_majority(bottom[r])
        cleaned[color] = np.full(bottom.shape[1], color, dtype=int)

    assert set(cleaned) == set(order), "stripe colors must match ring colors"
    return np.vstack([cleaned[color] for color in order])


def _render_top(shape, order, offsets):
    grid = np.zeros(shape, dtype=int)
    for color, offset in reversed(list(zip(order, offsets))):
        for rr, cc in _ring_cells(offset, shape):
            grid[rr, cc] = color
    return grid


def _visible_counts(scene, colors):
    return {color: int(np.count_nonzero(scene == color)) for color in colors}


def _offset_candidates(shape):
    h, w = shape
    return [(r0, c0) for r0 in range(h - _RING_H + 1) for c0 in range(w - _RING_W + 1)]


def _build_offsets(shape, count):
    candidates = _offset_candidates(shape)
    random.shuffle(candidates)
    offsets = [candidates.pop()]
    higher_cells = [_ring_cells(offsets[0], shape)]

    for _ in range(1, count):
        covered_by_higher = set().union(*higher_cells)
        covered_before_prev = set().union(*higher_cells[:-1]) if len(higher_cells) > 1 else set()
        visible_prev = higher_cells[-1] - covered_before_prev
        next_candidates = []
        for offset in candidates:
            cells = _ring_cells(offset, shape)
            visible = cells - covered_by_higher
            if not (_GEN_VISIBLE[0] <= len(visible) <= _GEN_VISIBLE[1]):
                continue
            if not (cells & visible_prev):
                continue
            if len(_candidate_offsets(visible, shape)) != 1:
                continue
            next_candidates.append(offset)
        if not next_candidates:
            return None
        offset = random.choice(next_candidates)
        candidates.remove(offset)
        offsets.append(offset)
        higher_cells.append(_ring_cells(offset, shape))
    return offsets


def _geometry_degenerate(order, offsets, scene):
    placement = {color: offsets[i] for i, color in enumerate(order)}
    visible = _visible_counts(scene, order)
    heuristics = [
        [color for color in sorted(order, key=lambda c: (placement[c][0], placement[c][1]))],
        [color for color in sorted(order, key=lambda c: (-placement[c][0], placement[c][1]))],
        [color for color in sorted(order, key=lambda c: (placement[c][1], placement[c][0]))],
        [color for color in sorted(order, key=lambda c: (-placement[c][1], placement[c][0]))],
        [color for color in sorted(order, key=lambda c: (-visible[c], placement[c][0], placement[c][1]))],
    ]
    return any(h == list(order) for h in heuristics)


def _full_row_coverage(scene):
    return all(np.any(scene[r] != 0) for r in range(scene.shape[0]))


def _make_bottom(order, width):
    row_order = random.sample(list(order), len(order))
    if row_order == list(order):
        return None

    bottom = np.zeros((len(order), width), dtype=int)
    for r, color in enumerate(row_order):
        bottom[r, :] = color
        noise_count = random.randint(*_GEN_NOISE)
        for c in random.sample(range(width), noise_count):
            choices = [v for v in _PALETTE if v != color]
            bottom[r, c] = random.choice(choices)
        if _stripe_majority(bottom[r]) != color:
            return None
    return bottom


def _gen_attempt():
    ring_count = random.randint(*_GEN_RING_COUNT)
    width = random.randint(*_GEN_WIDTH)
    top_h = random.randint(*_GEN_TOP_H)
    shape = (top_h, width)
    order = random.sample(_PALETTE, ring_count)

    offsets = _build_offsets(shape, ring_count)
    if offsets is None:
        return None

    top = _render_top(shape, order, offsets)
    if not _full_row_coverage(top):
        return None
    if _geometry_degenerate(order, offsets, top):
        return None
    if _recover_layer_order(top) != order:
        return None

    bottom = _make_bottom(order, width)
    if bottom is None:
        return None

    inp = np.vstack([top, np.zeros((1, width), dtype=int), bottom])
    expected = np.vstack([np.full((1, width), color, dtype=int) for color in order])

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
