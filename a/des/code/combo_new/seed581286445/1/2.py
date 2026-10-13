from __future__ import annotations

from collections import deque

import numpy as np


_RNG = np.random.default_rng()
_BG = 0
_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _connected_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            mins = arr.min(axis=0)
            maxs = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])),
                }
            )

    return comps


def _hole_bbox(component: dict) -> tuple[int, int, int, int]:
    coords = np.asarray(component["coords"], dtype=int)
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    mask = np.zeros((h, w), dtype=bool)
    local = coords - np.array([[r0, c0]])
    mask[local[:, 0], local[:, 1]] = True

    seen = np.zeros_like(mask, dtype=bool)
    q = deque()

    for rr in range(h):
        for cc in (0, w - 1):
            if not mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    for cc in range(w):
        for rr in (0, h - 1):
            if not mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))

    while q:
        rr, cc = q.popleft()
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    interior = (~mask) & (~seen)
    assert interior.any(), "each object must enclose a hole"

    hole_rows, hole_cols = np.where(interior)
    hr0 = int(hole_rows.min())
    hc0 = int(hole_cols.min())
    hr1 = int(hole_rows.max())
    hc1 = int(hole_cols.max())

    hole = interior[hr0 : hr1 + 1, hc0 : hc1 + 1]
    assert np.all(hole), "the enclosed cavity must be rectangular"
    assert int(interior.sum()) == hole.size, "each object must have exactly one rectangular cavity"

    return (r0 + hr0, c0 + hc0, r0 + hr1, c0 + hc1)


def _coords_inside_bbox(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return bool(
        np.all(coords[:, 0] >= r0)
        and np.all(coords[:, 0] <= r1)
        and np.all(coords[:, 1] >= c0)
        and np.all(coords[:, 1] <= c1)
    )


def _extract_chains(grid: np.ndarray, bg: int) -> list[list[int]]:
    comps = _connected_components(grid, bg)
    assert comps, "expected at least one non-background object"

    for comp in comps:
        comp["hole_bbox"] = _hole_bbox(comp)
        hb = comp["hole_bbox"]
        comp["hole_area"] = (hb[2] - hb[0] + 1) * (hb[3] - hb[1] + 1)

    parents: list[int | None] = [None] * len(comps)
    children: list[list[int]] = [[] for _ in comps]

    for i, comp in enumerate(comps):
        containers = []
        for j, other in enumerate(comps):
            if i == j:
                continue
            if _coords_inside_bbox(comp["coords"], other["hole_bbox"]):
                containers.append(j)
        if containers:
            parent = min(
                containers,
                key=lambda idx: (
                    int(comps[idx]["hole_area"]),
                    int(comps[idx]["bbox"][0]),
                    int(comps[idx]["bbox"][1]),
                ),
            )
            parents[i] = parent
            children[parent].append(i)

    for child_list in children:
        assert len(child_list) <= 1, "containment must form linear chains"

    roots = [idx for idx, parent in enumerate(parents) if parent is None]
    roots.sort(key=lambda idx: (int(comps[idx]["bbox"][1]), int(comps[idx]["bbox"][0])))

    chains: list[list[int]] = []
    for root in roots:
        chain: list[int] = []
        current = root
        visited = set()
        while current is not None:
            assert current not in visited, "containment graph must be acyclic"
            visited.add(current)
            chain.append(int(comps[current]["color"]))
            current = children[current][0] if children[current] else None
        chains.append(chain)

    return chains


def _render_chain(colors: list[int], bg: int) -> np.ndarray:
    size = 2 * len(colors) - 1
    out = np.full((size, size), bg, dtype=int)

    for depth, color in enumerate(colors):
        lo = depth
        hi = size - depth - 1
        if lo == hi:
            out[lo, lo] = int(color)
            continue
        out[lo, lo : hi + 1] = int(color)
        out[hi, lo : hi + 1] = int(color)
        out[lo : hi + 1, lo] = int(color)
        out[lo : hi + 1, hi] = int(color)

    return out


def _render_output(chains: list[list[int]], bg: int) -> np.ndarray:
    blocks = [_render_chain(chain, bg) for chain in chains]
    height = max(block.shape[0] for block in blocks)
    width = sum(block.shape[1] for block in blocks) + max(0, len(blocks) - 1)
    out = np.full((height, width), bg, dtype=int)

    x = 0
    for block in blocks:
        y = (height - block.shape[0]) // 2
        out[y : y + block.shape[0], x : x + block.shape[1]] = block
        x += block.shape[1] + 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered hollow ring objects by containment chain, then redraw each chain as canonical concentric frames.

    Input structure:
    - The background is black.
    - Each non-background 4-connected monochrome object is a closed ring with one rectangular cavity.
    - Some cavities fully contain another ring object, so the objects form several disjoint outer-to-inner linear chains.

    Rule:
    - Detect every ring object and recover its rectangular cavity by treating only that object's pixels as walls.
    - Use cavity containment to link each object to the unique smallest ring whose cavity contains it.
    - For each resulting chain, read colors from outermost to innermost and draw them as nested one-pixel rectangular frames inside a square of size (2L-1)×(2L-1), with the innermost color at the center.
    - Order chain blocks by the left-to-right order of their outermost input objects, separate blocks by one background column, and vertically center shorter blocks.

    Key insight:
    - The relevant relation is topological containment through each ring's cavity, not the original irregular geometry; once the containment chains are known, the output is a normalized concentric encoding of those color sequences.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    chains = _extract_chains(g, bg)
    return _render_output(chains, bg)


def _ring_mask(hole_h: int, hole_w: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    thickness = 1 if (max(hole_h, hole_w) >= 7 or _RNG.random() < 0.75) else 2
    base_h = hole_h + 2 * thickness
    base_w = hole_w + 2 * thickness

    coords: set[tuple[int, int]] = set()
    for r in range(base_h):
        for c in range(base_w):
            if thickness <= r < thickness + hole_h and thickness <= c < thickness + hole_w:
                continue
            coords.add((r, c))

    sides = ["top", "bottom", "left", "right"]
    _RNG.shuffle(sides)
    tab_count = int(_RNG.integers(0, 3))
    for side in sides[:tab_count]:
        if side in ("top", "bottom"):
            span = int(_RNG.integers(1, base_w + 1))
            start = int(_RNG.integers(0, base_w - span + 1))
            row = -1 if side == "top" else base_h
            for c in range(start, start + span):
                coords.add((row, c))
        else:
            span = int(_RNG.integers(1, base_h + 1))
            start = int(_RNG.integers(0, base_h - span + 1))
            col = -1 if side == "left" else base_w
            for r in range(start, start + span):
                coords.add((r, col))

    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    norm = {(r - min_r, c - min_c) for r, c in coords}
    max_r = max(r for r, _ in norm)
    max_c = max(c for _, c in norm)

    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in norm:
        mask[r, c] = True

    hole_top = thickness - min_r
    hole_left = thickness - min_c
    hole_bbox = (hole_top, hole_left, hole_top + hole_h - 1, hole_left + hole_w - 1)
    return mask, hole_bbox


def _build_chain(colors: list[int]) -> np.ndarray:
    if len(colors) == 1:
        hole_h = int(_RNG.integers(1, 4))
        hole_w = int(_RNG.integers(1, 4))
        ring, _hole_bbox = _ring_mask(hole_h, hole_w)
        grid = np.zeros(ring.shape, dtype=int)
        grid[ring] = int(colors[0])
        return grid

    child = _build_chain(colors[1:])
    child_h, child_w = child.shape
    top_gap = int(_RNG.integers(1, 3))
    bottom_gap = int(_RNG.integers(1, 3))
    left_gap = int(_RNG.integers(1, 3))
    right_gap = int(_RNG.integers(1, 3))

    hole_h = child_h + top_gap + bottom_gap
    hole_w = child_w + left_gap + right_gap
    ring, hole_bbox = _ring_mask(hole_h, hole_w)
    hr0, hc0, _hr1, _hc1 = hole_bbox

    grid = np.zeros(ring.shape, dtype=int)
    grid[ring] = int(colors[0])
    rr = hr0 + top_gap
    cc = hc0 + left_gap
    grid[rr : rr + child_h, cc : cc + child_w] = child
    return grid


def _sample_lengths() -> list[int]:
    chain_count = int(_RNG.integers(1, 4))
    if chain_count == 1:
        return [int(_RNG.integers(1, 5))]
    if chain_count == 2:
        options = [
            [1, 1],
            [2, 1],
            [3, 1],
            [2, 2],
            [3, 2],
            [4, 1],
            [4, 2],
        ]
        choice = list(options[int(_RNG.integers(0, len(options)))])
        if _RNG.random() < 0.5:
            choice.reverse()
        return choice

    options = [
        [1, 1, 1],
        [2, 1, 1],
        [2, 2, 1],
        [3, 1, 1],
        [3, 2, 1],
        [2, 2, 2],
    ]
    choice = list(options[int(_RNG.integers(0, len(options)))])
    return [int(x) for x in _RNG.permutation(choice)]


def _build_expected(chains: list[list[int]]) -> np.ndarray:
    return _render_output(chains, _BG)


def _gen_attempt() -> np.ndarray | None:
    lengths = _sample_lengths()
    total_objects = sum(lengths)
    colors = [int(x) for x in _RNG.choice(np.arange(1, 10), size=total_objects, replace=False)]

    chains: list[list[int]] = []
    start = 0
    for length in lengths:
        chains.append(colors[start : start + length])
        start += length

    chain_grids = []
    for chain in chains:
        local = _build_chain(chain)
        if max(local.shape) > 18:
            return None
        chain_grids.append((chain, local))

    order = [int(x) for x in _RNG.permutation(len(chain_grids))]
    ordered = [chain_grids[idx] for idx in order]
    ordered_chains = [chain for chain, _grid in ordered]
    ordered_grids = [grid for _chain, grid in ordered]

    left_margin = int(_RNG.integers(1, 4))
    right_margin = int(_RNG.integers(1, 4))
    gaps = [int(_RNG.integers(2, 5)) for _ in range(max(0, len(ordered_grids) - 1))]
    total_w = left_margin + right_margin + sum(grid.shape[1] for grid in ordered_grids) + sum(gaps)
    if total_w > 30:
        return None

    max_h = max(grid.shape[0] for grid in ordered_grids)
    top_margin = int(_RNG.integers(1, 4))
    bottom_margin = int(_RNG.integers(1, 4))
    extra_h = int(_RNG.integers(0, 4))
    total_h = max_h + top_margin + bottom_margin + extra_h
    if total_h > 30:
        return None

    grid = np.zeros((total_h, total_w), dtype=int)
    x = left_margin
    for idx, local in enumerate(ordered_grids):
        max_y = total_h - bottom_margin - local.shape[0]
        min_y = top_margin
        if min_y > max_y:
            return None
        y = int(_RNG.integers(min_y, max_y + 1))
        block = grid[y : y + local.shape[0], x : x + local.shape[1]]
        if np.any((block != 0) & (local != 0)):
            return None
        block[:] = np.where(local != 0, local, block)
        x += local.shape[1]
        if idx < len(gaps):
            x += gaps[idx]

    try:
        actual = main(grid)
    except AssertionError:
        return None

    expected = _build_expected(ordered_chains)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-ring puzzle instance")
