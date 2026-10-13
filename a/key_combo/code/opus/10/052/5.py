from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
MARKER = 7
RING_COLOR_POOL = [1, 2, 3, 4, 5, 6, 8, 9]

GEN_RETRIES = 400
GRID_SIZE_RANGE = (13, 22)
DEPTH_RANGE = (3, 5)
COMPONENT_SIZE_RANGE = (2, 4)
EXTRA_NOISE_RANGE = (0, 3)


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 8:
        deltas = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for dr, dc in deltas:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty foreground structure"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _depth_map(height: int, width: int) -> np.ndarray:
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    return np.minimum(
        np.minimum(rows, height - 1 - rows),
        np.minimum(cols, width - 1 - cols),
    )


def _extract_ring_structure(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray, list[int]]:
    mask = grid != BG
    bbox = _bbox(mask)
    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(sub != BG), "ring rectangle must be solid apart from orange overlays"

    depth = _depth_map(*sub.shape)
    layer_count = int(depth.max()) + 1
    colors: list[int] = []
    for layer in range(layer_count):
        visible = np.unique(sub[(depth == layer) & (sub != MARKER)])
        assert visible.size == 1, "each depth layer must have one recoverable base color"
        colors.append(int(visible[0]))

    assert len(set(colors)) == layer_count, "ring layers must use distinct visible colors"
    return bbox, depth, colors


def _render_histogram(counts: list[int], colors: list[int]) -> np.ndarray:
    height = max(counts)
    assert height > 0, "expected at least one non-isolated marker component"
    out = np.full((height, len(counts)), BG, dtype=int)
    for col, (count, color) in enumerate(zip(counts, colors)):
        if count:
            out[-count:, col] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Count non-isolated orange marker clusters by concentric-ring depth and compress them into a histogram.

    Input structure:
    - A black background surrounds one solid rectangular block made of 3-5 concentric one-pixel ring layers.
    - Each layer has its own color, while some ring cells are overwritten by orange marker pixels.
    - Orange markers form a mix of isolated singletons and small 8-connected clusters, all lying on ring-layer cells.

    Rule:
    - Recover the concentric rectangular depth map from the non-black block, with depth 1 at the outermost layer.
    - Find every 8-connected orange component and discard singleton components as noise.
    - Assign each remaining orange component to the depth value occurring most often under that component.
    - Count valid components per depth, then output a black histogram whose columns run from outermost to innermost depth and whose bar colors match the corresponding ring colors.

    Key insight:
    - The orange signal must be interpreted as connected components on top of a depth-coded ring frame: isolated orange pixels are distractors, and the output counts components per layer rather than orange pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bbox, depth_local, layer_colors = _extract_ring_structure(g)
    r0, c0, r1, c1 = bbox
    layer_count = len(layer_colors)
    counts = [0] * layer_count

    for comp in _components(g == MARKER, connectivity=8):
        if comp.shape[0] == 1:
            continue
        local_depths: list[int] = []
        for r, c in comp:
            r = int(r)
            c = int(c)
            assert r0 <= r <= r1 and c0 <= c <= c1, "markers must sit on the ring rectangle"
            local_depths.append(int(depth_local[r - r0, c - c0]))
        depth_counts = Counter(local_depths).most_common()
        assert depth_counts, "non-empty component expected"
        if len(depth_counts) > 1:
            assert depth_counts[0][1] > depth_counts[1][1], "component depth assignment must be unambiguous"
        counts[depth_counts[0][0]] += 1

    assert max(counts) > 0, "expected at least one valid marker cluster"
    return _render_histogram(counts, layer_colors)


def _shape_neighborhood(shape: tuple[int, int], cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _centered_offset(span: int) -> int:
    assert span >= 2
    choices = [v for v in range(1, span) if abs(v - (span - v)) <= 2]
    assert choices
    return random.choice(choices)


def _layer_coords(top: int, left: int, rect_h: int, rect_w: int, depth: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    r0 = top + depth
    c0 = left + depth
    r1 = top + rect_h - 1 - depth
    c1 = left + rect_w - 1 - depth
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                coords.append((r, c))
    return coords


def _grow_component(
    layer_set: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    target_size: int,
) -> set[tuple[int, int]] | None:
    starts = [cell for cell in layer_set if cell not in blocked]
    random.shuffle(starts)
    for start in starts:
        comp = {start}
        frontier = [start]
        while len(comp) < target_size:
            candidates: list[tuple[int, int]] = []
            for r, c in frontier:
                for nr, nc in _neighbors8(r, c):
                    if (nr, nc) in layer_set and (nr, nc) not in blocked and (nr, nc) not in comp:
                        candidates.append((nr, nc))
            if not candidates:
                break
            nxt = random.choice(candidates)
            comp.add(nxt)
            frontier.append(nxt)
        if len(comp) == target_size:
            return comp
    return None


def _sample_target_counts(layer_sizes: list[int]) -> tuple[list[int], int]:
    depth_count = len(layer_sizes)
    zero_layer = random.randrange(depth_count)
    counts = [0] * depth_count

    positive_layers = [d for d, size in enumerate(layer_sizes) if size >= 2 and d != zero_layer]
    assert positive_layers, "need at least one layer that can host a valid component"

    heavy_candidates = [d for d in positive_layers if layer_sizes[d] >= 10]
    heavy_layer = random.choice(heavy_candidates or positive_layers)
    counts[heavy_layer] = random.randint(2, 3) if layer_sizes[heavy_layer] >= 12 else 1

    for d in positive_layers:
        if d == heavy_layer:
            continue
        roll = random.random()
        if roll < 0.35:
            counts[d] = 0
        elif roll < 0.8:
            counts[d] = 1
        else:
            counts[d] = 2 if layer_sizes[d] >= 12 else 1

    assert counts[zero_layer] == 0
    if max(counts) == 0:
        counts[heavy_layer] = 1
    return counts, zero_layer


def _gen_attempt() -> np.ndarray | None:
    depth_count = random.randint(*DEPTH_RANGE)
    short_side = 2 * depth_count
    if random.random() < 0.5:
        rect_h = short_side
        rect_w = random.randint(short_side, min(short_side + 6, 18))
    else:
        rect_w = short_side
        rect_h = random.randint(short_side, min(short_side + 6, 18))

    grid_h = random.randint(max(GRID_SIZE_RANGE[0], rect_h + 2), min(GRID_SIZE_RANGE[1], rect_h + 8))
    grid_w = random.randint(max(GRID_SIZE_RANGE[0], rect_w + 2), min(GRID_SIZE_RANGE[1], rect_w + 8))
    if grid_h <= rect_h or grid_w <= rect_w:
        return None

    top = _centered_offset(grid_h - rect_h)
    left = _centered_offset(grid_w - rect_w)
    colors = random.sample(RING_COLOR_POOL, depth_count)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    depth_local = _depth_map(rect_h, rect_w)
    for d, color in enumerate(colors):
        rr, cc = np.where(depth_local == d)
        grid[top + rr, left + cc] = color

    layer_cells = [_layer_coords(top, left, rect_h, rect_w, d) for d in range(depth_count)]
    layer_sizes = [len(cells) for cells in layer_cells]
    target_counts, zero_layer = _sample_target_counts(layer_sizes)

    marker_cells: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    designed_components: list[set[tuple[int, int]]] = []
    counts = [0] * depth_count

    for depth, target in enumerate(target_counts):
        layer_set = set(layer_cells[depth])
        for _ in range(target):
            size = random.randint(*COMPONENT_SIZE_RANGE)
            comp = _grow_component(layer_set, blocked, size)
            if comp is None:
                return None
            designed_components.append(comp)
            counts[depth] += 1
            marker_cells |= comp
            blocked |= _shape_neighborhood(grid.shape, comp)

    zero_noise_layer = set(layer_cells[zero_layer])
    zero_noise_candidates = [cell for cell in zero_noise_layer if cell not in blocked]
    if not zero_noise_candidates:
        return None
    zero_noise = {random.choice(zero_noise_candidates)}
    marker_cells |= zero_noise
    blocked |= _shape_neighborhood(grid.shape, zero_noise)

    extra_noise_budget = random.randint(*EXTRA_NOISE_RANGE)
    noise_layers = list(range(depth_count))
    random.shuffle(noise_layers)
    for _ in range(extra_noise_budget):
        placed = False
        for depth in noise_layers:
            layer_set = set(layer_cells[depth])
            candidates = [cell for cell in layer_set if cell not in blocked]
            if not candidates:
                continue
            singleton = {random.choice(candidates)}
            marker_cells |= singleton
            blocked |= _shape_neighborhood(grid.shape, singleton)
            placed = True
            break
        if not placed:
            break

    for r, c in marker_cells:
        grid[r, c] = MARKER

    rect = grid[top : top + rect_h, left : left + rect_w]
    if np.any(rect == BG):
        return None

    for depth, cells in enumerate(layer_cells):
        marked_here = sum((r, c) in marker_cells for r, c in cells)
        if marked_here > int(0.4 * len(cells)):
            return None

    if counts[zero_layer] != 0 or max(counts) == 0:
        return None

    try:
        expected = _render_histogram(counts, colors)
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(expected, actual):
        return None

    comps = _components(grid == MARKER, connectivity=8)
    if not any(comp.shape[0] == 1 for comp in comps):
        return None
    if not any(comp.shape[0] > 1 for comp in comps):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
