from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
MAX_OUTLINE_LAYERS = 3
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 600
POSITION_ATTEMPTS = 120


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _connected_components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: set[tuple[int, int]] = set()
            while q:
                rr, cc = q.popleft()
                comp.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _component_boundary(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    boundary: set[tuple[int, int]] = set()
    for r, c in cells:
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            boundary.add((r, c))
            continue
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) not in cells:
                boundary.add((r, c))
                break
    return boundary


def _peel_layers(grid: np.ndarray, comp: set[tuple[int, int]]) -> tuple[list[set[tuple[int, int]]], list[int]]:
    h, w = grid.shape
    remaining = set(comp)
    outer_to_inner: list[set[tuple[int, int]]] = []
    colors_outer_to_inner: list[int] = []
    while remaining:
        boundary = _component_boundary(remaining, h, w)
        assert boundary
        cols = {int(grid[r, c]) for r, c in boundary}
        assert len(cols) == 1
        outer_color = next(iter(cols))
        layer = {(r, c) for r, c in remaining if int(grid[r, c]) == outer_color}
        outer_to_inner.append(layer)
        colors_outer_to_inner.append(outer_color)
        remaining -= layer
    layers = list(reversed(outer_to_inner))
    colors = list(reversed(colors_outer_to_inner))
    assert 2 <= len(layers) <= 3
    return layers, colors


def _infer_successor(objects: list[dict], colors_present: set[int]) -> dict[int, int]:
    succ: dict[int, int] = {}
    pred: dict[int, int] = {}
    for obj in objects:
        cols = obj["colors"]
        for a, b in zip(cols, cols[1:]):
            if a in succ:
                assert succ[a] == b
            else:
                succ[a] = b
            if b in pred:
                assert pred[b] == a
            else:
                pred[b] = a

    missing_out = list(colors_present - set(succ))
    missing_in = list(colors_present - set(pred))
    if len(missing_out) == 1 and len(missing_in) == 1:
        succ[missing_out[0]] = missing_in[0]
        pred[missing_in[0]] = missing_out[0]

    assert set(succ) == colors_present
    return succ


def _complete_from_objects(
    base_grid: np.ndarray,
    objects: list[dict],
    successor: dict[int, int],
) -> np.ndarray:
    h, w = base_grid.shape
    out = np.array(base_grid, copy=True)
    all_layers = [[set(layer) for layer in obj["layers"]] for obj in objects]
    all_colors: list[list[int]] = []
    for obj in objects:
        cols = list(obj["colors"])
        while len(cols) < MAX_OUTLINE_LAYERS + 1:
            cols.append(successor[cols[-1]])
        all_colors.append(cols)

    for depth in range(1, MAX_OUTLINE_LAYERS + 1):
        claims_by_cell: dict[tuple[int, int], list[int]] = {}
        new_layers = [set() for _ in objects]
        for idx, layers in enumerate(all_layers):
            if len(layers) > depth:
                continue
            frontier = layers[depth - 1]
            cand: set[tuple[int, int]] = set()
            for r, c in frontier:
                for nr, nc in _neighbors4(r, c, h, w):
                    if int(out[nr, nc]) == BG:
                        cand.add((nr, nc))
            for cell in cand:
                claims_by_cell.setdefault(cell, []).append(idx)

        conflicts = [cell for cell, owners in claims_by_cell.items() if len(owners) > 1]
        assert not conflicts

        for cell, owners in claims_by_cell.items():
            new_layers[owners[0]].add(cell)

        for idx, layers in enumerate(all_layers):
            if len(layers) > depth:
                continue
            layer = new_layers[idx]
            layers.append(layer)
            color = all_colors[idx][depth]
            for r, c in layer:
                out[r, c] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each seed's echo outlines to depth 3 using a shared cyclic color order.

    Each foreground component contains one seed plus 1-2 already drawn outer layers.
    Colors advance along one global ring, but each seed starts at its own seed color.
    Existing layers stay fixed; missing layers expand outward through currently black cells.
    """
    g = np.asarray(grid, dtype=int)
    colors_present = {int(v) for v in np.unique(g) if int(v) != BG}
    if not colors_present:
        return g.copy()

    comps = _connected_components(g != BG)
    objects: list[dict] = []
    for comp in comps:
        layers, colors = _peel_layers(g, comp)
        objects.append({"layers": layers, "colors": colors})

    successor = _infer_successor(objects, colors_present)
    return _complete_from_objects(g, objects, successor)


_BASE_SHAPES = [
    {(0, 0), (0, 1)},
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (0, 2), (0, 3)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 0), (0, 1), (0, 2), (1, 1)},
    {(0, 1), (0, 2), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)},
    {(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)},
    {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
    {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
]


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    norm = sorted((r - r0, c - c0) for r, c in cells)
    return tuple(norm)


def _rotate(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(c, -r) for r, c in cells}


def _flip(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(r, -c) for r, c in cells}


def _shape_variants(base: set[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    cur = set(base)
    seen: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(4):
        seen.add(_normalize(cur))
        seen.add(_normalize(_flip(cur)))
        cur = _rotate(cur)
    return list(seen)


_ALL_SHAPES: list[tuple[tuple[int, int], ...]] = []
for _base in _BASE_SHAPES:
    _ALL_SHAPES.extend(_shape_variants(_base))


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _translate(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r + r0, c + c0) for r, c in shape}


def _grow_layers(seed: set[tuple[int, int]], h: int, w: int, depth: int = MAX_OUTLINE_LAYERS) -> list[set[tuple[int, int]]]:
    layers = [set(seed)]
    seen = set(seed)
    frontier = set(seed)
    for _ in range(depth):
        nxt: set[tuple[int, int]] = set()
        for r, c in frontier:
            for nr, nc in _neighbors4(r, c, h, w):
                if (nr, nc) not in seen:
                    nxt.add((nr, nc))
        layers.append(nxt)
        seen |= nxt
        frontier = nxt
    return layers


def _neighbor_band(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    band = set(cells)
    for r, c in list(cells):
        for nr, nc in _neighbors4(r, c, h, w):
            band.add((nr, nc))
    return band


def _build_input_grid(h: int, w: int, objects: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for obj in objects:
        for depth in range(obj["input_depth"] + 1):
            color = obj["colors"][depth]
            for r, c in obj["full_layers"][depth]:
                grid[r, c] = color
    return grid


def _truth_successor(ring: list[int]) -> dict[int, int]:
    n = len(ring)
    return {ring[i]: ring[(i + 1) % n] for i in range(n)}


def _simulate_truth(h: int, w: int, objects: list[dict]) -> tuple[np.ndarray, bool, bool]:
    grid = _build_input_grid(h, w, objects)
    successor = _truth_successor(objects[0]["ring"])
    out = np.array(grid, copy=True)
    actual_layers = [[set(layer) for layer in obj["full_layers"][: obj["input_depth"] + 1]] for obj in objects]
    same_depth_conflict = False
    lower_depth_block = False

    for depth in range(1, MAX_OUTLINE_LAYERS + 1):
        claims: dict[tuple[int, int], list[int]] = {}
        new_layers = [set() for _ in objects]
        for idx, obj in enumerate(objects):
            if len(actual_layers[idx]) > depth:
                continue
            ideal = obj["full_layers"][depth]
            if any(int(out[r, c]) != BG for r, c in ideal):
                lower_depth_block = True
            frontier = actual_layers[idx][depth - 1]
            cand: set[tuple[int, int]] = set()
            for r, c in frontier:
                for nr, nc in _neighbors4(r, c, h, w):
                    if int(out[nr, nc]) == BG:
                        cand.add((nr, nc))
            for cell in cand:
                claims.setdefault(cell, []).append(idx)

        if any(len(owners) > 1 for owners in claims.values()):
            same_depth_conflict = True
            break

        for cell, owners in claims.items():
            new_layers[owners[0]].add(cell)

        for idx, obj in enumerate(objects):
            if len(actual_layers[idx]) > depth:
                continue
            actual_layers[idx].append(new_layers[idx])
            color = obj["colors"][depth]
            for r, c in new_layers[idx]:
                out[r, c] = color

    assert successor
    return out, same_depth_conflict, lower_depth_block


def _sample_depths(ring_len: int, want_overlap: bool) -> list[int]:
    if want_overlap:
        num_depth1 = random.randint(1, max(1, ring_len - 3))
    else:
        num_depth1 = random.randint(1, ring_len - 1)
    depths = [1] * num_depth1 + [2] * (ring_len - num_depth1)
    random.shuffle(depths)
    return depths


def _sample_object_specs(ring: list[int], want_overlap: bool) -> list[dict]:
    depths = _sample_depths(len(ring), want_overlap)
    objects: list[dict] = []
    for idx, seed_color in enumerate(ring):
        shape = random.choice(_ALL_SHAPES)
        colors = [ring[(idx + step) % len(ring)] for step in range(MAX_OUTLINE_LAYERS + 1)]
        objects.append(
            {
                "shape": shape,
                "seed_color": seed_color,
                "colors": colors,
                "input_depth": depths[idx],
                "ring": ring,
            }
        )
    return objects


def _place_objects(h: int, w: int, objects: list[dict]) -> bool:
    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    order = list(range(len(objects)))
    random.shuffle(order)
    for idx in order:
        obj = objects[idx]
        shape = obj["shape"]
        sh, sw = _shape_size(shape)
        if h < sh + 6 or w < sw + 6:
            return False
        placed = False
        for _ in range(POSITION_ATTEMPTS):
            r0 = random.randint(3, h - sh - 3)
            c0 = random.randint(3, w - sw - 3)
            seed = _translate(shape, r0, c0)
            full_layers = _grow_layers(seed, h, w)
            input_cells = set().union(*full_layers[: obj["input_depth"] + 1])
            if input_cells & blocked:
                continue
            obj["full_layers"] = full_layers
            placed = True
            occupied |= input_cells
            blocked = _neighbor_band(occupied, h, w)
            break
        if not placed:
            return False
    return True


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    ring_len = random.randint(4, 6)
    ring = random.sample(PALETTE, ring_len)
    want_overlap = random.random() < 0.4

    if want_overlap:
        h = random.randint(16, 21)
        w = random.randint(16, 21)
    else:
        h = random.randint(18, 26)
        w = random.randint(18, 26)

    objects = _sample_object_specs(ring, want_overlap)
    if not _place_objects(h, w, objects):
        return None

    inp = _build_input_grid(h, w, objects)
    expected, same_depth_conflict, lower_depth_block = _simulate_truth(h, w, objects)
    if same_depth_conflict:
        return None
    if want_overlap != lower_depth_block:
        return None

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    counts = Counter(int(v) for v in inp.ravel())
    if counts[BG] <= max(count for color, count in counts.items() if color != BG):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _sample_case()
        if case is None:
            continue
        inp, _expected = case
        return inp
    raise RuntimeError("generate_input: failed to sample a valid puzzle instance")
