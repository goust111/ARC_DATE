from __future__ import annotations

from collections import deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

PALETTE_SEQUENCE = [5, 7, 8, 4]
RING_COLORS = [2, 3, 1, 6]
GEN_RETRY_LIMIT = 400


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[Coord]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: list[list[Coord]] = []
    next_label = 1

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_label
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_label
                        q.append((nr, nc))
            components.append(comp)
            next_label += 1
    return labels, components


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    cells = np.argwhere(mask)
    for r, c in cells:
        r = int(r)
        c = int(c)
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(mask.shape, nr, nc) or not mask[nr, nc]:
                out[r, c] = True
                break
    return out


def _contour_from_shape(mask: np.ndarray) -> np.ndarray:
    rows = [int(r) for r in np.flatnonzero(np.any(mask, axis=1))]
    assert rows, "shape must be non-empty"
    top = rows[0]
    bottom = rows[-1]
    lefts = [int(np.flatnonzero(mask[r])[0]) for r in rows]
    rights = [int(np.flatnonzero(mask[r])[-1]) for r in rows]

    contour = np.zeros_like(mask, dtype=bool)
    contour[top, lefts[0] : rights[0] + 1] = True
    contour[bottom, lefts[-1] : rights[-1] + 1] = True
    for r, lc, rc in zip(rows, lefts, rights):
        contour[r, lc] = True
        contour[r, rc] = True
    for idx in range(len(rows) - 1):
        r = rows[idx]
        contour[r + 1, min(lefts[idx], lefts[idx + 1]) : max(lefts[idx], lefts[idx + 1]) + 1] = True
        contour[r + 1, min(rights[idx], rights[idx + 1]) : max(rights[idx], rights[idx + 1]) + 1] = True
    return contour


def _extract_palette(last_row: np.ndarray) -> list[int]:
    row = [int(x) for x in np.asarray(last_row, dtype=int)]
    target = PALETTE_SEQUENCE
    for start in range(len(row) - len(target) + 1):
        if row[start : start + len(target)] == target:
            return target[:]
    raise AssertionError("expected palette row containing 5,7,8,4")


def _ring_interior_mask(ring_cells: list[Coord], shape: tuple[int, int]) -> np.ndarray:
    ring_mask = np.zeros(shape, dtype=bool)
    for r, c in ring_cells:
        ring_mask[r, c] = True

    box = _bbox(ring_mask)
    assert box is not None
    r0, c0, r1, c1 = box
    patch_h = r1 - r0 + 3
    patch_w = c1 - c0 + 3
    blocked = np.zeros((patch_h, patch_w), dtype=bool)
    for r, c in ring_cells:
        blocked[r - r0 + 1, c - c0 + 1] = True

    outside = np.zeros_like(blocked, dtype=bool)
    q = deque()
    for r in range(patch_h):
        for c in range(patch_w):
            if r not in (0, patch_h - 1) and c not in (0, patch_w - 1):
                continue
            if blocked[r, c] or outside[r, c]:
                continue
            outside[r, c] = True
            q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(blocked.shape, nr, nc) and not blocked[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    interior_local = (~blocked) & (~outside)
    interior = np.zeros(shape, dtype=bool)
    interior[r0 : r1 + 1, c0 : c1 + 1] = interior_local[1:-1, 1:-1]
    return interior


def _region_depths(core: np.ndarray) -> tuple[np.ndarray, list[int]]:
    zero_labels, zero_components = _label_components(core == 0)
    ring_labels, ring_components = _label_components(core != 0)
    assert ring_components, "expected at least one closed ring"

    region_to_rings = [set() for _ in range(len(zero_components) + 1)]
    ring_to_regions = [set() for _ in range(len(ring_components) + 1)]
    reps = [(-1, -1)] * (len(zero_components) + 1)
    for rid, comp in enumerate(zero_components, start=1):
        reps[rid] = comp[0]

    for ring_id, ring_cells in enumerate(ring_components, start=1):
        for r, c in ring_cells:
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds(core.shape, nr, nc):
                    continue
                region_id = int(zero_labels[nr, nc])
                if region_id == 0:
                    continue
                region_to_rings[region_id].add(ring_id)
                ring_to_regions[ring_id].add(region_id)

    ring_outer: dict[int, int] = {}
    ring_inner: dict[int, int] = {}
    for ring_id, ring_cells in enumerate(ring_components, start=1):
        adj = sorted(ring_to_regions[ring_id])
        assert len(adj) == 2, f"ring {ring_id} must separate exactly two zero regions"
        interior_mask = _ring_interior_mask(ring_cells, core.shape)
        inside = [rid for rid in adj if interior_mask[reps[rid]]]
        assert len(inside) == 1, f"ring {ring_id} must have a unique inner zero region"
        inner = inside[0]
        outer = adj[0] if adj[1] == inner else adj[1]
        ring_outer[ring_id] = outer
        ring_inner[ring_id] = inner

    depths = [-1] * (len(zero_components) + 1)
    q = deque()
    for rid, comp in enumerate(zero_components, start=1):
        if any(r in (0, core.shape[0] - 1) or c in (0, core.shape[1] - 1) for r, c in comp):
            depths[rid] = 0
            q.append(rid)

    while q:
        outer_region = q.popleft()
        for ring_id in region_to_rings[outer_region]:
            if ring_outer[ring_id] != outer_region:
                continue
            inner_region = ring_inner[ring_id]
            want = depths[outer_region] + 1
            if depths[inner_region] == -1:
                depths[inner_region] = want
                q.append(inner_region)
            else:
                assert depths[inner_region] == want

    assert all(depth >= 0 for depth in depths[1:]), "every zero region must get a nesting depth"
    return zero_labels, depths


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each nested gap with a diagonal palette cycle whose length equals the enclosing ring depth.

    Input structure:
    - The background is black (0).
    - The bottom row contains a four-color palette run `5,7,8,4`.
    - Above that row sit 3-4 single-pixel-wide non-black closed contour rings, nested without touching or crossing.
    - Pixels between rings and inside the innermost ring are black.

    Rule:
    - Ignore the bottom palette row except as the visible ordered color source.
    - Determine each black connected region's nesting depth: outside the outermost ring is depth 0, the gap just inside the outermost ring is depth 1, and so on.
    - Leave depth-0 black pixels unchanged.
    - For every black pixel in a region of depth `D >= 1`, compute `(row + col) mod D` and replace the pixel with the palette color at that index.
    - Keep all contour pixels and the palette row unchanged.

    Key insight:
    - The same palette is reused everywhere, but each gap truncates it to a prefix whose length is fixed by geometric nesting depth rather than by ring color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 2

    out = g.copy()
    core = g[:-1, :]
    palette = _extract_palette(g[-1, :])
    zero_labels, depths = _region_depths(core)

    for r in range(core.shape[0]):
        for c in range(core.shape[1]):
            if core[r, c] != 0:
                continue
            depth = depths[int(zero_labels[r, c])]
            if depth <= 0:
                continue
            assert depth <= len(palette), "palette must cover all nesting depths"
            out[r, c] = palette[(r + c) % depth]
    return out


def _make_mask_from_rows(top: int, lefts: list[int], rights: list[int], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for offset, (lc, rc) in enumerate(zip(lefts, rights)):
        mask[top + offset, lc : rc + 1] = True
    return mask


def _box_mask(shape: tuple[int, int], box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _build_nested_shapes(core_h: int, w: int, depth_count: int) -> list[np.ndarray] | None:
    inner_min = 7 if depth_count == 4 else 5
    min_outer = inner_min + 6 * (depth_count - 1)
    max_outer_h = core_h - 2
    max_outer_w = w - 2
    if max_outer_h < min_outer or max_outer_w < min_outer:
        return None

    outer_h = random.randint(min_outer, min(max_outer_h, min_outer + 4))
    outer_w = random.randint(min_outer, min(max_outer_w, min_outer + 4))
    top = random.randint(1, core_h - outer_h - 1)
    left = random.randint(1, w - outer_w - 1)
    box = (top, left, top + outer_h - 1, left + outer_w - 1)

    boxes = [box]
    for level in range(1, depth_count):
        prev = boxes[-1]
        prev_h = prev[2] - prev[0] + 1
        prev_w = prev[3] - prev[1] + 1
        remaining = depth_count - level - 1
        min_child = inner_min + 6 * remaining

        pad_options = []
        for pt in range(3, 6):
            for pb in range(3, 6):
                for pl in range(3, 6):
                    for pr in range(3, 6):
                        child_h = prev_h - pt - pb
                        child_w = prev_w - pl - pr
                        if child_h >= min_child and child_w >= min_child:
                            pad_options.append((pt, pb, pl, pr))
        if not pad_options:
            return None
        pt, pb, pl, pr = random.choice(pad_options)
        box = (prev[0] + pt, prev[1] + pl, prev[2] - pb, prev[3] - pr)
        boxes.append(box)

    return [_box_mask((core_h, w), b) for b in boxes]


def _render_input(shapes: list[np.ndarray], width: int) -> np.ndarray:
    core_h = shapes[0].shape[0]
    grid = np.zeros((core_h + 1, width), dtype=int)
    for color, shape in zip(RING_COLORS, shapes):
        grid[:-1][_contour_from_shape(shape)] = color
    start = random.randint(1, width - len(PALETTE_SEQUENCE) - 1)
    grid[-1, start : start + len(PALETTE_SEQUENCE)] = np.array(PALETTE_SEQUENCE, dtype=int)
    return grid


def _expected_output_from_shapes(inp: np.ndarray, shapes: list[np.ndarray]) -> np.ndarray:
    out = inp.copy()
    boundaries = [_contour_from_shape(shape) for shape in shapes]
    for depth, shape in enumerate(shapes, start=1):
        region = shape & (~boundaries[depth - 1])
        if depth < len(shapes):
            region &= ~shapes[depth]
        for r, c in np.argwhere(region):
            r = int(r)
            c = int(c)
            out[r, c] = PALETTE_SEQUENCE[(r + c) % depth]
    return out


def _validate_instance(inp: np.ndarray, expected: np.ndarray, shapes: list[np.ndarray]) -> bool:
    if np.array_equal(inp, expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    ring_mask = inp[:-1] != 0
    _, ring_components = _label_components(ring_mask)
    if len(ring_components) != len(shapes):
        return False

    seen_period4 = False
    boundaries = [_contour_from_shape(shape) for shape in shapes]
    for depth, shape in enumerate(shapes, start=1):
        region = shape & (~boundaries[depth - 1])
        if depth < len(shapes):
            region &= ~shapes[depth]
        coords = np.argwhere(region)
        if coords.shape[0] < depth + 2:
            return False
        residues = {(int(r) + int(c)) % depth for r, c in coords}
        if residues != set(range(depth)):
            return False
        colors = {int(expected[int(r), int(c)]) for r, c in coords}
        if colors != {PALETTE_SEQUENCE[i] for i in range(depth)}:
            return False
        if depth == 4:
            seen_period4 = True

    if len(shapes) == 4 and not seen_period4:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(20, 30)
    width = random.randint(20, 30)
    can_make_four = height >= 28 and width >= 28
    depth_count = 4 if can_make_four and random.random() < 0.7 else 3
    shapes = _build_nested_shapes(height - 1, width, depth_count)
    if shapes is None:
        return None
    inp = _render_input(shapes, width)
    expected = _expected_output_from_shapes(inp, shapes)
    return inp if _validate_instance(inp, expected, shapes) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-period ARC input")
