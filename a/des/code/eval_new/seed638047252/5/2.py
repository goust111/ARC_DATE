from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


Coord = tuple[int, int]
ShapeSig = frozenset[Coord]


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors(r: int, c: int, h: int, w: int, connectivity: int):
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dr, dc in steps:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _bbox(coords: list[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _full_signature(h: int, w: int) -> ShapeSig:
    return frozenset((r, c) for r in range(h) for c in range(w))


def _shape_spans(sig: ShapeSig, h: int, w: int) -> bool:
    rows = {r for r, _ in sig}
    cols = {c for _, c in sig}
    return 0 in rows and (h - 1) in rows and 0 in cols and (w - 1) in cols


def _is_connected(sig: ShapeSig, h: int, w: int, connectivity: int) -> bool:
    if not sig:
        return False
    start = next(iter(sig))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors(r, c, h, w, connectivity):
            if (nr, nc) in sig and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(sig)


def _find_components(grid: np.ndarray, bg: int, connectivity: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue
            queue = deque([(r, c)])
            visited[r, c] = True
            coords: list[Coord] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w, connectivity):
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))
            r0, c0, r1, c1 = _bbox(coords)
            sig = frozenset((rr - r0, cc - c0) for rr, cc in coords)
            components.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "origin": (r0, c0),
                    "shape": (r1 - r0 + 1, c1 - c0 + 1),
                    "sig": sig,
                }
            )
    return components


def _solve_with_connectivity(grid: np.ndarray, bg: int, connectivity: int) -> np.ndarray:
    components = _find_components(grid, bg, connectivity)
    assert components and len(components) % 2 == 0, "expected an even number of colored fragments"

    by_shape: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, comp in enumerate(components):
        by_shape[comp["shape"]].append(idx)

    pairs: list[tuple[int, int]] = []
    for shape, indices in by_shape.items():
        assert len(indices) % 2 == 0, "each bbox-size group must contain an even number of fragments"
        h, w = shape
        full = _full_signature(h, w)
        sig_to_indices: dict[ShapeSig, list[int]] = defaultdict(list)
        for idx in indices:
            sig_to_indices[components[idx]["sig"]].append(idx)

        matched: set[int] = set()
        for idx in indices:
            if idx in matched:
                continue
            target = frozenset(full - components[idx]["sig"])
            candidates = [
                other
                for other in sig_to_indices.get(target, [])
                if other != idx and other not in matched
            ]
            assert len(candidates) == 1, "each fragment must have exactly one complementary partner"
            other = candidates[0]
            assert components[idx]["color"] != components[other]["color"], "paired fragments must have different colors"
            matched.add(idx)
            matched.add(other)
            pairs.append((idx, other))

        assert len(matched) == len(indices), "all fragments in a bbox-size group must be paired"

    out = np.full_like(grid, bg)
    for left_idx, right_idx in pairs:
        left = components[left_idx]
        right = components[right_idx]
        anchor = min(left["origin"], right["origin"])
        for comp in (left, right):
            dr = anchor[0] - comp["origin"][0]
            dc = anchor[1] - comp["origin"][1]
            for r, c in comp["coords"]:
                nr, nc = r + dr, c + dc
                assert 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1], "translated fragment must stay in bounds"
                assert out[nr, nc] == bg, "paired fragments must fill disjoint cells of the target rectangle"
                out[nr, nc] = comp["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Merge complementary scattered fragments by aligning each pair into one filled rectangle.

    Input structure:
    - bg is the most common color.
    - Several single-color non-bg regions are scattered across the grid.
    - The regions can be paired so that the two members of a pair have the same
      bounding-box dimensions and their relative occupied cells are
      complementary within that shared rectangle.
    - Each pair uses two different colors, and aligning their bounding boxes
      makes the pair fill one solid rectangle without overlap.

    Rule:
    - Find all monochrome connected fragments and group them by bounding-box
      size.
    - Within each size group, normalize every fragment to its own bounding box
      and match each fragment with the unique other fragment whose occupied-cell
      set is the exact complement inside that rectangle.
    - For every matched pair, choose the lexicographically earlier bounding-box
      origin as the anchor and translate the other fragment so both bounding
      boxes coincide there.
    - Clear the original scattered positions and keep only the merged rectangles,
      preserving each fragment's original color on the cells it contributes.

    Key insight:
    - Colors do not determine the pairing; the decisive signal is that two
      normalized fragment masks are exact positive/negative complements inside
      the same bounding box.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected an ARC-sized 2D grid"
    bg = _bg_color(g)

    last_error: AssertionError | None = None
    for connectivity in (4, 8):
        try:
            return _solve_with_connectivity(g, bg, connectivity)
        except AssertionError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


# --------------- generator ---------------

GEN_ATTEMPTS = 300
MASK_ATTEMPTS = 600
GRID_HEIGHT_RANGE = (20, 30)
GRID_WIDTH_RANGE = (20, 30)
PAIR_COUNT_CHOICES = (2, 3, 4)
RECT_SIDE_CHOICES = (3, 4, 5, 6)
AREA_RATIO_RANGE = (0.3, 0.7)
PLACEMENT_GAP = 1
SAME_ROW_PROBABILITY = 0.5
REPEAT_DIMS_PROBABILITY = 0.5
PALETTE = tuple(range(10))


def _paint_signature(grid: np.ndarray, sig: ShapeSig, top: int, left: int, color: int) -> None:
    for r, c in sig:
        grid[top + r, left + c] = color


def _reserve(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    r0 = max(0, top - PLACEMENT_GAP)
    c0 = max(0, left - PLACEMENT_GAP)
    r1 = min(occupied.shape[0], top + h + PLACEMENT_GAP)
    c1 = min(occupied.shape[1], left + w + PLACEMENT_GAP)
    occupied[r0:r1, c0:c1] = True


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    r0 = max(0, top - PLACEMENT_GAP)
    c0 = max(0, left - PLACEMENT_GAP)
    r1 = min(occupied.shape[0], top + h + PLACEMENT_GAP)
    c1 = min(occupied.shape[1], left + w + PLACEMENT_GAP)
    return not np.any(occupied[r0:r1, c0:c1])


def _sample_dims(pair_count: int, rng: random.Random) -> list[tuple[int, int]]:
    all_dims = [(h, w) for h in RECT_SIDE_CHOICES for w in RECT_SIDE_CHOICES]
    dims: list[tuple[int, int]] = []
    if pair_count >= 2 and rng.random() < REPEAT_DIMS_PROBABILITY:
        repeated = rng.choice(all_dims)
        repeat_count = 2 if pair_count == 2 else rng.choice((2, min(3, pair_count)))
        dims.extend([repeated] * repeat_count)
        remaining = [d for d in all_dims if d != repeated]
        dims.extend(rng.sample(remaining, pair_count - repeat_count))
    else:
        dims = rng.sample(all_dims, pair_count)
    rng.shuffle(dims)
    return dims


def _sample_complementary_shapes(h: int, w: int, used: set[ShapeSig], rng: random.Random) -> tuple[ShapeSig, ShapeSig] | None:
    full = _full_signature(h, w)
    min_area = max(2, int(np.ceil(h * w * AREA_RATIO_RANGE[0])))
    max_area = min(h * w - 2, int(np.floor(h * w * AREA_RATIO_RANGE[1])))
    for _ in range(MASK_ATTEMPTS):
        fill_prob = rng.uniform(*AREA_RATIO_RANGE)
        sig = frozenset((r, c) for r in range(h) for c in range(w) if rng.random() < fill_prob)
        if not (min_area <= len(sig) <= max_area):
            continue
        comp = frozenset(full - sig)
        if sig in used or comp in used:
            continue
        if not _shape_spans(sig, h, w) or not _shape_spans(comp, h, w):
            continue
        if not _is_connected(sig, h, w, 8) or not _is_connected(comp, h, w, 8):
            continue
        used.add(sig)
        used.add(comp)
        return sig, comp
    return None


def _place_pair(
    occupied: np.ndarray,
    h: int,
    w: int,
    same_row: bool,
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    grid_h, grid_w = occupied.shape
    anchors = [(r, c) for r in range(grid_h - h + 1) for c in range(grid_w - w + 1)]
    rng.shuffle(anchors)

    for anchor_r, anchor_c in anchors:
        if not _can_place(occupied, anchor_r, anchor_c, h, w):
            continue
        trial = occupied.copy()
        _reserve(trial, anchor_r, anchor_c, h, w)
        if same_row:
            scatters = [(anchor_r, c) for c in range(anchor_c + 1, grid_w - w + 1)]
        else:
            scatters = [(r, c) for r in range(anchor_r + 1, grid_h - h + 1) for c in range(grid_w - w + 1)]
        rng.shuffle(scatters)
        for scatter_r, scatter_c in scatters:
            if not _can_place(trial, scatter_r, scatter_c, h, w):
                continue
            _reserve(occupied, anchor_r, anchor_c, h, w)
            _reserve(occupied, scatter_r, scatter_c, h, w)
            return (anchor_r, anchor_c), (scatter_r, scatter_c)
    return None


def _build_grids(specs: list[dict], grid_h: int, grid_w: int, bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((grid_h, grid_w), bg, dtype=int)
    out = np.full((grid_h, grid_w), bg, dtype=int)
    for spec in specs:
        a_sig, b_sig = spec["shapes"]
        a_color, b_color = spec["colors"]
        anchor = spec["anchor"]
        scatter = spec["scatter"]

        if spec["anchor_piece"] == 0:
            _paint_signature(inp, a_sig, anchor[0], anchor[1], a_color)
            _paint_signature(inp, b_sig, scatter[0], scatter[1], b_color)
        else:
            _paint_signature(inp, b_sig, anchor[0], anchor[1], b_color)
            _paint_signature(inp, a_sig, scatter[0], scatter[1], a_color)

        _paint_signature(out, a_sig, anchor[0], anchor[1], a_color)
        _paint_signature(out, b_sig, anchor[0], anchor[1], b_color)
    return inp, out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(*GRID_HEIGHT_RANGE)
    grid_w = rng.randint(*GRID_WIDTH_RANGE)
    pair_count = rng.choice(PAIR_COUNT_CHOICES)
    bg = rng.choice(PALETTE)
    colors = rng.sample([c for c in PALETTE if c != bg], 2 * pair_count)

    dims = _sample_dims(pair_count, rng)
    used_by_dims: dict[tuple[int, int], set[ShapeSig]] = defaultdict(set)
    shapes: list[tuple[ShapeSig, ShapeSig]] = []
    for h, w in dims:
        sampled = _sample_complementary_shapes(h, w, used_by_dims[(h, w)], rng)
        if sampled is None:
            return None
        shapes.append(sampled)

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    same_row_flags = [rng.random() < SAME_ROW_PROBABILITY for _ in range(pair_count)]
    specs: list[dict | None] = [None] * pair_count
    order = sorted(range(pair_count), key=lambda idx: dims[idx][0] * dims[idx][1], reverse=True)
    for idx in order:
        h, w = dims[idx]
        placement = _place_pair(occupied, h, w, same_row_flags[idx], rng)
        if placement is None:
            return None
        specs[idx] = {
            "shapes": shapes[idx],
            "colors": (colors[2 * idx], colors[2 * idx + 1]),
            "anchor": placement[0],
            "scatter": placement[1],
            "anchor_piece": rng.randrange(2),
        }

    final_specs = [spec for spec in specs if spec is not None]
    inp, expected = _build_grids(final_specs, grid_h, grid_w, bg)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment input")
