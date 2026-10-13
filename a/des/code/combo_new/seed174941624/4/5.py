from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
YELLOW = 4
OVERLAP = 5
BASE_COLORS = (1, 2, 3, 6, 7, 8, 9)
MAX_PERIOD = 3
GENERATION_RETRIES = 400

PATTERN_LIBRARY = (
    np.array([[1, 0], [0, 1]], dtype=int),
    np.array([[0, 1], [1, 0]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    np.array([[1, 0, 1], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=int),
    np.array([[0, 1], [1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _intersect_boxes(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _inside(box: tuple[int, int, int, int], r: int, c: int) -> bool:
    return box[0] <= r <= box[2] and box[1] <= c <= box[3]


def _repeat_tile(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    reps_r = (height + tile.shape[0] - 1) // tile.shape[0]
    reps_c = (width + tile.shape[1] - 1) // tile.shape[1]
    return np.tile(tile, (reps_r, reps_c))[:height, :width]


def _infer_periodic_mask(region: np.ndarray, color: int) -> np.ndarray:
    h, w = region.shape
    known = region != OVERLAP
    observed = np.zeros((h, w), dtype=int)
    observed[region == color] = 1

    candidates: list[tuple[int, int, int, int, np.ndarray]] = []
    for pr in range(1, min(MAX_PERIOD, h) + 1):
        for pc in range(1, min(MAX_PERIOD, w) + 1):
            tile = np.full((pr, pc), -1, dtype=int)
            ok = True
            for r, c in np.argwhere(known):
                rr = int(r) % pr
                cc = int(c) % pc
                value = int(observed[r, c])
                prev = int(tile[rr, cc])
                if prev == -1:
                    tile[rr, cc] = value
                elif prev != value:
                    ok = False
                    break
            if not ok or np.any(tile < 0):
                continue
            full = _repeat_tile(tile, h, w)
            candidates.append((pr * pc, pr + pc, pr, pc, full))

    assert candidates, "expected a visible periodic stencil"
    candidates.sort(key=lambda item: item[:4])
    best = candidates[0]
    for cand in candidates:
        if cand[:2] == best[:2]:
            assert np.array_equal(cand[4], best[4]), "minimal periodic completion must be unique"
    return best[4]


def _solve_layers(grid: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], np.ndarray]]:
    colors = sorted(set(map(int, np.unique(grid))) - {BACKGROUND, OVERLAP})
    assert len(colors) == 2, "expected exactly two visible stencil colors"
    assert YELLOW not in colors, "yellow is reserved for the merged overlap color"

    overlap_box = _bbox(grid == OVERLAP)
    assert overlap_box is not None, "expected a gray overlap rectangle"
    or0, oc0, or1, oc1 = overlap_box
    assert np.all(grid[or0 : or1 + 1, oc0 : oc1 + 1] == OVERLAP), "overlap must be a solid gray rectangle"

    layers: list[tuple[int, tuple[int, int, int, int], np.ndarray]] = []
    for color in colors:
        box = _bbox((grid == color) | (grid == OVERLAP))
        assert box is not None
        r0, c0, r1, c1 = box
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        allowed = {BACKGROUND, OVERLAP, color}
        assert set(map(int, np.unique(region))).issubset(allowed), "each rectangle crop must isolate one stencil"
        mask = _infer_periodic_mask(region, color)
        layers.append((color, box, mask))

    intersection = _intersect_boxes(layers[0][1], layers[1][1])
    assert intersection == overlap_box, "gray rectangle must equal the rectangles' geometric overlap"
    return layers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct two overlapping hollow stencils and resolve their gray overlap by layer-combining the hidden patterns.

    Input structure:
    - The background is black.
    - Two axis-aligned rectangular stencil regions use different non-black colors, with regular black holes repeating across each rectangle.
    - The rectangles partially overlap, and every cell of the overlap is shown as a solid gray rectangle instead of the hidden stencil values.
    - Outside the gray overlap, each rectangle's visible area already shows its own color-or-hole pattern.

    Rule:
    - Infer each rectangle's full bbox by combining its visible color with the shared gray overlap block.
    - Within each bbox, treat the non-gray cells as observations of a small repeating binary stencil pattern and extend that periodic pattern through the hidden overlap.
    - Rebuild the whole scene from the two completed stencils: colored cells stay in their rectangle color, holes stay black, and cells covered by both colored layers become yellow.
    - Cells covered by exactly one colored layer keep that layer's color; cells where both layers are holes remain black.

    Key insight:
    - The gray block is not an object but an occluded rectangle intersection; the visible non-overlap strips reveal enough periodic structure to extrapolate both stencils and then combine them with a boolean-style overlay rule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    layers = _solve_layers(g)
    h, w = g.shape
    out = np.full((h, w), BACKGROUND, dtype=int)

    for r in range(h):
        for c in range(w):
            active: list[int] = []
            for color, box, mask in layers:
                if not _inside(box, r, c):
                    continue
                if int(mask[r - box[0], c - box[1]]) == 1:
                    active.append(color)
            if len(active) == 2:
                out[r, c] = YELLOW
            elif len(active) == 1:
                out[r, c] = active[0]

    return out


def _build_boxes(
    h1: int,
    w1: int,
    h2: int,
    w2: int,
    overlap_h: int,
    overlap_w: int,
    row_sign: int,
    col_sign: int,
    top: int,
    left: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    if row_sign == 1:
        local_r1, local_r2 = 0, h1 - overlap_h
    else:
        local_r1, local_r2 = h2 - overlap_h, 0

    if col_sign == 1:
        local_c1, local_c2 = 0, w1 - overlap_w
    else:
        local_c1, local_c2 = w2 - overlap_w, 0

    box1 = (top + local_r1, left + local_c1, top + local_r1 + h1 - 1, left + local_c1 + w1 - 1)
    box2 = (top + local_r2, left + local_c2, top + local_r2 + h2 - 1, left + local_c2 + w2 - 1)
    return box1, box2


def _paint_layer(
    layer: np.ndarray, box: tuple[int, int, int, int], tile: np.ndarray
) -> None:
    r0, c0, r1, c1 = box
    patch = _repeat_tile(tile, r1 - r0 + 1, c1 - c0 + 1).astype(bool)
    layer[r0 : r1 + 1, c0 : c1 + 1] = patch


def _render_scene(
    shape: tuple[int, int],
    box1: tuple[int, int, int, int],
    box2: tuple[int, int, int, int],
    color1: int,
    color2: int,
    tile1: np.ndarray,
    tile2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    layer1 = np.zeros(shape, dtype=bool)
    layer2 = np.zeros(shape, dtype=bool)
    _paint_layer(layer1, box1, tile1)
    _paint_layer(layer2, box2, tile2)

    out = np.full(shape, BACKGROUND, dtype=int)
    out[layer1 & ~layer2] = color1
    out[layer2 & ~layer1] = color2
    out[layer1 & layer2] = YELLOW

    inp = out.copy()
    overlap_box = _intersect_boxes(box1, box2)
    assert overlap_box is not None
    r0, c0, r1, c1 = overlap_box
    inp[r0 : r1 + 1, c0 : c1 + 1] = OVERLAP
    return inp, out


def _residues_visible(inp: np.ndarray, color: int, box: tuple[int, int, int, int], tile: np.ndarray) -> bool:
    recovered_box = _bbox((inp == color) | (inp == OVERLAP))
    if recovered_box != box:
        return False

    r0, c0, r1, c1 = box
    region = inp[r0 : r1 + 1, c0 : c1 + 1]
    if not set(map(int, np.unique(region))).issubset({BACKGROUND, OVERLAP, color}):
        return False

    seen = set()
    pr, pc = tile.shape
    for r, c in np.argwhere(region != OVERLAP):
        seen.add((int(r) % pr, int(c) % pc))
    if len(seen) != pr * pc:
        return False

    return bool(np.any(region == color) and np.any(region == BACKGROUND))


def _sample_dimensions(tile: np.ndarray) -> tuple[int, int]:
    min_h = max(5, tile.shape[0] + 2)
    min_w = max(5, tile.shape[1] + 2)
    return random.randint(min_h, 10), random.randint(min_w, 10)


def _gen_attempt() -> np.ndarray | None:
    color1, color2 = random.sample(BASE_COLORS, 2)
    tile1 = random.choice(PATTERN_LIBRARY)
    tile2 = random.choice(PATTERN_LIBRARY)
    h1, w1 = _sample_dimensions(tile1)
    h2, w2 = _sample_dimensions(tile2)

    max_overlap_h = min(h1, h2) - 2
    max_overlap_w = min(w1, w2) - 2
    if max_overlap_h < 2 or max_overlap_w < 2:
        return None

    overlap_h = random.randint(2, max_overlap_h)
    overlap_w = random.randint(2, max_overlap_w)
    row_sign = random.choice((-1, 1))
    col_sign = random.choice((-1, 1))

    union_h = h1 + h2 - overlap_h
    union_w = w1 + w2 - overlap_w
    if union_h > 30 or union_w > 30:
        return None

    extra_h = random.randint(0, min(6, 30 - union_h))
    extra_w = random.randint(0, min(6, 30 - union_w))
    top = random.randint(0, extra_h)
    left = random.randint(0, extra_w)
    bottom = extra_h - top
    right = extra_w - left
    shape = (union_h + top + bottom, union_w + left + right)

    box1, box2 = _build_boxes(h1, w1, h2, w2, overlap_h, overlap_w, row_sign, col_sign, top, left)
    inp, out = _render_scene(shape, box1, box2, color1, color2, tile1, tile2)

    if not _residues_visible(inp, color1, box1, tile1):
        return None
    if not _residues_visible(inp, color2, box2, tile2):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid overlapping-stencil sample")
