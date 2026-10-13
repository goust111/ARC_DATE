from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
LAYER_COUNT = 4
Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_cross(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = g.shape
    solid_rows = [
        (r, int(g[r, 0]))
        for r in range(h)
        if int(g[r, 0]) != bg and np.all(g[r, :] == int(g[r, 0]))
    ]
    solid_cols = [
        (c, int(g[0, c]))
        for c in range(w)
        if int(g[0, c]) != bg and np.all(g[:, c] == int(g[0, c]))
    ]
    crosses = [
        (r, c, row_color)
        for r, row_color in solid_rows
        for c, col_color in solid_cols
        if row_color == col_color
    ]
    assert len(crosses) == 1, "expected exactly one same-color separator cross"
    cross_r, cross_c, sep = crosses[0]
    assert cross_r > 0 and cross_c > 0, "separator must split non-empty regions"
    assert h == 2 * cross_r + 1 and w == 2 * cross_c + 1, "regions must be equal size"
    return int(cross_r), int(cross_c), int(sep)


def _extract_regions(g: np.ndarray, cross_r: int, cross_c: int) -> list[np.ndarray]:
    return [
        g[:cross_r, :cross_c],
        g[:cross_r, cross_c + 1 :],
        g[cross_r + 1 :, :cross_c],
        g[cross_r + 1 :, cross_c + 1 :],
    ]


def _stamp_mask(shape: np.ndarray, anchor: Coord, size: tuple[int, int]) -> np.ndarray:
    out = np.zeros(size, dtype=bool)
    r0, c0 = anchor
    h, w = shape.shape
    out[r0 : r0 + h, c0 : c0 + w] = shape
    return out


def _anchors_for(fragment: np.ndarray, shape: np.ndarray) -> list[Coord]:
    h, w = fragment.shape
    sh, sw = shape.shape
    if sh > h or sw > w:
        return []

    anchors: list[Coord] = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            full = _stamp_mask(shape, (r0, c0), fragment.shape)
            if np.all(full[fragment]):
                anchors.append((r0, c0))
    return anchors


def _recover_order(complete: list[np.ndarray], fragments: list[np.ndarray]) -> list[int]:
    missing = [full & ~frag for full, frag in zip(complete, fragments)]
    empty = [i for i, miss in enumerate(missing) if not np.any(miss)]
    assert len(empty) == 1, "exactly one layer must be complete"

    order: list[int] = []
    remaining = set(range(len(complete)))
    above = np.zeros_like(complete[0], dtype=bool)
    while remaining:
        choices = [
            i
            for i in remaining
            if np.array_equal(missing[i], complete[i] & above)
        ]
        assert len(choices) == 1, "occlusion evidence must choose one next layer"
        layer = choices[0]
        order.append(layer)
        remaining.remove(layer)
        above |= complete[layer]

    assert len(order) == len(complete)
    return order


def _solve_layers(fragments: list[np.ndarray]) -> tuple[list[int], list[np.ndarray]]:
    solutions: list[tuple[list[int], list[np.ndarray]]] = []
    for source_idx, fragment in enumerate(fragments):
        if not np.any(fragment):
            continue
        shape = _crop(fragment, _bbox(fragment))

        complete: list[np.ndarray] = []
        try:
            for other in fragments:
                anchors = _anchors_for(other, shape)
                assert len(anchors) == 1, "each fragment must align to the shape uniquely"
                complete.append(_stamp_mask(shape, anchors[0], other.shape))
                assert np.all(complete[-1][other]), "fragment must be a subset of its completion"

            order = _recover_order(complete, fragments)
            assert order[0] == source_idx, "the dictionary fragment must be the complete top layer"
            for layer in order[1:]:
                assert np.any(complete[layer] & ~fragments[layer]), "lower layers must show damage"
        except AssertionError:
            continue

        solutions.append((order, complete))

    assert len(solutions) == 1, "complete shape and layer order must be unique"
    return solutions[0]


def _overlay(masks: list[np.ndarray], colors: list[int], top_to_bottom: list[int], bg: int) -> np.ndarray:
    out = np.full(masks[0].shape, bg, dtype=int)
    empty = np.ones_like(masks[0], dtype=bool)
    for layer in top_to_bottom:
        paint = masks[layer] & empty
        out[paint] = int(colors[layer])
        empty &= ~paint
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer four damaged same-shape layers from quadrant fragments and restack them in reverse order.

    Input structure:
    - One solid non-background row and one solid same-color column form a separator cross.
    - The cross divides the grid into four equal regions sharing the same local coordinates.
    - Each region contains one monochrome fragment of the same sprite shape; lower layers have cells removed where complete upper layers would cover them.

    Rule:
    - Find the one undamaged fragment and use its cropped mask as the common sprite dictionary.
    - Complete every fragment by aligning that dictionary to its unique anchor in the region.
    - Recover the original top-to-bottom layer order from the equality between each layer's missing cells and the union of complete layers above it.
    - Output one region-sized grid by transparently stacking the completed sprites in the opposite order.

    Key insight:
    - The original stack is never directly drawn; the missing cells are the witnesses that identify the hidden layer order, and the requested output flips that order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    cross_r, cross_c, sep = _find_cross(g, bg)
    regions = _extract_regions(g, cross_r, cross_c)

    fragments: list[np.ndarray] = []
    colors: list[int] = []
    for region in regions:
        present = [int(v) for v in np.unique(region) if int(v) not in (bg, sep)]
        assert len(present) == 1, "each quadrant must contain exactly one sprite color"
        color = present[0]
        assert color != sep
        mask = region == color
        assert np.any(mask), "sprite fragment must be visible"
        colors.append(color)
        fragments.append(mask)
    assert len(set(colors)) == len(colors), "layer colors must be distinct"

    order, complete = _solve_layers(fragments)
    return _overlay(complete, colors, list(reversed(order)), bg)


# --------------- generator ---------------


_MAX_ATTEMPTS = 900
_Q_RANGE = (8, 13)
_SHAPE_SIZES = ((4, 4), (4, 5), (5, 4), (5, 5), (5, 6), (6, 5))
_PALETTE = tuple(range(1, 10))
_DIRECTIONS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int, h: int, w: int) -> list[Coord]:
    out: list[Coord] = []
    for dr, dc in _DIRECTIONS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set(_neighbors4(start[0], start[1], h, w))
    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nb in _neighbors4(cell[0], cell[1], h, w):
            if nb not in cells:
                frontier.add(nb)
    if len(cells) != area:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_shape() -> np.ndarray | None:
    h, w = random.choice(_SHAPE_SIZES)
    min_area = max(h + w - 2, 8)
    max_area = min(h * w - 4, 15)
    if min_area > max_area:
        return None
    area = random.randint(min_area, max_area)
    shape = _random_connected_mask(h, w, area)
    if shape is None:
        return None
    if not (np.all(np.any(shape, axis=0)) and np.all(np.any(shape, axis=1))):
        return None
    if np.array_equal(shape, np.flipud(shape)) or np.array_equal(shape, np.fliplr(shape)):
        return None
    return shape


def _nearby_anchor_options(prev: Coord, q: int, shape: np.ndarray) -> list[Coord]:
    sh, sw = shape.shape
    out: list[Coord] = []
    for r in range(q - sh + 1):
        for c in range(q - sw + 1):
            dist = max(abs(r - prev[0]), abs(c - prev[1]))
            if 1 <= dist <= 3:
                out.append((r, c))
    random.shuffle(out)
    return out


def _sample_anchors(q: int, shape: np.ndarray) -> list[Coord] | None:
    sh, sw = shape.shape
    all_anchors = [(r, c) for r in range(q - sh + 1) for c in range(q - sw + 1)]
    if len(all_anchors) < LAYER_COUNT:
        return None
    first = random.choice(all_anchors)
    anchors = [first]
    used = {first}
    for _ in range(1, LAYER_COUNT):
        options = [pt for pt in _nearby_anchor_options(anchors[-1], q, shape) if pt not in used]
        if not options:
            return None
        nxt = random.choice(options[: min(12, len(options))])
        anchors.append(nxt)
        used.add(nxt)
    return anchors


def _render_fragments(
    shape: np.ndarray,
    anchors: list[Coord],
    order: list[int],
    q: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    complete = [_stamp_mask(shape, anchor, (q, q)) for anchor in anchors]
    fragments = [np.zeros((q, q), dtype=bool) for _ in range(LAYER_COUNT)]
    above = np.zeros((q, q), dtype=bool)
    for layer in order:
        fragments[layer] = complete[layer] & ~above
        above |= complete[layer]
    return complete, fragments


def _has_chain_witnesses(complete: list[np.ndarray], order: list[int]) -> bool:
    earlier = np.zeros_like(complete[0], dtype=bool)
    for depth in range(1, len(order)):
        current = order[depth]
        prev = order[depth - 1]
        witness = complete[current] & complete[prev] & ~earlier
        if not np.any(witness):
            return False
        earlier |= complete[prev]
    return True


def _assemble_grid(fragments: list[np.ndarray], colors: list[int], q: int, sep: int) -> np.ndarray:
    grid = np.full((2 * q + 1, 2 * q + 1), BLACK, dtype=int)
    grid[q, :] = sep
    grid[:, q] = sep
    starts = ((0, 0), (0, q + 1), (q + 1, 0), (q + 1, q + 1))
    for layer, (r0, c0) in enumerate(starts):
        region = grid[r0 : r0 + q, c0 : c0 + q]
        region[fragments[layer]] = int(colors[layer])
    return grid


def _same_order_by_area(fragments: list[np.ndarray], order: list[int]) -> bool:
    counts = [int(np.count_nonzero(fragment)) for fragment in fragments]
    if len(set(counts)) != len(counts):
        return False
    ranked = sorted(range(len(fragments)), key=lambda i: counts[i], reverse=True)
    return ranked == order


def _semantic_checks(
    grid: np.ndarray,
    complete: list[np.ndarray],
    fragments: list[np.ndarray],
    colors: list[int],
    order: list[int],
) -> bool:
    try:
        expected = _overlay(complete, colors, list(reversed(order)), BLACK)
        if not np.array_equal(main(grid), expected):
            return False

        recovered_order, recovered_complete = _solve_layers(fragments)
        if recovered_order != order:
            return False
        for a, b in zip(recovered_complete, complete):
            if not np.array_equal(a, b):
                return False

        missing_counts = [int(np.count_nonzero(complete[i] & ~fragments[i])) for i in order]
        if missing_counts[0] != 0 or any(count == 0 for count in missing_counts[1:]):
            return False
        if _same_order_by_area(fragments, order):
            return False
        if not _has_chain_witnesses(complete, order):
            return False

        fragment_wrong = _overlay(fragments, colors, list(reversed(order)), BLACK)
        original_stack_wrong = _overlay(complete, colors, order, BLACK)
        row_major_wrong = _overlay(complete, colors, list(range(LAYER_COUNT)), BLACK)
        for wrong in (fragment_wrong, original_stack_wrong, row_major_wrong):
            if np.array_equal(wrong, expected):
                return False

        if np.count_nonzero(expected != BLACK) < 8:
            return False
        if set(map(int, np.unique(expected))) <= {BLACK, colors[order[-1]]}:
            return False

        counts = Counter(map(int, grid.ravel()))
        if counts[BLACK] <= max(v for k, v in counts.items() if k != BLACK):
            return False
    except Exception:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    q = random.randint(*_Q_RANGE)
    shape = _make_shape()
    if shape is None or max(2 * q + 1, 2 * q + 1) > 30:
        return None

    order = random.sample(range(LAYER_COUNT), LAYER_COUNT)
    anchors_in_order = _sample_anchors(q, shape)
    if anchors_in_order is None:
        return None
    anchors = [(-1, -1)] * LAYER_COUNT
    for layer, anchor in zip(order, anchors_in_order):
        anchors[layer] = anchor

    complete, fragments = _render_fragments(shape, anchors, order, q)
    if any(not np.any(fragment) for fragment in fragments):
        return None
    if any(np.array_equal(complete[i], complete[j]) for i in range(LAYER_COUNT) for j in range(i + 1, LAYER_COUNT)):
        return None

    palette = list(_PALETTE)
    random.shuffle(palette)
    sep = palette[0]
    colors = palette[1 : 1 + LAYER_COUNT]
    grid = _assemble_grid(fragments, colors, q, sep)
    if _semantic_checks(grid, complete, fragments, colors, order):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid layer-order inversion puzzle")
