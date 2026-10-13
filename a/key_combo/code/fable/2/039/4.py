from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, find_components, get_bg


GEN_RETRIES = 500
BLOCK_H = 6
BLOCK_W = 6
LOCK_H = 5
LOCK_W = 5
KEY_COUNT_RANGE = (2, 3)
LOCK_COUNT_RANGE = (2, 3)


def _as_bool(a: list[list[int]]) -> np.ndarray:
    return np.array(a, dtype=bool)


_BASE_SHAPES = (
    _as_bool([[1, 0], [1, 1]]),
    _as_bool([[1, 1, 1], [0, 1, 0]]),
    _as_bool([[1, 1, 0], [0, 1, 1]]),
    _as_bool([[1, 0], [1, 1], [0, 1]]),
    _as_bool([[1, 1, 1], [1, 0, 0]]),
    _as_bool([[1, 1, 1], [0, 0, 1]]),
    _as_bool([[1, 0, 0], [1, 1, 1]]),
    _as_bool([[1, 0, 1], [1, 1, 1]]),
    _as_bool([[1, 1, 1], [0, 1, 0], [0, 1, 0]]),
    _as_bool([[1, 0, 0], [1, 1, 0], [0, 1, 1]]),
    _as_bool([[1, 1, 0], [0, 1, 0], [0, 1, 1]]),
)


def _mask_tuple(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in np.asarray(mask, dtype=bool))


def _rotations(mask: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(np.asarray(mask, dtype=bool), k).copy() for k in range(4)]


def _rotation_class(mask: np.ndarray) -> str:
    return min(repr(_mask_tuple(rot)) for rot in _rotations(mask))


def _shape_classes() -> list[np.ndarray]:
    seen: set[str] = set()
    classes: list[np.ndarray] = []
    for shape in _BASE_SHAPES:
        if len(components_mask(shape, connectivity=4)) != 1:
            continue
        key = _rotation_class(shape)
        if key in seen:
            continue
        seen.add(key)
        classes.append(shape.copy())
    return classes


SHAPE_CLASSES = _shape_classes()


def _segments(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1, *sorted(int(x) for x in separators), int(length)]
    spans: list[tuple[int, int]] = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a > 1:
            spans.append((a + 1, b - 1))
    return spans


def _span_len(span: tuple[int, int]) -> int:
    return int(span[1] - span[0] + 1)


def _find_table(g: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]:
    bg = int(get_bg(g))
    candidates: list[tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]] = []

    for color in sorted(map(int, np.unique(g))):
        if color == bg:
            continue
        sep_rows = [r for r in range(g.shape[0]) if bool(np.all(g[r, :] == color))]
        sep_cols = [c for c in range(g.shape[1]) if bool(np.all(g[:, c] == color))]
        if not sep_rows or not sep_cols:
            continue

        row_spans = _segments(g.shape[0], sep_rows)
        col_spans = _segments(g.shape[1], sep_cols)
        if len(row_spans) < 2 or len(col_spans) < 2:
            continue
        if len({_span_len(span) for span in row_spans}) != 1:
            continue
        if len({_span_len(span) for span in col_spans}) != 1:
            continue

        tr0, tr1 = row_spans[0]
        tc0, tc1 = col_spans[0]
        if not bool(np.all(g[tr0 : tr1 + 1, tc0 : tc1 + 1] == bg)):
            continue
        candidates.append((bg, color, row_spans, col_spans))

    assert len(candidates) == 1, "expected one regular table separator color"
    return candidates[0]


def _block(g: np.ndarray, row_span: tuple[int, int], col_span: tuple[int, int]) -> np.ndarray:
    r0, r1 = row_span
    c0, c1 = col_span
    return g[r0 : r1 + 1, c0 : c1 + 1]


def _extract_key(block: np.ndarray, bg: int) -> dict:
    comps = find_components(block, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert len(comps) == 1, "row label must contain one connected monochrome key glyph"
    comp = comps[0]
    color = int(comp["color"])
    assert color != bg
    r0, c0, r1, c1 = map(int, comp["bbox"])
    mask = np.zeros(block.shape, dtype=bool)
    coords = comp["coords"]
    mask[coords[:, 0], coords[:, 1]] = True
    shape = mask[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert shape.any()
    return {"color": color, "shape": shape}


def _extract_lock(block: np.ndarray, bg: int) -> dict:
    fg_colors = [int(c) for c in np.unique(block) if int(c) != bg]
    assert len(fg_colors) == 1, "column label must contain one lock color"
    color = fg_colors[0]
    comps = find_components(block, bg=bg, connectivity=4, monochromatic=True, colors=[color], return_format="dict")
    assert len(comps) == 1, "lock body must be one connected component"
    comp = comps[0]
    r0, c0, r1, c1 = map(int, comp["bbox"])
    crop = block[r0 : r1 + 1, c0 : c1 + 1]

    holes: list[np.ndarray] = []
    for coords in components_mask(crop == bg, connectivity=4):
        touches_edge = bool(
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == crop.shape[0] - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == crop.shape[1] - 1)
        )
        if not touches_edge:
            holes.append(coords)
    assert len(holes) == 1, "lock must contain exactly one enclosed background hole"

    hole_coords = holes[0]
    hr0, hc0, hr1, hc1 = bbox_coords(hole_coords)
    hole_crop = np.zeros(crop.shape, dtype=bool)
    hole_crop[hole_coords[:, 0], hole_coords[:, 1]] = True
    hole_shape = hole_crop[hr0 : hr1 + 1, hc0 : hc1 + 1].copy()

    body_mask = block == color
    hole_mask = np.zeros(block.shape, dtype=bool)
    hole_mask[r0 + hole_coords[:, 0], c0 + hole_coords[:, 1]] = True
    assert hole_shape.any()
    return {
        "color": color,
        "body_mask": body_mask,
        "hole_mask": hole_mask,
        "hole_shape": hole_shape,
    }


def _matching_rotation(key_shape: np.ndarray, hole_shape: np.ndarray) -> int | None:
    for k, rotated in enumerate(_rotations(key_shape)):
        if rotated.shape == hole_shape.shape and np.array_equal(rotated, hole_shape):
            return k
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill table cells where a row key rotates to fit a column lock's hole.

    Input structure:
    - A single separator color forms full rows and full columns that divide the
      grid into a regular block table.
    - The upper-left block is background, the rest of the top row are one-color
      lock bodies with one enclosed background hole, and the rest of the first
      column are one-color connected key glyphs.
    - Every non-header table block starts as background.

    Rule:
    - Keep separators and all header blocks unchanged.
    - For each internal block, compare the row key's binary glyph after 0, 90,
      180, or 270 degree rotation with the column lock's enclosed hole shape.
    - If they match exactly, copy that column lock into the internal block at
      the same local position as in the header and recolor the hole cells with
      the row key color; otherwise leave the block as background.

    Key insight:
    - The match is between the key and the lock's negative-space hole, not the
      lock exterior or any color relation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, _sep, row_spans, col_spans = _find_table(g)

    keys = [_extract_key(_block(g, row_spans[i], col_spans[0]), bg) for i in range(1, len(row_spans))]
    locks = [_extract_lock(_block(g, row_spans[0], col_spans[j]), bg) for j in range(1, len(col_spans))]

    out = g.copy()
    for ri, row_span in enumerate(row_spans[1:]):
        for cj, col_span in enumerate(col_spans[1:]):
            source = _block(g, row_span, col_span)
            assert bool(np.all(source == bg)), "internal table blocks must start as background"
            if _matching_rotation(keys[ri]["shape"], locks[cj]["hole_shape"]) is None:
                continue
            target = _block(out, row_span, col_span)
            lock = locks[cj]
            target[lock["body_mask"]] = int(lock["color"])
            target[lock["hole_mask"]] = int(keys[ri]["color"])
    return out


def _grid_shape(n_keys: int, n_locks: int) -> tuple[int, int]:
    rows = n_keys + 1
    cols = n_locks + 1
    return rows * BLOCK_H + rows + 1, cols * BLOCK_W + cols + 1


def _separator_indices(block_count: int, block_size: int) -> list[int]:
    return [i * (block_size + 1) for i in range(block_count + 1)]


def _block_spans(block_count: int, block_size: int) -> list[tuple[int, int]]:
    seps = _separator_indices(block_count, block_size)
    return [(seps[i] + 1, seps[i + 1] - 1) for i in range(block_count)]


def _place_shape(dst: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    h, w = mask.shape
    patch = dst[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = int(color)


def _draw_key(block: np.ndarray, key_shape: np.ndarray, color: int) -> tuple[int, int]:
    kh, kw = key_shape.shape
    r_choices = range(1, block.shape[0] - kh)
    c_choices = range(1, block.shape[1] - kw)
    r0 = random.choice(list(r_choices))
    c0 = random.choice(list(c_choices))
    _place_shape(block, key_shape, color, r0, c0)
    return r0, c0


def _draw_lock(
    block: np.ndarray,
    hole_shape: np.ndarray,
    color: int,
    bg: int,
    body_pos: tuple[int, int],
) -> tuple[int, int]:
    br, bc = body_pos
    block[br : br + LOCK_H, bc : bc + LOCK_W] = int(color)
    hh, hw = hole_shape.shape
    hr = random.randint(br + 1, br + LOCK_H - hh - 1)
    hc = random.randint(bc + 1, bc + LOCK_W - hw - 1)
    block[hr : hr + hh, hc : hc + hw][hole_shape] = int(bg)
    return hr, hc


def _render_expected(
    inp: np.ndarray,
    bg: int,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    key_shapes: list[np.ndarray],
    key_colors: list[int],
    lock_shapes: list[np.ndarray],
    lock_colors: list[int],
) -> np.ndarray:
    out = inp.copy()
    parsed_locks = [_extract_lock(_block(inp, row_spans[0], col_spans[j + 1]), bg) for j in range(len(lock_shapes))]
    for ri, row_span in enumerate(row_spans[1:]):
        for cj, col_span in enumerate(col_spans[1:]):
            if _matching_rotation(key_shapes[ri], lock_shapes[cj]) is None:
                continue
            target = _block(out, row_span, col_span)
            lock = parsed_locks[cj]
            target[lock["body_mask"]] = int(lock_colors[cj])
            target[lock["hole_mask"]] = int(key_colors[ri])
    return out


def _sample_lock_classes(row_classes: list[int], lock_count: int) -> list[int] | None:
    outside = [i for i in range(len(SHAPE_CLASSES)) if i not in set(row_classes)]
    if not outside:
        return None

    chosen = [random.choice(row_classes), random.choice(outside)]
    while len(chosen) < lock_count:
        pool = row_classes if random.random() < 0.55 else row_classes + outside
        chosen.append(random.choice(pool))
    random.shuffle(chosen)
    return chosen[:lock_count]


def _sample_body_positions(lock_count: int) -> list[tuple[int, int]] | None:
    positions = [(r, c) for r in range(BLOCK_H - LOCK_H + 1) for c in range(BLOCK_W - LOCK_W + 1)]
    if lock_count > len(positions):
        return None
    chosen = random.sample(positions, lock_count)
    if len(set(chosen)) < min(2, lock_count):
        return None
    return chosen


def _match_matrix(key_shapes: list[np.ndarray], lock_shapes: list[np.ndarray]) -> np.ndarray:
    return np.array(
        [
            [_matching_rotation(key_shape, lock_shape) is not None for lock_shape in lock_shapes]
            for key_shape in key_shapes
        ],
        dtype=bool,
    )


def _gen_attempt() -> np.ndarray | None:
    n_keys = random.randint(*KEY_COUNT_RANGE)
    n_locks = random.randint(*LOCK_COUNT_RANGE)
    height, width = _grid_shape(n_keys, n_locks)
    if max(height, width) > 30:
        return None

    bg, sep = random.sample(range(10), 2)
    palette = [c for c in range(10) if c not in (bg, sep)]
    if len(palette) < n_keys + n_locks:
        return None
    colors = random.sample(palette, n_keys + n_locks)
    key_colors = colors[:n_keys]
    lock_colors = colors[n_keys:]

    row_classes = random.sample(range(len(SHAPE_CLASSES)), n_keys)
    lock_classes = _sample_lock_classes(row_classes, n_locks)
    body_positions = _sample_body_positions(n_locks)
    if lock_classes is None or body_positions is None:
        return None

    key_shapes = [random.choice(_rotations(SHAPE_CLASSES[idx])) for idx in row_classes]
    lock_shapes = [random.choice(_rotations(SHAPE_CLASSES[idx])) for idx in lock_classes]
    matches = _match_matrix(key_shapes, lock_shapes)
    if not matches.any() or bool(matches.all()):
        return None

    block_rows = n_keys + 1
    block_cols = n_locks + 1
    row_spans = _block_spans(block_rows, BLOCK_H)
    col_spans = _block_spans(block_cols, BLOCK_W)

    grid = np.full((height, width), int(bg), dtype=int)
    for r in _separator_indices(block_rows, BLOCK_H):
        grid[r, :] = int(sep)
    for c in _separator_indices(block_cols, BLOCK_W):
        grid[:, c] = int(sep)

    for i, key_shape in enumerate(key_shapes):
        block = _block(grid, row_spans[i + 1], col_spans[0])
        _draw_key(block, key_shape, key_colors[i])

    for j, lock_shape in enumerate(lock_shapes):
        block = _block(grid, row_spans[0], col_spans[j + 1])
        _draw_lock(block, lock_shape, lock_colors[j], bg, body_positions[j])

    expected = _render_expected(
        grid,
        bg,
        row_spans,
        col_spans,
        key_shapes,
        key_colors,
        lock_shapes,
        lock_colors,
    )

    if np.array_equal(expected, grid):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    internal = np.zeros(grid.shape, dtype=bool)
    for row_span in row_spans[1:]:
        for col_span in col_spans[1:]:
            r0, r1 = row_span
            c0, c1 = col_span
            internal[r0 : r1 + 1, c0 : c1 + 1] = True
    changed_internal = (actual != grid) & internal
    if not changed_internal.any():
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid lock-and-key table input")
