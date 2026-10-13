from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


def _load_arc_utils():
    here = Path(__file__).resolve()
    candidates = [Path("/home/lixinyu/snap/new/solvers")]
    if len(here.parents) >= 7:
        candidates.append(here.parents[6] / "new" / "solvers")
    for candidate in candidates:
        if candidate.exists():
            path = str(candidate)
            if path not in sys.path:
                sys.path.insert(0, path)
            break
    from arc_utils import Color, bbox, crop_bbox, paste

    return Color, bbox, crop_bbox, paste


Color, bbox, crop_bbox, paste = _load_arc_utils()


def _uniform_block_color(block: np.ndarray) -> int:
    color = int(block[0, 0])
    assert np.all(block == color), "macro-cell must be uniform"
    return color


def _selector_from_right(right: np.ndarray, stamp_shape: tuple[int, int]) -> np.ndarray:
    stamp_h, stamp_w = map(int, stamp_shape)
    h, w = map(int, right.shape)
    assert h % stamp_h == 0 and w % stamp_w == 0, "right zone must tile by the stamp size"

    macro_h = h // stamp_h
    macro_w = w // stamp_w
    assert 2 <= macro_h <= 4 and 2 <= macro_w <= 4

    selector = np.zeros((macro_h, macro_w), dtype=bool)
    for mr in range(macro_h):
        for mc in range(macro_w):
            block = right[
                mr * stamp_h : (mr + 1) * stamp_h,
                mc * stamp_w : (mc + 1) * stamp_w,
            ]
            color = _uniform_block_color(block)
            assert color in (Color.BLACK, Color.ORANGE), "right zone must use only black/orange macro-cells"
            selector[mr, mc] = color == Color.ORANGE

    assert np.any(selector), "expected at least one active macro-cell"
    return selector


def _completed_stamp_from_left(left: np.ndarray) -> np.ndarray:
    marker_coords = np.argwhere(left == Color.PINK)
    assert marker_coords.shape == (1, 2), "expected one magenta center marker"
    cr, cc = map(int, marker_coords[0])

    expected_divider = np.full(left.shape[0], Color.YELLOW, dtype=int)
    expected_divider[cr] = Color.PINK
    assert np.array_equal(left[:, cc], expected_divider), "yellow divider column must pass through the center marker"
    assert int(np.count_nonzero(left == Color.YELLOW)) == left.shape[0] - 1

    source_mask = (left != Color.BLACK) & (left != Color.YELLOW) & (left != Color.PINK)
    source_coords = np.argwhere(source_mask)
    assert 3 <= len(source_coords) <= 8
    assert np.all(source_coords[:, 0] < cr) and np.all(source_coords[:, 1] < cc), "source must lie strictly in the upper-left quadrant"

    completed = np.zeros_like(left)
    completed[source_mask] = left[source_mask]
    for r, c in source_coords:
        rr = 2 * cr - int(r)
        rc = 2 * cc - int(c)
        assert 0 <= rr < left.shape[0] and 0 <= rc < left.shape[1]
        assert completed[rr, rc] == Color.BLACK, "reflected target quadrant must be empty before completion"
        completed[rr, rc] = int(left[r, c])

    completed_colors = {int(v) for v in np.unique(completed) if int(v) != Color.BLACK}
    assert completed_colors == {Color.BLUE, Color.RED}, "stamp colors must be exactly blue and red"

    stamp_box = bbox(completed != Color.BLACK)
    assert stamp_box is not None
    stamp = crop_bbox(completed, stamp_box)
    assert stamp.shape[0] % 2 == 1 and stamp.shape[1] % 2 == 1
    return stamp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a half-pattern by point symmetry and use the result as a stamp over the activated macro-cells on the right.

    Input structure:
    - A grey full-height separator column splits the black-background grid into a left clue zone and a right macro-grid zone.
    - In the left zone, one magenta center marker lies on a full yellow vertical divider column, and a blue/red 4-connected source pattern appears only in the upper-left quadrant of that center.
    - In the right zone, the cells are partitioned into uniform black/orange macro-cells whose size matches the completed left-side stamp bbox.

    Rule:
    - Reflect every blue/red source pixel through the magenta center by 180 degrees, drawing the reflected copy into the lower-right quadrant with the same colors.
    - Ignore the yellow divider, the grey separator, and the magenta marker; crop the bbox of the completed blue/red foreground to obtain a stamp whose black cells are transparent.
    - Output only the right zone, replacing each orange macro-cell by one copy of that stamp while leaving black macro-cells empty.

    Key insight:
    - The left figure is not the final answer: it is a half-defined template whose full stamp shape is determined only after point-symmetry completion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_cols = np.flatnonzero(np.all(g == Color.GREY, axis=0))
    assert len(sep_cols) == 1, "expected one grey separator column"
    sep_c = int(sep_cols[0])
    assert 0 < sep_c < g.shape[1] - 1

    left = g[:, :sep_c]
    right = g[:, sep_c + 1 :]
    stamp = _completed_stamp_from_left(left)
    selector = _selector_from_right(right, stamp.shape)

    out = np.zeros_like(right)
    stamp_h, stamp_w = stamp.shape
    for mr, mc in np.argwhere(selector):
        paste(out, stamp, int(mr) * stamp_h, int(mc) * stamp_w, transparent=Color.BLACK)
    return out


_STAMP_HALF_RANGE = (2, 3)
_SOURCE_SIZE_RANGE = (3, 8)
_MACRO_RANGE = (2, 4)
_LEFT_MARGIN_RANGE = (1, 3)
_RIGHT_MARGIN_RANGE = (1, 3)
_SELECTOR_ACTIVE_FRAC = (0.25, 0.7)
_GEN_ATTEMPTS = 400


def _grow_connected_shape(height: int, width: int, size: int) -> np.ndarray:
    cells = {(0, 0)}
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        if 0 <= r < height and 0 <= c < width and (r, c) not in cells:
            frontier.add((r, c))

    add_frontier(1, 0)
    add_frontier(0, 1)
    while len(cells) < size:
        assert frontier
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        cells.add((r, c))
        add_frontier(r - 1, c)
        add_frontier(r + 1, c)
        add_frontier(r, c - 1)
        add_frontier(r, c + 1)

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _color_shape(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    order = coords[np.argsort(coords[:, 0] + coords[:, 1])]
    values = [Color.BLUE] * len(order)
    red_count = random.randint(1, len(order) - 1)
    for idx in random.sample(range(len(order)), red_count):
        values[idx] = Color.RED

    sprite = np.zeros(mask.shape, dtype=int)
    for (r, c), color in zip(order, values):
        sprite[int(r), int(c)] = int(color)
    assert Color.BLUE in sprite and Color.RED in sprite
    return sprite


def _sample_selector(macro_h: int, macro_w: int) -> np.ndarray:
    total = macro_h * macro_w
    lo = max(1, int(np.ceil(total * _SELECTOR_ACTIVE_FRAC[0])))
    hi = min(total - 1 if total > 1 else 1, int(np.floor(total * _SELECTOR_ACTIVE_FRAC[1])))
    if hi < lo:
        hi = lo
    active = random.randint(lo, hi)

    selector = np.zeros((macro_h, macro_w), dtype=bool)
    selector.ravel()[random.sample(range(total), active)] = True
    return selector


def _render_stamp_output(stamp: np.ndarray, selector: np.ndarray) -> np.ndarray:
    stamp_h, stamp_w = stamp.shape
    macro_h, macro_w = selector.shape
    out = np.zeros((stamp_h * macro_h, stamp_w * macro_w), dtype=int)
    for mr, mc in np.argwhere(selector):
        paste(out, stamp, int(mr) * stamp_h, int(mc) * stamp_w, transparent=Color.BLACK)
    return out


def _build_left_zone(
    height: int,
    stamp_h: int,
    stamp_w: int,
    source: np.ndarray,
) -> np.ndarray:
    top_margin = random.randint(0, height - stamp_h)
    bottom_margin = height - stamp_h - top_margin
    left_margin = random.randint(*_LEFT_MARGIN_RANGE)
    right_margin = random.randint(*_RIGHT_MARGIN_RANGE)
    left_width = left_margin + stamp_w + right_margin

    left = np.zeros((height, left_width), dtype=int)
    a = stamp_h // 2
    b = stamp_w // 2
    cr = top_margin + a
    cc = left_margin + b
    left[:, cc] = Color.YELLOW
    left[cr, cc] = Color.PINK

    src_h, src_w = source.shape
    start_r = cr - a
    start_c = cc - b
    left[start_r : start_r + src_h, start_c : start_c + src_w][source != Color.BLACK] = source[source != Color.BLACK]
    assert bottom_margin >= 0
    return left


def _build_input(left: np.ndarray, selector: np.ndarray, stamp_shape: tuple[int, int]) -> np.ndarray:
    stamp_h, stamp_w = map(int, stamp_shape)
    right = np.kron(selector.astype(int), np.ones((stamp_h, stamp_w), dtype=int)) * Color.ORANGE
    grid = np.zeros((left.shape[0], left.shape[1] + 1 + right.shape[1]), dtype=int)
    grid[:, : left.shape[1]] = left
    grid[:, left.shape[1]] = Color.GREY
    grid[:, left.shape[1] + 1 :] = right
    return grid


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> None:
    sep_c = int(np.flatnonzero(np.all(inp == Color.GREY, axis=0))[0])
    left = inp[:, :sep_c]
    stamp = _completed_stamp_from_left(left)
    stamp_h, stamp_w = stamp.shape

    marker = np.argwhere(left == Color.PINK)[0]
    cr, cc = map(int, marker)
    source = (left != Color.BLACK) & (left != Color.YELLOW) & (left != Color.PINK)
    reflected_coords = {(2 * cr - int(r), 2 * cc - int(c)) for r, c in np.argwhere(source)}
    assert reflected_coords
    assert all(left[rr, rc] == Color.BLACK for rr, rc in reflected_coords), "input must leave the reflected half empty"

    assert out.shape[0] % stamp_h == 0 and out.shape[1] % stamp_w == 0
    assert np.count_nonzero(stamp == Color.BLACK) > 0, "stamp must contain transparent cells"

    color_set = {int(v) for v in np.unique(out) if int(v) != Color.BLACK}
    assert color_set <= {Color.BLUE, Color.RED}


def _gen_attempt() -> np.ndarray | None:
    half_h = random.randint(*_STAMP_HALF_RANGE)
    half_w = random.randint(*_STAMP_HALF_RANGE)
    src_area = half_h * half_w
    if src_area < _SOURCE_SIZE_RANGE[0]:
        return None

    source_size = random.randint(_SOURCE_SIZE_RANGE[0], min(_SOURCE_SIZE_RANGE[1], src_area))
    source_mask = _grow_connected_shape(half_h, half_w, source_size)
    source = _color_shape(source_mask)

    stamp_h = 2 * half_h + 1
    stamp_w = 2 * half_w + 1
    stamp = np.zeros((stamp_h, stamp_w), dtype=int)
    stamp[:half_h, :half_w] = source
    stamp = _completed_stamp_from_left(_build_left_zone(stamp_h, stamp_h, stamp_w, source))

    macro_h = random.randint(*_MACRO_RANGE)
    macro_w = random.randint(*_MACRO_RANGE)
    height = stamp_h * macro_h

    left = _build_left_zone(height, stamp_h, stamp_w, source)
    if left.shape[1] + 1 + stamp_w * macro_w > 30 or height > 30:
        return None

    selector = _sample_selector(macro_h, macro_w)
    inp = _build_input(left, selector, (stamp_h, stamp_w))
    expected = _render_stamp_output(_completed_stamp_from_left(left), selector)

    if inp.shape[0] > 30 or inp.shape[1] > 30:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    _semantic_checks(inp, expected)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 2")
