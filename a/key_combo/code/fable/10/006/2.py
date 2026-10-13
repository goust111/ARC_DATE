from __future__ import annotations

import random
from collections import Counter

import numpy as np


PALETTE = list(range(10))
SIZE_RANGE = (3, 5)
MAX_ATTEMPTS = 400


def _colors_present(grid: np.ndarray) -> list[int]:
    return sorted(set(map(int, grid.flat)))


def _separator_row(grid: np.ndarray) -> int:
    height, width = map(int, grid.shape)
    counts = Counter(map(int, grid.flat))
    rows = []
    for r in range(height):
        color = int(grid[r, 0])
        if np.all(grid[r] == color) and counts[color] == width:
            rows.append(r)
    assert len(rows) == 1, "expected exactly one separator row"
    return int(rows[0])


def _decode_layers(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep = _separator_row(g)
    assert 2 * sep + 1 == g.shape[0], "separator must split the grid into equal-height layers"

    upper = g[:sep].copy()
    lower = g[sep + 1 :].copy()
    assert upper.shape == lower.shape and upper.size > 0

    upper_colors = set(_colors_present(upper))
    lower_colors = set(_colors_present(lower))
    shared = upper_colors & lower_colors
    assert len(shared) == 1, "layers must share exactly one background color"
    bg = int(next(iter(shared)))

    upper_fg = upper_colors - {bg}
    lower_fg = lower_colors - {bg}
    assert len(upper_fg) == 1, "upper layer must have one foreground color"
    assert len(lower_fg) == 1, "lower layer must have one foreground color"
    top_fg = int(next(iter(upper_fg)))
    bottom_fg = int(next(iter(lower_fg)))
    assert top_fg != bottom_fg

    return upper, lower, bg, top_fg, bottom_fg


def _symbol_and_sources(
    upper: np.ndarray, lower: np.ndarray, bg: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    top_mask = upper != bg
    bottom_mask = lower != bg
    top_only = top_mask & ~bottom_mask
    bottom_only = bottom_mask & ~top_mask
    overlap = top_mask & bottom_mask
    xor_mask = top_only | bottom_only
    return xor_mask, top_only, bottom_only, overlap


def _render_output(
    xor_mask: np.ndarray,
    top_only: np.ndarray,
    bottom_only: np.ndarray,
    bg: int,
    top_fg: int,
    bottom_fg: int,
) -> np.ndarray:
    h, w = map(int, xor_mask.shape)
    out = np.full((h * h, w * w), bg, dtype=int)
    for r in range(h):
        for c in range(w):
            if not xor_mask[r, c]:
                continue
            color = top_fg if top_only[r, c] else bottom_fg
            block = out[r * h : (r + 1) * h, c * w : (c + 1) * w]
            block[xor_mask] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge separator-stacked masks by xor and stamp the xor symbol into each active cell with outer-source coloring.

    Input structure:
    - One full-width separator row of a unique color splits the grid into equal-size upper and lower layers.
    - The two layers share one background color, while the upper and lower layers each use one different foreground color.
    - Aligned cells can therefore be upper-only foreground, lower-only foreground, both foreground, or both background.

    Rule:
    - Remove the separator row and align the two layers cell by cell.
    - Build a binary symbol whose cells are on exactly where one layer is foreground and the other is background.
    - Produce a self-nested output of shape (k*k, m*m): every on cell becomes a full copy of that binary symbol, every off cell becomes a blank block, and each copied symbol is colored uniformly by the layer that made the outer on cell active.

    Key insight:
    - The visible symbol is the xor of the two masks, but the color of each stamped copy still records whether that outer xor cell came from the upper layer or the lower layer.
    """
    upper, lower, bg, top_fg, bottom_fg = _decode_layers(np.asarray(grid, dtype=int))
    xor_mask, top_only, bottom_only, overlap = _symbol_and_sources(upper, lower, bg)

    assert overlap.any(), "expected at least one overlapped foreground cell"
    assert int(top_only.sum()) > 1 and int(bottom_only.sum()) > 1, "each layer needs multiple exclusive cells"
    assert xor_mask.any() and (~xor_mask).any(), "symbol must contain both on and off cells"

    return _render_output(xor_mask, top_only, bottom_only, bg, top_fg, bottom_fg)


def _full_span(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()) == 0
        and int(rows.max()) == mask.shape[0] - 1
        and int(cols.min()) == 0
        and int(cols.max()) == mask.shape[1] - 1
    )


def _mixed_axes(mask: np.ndarray) -> bool:
    row_mixed = sum(0 < int(row.sum()) < mask.shape[1] for row in mask) >= 2
    col_mixed = sum(0 < int(col.sum()) < mask.shape[0] for col in mask.T) >= 2
    return row_mixed and col_mixed


def _sample_xor_mask(h: int, w: int) -> np.ndarray | None:
    total = h * w
    lo = max(4, (total + 2) // 3)
    hi = min(total - 2, (2 * total) // 3)
    if lo > hi:
        return None

    for _ in range(120):
        mask = np.zeros((h, w), dtype=bool)
        n_open = random.randint(lo, hi)
        cells = random.sample(range(total), n_open)
        mask.flat[cells] = True
        if not _full_span(mask):
            continue
        if not _mixed_axes(mask):
            continue
        return mask
    return None


def _sample_states(h: int, w: int) -> np.ndarray | None:
    xor_mask = _sample_xor_mask(h, w)
    if xor_mask is None:
        return None

    states = np.zeros((h, w), dtype=int)
    open_idx = [int(i) for i in np.flatnonzero(xor_mask)]
    closed_idx = [int(i) for i in np.flatnonzero(~xor_mask)]
    if len(open_idx) < 4 or len(closed_idx) < 2:
        return None

    random.shuffle(open_idx)
    random.shuffle(closed_idx)

    top_need = random.randint(2, len(open_idx) - 2)
    bottom_need = len(open_idx) - top_need
    if bottom_need < 2:
        return None

    states.flat[open_idx[:top_need]] = 1
    states.flat[open_idx[top_need:]] = 2
    states.flat[closed_idx[0]] = 3
    for idx in closed_idx[1:]:
        states.flat[idx] = 0 if random.random() < 0.7 else 3

    top_only = states == 1
    bottom_only = states == 2
    overlap = states == 3
    if int(top_only.sum()) < 2 or int(bottom_only.sum()) < 2 or int(overlap.sum()) < 1:
        return None
    if not (states == 0).any():
        return None
    return states


def _states_to_input(states: np.ndarray, bg: int, top_fg: int, bottom_fg: int, sep: int) -> np.ndarray:
    upper = np.full(states.shape, bg, dtype=int)
    lower = np.full(states.shape, bg, dtype=int)
    upper[(states == 1) | (states == 3)] = top_fg
    lower[(states == 2) | (states == 3)] = bottom_fg
    return np.vstack([upper, np.full((1, states.shape[1]), sep, dtype=int), lower])


def _wrong_union_output(
    states: np.ndarray, bg: int, top_fg: int, bottom_fg: int
) -> np.ndarray:
    top_mask = (states == 1) | (states == 3)
    bottom_mask = (states == 2) | (states == 3)
    union_mask = top_mask | bottom_mask
    topish = states == 1
    bottomish = states == 2
    overlap = states == 3
    h, w = states.shape
    out = np.full((h * h, w * w), bg, dtype=int)
    for r in range(h):
        for c in range(w):
            if not union_mask[r, c]:
                continue
            color = top_fg if (topish[r, c] or overlap[r, c]) else bottom_fg
            block = out[r * h : (r + 1) * h, c * w : (c + 1) * w]
            block[union_mask] = color
    return out


def _wrong_single_color_output(states: np.ndarray, bg: int, top_fg: int) -> np.ndarray:
    xor_mask = (states == 1) | (states == 2)
    h, w = states.shape
    out = np.full((h * h, w * w), bg, dtype=int)
    for r in range(h):
        for c in range(w):
            if not xor_mask[r, c]:
                continue
            block = out[r * h : (r + 1) * h, c * w : (c + 1) * w]
            block[xor_mask] = top_fg
    return out


def _wrong_solid_block_output(
    states: np.ndarray, bg: int, top_fg: int, bottom_fg: int
) -> np.ndarray:
    h, w = states.shape
    out = np.full((h * h, w * w), bg, dtype=int)
    for r in range(h):
        for c in range(w):
            if states[r, c] == 1:
                out[r * h : (r + 1) * h, c * w : (c + 1) * w] = top_fg
            elif states[r, c] == 2:
                out[r * h : (r + 1) * h, c * w : (c + 1) * w] = bottom_fg
    return out


def _wrong_inner_source_output(
    states: np.ndarray, bg: int, top_fg: int, bottom_fg: int
) -> np.ndarray:
    top_only = states == 1
    bottom_only = states == 2
    xor_mask = top_only | bottom_only
    h, w = states.shape
    out = np.full((h * h, w * w), bg, dtype=int)
    for r in range(h):
        for c in range(w):
            if not xor_mask[r, c]:
                continue
            block = out[r * h : (r + 1) * h, c * w : (c + 1) * w]
            block[top_only] = top_fg
            block[bottom_only] = bottom_fg
    return out


def _reject_shortcuts(states: np.ndarray, expected: np.ndarray, bg: int, top_fg: int, bottom_fg: int) -> bool:
    wrong_candidates = [
        _wrong_union_output(states, bg, top_fg, bottom_fg),
        _wrong_single_color_output(states, bg, top_fg),
        _wrong_solid_block_output(states, bg, top_fg, bottom_fg),
        _wrong_inner_source_output(states, bg, top_fg, bottom_fg),
    ]
    return all(not np.array_equal(expected, wrong) for wrong in wrong_candidates)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*SIZE_RANGE)
    w = random.randint(*SIZE_RANGE)
    states = _sample_states(h, w)
    if states is None:
        return None

    bg, top_fg, bottom_fg, sep = random.sample(PALETTE, 4)
    inp = _states_to_input(states, bg, top_fg, bottom_fg, sep)
    xor_mask = (states == 1) | (states == 2)
    expected = _render_output(xor_mask, states == 1, states == 2, bg, top_fg, bottom_fg)

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if not _reject_shortcuts(states, expected, bg, top_fg, bottom_fg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid layered xor nesting input")
