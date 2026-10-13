from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
NON_BG_COLORS = tuple(range(1, 10))
MIN_SEGMENTS = 3
MAX_SEGMENTS = 5
MIN_SEGMENT_H = 4
MAX_SEGMENT_H = 8
MIN_SEGMENT_W = 4
MAX_SEGMENT_W = 6
GEN_RETRIES = 400


def _top_view(layers: list[np.ndarray], bg: int = BACKGROUND) -> np.ndarray:
    out = np.full(layers[0].shape, bg, dtype=int)
    filled = np.zeros(layers[0].shape, dtype=bool)
    for layer in reversed(layers):
        mask = (layer != bg) & (~filled)
        out[mask] = layer[mask]
        filled |= mask
    return out


def _fold_stack_onto_segment(stack: list[np.ndarray], segment: np.ndarray) -> list[np.ndarray]:
    moved = [np.fliplr(layer) for layer in reversed(stack)]
    return [segment.copy(), *moved]


def _separator_spec(grid: np.ndarray) -> tuple[int, list[int], int]:
    h, w = map(int, grid.shape)
    candidates: list[tuple[int, list[int], int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BACKGROUND:
            continue
        sep_cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if len(sep_cols) < 2:
            continue
        occ_cols = set(np.flatnonzero(np.any(grid == color, axis=0)).tolist())
        if occ_cols != set(sep_cols):
            continue
        widths = [sep_cols[0]]
        widths.extend(sep_cols[i] - sep_cols[i - 1] - 1 for i in range(1, len(sep_cols)))
        widths.append(w - sep_cols[-1] - 1)
        if min(widths) <= 0 or len(set(widths)) != 1:
            continue
        seg_w = int(widths[0])
        if any(col % (seg_w + 1) != seg_w for col in sep_cols):
            continue
        candidates.append((color, sep_cols, seg_w))
    assert len(candidates) == 1, "expected one fold-line color partitioning equal-width segments"
    return candidates[0]


def _extract_segments(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    fold_color, sep_cols, seg_w = _separator_spec(grid)
    starts = [0] + [col + 1 for col in sep_cols]
    segments = [grid[:, start : start + seg_w].copy() for start in starts]
    assert len(segments) >= 3, "expected at least three strip segments"
    assert all(seg.shape == segments[0].shape for seg in segments)
    assert not np.any(np.isin(np.concatenate(segments, axis=1), [fold_color])), "fold color may only appear on fold lines"
    return segments, fold_color


def _solve_segments(segments: list[np.ndarray]) -> np.ndarray:
    stack = [segments[0].copy()]
    for seg in segments[1:]:
        stack = _fold_stack_onto_segment(stack, seg)
    return _top_view(stack)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the segmented strip from left to right like a rolled map and show the final top view on the rightmost segment.

    Input structure:
    - A black-background strip is divided by one color's full-height vertical fold lines into at least three equal-width segments.
    - The fold-line color appears only on those separator columns.
    - Each segment may contain sparse colored pixels on the black background.

    Rule:
    - Repeatedly take the whole stack lying left of the current leftmost fold line and fold it rightward onto the adjacent segment.
    - Every fold mirrors the moved stack left-right, places it above the destination stack, and reverses the moved stack's internal layer order.
    - After all folds, output only the size of the final rightmost segment, taking at each cell the topmost non-black pixel and leaving black where every layer is black.

    Key insight:
    - The result depends on both composed horizontal reflections and the fact that each whole-stack fold reverses layer order, so neither plain translation nor one-shot mirroring is sufficient.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    segments, _fold_color = _extract_segments(g)
    return _solve_segments(segments)


def _build_tokens(
    n_segments: int,
    *,
    reverse_left: bool = True,
    mirror_left: bool = True,
) -> list[tuple[int, bool]]:
    tokens = [(0, False)]
    for seg in range(1, n_segments):
        left = list(reversed(tokens)) if reverse_left else list(tokens)
        moved = [(idx, (not flipped) if mirror_left else flipped) for idx, flipped in left]
        tokens = [(seg, False), *moved]
    return tokens


def _render_tokens(
    segments: list[np.ndarray],
    tokens: list[tuple[int, bool]],
    *,
    top_nonblack: bool = True,
) -> np.ndarray:
    layers = [np.fliplr(segments[idx]) if flipped else segments[idx] for idx, flipped in tokens]
    if top_nonblack:
        return _top_view(layers)
    return layers[-1].copy()


def _visible_segment_ids(
    segments: list[np.ndarray],
    tokens: list[tuple[int, bool]],
) -> set[int]:
    h, w = map(int, segments[0].shape)
    visible = -np.ones((h, w), dtype=int)
    filled = np.zeros((h, w), dtype=bool)
    layers = [(idx, np.fliplr(segments[idx]) if flipped else segments[idx]) for idx, flipped in tokens]
    for idx, layer in reversed(layers):
        mask = (layer != BACKGROUND) & (~filled)
        visible[mask] = idx
        filled |= mask
    return {int(x) for x in visible[visible >= 0].ravel()}


def _pick_cell(
    height: int,
    width: int,
    used: set[tuple[int, int]],
    *,
    require_asymmetric_col: bool = False,
    mirror_cell_must_be_free: bool = False,
) -> tuple[int, int] | None:
    candidates = []
    for r in range(height):
        for c in range(width):
            if (r, c) in used:
                continue
            if require_asymmetric_col and c == width - 1 - c:
                continue
            if mirror_cell_must_be_free and (r, width - 1 - c) in used:
                continue
            candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _paint_for_final_cell(
    segments: list[np.ndarray],
    final_flips: dict[int, bool],
    seg_idx: int,
    final_cell: tuple[int, int],
    color: int,
) -> None:
    r, c = final_cell
    seg_w = int(segments[seg_idx].shape[1])
    src_c = seg_w - 1 - c if final_flips[seg_idx] else c
    segments[seg_idx][r, src_c] = color


def _assemble_input_grid(segments: list[np.ndarray], fold_color: int) -> np.ndarray:
    rows = []
    for idx, seg in enumerate(segments):
        rows.append(seg)
        if idx + 1 != len(segments):
            rows.append(np.full((seg.shape[0], 1), fold_color, dtype=int))
    return np.concatenate(rows, axis=1)


def _gen_attempt() -> np.ndarray | None:
    n_segments = random.randint(MIN_SEGMENTS, MAX_SEGMENTS)
    max_seg_w = min(MAX_SEGMENT_W, (30 - (n_segments - 1)) // n_segments)
    if max_seg_w < MIN_SEGMENT_W:
        return None
    seg_h = random.randint(MIN_SEGMENT_H, MAX_SEGMENT_H)
    seg_w = random.randint(MIN_SEGMENT_W, max_seg_w)

    fold_color = random.choice(NON_BG_COLORS)
    paint_colors = [c for c in NON_BG_COLORS if c != fold_color]
    segments = [np.zeros((seg_h, seg_w), dtype=int) for _ in range(n_segments)]

    correct_tokens = _build_tokens(n_segments)
    no_reverse_tokens = _build_tokens(n_segments, reverse_left=False, mirror_left=True)
    no_mirror_tokens = _build_tokens(n_segments, reverse_left=True, mirror_left=False)
    final_flips = {idx: flipped for idx, flipped in correct_tokens}
    correct_pos = {idx: pos for pos, (idx, _flipped) in enumerate(correct_tokens)}
    wrong_pos = {idx: pos for pos, (idx, _flipped) in enumerate(no_reverse_tokens)}

    inversion_pairs = [
        (a, b)
        for a in range(n_segments)
        for b in range(a + 1, n_segments)
        if (correct_pos[a] - correct_pos[b]) * (wrong_pos[a] - wrong_pos[b]) < 0
    ]
    if not inversion_pairs:
        return None

    flipped_segments = [idx for idx, flipped in final_flips.items() if flipped]
    if not flipped_segments:
        return None
    mirror_seg = random.choice(flipped_segments)

    used_cells: set[tuple[int, int]] = set()

    for seg_idx in range(n_segments):
        if seg_idx == mirror_seg:
            cell = _pick_cell(
                seg_h,
                seg_w,
                used_cells,
                require_asymmetric_col=True,
                mirror_cell_must_be_free=True,
            )
            if cell is None:
                return None
            used_cells.add(cell)
            used_cells.add((cell[0], seg_w - 1 - cell[1]))
        else:
            cell = _pick_cell(seg_h, seg_w, used_cells)
            if cell is None:
                return None
            used_cells.add(cell)
        _paint_for_final_cell(segments, final_flips, seg_idx, cell, random.choice(paint_colors))

    a, b = random.choice(inversion_pairs)
    inversion_cell = _pick_cell(seg_h, seg_w, used_cells)
    if inversion_cell is None:
        return None
    color_a, color_b = random.sample(paint_colors, 2)
    _paint_for_final_cell(segments, final_flips, a, inversion_cell, color_a)
    _paint_for_final_cell(segments, final_flips, b, inversion_cell, color_b)

    grid = _assemble_input_grid(segments, fold_color)
    correct = _render_tokens(segments, correct_tokens)
    no_reverse = _render_tokens(segments, no_reverse_tokens)
    no_mirror = _render_tokens(segments, no_mirror_tokens)
    top_layer_only = _render_tokens(segments, correct_tokens, top_nonblack=False)

    if not np.array_equal(main(grid), correct):
        return None
    if np.array_equal(correct, no_reverse):
        return None
    if np.array_equal(correct, no_mirror):
        return None
    if np.array_equal(correct, top_layer_only):
        return None
    if _visible_segment_ids(segments, correct_tokens) != set(range(n_segments)):
        return None
    if len(np.unique(correct[correct != BACKGROUND])) < min(3, n_segments):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folding-strip input")
