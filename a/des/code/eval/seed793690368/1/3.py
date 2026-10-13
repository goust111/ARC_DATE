from __future__ import annotations

from collections import deque
import numpy as np


BG = 0
FRAME = 5
NON_FRAME_COLORS = tuple(color for color in range(1, 10) if color != FRAME)
GRID_SHAPES = (
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 2),
    (4, 3),
)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_PAD = 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _majority_color(values: np.ndarray) -> int:
    colors, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(colors[np.argmax(counts)])


def _frame_structure(grid: np.ndarray) -> dict[str, object]:
    grey = np.argwhere(grid == FRAME)
    assert grey.size > 0, "expected a grey frame"
    r0, c0 = map(int, grey.min(axis=0))
    r1, c1 = map(int, grey.max(axis=0))

    h_lines = [
        r for r in range(r0, r1 + 1) if np.all(grid[r, c0 : c1 + 1] == FRAME)
    ]
    v_lines = [
        c for c in range(c0, c1 + 1) if np.all(grid[r0 : r1 + 1, c] == FRAME)
    ]
    assert len(h_lines) >= 2 and len(v_lines) >= 2, "frame must have grid lines"

    row_spans = [(a + 1, b - 1) for a, b in zip(h_lines, h_lines[1:]) if b - a > 1]
    col_spans = [(a + 1, b - 1) for a, b in zip(v_lines, v_lines[1:]) if b - a > 1]
    assert row_spans and col_spans, "frame cells must have non-empty interiors"

    cell_heights = {re - rs + 1 for rs, re in row_spans}
    cell_widths = {ce - cs + 1 for cs, ce in col_spans}
    assert len(cell_heights) == 1 and len(cell_widths) == 1, "cells must be uniform"

    return {
        "bbox": (r0, c0, r1, c1),
        "h_lines": h_lines,
        "v_lines": v_lines,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "cell_h": int(next(iter(cell_heights))),
        "cell_w": int(next(iter(cell_widths))),
    }


def _single_marker(values: np.ndarray) -> int:
    idx = np.flatnonzero(np.asarray(values, dtype=int) != BG)
    assert len(idx) == 1, "expected exactly one marker pixel in each segment"
    return int(values[int(idx[0])])


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    info = _frame_structure(g)
    frame_r0, frame_c0, frame_r1, frame_c1 = info["bbox"]
    row_spans: list[tuple[int, int]] = info["row_spans"]  # type: ignore[assignment]
    col_spans: list[tuple[int, int]] = info["col_spans"]  # type: ignore[assignment]
    cell_h = int(info["cell_h"])
    cell_w = int(info["cell_w"])

    top_marker_row = frame_r0 - 1
    left_marker_col = frame_c0 - 1
    assert top_marker_row >= 0 and left_marker_col >= 0, "markers must sit outside frame"

    col_colors = [
        _single_marker(g[top_marker_row, cs : ce + 1]) for cs, ce in col_spans
    ]
    row_colors = [
        _single_marker(g[rs : re + 1, left_marker_col]) for rs, re in row_spans
    ]
    assert len(set(col_colors)) == len(col_colors), "column colors must be unique"
    assert len(set(row_colors)) == len(row_colors), "row colors must be unique"
    col_lookup = {color: idx for idx, color in enumerate(col_colors)}
    row_lookup = {color: idx for idx, color in enumerate(row_colors)}

    out = g.copy()
    mask = (g != BG) & (g != FRAME)
    for coords in _components(mask):
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        if frame_r0 <= r0 and r1 <= frame_r1 and frame_c0 <= c0 and c1 <= frame_c1:
            continue
        if int(coords.shape[0]) == 1:
            continue

        box_h = r1 - r0 + 1
        box_w = c1 - c0 + 1
        if box_h != cell_h + 1 or box_w != cell_w + 1:
            continue
        if int(coords.shape[0]) != box_h * box_w:
            continue

        piece = g[r0 : r1 + 1, c0 : c1 + 1]
        col_color = _majority_color(piece[0, :])
        row_color = _majority_color(piece[:, 0])
        if col_color not in col_lookup or row_color not in row_lookup:
            continue

        out[coords[:, 0], coords[:, 1]] = BG
        row_idx = row_lookup[row_color]
        col_idx = col_lookup[col_color]
        rs, re = row_spans[row_idx]
        cs, ce = col_spans[col_idx]
        out[rs : re + 1, cs : ce + 1] = piece[1:, 1:]

    return out


def _choose_shape(rng: np.random.Generator) -> tuple[int, int]:
    idx = int(rng.integers(0, len(GRID_SHAPES)))
    return GRID_SHAPES[idx]


def _pattern_colors(rng: np.random.Generator) -> np.ndarray:
    count = int(rng.integers(2, 5))
    return rng.choice(np.array(NON_FRAME_COLORS, dtype=int), size=count, replace=False)


def _make_pattern(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    colors = _pattern_colors(rng)
    pattern = np.empty((h, w), dtype=int)
    mode = int(rng.integers(0, 6))

    if mode == 0:
        pattern[:, :] = int(colors[0])
        r0 = int(rng.integers(0, h - 1))
        r1 = int(rng.integers(r0 + 1, h + 1))
        c0 = int(rng.integers(0, w - 1))
        c1 = int(rng.integers(c0 + 1, w + 1))
        pattern[r0:r1, c0:c1] = int(colors[1])
        if len(colors) > 2:
            stripe = int(rng.integers(0, h))
            pattern[stripe, :] = int(colors[2])
    elif mode == 1:
        period = int(rng.integers(1, min(3, h) + 1))
        for r in range(h):
            pattern[r, :] = int(colors[(r // period) % len(colors)])
    elif mode == 2:
        period = int(rng.integers(1, min(3, w) + 1))
        for c in range(w):
            pattern[:, c] = int(colors[(c // period) % len(colors)])
    elif mode == 3:
        step_r = int(rng.integers(1, 3))
        step_c = int(rng.integers(1, 3))
        for r in range(h):
            for c in range(w):
                pattern[r, c] = int(colors[((r // step_r) + (c // step_c)) % len(colors)])
    elif mode == 4:
        mid_r = int(rng.integers(1, h))
        mid_c = int(rng.integers(1, w))
        pattern[:mid_r, :mid_c] = int(colors[0])
        pattern[:mid_r, mid_c:] = int(colors[1 % len(colors)])
        pattern[mid_r:, :mid_c] = int(colors[2 % len(colors)])
        pattern[mid_r:, mid_c:] = int(colors[3 % len(colors)])
    else:
        for r in range(h):
            for c in range(w):
                pattern[r, c] = int(colors[(2 * r + c) % len(colors)])

    assert np.all(pattern != BG) and np.all(pattern != FRAME)
    return pattern


def _reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = PLACEMENT_PAD) -> None:
    rr0 = max(0, r0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = PLACEMENT_PAD) -> bool:
    rr0 = max(0, r0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _scatter_positions(
    rng: np.random.Generator,
    occupied: np.ndarray,
    piece_h: int,
    piece_w: int,
    count: int,
) -> list[tuple[int, int]] | None:
    h, w = occupied.shape
    candidates = [
        (r, c)
        for r in range(h - piece_h + 1)
        for c in range(w - piece_w + 1)
        if _can_place(occupied, r, c, piece_h, piece_w)
    ]
    if len(candidates) < count:
        return None

    order = rng.permutation(len(candidates))
    chosen: list[tuple[int, int]] = []
    for idx in order:
        r0, c0 = candidates[int(idx)]
        if not _can_place(occupied, r0, c0, piece_h, piece_w):
            continue
        _reserve(occupied, r0, c0, piece_h, piece_w)
        chosen.append((r0, c0))
        if len(chosen) == count:
            return chosen
    return None


def _build_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    n_rows, n_cols = _choose_shape(rng)
    cell_h = int(rng.integers(3, 6))
    cell_w = int(rng.integers(3, 6))
    piece_h = cell_h + 1
    piece_w = cell_w + 1

    label_colors = rng.choice(
        np.array(NON_FRAME_COLORS, dtype=int),
        size=n_rows + n_cols,
        replace=False,
    )
    row_colors = [int(color) for color in label_colors[:n_rows]]
    col_colors = [int(color) for color in label_colors[n_rows:]]

    frame_h = n_rows * cell_h + (n_rows + 1)
    frame_w = n_cols * cell_w + (n_cols + 1)
    top_margin = piece_h + 3 + int(rng.integers(0, 3))
    bottom_margin = piece_h + 3 + int(rng.integers(0, 3))
    left_margin = piece_w + 3 + int(rng.integers(0, 3))
    right_margin = piece_w + 3 + int(rng.integers(0, 3))

    height = top_margin + frame_h + bottom_margin
    width = left_margin + frame_w + right_margin
    frame_r0 = top_margin
    frame_c0 = left_margin
    frame_r1 = frame_r0 + frame_h - 1
    frame_c1 = frame_c0 + frame_w - 1

    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)

    h_lines = [frame_r0 + i * (cell_h + 1) for i in range(n_rows + 1)]
    v_lines = [frame_c0 + j * (cell_w + 1) for j in range(n_cols + 1)]
    row_spans = [(h_lines[i] + 1, h_lines[i + 1] - 1) for i in range(n_rows)]
    col_spans = [(v_lines[j] + 1, v_lines[j + 1] - 1) for j in range(n_cols)]

    for r in h_lines:
        inp[r, frame_c0 : frame_c1 + 1] = FRAME
        out[r, frame_c0 : frame_c1 + 1] = FRAME
    for c in v_lines:
        inp[frame_r0 : frame_r1 + 1, c] = FRAME
        out[frame_r0 : frame_r1 + 1, c] = FRAME

    top_marker_row = frame_r0 - 1
    left_marker_col = frame_c0 - 1
    marker_cells: list[tuple[int, int]] = []
    for j, color in enumerate(col_colors):
        cs, ce = col_spans[j]
        marker_c = (cs + ce) // 2
        inp[top_marker_row, marker_c] = color
        out[top_marker_row, marker_c] = color
        marker_cells.append((top_marker_row, marker_c))
    for i, color in enumerate(row_colors):
        rs, re = row_spans[i]
        marker_r = (rs + re) // 2
        inp[marker_r, left_marker_col] = color
        out[marker_r, left_marker_col] = color
        marker_cells.append((marker_r, left_marker_col))

    cells = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    if bool(rng.integers(0, 2)):
        chosen_cells = cells
    else:
        piece_count = int(rng.integers(1, len(cells)))
        chosen_idx = rng.choice(len(cells), size=piece_count, replace=False)
        chosen_cells = [cells[int(idx)] for idx in np.sort(chosen_idx)]

    pieces: list[np.ndarray] = []
    for i, j in chosen_cells:
        pattern = _make_pattern(rng, cell_h, cell_w)
        rs, re = row_spans[i]
        cs, ce = col_spans[j]
        out[rs : re + 1, cs : ce + 1] = pattern

        piece = np.empty((piece_h, piece_w), dtype=int)
        piece[0, :] = col_colors[j]
        piece[1:, 0] = row_colors[i]
        piece[1:, 1:] = pattern
        pieces.append(piece)

    occupied = np.zeros((height, width), dtype=bool)
    _reserve(occupied, frame_r0, frame_c0, frame_h, frame_w)
    for marker_r, marker_c in marker_cells:
        _reserve(occupied, marker_r, marker_c, 1, 1)

    positions = _scatter_positions(rng, occupied, piece_h, piece_w, len(pieces))
    if positions is None:
        return None

    order = rng.permutation(len(pieces))
    for piece_idx, pos_idx in enumerate(order):
        piece = pieces[piece_idx]
        r0, c0 = positions[int(pos_idx)]
        inp[r0 : r0 + piece_h, c0 : c0 + piece_w] = piece

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _build_case(rng)
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid input")
