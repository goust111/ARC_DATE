from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _separator_color(g: np.ndarray) -> int:
    for color in sorted(set(map(int, g.ravel()))):
        full_rows = np.all(g == color, axis=1)
        full_cols = np.all(g == color, axis=0)
        if int(full_rows.sum()) >= 2 and int(full_cols.sum()) >= 2:
            return int(color)
    raise AssertionError("expected a panel separator color spanning full rows and columns")


def _panel_spans(indexes: list[int], size: int) -> list[tuple[int, int]]:
    lines = sorted(set(int(v) for v in indexes))
    if not lines or lines[0] != 0:
        lines = [0] + lines
    if lines[-1] != size - 1:
        lines.append(size - 1)

    spans = []
    for a, b in zip(lines, lines[1:]):
        if b - a > 1:
            spans.append((a + 1, b - 1))
    assert spans
    return spans


def _object_color(region: np.ndarray, bg: int) -> int | None:
    colors = [int(v) for v in np.unique(region) if int(v) != int(bg)]
    if not colors:
        return None
    assert len(colors) == 1
    return colors[0]


def _skeleton_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    assert len(rows) > 0 and len(cols) > 0

    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)

    main_rows = np.flatnonzero(row_counts == row_counts.max())
    main_cols = np.flatnonzero(col_counts == col_counts.max())
    center_row = int(main_rows[len(main_rows) // 2])
    center_col = int(main_cols[len(main_cols) // 2])

    skel = np.zeros_like(mask, dtype=bool)
    skel[center_row, mask[center_row]] = True
    skel[mask[:, center_col], center_col] = True
    return skel


def main(grid: np.ndarray) -> np.ndarray:
    """
    Thin each red object inside a separator-defined panel to its one-pixel centerline skeleton.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    sep = _separator_color(g)
    counts = Counter(map(int, g.ravel()))
    bg_candidates = [color for color, _count in counts.most_common() if int(color) != int(sep)]
    assert bg_candidates
    bg = int(bg_candidates[0])

    out = g.copy()
    full_rows = [int(r) for r in np.flatnonzero(np.all(g == sep, axis=1))]
    full_cols = [int(c) for c in np.flatnonzero(np.all(g == sep, axis=0))]
    row_spans = _panel_spans(full_rows, g.shape[0])
    col_spans = _panel_spans(full_cols, g.shape[1])

    heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(heights) == 1 and len(widths) == 1

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            region = g[r0 : r1 + 1, c0 : c1 + 1]
            obj = _object_color(region, bg)
            if obj is None:
                continue
            mask = region == obj
            skel = _skeleton_mask(mask)
            patch = np.full(region.shape, bg, dtype=int)
            patch[skel] = sep
            out[r0 : r1 + 1, c0 : c1 + 1] = patch

    return out


_BG = 5
_SEP = 1
_OBJ = 2
_PANEL_ROWS = (2, 4)
_PANEL_COLS = (2, 4)
_PANEL_H = (7, 11)
_PANEL_W = (7, 11)
_THICKNESSES = (2, 3)
_MAX_ATTEMPTS = 300


def _horizontal_patch(h: int, w: int) -> np.ndarray:
    patch = np.full((h, w), _BG, dtype=int)
    thickness = random.choice(_THICKNESSES)
    length = random.randint(max(4, thickness + 1), w - 2)
    top = random.randint(1, h - thickness - 1)
    left = random.randint(1, w - length - 1)
    patch[top : top + thickness, left : left + length] = _OBJ
    return patch


def _vertical_patch(h: int, w: int) -> np.ndarray:
    patch = np.full((h, w), _BG, dtype=int)
    thickness = random.choice(_THICKNESSES)
    length = random.randint(max(4, thickness + 1), h - 2)
    top = random.randint(1, h - length - 1)
    left = random.randint(1, w - thickness - 1)
    patch[top : top + length, left : left + thickness] = _OBJ
    return patch


def _l_patch(h: int, w: int) -> np.ndarray:
    patch = np.full((h, w), _BG, dtype=int)
    t_h = random.choice(_THICKNESSES)
    t_v = random.choice(_THICKNESSES)
    corner = random.choice(["tl", "tr", "bl", "br"])

    max_h_len = w - 2
    max_v_len = h - 2
    h_len = random.randint(max(4, t_v + 1), max_h_len)
    v_len = random.randint(max(4, t_h + 1), max_v_len)

    if corner == "tl":
        top = random.randint(1, h - v_len - 1)
        left = random.randint(1, w - h_len - 1)
        patch[top : top + t_h, left : left + h_len] = _OBJ
        patch[top : top + v_len, left : left + t_v] = _OBJ
    elif corner == "tr":
        top = random.randint(1, h - v_len - 1)
        right = random.randint(1, w - h_len - 1) + h_len
        patch[top : top + t_h, right - h_len : right] = _OBJ
        patch[top : top + v_len, right - t_v : right] = _OBJ
    elif corner == "bl":
        bottom = random.randint(1, h - v_len - 1) + v_len
        left = random.randint(1, w - h_len - 1)
        patch[bottom - t_h : bottom, left : left + h_len] = _OBJ
        patch[bottom - v_len : bottom, left : left + t_v] = _OBJ
    else:
        bottom = random.randint(1, h - v_len - 1) + v_len
        right = random.randint(1, w - h_len - 1) + h_len
        patch[bottom - t_h : bottom, right - h_len : right] = _OBJ
        patch[bottom - v_len : bottom, right - t_v : right] = _OBJ

    return patch


def _shape_sequence(n: int) -> list[str]:
    base = ["h", "v", "l"]
    start = random.randrange(len(base))
    seq = [base[(start + i) % len(base)] for i in range(n)]
    random.shuffle(seq)
    return seq


def _make_panel(shape: str, h: int, w: int) -> np.ndarray:
    if shape == "h":
        return _horizontal_patch(h, w)
    if shape == "v":
        return _vertical_patch(h, w)
    if shape == "l":
        return _l_patch(h, w)
    raise AssertionError(f"unknown shape {shape}")


def _gen_attempt() -> np.ndarray | None:
    n_pr = random.randint(*_PANEL_ROWS)
    n_pc = random.randint(*_PANEL_COLS)
    ph = random.randint(*_PANEL_H)
    pw = random.randint(*_PANEL_W)

    h = n_pr * ph + (n_pr + 1)
    w = n_pc * pw + (n_pc + 1)
    grid = np.full((h, w), _BG, dtype=int)
    grid[0, :] = _SEP
    grid[-1, :] = _SEP
    grid[:, 0] = _SEP
    grid[:, -1] = _SEP

    for r in range(1, n_pr):
        grid[r * (ph + 1), :] = _SEP
    for c in range(1, n_pc):
        grid[:, c * (pw + 1)] = _SEP

    shapes = _shape_sequence(n_pr * n_pc)
    idx = 0
    for pr in range(n_pr):
        for pc in range(n_pc):
            r0 = 1 + pr * (ph + 1)
            c0 = 1 + pc * (pw + 1)
            grid[r0 : r0 + ph, c0 : c0 + pw] = _make_panel(shapes[idx], ph, pw)
            idx += 1

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
