from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
SEPARATOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

PANEL_ROWS_RANGE = (2, 4)
PANEL_COLS_RANGE = (2, 4)
PANEL_HEIGHT_RANGE = (4, 7)
PANEL_WIDTH_RANGE = (4, 7)
SEPARATOR_WIDTH_RANGE = (1, 2)
MARGIN_RANGE = (0, 2)
GRID_SIDE_MAX = 30

SHAPE_KINDS = ("rect", "frame", "plus", "ell", "diag", "tee")
GEN_RETRIES = 400


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(mask):
        if not bool(mask[i]):
            i += 1
            continue
        j = i + 1
        while j < len(mask) and bool(mask[j]):
            j += 1
        spans.append((i, j))
        i = j
    return spans


def _gaps(runs: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for start, end in runs:
        if cursor < start:
            spans.append((cursor, start))
        cursor = end
    if cursor < limit:
        spans.append((cursor, limit))
    return spans


def _nonzero_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(g != BG)
    assert coords.size, "expected a non-empty board"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _signature_color(panel: np.ndarray) -> int:
    values = panel[panel != BG]
    counts = Counter(map(int, values.reshape(-1)))
    assert len(counts) >= 2, "each panel must contain a main color and a signature color"
    singletons = [color for color, count in counts.items() if count == 1]
    assert len(singletons) == 1, "expected exactly one signature color"
    return int(singletons[0])


def _extract_thumbnail(board: np.ndarray) -> np.ndarray:
    h, w = map(int, board.shape)
    candidates: list[np.ndarray] = []

    for color in sorted(set(map(int, board.reshape(-1)))):
        if color == BG:
            continue

        row_mask = np.all(board == color, axis=1)
        col_mask = np.all(board == color, axis=0)
        if not row_mask.any() or not col_mask.any():
            continue

        row_runs = _runs(row_mask)
        col_runs = _runs(col_mask)
        if not row_runs or not col_runs:
            continue
        if row_runs[0][0] != 0 or row_runs[-1][1] != h:
            continue
        if col_runs[0][0] != 0 or col_runs[-1][1] != w:
            continue

        row_spans = _gaps(row_runs, h)
        col_spans = _gaps(col_runs, w)
        if not row_spans or not col_spans:
            continue

        row_heights = {r1 - r0 for r0, r1 in row_spans}
        col_widths = {c1 - c0 for c0, c1 in col_spans}
        if len(row_heights) != 1 or len(col_widths) != 1:
            continue

        try:
            out = np.empty((len(row_spans), len(col_spans)), dtype=int)
            for rr, (r0, r1) in enumerate(row_spans):
                for cc, (c0, c1) in enumerate(col_spans):
                    panel = board[r0:r1, c0:c1]
                    out[rr, cc] = _signature_color(panel)
        except AssertionError:
            continue

        candidates.append(out)

    assert len(candidates) == 1, "expected exactly one valid panel grid"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Extract each panel's singleton signature color and arrange them as a coarse thumbnail."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    r0, c0, r1, c1 = _nonzero_bbox(g)
    board = g[r0 : r1 + 1, c0 : c1 + 1]
    return _extract_thumbnail(board).copy()


def _rect_mask(h: int, w: int) -> np.ndarray:
    rh = random.randint(2, h)
    rw = random.randint(2, w)
    r0 = random.randint(0, h - rh)
    c0 = random.randint(0, w - rw)
    mask = np.zeros((h, w), dtype=bool)
    mask[r0 : r0 + rh, c0 : c0 + rw] = True
    return mask


def _frame_mask(h: int, w: int) -> np.ndarray | None:
    if h < 3 or w < 3:
        return None
    rh = random.randint(3, h)
    rw = random.randint(3, w)
    r0 = random.randint(0, h - rh)
    c0 = random.randint(0, w - rw)
    mask = np.zeros((h, w), dtype=bool)
    mask[r0, c0 : c0 + rw] = True
    mask[r0 + rh - 1, c0 : c0 + rw] = True
    mask[r0 : r0 + rh, c0] = True
    mask[r0 : r0 + rh, c0 + rw - 1] = True
    return mask


def _plus_mask(h: int, w: int) -> np.ndarray:
    rh = random.randint(3, h)
    rw = random.randint(3, w)
    r0 = random.randint(0, h - rh)
    c0 = random.randint(0, w - rw)
    center_r = r0 + random.randint(0, rh - 1)
    center_c = c0 + random.randint(0, rw - 1)
    mask = np.zeros((h, w), dtype=bool)
    mask[center_r, c0 : c0 + rw] = True
    mask[r0 : r0 + rh, center_c] = True
    return mask


def _ell_mask(h: int, w: int) -> np.ndarray:
    rh = random.randint(3, h)
    rw = random.randint(3, w)
    r0 = random.randint(0, h - rh)
    c0 = random.randint(0, w - rw)
    orient = random.choice(("tl", "tr", "bl", "br"))
    mask = np.zeros((h, w), dtype=bool)
    if orient == "tl":
        mask[r0 : r0 + rh, c0] = True
        mask[r0, c0 : c0 + rw] = True
    elif orient == "tr":
        mask[r0 : r0 + rh, c0 + rw - 1] = True
        mask[r0, c0 : c0 + rw] = True
    elif orient == "bl":
        mask[r0 : r0 + rh, c0] = True
        mask[r0 + rh - 1, c0 : c0 + rw] = True
    else:
        mask[r0 : r0 + rh, c0 + rw - 1] = True
        mask[r0 + rh - 1, c0 : c0 + rw] = True
    return mask


def _diag_mask(h: int, w: int) -> np.ndarray:
    length = random.randint(3, min(h, w))
    r0 = random.randint(0, h - length)
    c0 = random.randint(0, w - length)
    mask = np.zeros((h, w), dtype=bool)
    if random.random() < 0.5:
        for k in range(length):
            mask[r0 + k, c0 + k] = True
    else:
        for k in range(length):
            mask[r0 + k, c0 + length - 1 - k] = True
    return mask


def _tee_mask(h: int, w: int) -> np.ndarray:
    rh = random.randint(3, h)
    rw = random.randint(3, w)
    r0 = random.randint(0, h - rh)
    c0 = random.randint(0, w - rw)
    orient = random.choice(("up", "down", "left", "right"))
    mask = np.zeros((h, w), dtype=bool)
    if orient in ("up", "down"):
        mid = c0 + random.randint(0, rw - 1)
        if orient == "up":
            bar_r = r0
            stem_rows = slice(r0, r0 + rh)
        else:
            bar_r = r0 + rh - 1
            stem_rows = slice(r0, r0 + rh)
        mask[bar_r, c0 : c0 + rw] = True
        mask[stem_rows, mid] = True
    else:
        mid = r0 + random.randint(0, rh - 1)
        if orient == "left":
            bar_c = c0
            stem_cols = slice(c0, c0 + rw)
        else:
            bar_c = c0 + rw - 1
            stem_cols = slice(c0, c0 + rw)
        mask[r0 : r0 + rh, bar_c] = True
        mask[mid, stem_cols] = True
    return mask


def _shape_mask(h: int, w: int) -> np.ndarray | None:
    kind = random.choice(SHAPE_KINDS)
    if kind == "rect":
        mask = _rect_mask(h, w)
    elif kind == "frame":
        mask = _frame_mask(h, w)
    elif kind == "plus":
        mask = _plus_mask(h, w)
    elif kind == "ell":
        mask = _ell_mask(h, w)
    elif kind == "diag":
        mask = _diag_mask(h, w)
    else:
        mask = _tee_mask(h, w)
    if mask is None or int(mask.sum()) < 3:
        return None
    return mask


def _paint_panel(h: int, w: int, main_color: int, signature_color: int) -> np.ndarray | None:
    mask = _shape_mask(h, w)
    if mask is None:
        return None
    panel = np.full((h, w), BG, dtype=int)
    panel[mask] = int(main_color)
    coords = np.argwhere(mask)
    sr, sc = coords[random.randrange(len(coords))]
    panel[int(sr), int(sc)] = int(signature_color)
    return panel


def _board_shape(nr: int, nc: int, ph: int, pw: int, sep: int) -> tuple[int, int]:
    return nr * ph + (nr + 1) * sep, nc * pw + (nc + 1) * sep


def _sample_palettes() -> tuple[list[int], list[int]]:
    main_palette = random.sample(COLOR_POOL, random.randint(2, 4))
    sig_palette = random.sample(COLOR_POOL, random.randint(2, 4))
    return main_palette, sig_palette


def _gen_attempt() -> np.ndarray | None:
    n_panel_rows = random.randint(*PANEL_ROWS_RANGE)
    n_panel_cols = random.randint(*PANEL_COLS_RANGE)
    panel_h = random.randint(*PANEL_HEIGHT_RANGE)
    panel_w = random.randint(*PANEL_WIDTH_RANGE)
    sep = random.randint(*SEPARATOR_WIDTH_RANGE)

    board_h, board_w = _board_shape(n_panel_rows, n_panel_cols, panel_h, panel_w, sep)
    margin_top = random.randint(*MARGIN_RANGE)
    margin_bottom = random.randint(*MARGIN_RANGE)
    margin_left = random.randint(*MARGIN_RANGE)
    margin_right = random.randint(*MARGIN_RANGE)
    full_h = board_h + margin_top + margin_bottom
    full_w = board_w + margin_left + margin_right
    if full_h > GRID_SIDE_MAX or full_w > GRID_SIDE_MAX:
        return None

    main_palette, sig_palette = _sample_palettes()
    grid = np.full((full_h, full_w), BG, dtype=int)
    out = np.empty((n_panel_rows, n_panel_cols), dtype=int)

    top = margin_top
    left = margin_left
    grid[top : top + board_h, left : left + board_w] = SEPARATOR

    main_used: list[int] = []
    sig_used: list[int] = []
    for pr in range(n_panel_rows):
        for pc in range(n_panel_cols):
            panel_top = top + sep + pr * (panel_h + sep)
            panel_left = left + sep + pc * (panel_w + sep)

            main_color = random.choice(main_palette)
            sig_choices = [color for color in sig_palette if color != main_color]
            if not sig_choices:
                sig_choices = [color for color in COLOR_POOL if color != main_color]
            signature_color = random.choice(sig_choices)

            panel = _paint_panel(panel_h, panel_w, main_color, signature_color)
            if panel is None:
                return None

            grid[panel_top : panel_top + panel_h, panel_left : panel_left + panel_w] = panel
            out[pr, pc] = int(signature_color)
            main_used.append(int(main_color))
            sig_used.append(int(signature_color))

    if len(set(main_used)) < 2 or len(set(sig_used)) < 2:
        return None

    try:
        if np.array_equal(main(grid), out):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
