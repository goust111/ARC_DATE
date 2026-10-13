from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_arc_utils():
    """Load arc_utils both in normal imports and when loaded by SourceFileLoader."""
    try:
        import arc_utils as mod  # type: ignore
        return mod
    except Exception:
        candidates = [
            Path(__file__).resolve().parents[6] / "new" / "solvers" / "arc_utils.py",
            Path("/home/lixinyu/snap/new/solvers/arc_utils.py"),
        ]
        for path in candidates:
            if not path.exists():
                continue
            spec = importlib.util.spec_from_file_location("_arc_utils", path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise RuntimeError("cannot load arc_utils")


_UTILS = _load_arc_utils()
Color = _UTILS.Color
get_bg = _UTILS.get_bg
bbox = _UTILS.bbox

BG = Color.BLACK


def _settle_left_board(left_board: np.ndarray, bg: int) -> np.ndarray:
    """Apply vertical gravity drop to single pixels in each column of the left board."""
    h, w = left_board.shape
    settled = np.full((h, w), bg, dtype=int)

    for c in range(w):
        column_pixels = [int(left_board[r, c]) for r in range(h - 1, -1, -1) if int(left_board[r, c]) != bg]
        for i, color in enumerate(column_pixels):
            settled[h - 1 - i, c] = color
    return settled


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "main expects a 2D grid"

    h, w = g.shape
    assert h > 0 and w > 1, "grid must be at least 1x2"
    bg = int(get_bg(g))

    sep_cols = [
        c
        for c in range(w)
        if np.all(g[:, c] == int(g[0, c])) and int(g[0, c]) != bg
    ]
    assert len(sep_cols) == 1, "expected exactly one full-height separator column"
    sep = sep_cols[0]
    assert 0 < sep < w - 1, "separator must separate two non-empty boards"

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.size > 0 and right.size > 0, "left/right boards must be non-empty"

    pattern_mask = right != bg
    assert bool(pattern_mask.any()), "right board must contain non-background pixels"
    assert bbox(pattern_mask) is not None, "right board must have a non-empty mask"

    pattern_colors = np.unique(right[pattern_mask])
    assert len(pattern_colors) == 1, "right board must be single-color binary mask"

    settled = _settle_left_board(left, bg)
    ph, pw = right.shape
    lh, lw = settled.shape
    out_h = lh * ph
    out_w = lw * pw
    assert out_h <= 30 and out_w <= 30, "expanded layout must stay within ARC bounds"

    out = np.full((out_h, out_w), bg, dtype=int)
    for r in range(lh):
        for c in range(lw):
            color = int(settled[r, c])
            if color == bg:
                continue

            top = r * ph
            left_col = c * pw
            tile = out[top : top + ph, left_col : left_col + pw]
            tile[pattern_mask] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop colored pixels in the left board by gravity and stamp right-board binary tiles at each settled pixel location.

    Input structure
    - The input is split into two vertical boards by one full-height non-background separator column.
    - The left board is black-background with several colored single pixels in arbitrary rows/columns.
    - The right board is a single-color binary pattern (foreground color + black background) sharing the same height as the input.

    Rule:
    - First, independently let each colored pixel in each left-board column fall vertically with gravity to the bottom, preserving left-to-right column order.
    - Then build an output canvas with height `left_height * pattern_height` and width `left_width * pattern_width`.
    - For each occupied cell in the settled left layout, stamp a copy of the right-board pattern in its output tile, painting pattern foreground pixels with that cell color and background pixels black.
    - Empty settled cells correspond to all-black tiles.

    Key insight:
    - The visible output layout is a Kronecker-style expansion of the post-settled left occupancy, using the right-board pattern as tile template.
    """
    return _apply_rule(grid)


# -----------------------------
gen_rng = np.random.default_rng()
GEN_H_RANGE = (4, 6)
GEN_LEFT_W_RANGE = (2, 8)
GEN_RIGHT_W_RANGE = (2, 8)
GEN_ATTEMPT_LIMIT = 300


def _gen_left_panel(rng: np.random.Generator, h: int, w: int, palette: list[int]) -> np.ndarray:
    left = np.full((h, w), BG, dtype=int)
    if w <= 0:
        return left

    n_cols = int(rng.integers(1, w + 1))
    cols = rng.choice(w, size=n_cols, replace=False)
    force_multi = float(rng.random()) < 0.55 and h >= 2

    floating_seen = False
    for idx, c in enumerate(cols):
        if force_multi and idx == 0:
            stack = int(rng.integers(2, min(4, h) + 1))
        else:
            stack = 1 if float(rng.random()) < 0.45 else int(rng.integers(2, min(4, h) + 1))
        stack = max(1, min(int(stack), h))

        rows = np.sort(rng.choice(h, size=stack, replace=False))
        if float(rng.random()) < 0.6 and h > 1 and not np.array_equal(rows, np.array([h - 1])):
            rows[0] = int(rng.integers(0, h - 1))
            rows = np.sort(rows)

        for r in rows:
            color = int(rng.choice(palette))
            left[int(r), int(c)] = color
            floating_seen = floating_seen or (int(r) < h - 1)

    if h > 1 and not floating_seen:
        # force one visible fall distance branch for determinism coverage
        c = int(cols[0])
        occupied = np.flatnonzero(left[:, c] != BG)
        if occupied.size == 0:
            left[h - 1, c] = int(rng.choice(palette))
            occupied = np.array([h - 1], dtype=int)
        r0 = int(rng.choice(occupied))
        candidate_rows = [r for r in range(h - 1) if r != r0]
        r1 = int(rng.choice(candidate_rows))
        color = int(left[r0, c])
        left[r0, c] = BG
        left[r1, c] = color

    return left


def _gen_right_pattern(rng: np.random.Generator, h: int, w: int, palette: list[int]) -> np.ndarray:
    fg = int(rng.choice(palette))
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]

    mode = int(rng.integers(0, 3))
    if mode == 0:
        p = float(rng.uniform(0.2, 0.8))
        mask = rng.random((h, w)) < p
    elif mode == 1:
        stripe = int(rng.integers(2, max(2, min(5, w)) + 1))
        phase = int(rng.integers(stripe))
        mask = ((cols + phase) % stripe) == 0
    else:
        parity = int(rng.integers(2))
        mask = ((rows + cols + parity) % 2) == 0

    if float(rng.random()) < 0.5:
        mask = np.flipud(mask)
    if float(rng.random()) < 0.5:
        mask = np.fliplr(mask)

    if not mask.any() or mask.all():
        for _ in range(8):
            p = float(rng.uniform(0.2, 0.8))
            mask = rng.random((h, w)) < p
            if mask.any() and not mask.all():
                break

    if not mask.any():
        mask[0, 0] = True
    if mask.all():
        mask[0, 0] = False

    right = np.full((h, w), BG, dtype=int)
    right[mask] = fg
    return right


def _gen_candidate() -> np.ndarray | None:
    h = int(gen_rng.integers(*GEN_H_RANGE))
    left_w = int(gen_rng.integers(*GEN_LEFT_W_RANGE))
    right_w = int(gen_rng.integers(*GEN_RIGHT_W_RANGE))

    if h * h > 30:
        return None
    if left_w * right_w > 30:
        return None

    bg = BG
    sep_color_candidates = [c for c in Color.NOT_BLACK if c != bg]
    sep_color = int(gen_rng.choice(sep_color_candidates))
    left_palette = [c for c in Color.NOT_BLACK if c not in (bg, sep_color)]

    left = _gen_left_panel(gen_rng, h, left_w, left_palette)
    right = _gen_right_pattern(gen_rng, h, right_w, [c for c in Color.NOT_BLACK if c != bg])

    inp = np.full((h, left_w + 1 + right_w), bg, dtype=int)
    inp[:, left_w] = sep_color
    inp[:, :left_w] = left
    inp[:, left_w + 1 :] = right

    out = _apply_rule(inp)
    if inp.shape[0] > 30 or inp.shape[1] > 30:
        return None
    if out.shape[0] > 30 or out.shape[1] > 30:
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate diverse valid inputs covering gravity-branch and tiling-branch cases."""
    for _ in range(GEN_ATTEMPT_LIMIT):
        try:
            inp = _gen_candidate()
        except Exception:
            continue
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid 016-3 input")
