from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
COLOR_COUNT = 10
GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _fold_columns(g: np.ndarray, bg: int) -> tuple[list[int], int]:
    cols: list[tuple[int, int]] = []
    for c in range(g.shape[1]):
        color = int(g[0, c])
        if color != bg and np.all(g[:, c] == color):
            cols.append((c, color))

    assert cols, "expected at least one full-height fold line"
    colors = {color for _, color in cols}
    assert len(colors) == 1, "all fold lines must use the same dedicated color"
    fold_color = colors.pop()

    line_cols = [c for c, _ in cols]
    assert line_cols[0] > 0 and line_cols[-1] < g.shape[1] - 1, "fold lines must split nonempty segments"
    assert all(b - a > 1 for a, b in zip(line_cols, line_cols[1:])), "segments between fold lines must be nonempty"
    line_mask = np.zeros(g.shape[1], dtype=bool)
    line_mask[line_cols] = True
    assert not np.any((g == fold_color) & ~line_mask[None, :]), "fold color may only appear in fold-line columns"
    return line_cols, int(fold_color)


def _overlay_folded(left: np.ndarray, right: np.ndarray, bg: int, *, overwrite: bool = True) -> np.ndarray:
    assert left.shape[0] == right.shape[0], "folded pieces must have the same height"
    assert left.shape[1] <= right.shape[1], "left side must fit on the remaining right side"

    folded = left[:, ::-1]
    out = right.copy()
    view = out[:, : folded.shape[1]]
    mask = folded != int(bg)
    if not overwrite:
        mask &= view == int(bg)
    view[mask] = folded[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold a horizontal strip across its vertical fold lines from left to right.

    Input structure:
    - The grid is a horizontal strip whose dominant color is the background.
    - One dedicated non-background color forms one or more full-height vertical
      fold-line columns, and that fold color is not used as pattern color.
    - The fold lines split the strip into nonempty painted segments; before each
      fold, the current left segment is no wider than the remaining right canvas.

    Rule:
    - Repeatedly take the leftmost remaining fold line.
    - Mirror everything to its left across that line onto the first columns on
      the right: a cell at distance k to the left lands at distance k to the right.
    - Mirrored non-background cells overwrite existing cells, mirrored background
      cells are transparent, and the fold-line column itself is removed.
    - Return the canvas remaining after the last fold line has been removed.

    Key insight:
    - The fold lines are ordered mirror operations, not passive separators: an
      early segment can be folded again by later lines, and each newly folded
      layer sits above the older right-hand content.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 1 and g.shape[1] >= 3, "grid too small for a fold strip"

    bg = _get_bg(g)
    _line_cols, fold_color = _fold_columns(g, bg)

    cur = g.copy()
    folded_count = 0
    while True:
        lines = [c for c in range(cur.shape[1]) if np.all(cur[:, c] == fold_color)]
        if not lines:
            break

        c = int(lines[0])
        assert 0 < c < cur.shape[1] - 1, "each fold must have content on both sides"
        left = cur[:, :c]
        right = cur[:, c + 1 :]
        cur = _overlay_folded(left, right, bg, overwrite=True)
        folded_count += 1

    assert folded_count >= 1, "expected at least one fold"
    assert not np.any(cur == fold_color), "all fold-line columns should disappear"
    return cur


# --------------- generator ---------------


def _sample_widths(n_folds: int) -> list[int] | None:
    for _ in range(200):
        widths = [random.randint(2, 4)]
        for _i in range(n_folds):
            lo = widths[-1]
            widths.append(random.randint(lo, min(8, lo + 3)))
        if sum(widths) + n_folds <= 30 and widths[-1] >= 3:
            return widths
    return None


def _has_full_nonbg_col(seg: np.ndarray) -> bool:
    for c in range(seg.shape[1]):
        color = int(seg[0, c])
        if color != BACKGROUND and np.all(seg[:, c] == color):
            return True
    return False


def _paint_asymmetric_seed(seg: np.ndarray, color: int) -> None:
    h, w = seg.shape
    shapes = [
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 1), (1, 1), (2, 1), (2, 0)],
    ]
    fits = []
    for shape in shapes:
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)
        if max_r < h and max_c < w:
            fits.append(shape)
    shape = random.choice(fits)
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    r0 = random.randint(0, h - max_r - 1)
    c0 = random.randint(0, w - max_c - 1)
    for dr, dc in shape:
        seg[r0 + dr, c0 + dc] = int(color)


def _scatter_own_color(seg: np.ndarray, color: int, count: int) -> None:
    choices = [(int(r), int(c)) for r, c in np.argwhere(seg == BACKGROUND)]
    random.shuffle(choices)
    for r, c in choices[:count]:
        seg[r, c] = int(color)


def _force_collision(seg: np.ndarray, source: np.ndarray, color: int) -> bool:
    choices = [(int(r), int(c)) for r, c in np.argwhere(source != BACKGROUND)]
    if not choices:
        return False
    r, c = random.choice(choices)
    seg[r, c] = int(color)
    return True


def _place_survivor(seg: np.ndarray, source: np.ndarray, color: int) -> bool:
    h, w = seg.shape
    sw = source.shape[1]
    choices: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if c >= sw or int(source[r, c]) == BACKGROUND:
                choices.append((r, c))
    if not choices:
        return False

    random.shuffle(choices)
    r, c = choices[0]
    seg[r, c] = int(color)

    neighbors = [(r, c + 1), (r + 1, c), (r, c - 1), (r - 1, c)]
    random.shuffle(neighbors)
    for nr, nc in neighbors:
        if 0 <= nr < h and 0 <= nc < w:
            if nc >= sw or int(source[nr, nc]) == BACKGROUND:
                if int(seg[nr, nc]) == BACKGROUND and random.random() < 0.7:
                    seg[nr, nc] = int(color)
                    break
    return True


def _fold_segments(segments: list[np.ndarray], *, overwrite: bool = True) -> np.ndarray:
    acc = segments[0].copy()
    for seg in segments[1:]:
        acc = _overlay_folded(acc, seg, BACKGROUND, overwrite=overwrite)
    return acc


def _join_segments(segments: list[np.ndarray], fold_color: int) -> np.ndarray:
    h = segments[0].shape[0]
    width = sum(seg.shape[1] for seg in segments) + len(segments) - 1
    grid = np.full((h, width), BACKGROUND, dtype=int)
    c = 0
    for i, seg in enumerate(segments):
        w = seg.shape[1]
        grid[:, c : c + w] = seg
        c += w
        if i < len(segments) - 1:
            grid[:, c] = int(fold_color)
            c += 1
    return grid


def _build_segments(h: int, widths: list[int], colors: list[int]) -> tuple[list[np.ndarray], np.ndarray] | None:
    segments: list[np.ndarray] = []
    first = np.full((h, widths[0]), BACKGROUND, dtype=int)
    _paint_asymmetric_seed(first, colors[0])
    _scatter_own_color(first, colors[0], random.randint(0, 2))
    if _has_full_nonbg_col(first):
        return None
    segments.append(first)

    acc = first.copy()
    for i, width in enumerate(widths[1:], start=1):
        seg = np.full((h, width), BACKGROUND, dtype=int)
        incoming = acc[:, ::-1]
        if not _force_collision(seg, incoming, colors[i]):
            return None
        if not _place_survivor(seg, incoming, colors[i]):
            return None
        _scatter_own_color(seg, colors[i], random.randint(0, 2))
        if _has_full_nonbg_col(seg):
            return None
        segments.append(seg)
        acc = _overlay_folded(acc, seg, BACKGROUND, overwrite=True)

    return segments, acc


def _last_fold_only(segments: list[np.ndarray]) -> np.ndarray:
    return _overlay_folded(segments[-2], segments[-1], BACKGROUND, overwrite=True)


def _semantic_valid(inp: np.ndarray, expected: np.ndarray, segments: list[np.ndarray], fold_color: int, colors: list[int]) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if _get_bg(inp) != BACKGROUND:
        return False
    if expected.shape[1] != segments[-1].shape[1]:
        return False
    if len(segments) < 3:
        return False
    if any(_has_full_nonbg_col(seg) for seg in segments):
        return False
    try:
        line_cols, detected_fold = _fold_columns(inp, BACKGROUND)
    except AssertionError:
        return False
    if detected_fold != int(fold_color) or len(line_cols) != len(segments) - 1:
        return False
    if any(int(color) not in set(map(int, expected.ravel())) for color in colors):
        return False
    if not np.any(expected == BACKGROUND):
        return False
    if np.mean(expected != BACKGROUND) > 0.55:
        return False
    try:
        solved = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(expected, segments[-1]):
        return False
    if np.array_equal(expected, _last_fold_only(segments)):
        return False
    if np.array_equal(expected, _fold_segments(segments, overwrite=False)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_folds = random.choice([2, 3, 4])
    widths = _sample_widths(n_folds)
    if widths is None:
        return None

    h = random.randint(5, 10)
    total_w = sum(widths) + n_folds
    if total_w <= h + 3 or total_w > 30:
        return None

    fold_color = random.choice(range(1, COLOR_COUNT))
    palette = [c for c in range(1, COLOR_COUNT) if c != fold_color]
    if len(palette) < len(widths):
        return None
    colors = random.sample(palette, len(widths))

    built = _build_segments(h, widths, colors)
    if built is None:
        return None
    segments, expected = built
    if not np.array_equal(expected, _fold_segments(segments, overwrite=True)):
        return None

    inp = _join_segments(segments, fold_color)
    if not _semantic_valid(inp, expected, segments, fold_color, colors):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
