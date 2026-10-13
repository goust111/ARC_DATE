from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_RETRIES = 400


def _full_rows(mask: np.ndarray) -> list[int]:
    return [int(r) for r in range(mask.shape[0]) if bool(np.all(mask[r, :]))]


def _full_cols(mask: np.ndarray) -> list[int]:
    return [int(c) for c in range(mask.shape[1]) if bool(np.all(mask[:, c]))]


def _parse_control(grid: np.ndarray) -> tuple[str, int, int]:
    colors = [int(c) for c in np.unique(grid) if int(c) != BG]
    assert len(colors) == 2, "expected exactly two foreground colors"

    candidates: list[tuple[str, int, int]] = []
    for color in colors:
        mask = grid == color
        rows = _full_rows(mask)
        cols = _full_cols(mask)

        if len(cols) == 1 and not rows:
            axis = cols[0]
            extras = [(int(r), int(c)) for r, c in np.argwhere(mask) if int(c) != axis]
            if len(extras) == 2 and extras[0][0] == extras[1][0]:
                row = extras[0][0]
                deltas = sorted(c - axis for _r, c in extras)
                if deltas in ([-2, -1], [1, 2]):
                    side = -1 if deltas[1] < 0 else 1
                    candidates.append(("vertical", axis, side))

        if len(rows) == 1 and not cols:
            axis = rows[0]
            extras = [(int(r), int(c)) for r, c in np.argwhere(mask) if int(r) != axis]
            if len(extras) == 2 and extras[0][1] == extras[1][1]:
                col = extras[0][1]
                deltas = sorted(r - axis for r, _c in extras)
                if deltas in ([-2, -1], [1, 2]):
                    side = -1 if deltas[1] < 0 else 1
                    candidates.append(("horizontal", axis, side))

    assert len(candidates) == 1, "control line must be unique"
    return candidates[0]


def _content_color(grid: np.ndarray, orientation: str, axis: int, side: int) -> int:
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        mask = grid == color
        if orientation == "vertical" and len(_full_cols(mask)) == 1 and not _full_rows(mask):
            continue
        if orientation == "horizontal" and len(_full_rows(mask)) == 1 and not _full_cols(mask):
            continue
        return color
    raise AssertionError("content color not found")


def _fold_vertical(content: np.ndarray, axis: int, fold_side: int) -> np.ndarray:
    h, w = content.shape
    if fold_side < 0:
        out = content[:, axis + 1 :].copy()
        for dist in range(1, axis + 1):
            src = axis - dist
            dst = dist - 1
            assert dst < out.shape[1]
            fill = (out[:, dst] == BG) & (content[:, src] != BG)
            out[fill, dst] = content[:, src][fill]
        return out

    out = content[:, :axis].copy()
    right_w = w - axis - 1
    for dist in range(1, right_w + 1):
        src = axis + dist
        dst = axis - dist
        assert dst >= 0
        fill = (out[:, dst] == BG) & (content[:, src] != BG)
        out[fill, dst] = content[:, src][fill]
    return out


def _fold_horizontal(content: np.ndarray, axis: int, fold_side: int) -> np.ndarray:
    h, w = content.shape
    if fold_side < 0:
        out = content[axis + 1 :, :].copy()
        for dist in range(1, axis + 1):
            src = axis - dist
            dst = dist - 1
            assert dst < out.shape[0]
            fill = (out[dst, :] == BG) & (content[src, :] != BG)
            out[dst, fill] = content[src, fill]
        return out

    out = content[:axis, :].copy()
    below_h = h - axis - 1
    for dist in range(1, below_h + 1):
        src = axis + dist
        dst = axis - dist
        assert dst >= 0
        fill = (out[dst, :] == BG) & (content[src, :] != BG)
        out[dst, fill] = content[src, fill]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a single fold line plus 2-pixel arrow, fold the indicated side, and
    return only the retained half with overlaid non-background content.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    orientation, axis, fold_side = _parse_control(g)
    content_color = _content_color(g, orientation, axis, fold_side)
    content = np.where(g == content_color, content_color, BG).astype(int)

    if orientation == "vertical":
        return _fold_vertical(content, axis, fold_side)
    return _fold_horizontal(content, axis, fold_side)


def _add_patch(mask: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    mask[r0 : r0 + h, c0 : c0 + w] = True


def _random_mask(h: int, w: int, min_patches: int, max_patches: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    n_patches = random.randint(min_patches, max_patches)
    for _ in range(n_patches):
        style = random.choice(("rect", "hline", "vline", "point", "ell"))
        if style == "rect":
            ph = random.randint(1, max(1, min(3, h)))
            pw = random.randint(1, max(1, min(3, w)))
        elif style == "hline":
            ph = 1
            pw = random.randint(1, max(1, min(4, w)))
        elif style == "vline":
            ph = random.randint(1, max(1, min(4, h)))
            pw = 1
        elif style == "point":
            ph = 1
            pw = 1
        else:
            ph = random.randint(2, max(2, min(4, h)))
            pw = random.randint(2, max(2, min(4, w)))

        r0 = random.randint(0, h - ph)
        c0 = random.randint(0, w - pw)
        _add_patch(mask, r0, c0, ph, pw)
        if style == "ell" and ph >= 2 and pw >= 2:
            mask[r0, c0] = False
            mask[r0 + ph - 1, c0 + pw - 1] = False

    return mask


def _ensure_points(mask: np.ndarray, points: list[tuple[int, int]]) -> None:
    for r, c in points:
        mask[r, c] = True


def _build_halves(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    retained = _random_mask(h, w, 2, 6)
    folded = _random_mask(h, w, 2, 6)

    rr = random.randrange(h)
    cc = random.randrange(w)
    _ensure_points(folded, [(rr, cc)])
    if random.random() < 0.5:
        _ensure_points(retained, [(rr, cc)])
    else:
        retained[rr, cc] = False

    if not retained.any():
        retained[random.randrange(h), random.randrange(w)] = True
    if not folded.any():
        folded[random.randrange(h), random.randrange(w)] = True

    density = float(retained.mean() + folded.mean())
    if density > 0.5:
        keep_a = np.random.random((h, w)) < 0.7
        keep_b = np.random.random((h, w)) < 0.7
        retained &= keep_a
        folded &= keep_b
        retained[rr, cc] = retained[rr, cc] or random.random() < 0.5
        folded[rr, cc] = True

    return retained, folded


def _encode_vertical(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    axis_col: int,
    content_color: int,
    control_color: int,
    fold_side: int,
) -> np.ndarray:
    h, half = left_mask.shape
    grid = np.zeros((h, 2 * half + 1), dtype=int)
    grid[:, :half][left_mask] = content_color
    grid[:, half + 1 :][right_mask] = content_color
    grid[:, axis_col] = control_color

    arrow_row = random.randrange(h)
    if fold_side < 0:
        grid[arrow_row, axis_col - 2 : axis_col] = control_color
    else:
        grid[arrow_row, axis_col + 1 : axis_col + 3] = control_color
    return grid


def _encode_horizontal(
    top_mask: np.ndarray,
    bottom_mask: np.ndarray,
    axis_row: int,
    content_color: int,
    control_color: int,
    fold_side: int,
) -> np.ndarray:
    half, w = top_mask.shape
    grid = np.zeros((2 * half + 1, w), dtype=int)
    grid[:half, :][top_mask] = content_color
    grid[half + 1 :, :][bottom_mask] = content_color
    grid[axis_row, :] = control_color

    arrow_col = random.randrange(w)
    if fold_side < 0:
        grid[axis_row - 2 : axis_row, arrow_col] = control_color
    else:
        grid[axis_row + 1 : axis_row + 3, arrow_col] = control_color
    return grid


def _expected_from_halves(retained: np.ndarray, folded: np.ndarray) -> np.ndarray:
    out = retained.copy()
    out[folded] = True
    return out


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(("vertical", "horizontal"))
    fold_side = random.choice((-1, 1))
    content_color, control_color = random.sample(PALETTE, 2)

    if orientation == "vertical":
        h = random.randint(6, 16)
        half = random.randint(3, 9)
        retained, folded = _build_halves(h, half)
        if fold_side < 0:
            left_mask = np.fliplr(folded)
            right_mask = retained
        else:
            left_mask = retained
            right_mask = np.fliplr(folded)
        grid = _encode_vertical(left_mask, right_mask, half, content_color, control_color, fold_side)
        expected = np.where(_expected_from_halves(retained, folded), content_color, BG).astype(int)
    else:
        half = random.randint(3, 9)
        w = random.randint(6, 16)
        retained, folded = _build_halves(half, w)
        if fold_side < 0:
            top_mask = np.flipud(folded)
            bottom_mask = retained
        else:
            top_mask = retained
            bottom_mask = np.flipud(folded)
        grid = _encode_horizontal(top_mask, bottom_mask, half, content_color, control_color, fold_side)
        expected = np.where(_expected_from_halves(retained, folded), content_color, BG).astype(int)

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fold puzzle input")
