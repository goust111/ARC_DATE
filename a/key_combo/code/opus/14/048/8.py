import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
PANEL_COLORS = (BLUE, GREEN)

_MAX_GEN_ATTEMPTS = 400


def _separator_columns(grid: np.ndarray) -> tuple[int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % 2 == 1 and height >= 3 and width >= 3, "expected an odd-height grid with two panels"

    sep_row = height // 2
    separator = g[sep_row]
    assert np.all(np.isin(separator, (RED, GREY))), "middle row must contain only grey and red separator pixels"
    assert np.any(separator == GREY), "separator row must contain grey pixels"
    return sep_row, separator == RED


def _compose_output(top: np.ndarray, bottom: np.ndarray, suppressed_cols: np.ndarray) -> np.ndarray:
    top = np.asarray(top, dtype=int)
    bottom = np.asarray(bottom, dtype=int)
    assert top.shape == bottom.shape and top.ndim == 2

    width = top.shape[1]
    assert suppressed_cols.shape == (width,)

    normal_cols = ~suppressed_cols
    overlap = (top != BLACK) & (bottom != BLACK) & normal_cols[None, :]
    assert not np.any(overlap), "non-marker columns must remain collision-free"

    out = bottom.copy()
    take_top = (top != BLACK) & (bottom == BLACK) & normal_cols[None, :]
    out[take_top] = top[take_top]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the encoded separator row and overlay the two panels while suppressing the top panel in marked columns.

    Input structure:
    - The grid has a black background and odd height.
    - The exact middle row is a full-width separator made of grey(5) pixels, with 1-3 red(2) pixels embedded as column markers.
    - Above and below the separator are equal-size panels whose foreground pixels occupy local coordinates on the same canvas.

    Rule:
    - Read the red separator pixels to obtain the set of marked columns.
    - Delete the separator row, keeping the top and bottom panels aligned by local row and column.
    - In unmarked columns, transparently overlay the two panels, taking the unique non-black pixel at each position.
    - In marked columns, ignore every top-panel pixel and keep only the bottom-panel content.

    Key insight:
    - The separator row has dual meaning: it is both the structural divider to remove and the control code that says which columns cancel the top panel before overlay.
    """
    g = np.asarray(grid, dtype=int)
    sep_row, suppressed_cols = _separator_columns(g)
    top = g[:sep_row, :]
    bottom = g[sep_row + 1 :, :]
    assert top.shape == bottom.shape and top.size > 0, "separator must split two non-empty equal panels"
    return _compose_output(top, bottom, suppressed_cols)


def _plain_overlay(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    overlap = (top != BLACK) & (bottom != BLACK)
    assert not np.any(overlap), "plain overlay baseline only applies to collision-free panels"
    out = bottom.copy()
    out[(bottom == BLACK) & (top != BLACK)] = top[(bottom == BLACK) & (top != BLACK)]
    return out


def _top_over_bottom(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    out = bottom.copy()
    out[top != BLACK] = top[top != BLACK]
    return out


def _rand_color(rng: np.random.Generator, exclude: int | None = None) -> int:
    choices = [color for color in PANEL_COLORS if color != exclude]
    return int(rng.choice(choices))


def _set_if_empty(panel: np.ndarray, r: int, c: int, color: int) -> None:
    if panel[r, c] == BLACK:
        panel[r, c] = color


def _force_second_color(rng: np.random.Generator, top: np.ndarray, bottom: np.ndarray, suppressed_cols: np.ndarray) -> None:
    combined = np.concatenate((top.ravel(), bottom.ravel()))
    used = {int(v) for v in combined if int(v) != BLACK}
    if len(used) >= 2:
        return

    panel_h, width = top.shape
    normal_cols = [c for c in range(width) if not suppressed_cols[c]]
    target_color = BLUE if GREEN in used else GREEN
    if not normal_cols:
        normal_cols = list(range(width))

    for _ in range(80):
        c = int(rng.choice(normal_cols))
        r = int(rng.integers(panel_h))
        if top[r, c] == BLACK and bottom[r, c] == BLACK:
            if rng.random() < 0.5:
                top[r, c] = target_color
            else:
                bottom[r, c] = target_color
            return


def _build_candidate(rng: np.random.Generator) -> np.ndarray:
    panel_h = int(rng.integers(4, 8))
    width = int(rng.integers(8, 13))
    marker_count = int(rng.integers(1, min(3, width - 2) + 1))

    suppressed_cols = np.zeros(width, dtype=bool)
    suppressed_cols[rng.choice(width, size=marker_count, replace=False)] = True
    normal_cols = [c for c in range(width) if not suppressed_cols[c]]
    assert len(normal_cols) >= 2

    top = np.zeros((panel_h, width), dtype=int)
    bottom = np.zeros((panel_h, width), dtype=int)

    top_demo_col, bottom_demo_col = map(int, rng.choice(normal_cols, size=2, replace=False))
    top_demo_row = int(rng.integers(panel_h))
    bottom_demo_row = int(rng.integers(panel_h))
    top[top_demo_row, top_demo_col] = _rand_color(rng)
    bottom[bottom_demo_row, bottom_demo_col] = _rand_color(rng)

    suppress_col = int(rng.choice(np.flatnonzero(suppressed_cols)))
    suppress_row = int(rng.integers(panel_h))
    top_hidden_color = _rand_color(rng)
    top[suppress_row, suppress_col] = top_hidden_color
    if rng.random() < 0.5:
        bottom[suppress_row, suppress_col] = BLACK
    else:
        bottom[suppress_row, suppress_col] = _rand_color(rng, exclude=top_hidden_color)

    density = rng.uniform(0.10, 0.22)
    target_extra = int(round(panel_h * width * density))
    protected = {
        (top_demo_row, top_demo_col),
        (bottom_demo_row, bottom_demo_col),
        (suppress_row, suppress_col),
    }

    for _ in range(target_extra):
        r = int(rng.integers(panel_h))
        c = int(rng.integers(width))
        if (r, c) in protected:
            continue
        if suppressed_cols[c]:
            state = int(rng.choice(4, p=(0.48, 0.18, 0.18, 0.16)))
            if state in (1, 3):
                _set_if_empty(top, r, c, _rand_color(rng))
            if state in (2, 3):
                bottom[r, c] = _rand_color(rng)
        else:
            if rng.random() < 0.72:
                continue
            if rng.random() < 0.5:
                if top[r, c] == BLACK and bottom[r, c] == BLACK:
                    top[r, c] = _rand_color(rng)
            elif top[r, c] == BLACK:
                bottom[r, c] = _rand_color(rng)

    if np.count_nonzero(top) < 3:
        for _ in range(30):
            r = int(rng.integers(panel_h))
            c = int(rng.choice(normal_cols))
            if top[r, c] == BLACK and bottom[r, c] == BLACK:
                top[r, c] = _rand_color(rng)
                if np.count_nonzero(top) >= 3:
                    break

    if np.count_nonzero(bottom) < 3:
        for _ in range(30):
            r = int(rng.integers(panel_h))
            c = int(rng.choice(normal_cols))
            if top[r, c] == BLACK and bottom[r, c] == BLACK:
                bottom[r, c] = _rand_color(rng)
                if np.count_nonzero(bottom) >= 3:
                    break

    _force_second_color(rng, top, bottom, suppressed_cols)

    separator = np.full(width, GREY, dtype=int)
    separator[suppressed_cols] = RED
    return np.vstack((top, separator[None, :], bottom))


def _semantic_checks(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    sep_row, suppressed_cols = _separator_columns(g)
    top = g[:sep_row, :]
    bottom = g[sep_row + 1 :, :]
    out = main(g)

    assert np.all(np.isin(top, (BLACK, BLUE, GREEN))), "top panel must use only black/blue/green"
    assert np.all(np.isin(bottom, (BLACK, BLUE, GREEN))), "bottom panel must use only black/blue/green"
    assert 1 <= int(np.sum(suppressed_cols)) <= 3, "expected 1-3 control columns"

    normal_cols = ~suppressed_cols
    top_normal_cols = {int(c) for c in np.where(np.any((top != BLACK) & normal_cols[None, :], axis=0))[0]}
    bottom_normal_cols = {int(c) for c in np.where(np.any((bottom != BLACK) & normal_cols[None, :], axis=0))[0]}
    assert any(c1 != c2 for c1 in top_normal_cols for c2 in bottom_normal_cols), "need distinct unmarked columns sourced from both panels"

    hidden_top = (top != BLACK) & suppressed_cols[None, :]
    assert np.any(hidden_top), "need top foreground in a control column"

    expected = _compose_output(top, bottom, suppressed_cols)
    assert np.array_equal(out, expected), "solver output must match the encoded overlay rule"

    visible_if_unsuppressed = _top_over_bottom(top, bottom)
    control_conflict = hidden_top & ((bottom == BLACK) | (bottom != top))
    assert np.any(control_conflict), "need at least one visibly suppressed top pixel"
    assert not np.array_equal(out, visible_if_unsuppressed), "ignoring the control columns must fail"
    assert not np.array_equal(out, bottom), "bottom-only shortcut must fail"
    assert not np.array_equal(out, top), "top-only shortcut must fail"
    assert np.count_nonzero(out) >= 3, "output should remain visually informative"


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _build_candidate(rng)
        try:
            _semantic_checks(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid separator-overlay puzzle")
