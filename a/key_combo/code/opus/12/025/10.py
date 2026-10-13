from __future__ import annotations

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5
TEAL = 8

_RNG = np.random.default_rng()

_MIN_BARS = 4
_MAX_BARS = 6
_VALID_BAR_COUNTS = (4, 6)
_START_COL_OPTIONS = (1, 3, 5)
_MAX_TEAL_RANGE = (2, 4)
_BLUE_HEIGHT_RANGE = (1, 3)
_RIGHT_PAD_RANGE = (0, 4)
_TOP_PAD_RANGE = (0, 4)
_BOTTOM_PAD_RANGE = (1, 4)
_GEN_ATTEMPTS = 400


def _is_contiguous(rows: np.ndarray) -> bool:
    if rows.size == 0:
        return False
    return bool(np.array_equal(rows, np.arange(int(rows[0]), int(rows[0]) + len(rows))))


def _extract_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape

    red_coords = np.argwhere(g == RED)
    grey_coords = np.argwhere(g == GREY)
    assert red_coords.size > 0 and grey_coords.size > 0

    red_rows = np.unique(red_coords[:, 0])
    grey_rows = np.unique(grey_coords[:, 0])
    assert len(red_rows) == 1 and len(grey_rows) == 1

    red_row = int(red_rows[0])
    grey_row = int(grey_rows[0])
    assert grey_row == red_row + 1

    marker_cols = np.where(g[red_row] == RED)[0]
    assert marker_cols.size >= 2
    assert np.array_equal(marker_cols, np.where(g[grey_row] == GREY)[0])
    assert np.all(np.diff(marker_cols) == 2)

    bar_cols = [int((a + b) // 2) for a, b in zip(marker_cols[:-1], marker_cols[1:])]
    assert _MIN_BARS <= len(bar_cols) <= _MAX_BARS

    teal_heights: list[int] = []
    blue_heights: list[int] = []
    blue_starts: list[int] = []

    for c in bar_cols:
        assert g[red_row, c] == BG
        assert g[grey_row, c] == BG

        col = g[:, c]
        non_bg_rows = np.where(col != BG)[0]
        teal_rows = np.where(col[:red_row] == TEAL)[0]
        blue_rows = np.where(col[grey_row + 1 :] == BLUE)[0] + grey_row + 1

        assert teal_rows.size > 0 and blue_rows.size > 0
        assert _is_contiguous(teal_rows)
        assert _is_contiguous(blue_rows)

        allowed = set(map(int, teal_rows)) | set(map(int, blue_rows))
        assert all((int(r) in allowed) for r in non_bg_rows)

        teal_heights.append(int(teal_rows.size))
        blue_heights.append(int(blue_rows.size))
        blue_starts.append(int(blue_rows[0]))

    return {
        "shape": (H, W),
        "red_row": red_row,
        "grey_row": grey_row,
        "marker_cols": [int(c) for c in marker_cols],
        "bar_cols": bar_cols,
        "teal_heights": teal_heights,
        "blue_heights": blue_heights,
        "blue_starts": blue_starts,
    }


def _selected_bar_indices(n_bars: int, from_right: bool) -> set[int]:
    if from_right:
        return {idx for idx in range(n_bars) if (n_bars - 1 - idx) % 2 == 0}
    return {idx for idx in range(n_bars) if idx % 2 == 0}


def _render_output(
    grid: np.ndarray,
    structure: dict,
    *,
    normalize_upper: bool,
    select_from_right: bool,
    recolor_selected_only: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    red_row = int(structure["red_row"])
    grey_row = int(structure["grey_row"])
    bar_cols = list(structure["bar_cols"])
    teal_heights = list(structure["teal_heights"])
    blue_heights = list(structure["blue_heights"])

    target_upper = max(teal_heights) if normalize_upper else None
    selected = _selected_bar_indices(len(bar_cols), from_right=select_from_right)

    for idx, (c, teal_h, blue_h) in enumerate(zip(bar_cols, teal_heights, blue_heights)):
        out[:, c] = BG

        upper_h = target_upper if normalize_upper else int(teal_h)
        if normalize_upper:
            top = red_row - upper_h + 1
            out[top : red_row + 1, c] = TEAL
        else:
            top = red_row - teal_h
            out[top:red_row, c] = TEAL

        lower_color = TEAL if (recolor_selected_only and idx in selected) else BLUE
        out[grey_row + 1 : grey_row + 1 + blue_h, c] = lower_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Normalize the upper halves of marker-separated vertical bars, then recolor alternating lower halves from the right.

    Input structure:
    - A black background contains one red marker row directly above one grey marker row.
    - The marker pixels appear in alternating columns, and each gap column between consecutive markers is a bar column.
    - Every bar column has one contiguous teal segment above the red row and one contiguous blue segment below the grey row.

    Rule:
    - Find the maximum teal height among all bar columns.
    - In every bar column, fill a teal segment of that maximum height ending on the red-row baseline.
    - Keep each blue segment's height, move it to start immediately below the grey row, and sort bars from right to left.
    - Recolor the lower segment of right-ranks 0, 2, 4, ... from blue to teal; leave the other lower segments blue.

    Key insight:
    - The red/grey marker pair defines a shared baseline: first normalize every upper segment against that baseline, then apply right-anchored alternating recoloring to the normalized bars.
    """
    structure = _extract_structure(np.asarray(grid, dtype=int))
    return _render_output(
        np.asarray(grid, dtype=int),
        structure,
        normalize_upper=True,
        select_from_right=True,
        recolor_selected_only=True,
    )


def _sample_teal_heights(n_bars: int, max_teal: int) -> np.ndarray:
    selected = _selected_bar_indices(n_bars, from_right=True)
    unselected = set(range(n_bars)) - selected

    for _ in range(200):
        heights = _RNG.integers(1, max_teal + 1, size=n_bars).astype(int)
        heights[int(_RNG.integers(0, n_bars))] = max_teal
        if len(set(map(int, heights))) < 2:
            continue
        if not any(int(heights[i]) < max_teal for i in selected):
            continue
        if not any(int(heights[i]) < max_teal for i in unselected):
            continue
        return heights
    raise RuntimeError("failed to sample non-degenerate teal heights")


def _sample_blue_heights(n_bars: int) -> np.ndarray:
    heights = _RNG.integers(_BLUE_HEIGHT_RANGE[0], _BLUE_HEIGHT_RANGE[1] + 1, size=n_bars).astype(int)
    if len(set(map(int, heights))) == 1 and _RNG.random() < 0.6:
        idx = int(_RNG.integers(0, n_bars))
        choices = [h for h in range(_BLUE_HEIGHT_RANGE[0], _BLUE_HEIGHT_RANGE[1] + 1) if h != int(heights[idx])]
        heights[idx] = int(_RNG.choice(np.array(choices, dtype=int)))
    return heights


def _place_upper_segment(grid: np.ndarray, col: int, red_row: int, teal_h: int, floating: bool) -> None:
    if floating:
        max_start = red_row - teal_h
        start = int(_RNG.integers(0, max_start + 1))
    else:
        start = red_row - teal_h
    grid[start : start + teal_h, col] = TEAL


def _place_lower_segments(
    grid: np.ndarray,
    bar_cols: list[int],
    grey_row: int,
    blue_heights: np.ndarray,
    *,
    floating: bool,
) -> None:
    H = grid.shape[0]
    starts = []
    for blue_h in blue_heights:
        min_start = grey_row + 1
        max_start = H - int(blue_h)
        if floating:
            start = int(_RNG.integers(min_start, max_start + 1))
        else:
            start = max_start
        starts.append(start)

    if floating and not any(start > grey_row + 1 for start in starts):
        idx = int(_RNG.integers(0, len(starts)))
        blue_h = int(blue_heights[idx])
        max_start = H - blue_h
        if max_start > grey_row + 1:
            starts[idx] = int(_RNG.integers(grey_row + 2, max_start + 1))

    for c, blue_h, start in zip(bar_cols, blue_heights, starts):
        grid[start : start + int(blue_h), c] = BLUE


def _build_input(layout: dict) -> np.ndarray:
    H = int(layout["H"])
    W = int(layout["W"])
    red_row = int(layout["red_row"])
    grey_row = int(layout["grey_row"])
    marker_cols = list(layout["marker_cols"])
    bar_cols = list(layout["bar_cols"])
    teal_heights = np.asarray(layout["teal_heights"], dtype=int)
    blue_heights = np.asarray(layout["blue_heights"], dtype=int)

    grid = np.full((H, W), BG, dtype=int)
    grid[red_row, marker_cols] = RED
    grid[grey_row, marker_cols] = GREY

    upper_floating = bool(layout["upper_floating"])
    lower_floating = bool(layout["lower_floating"])
    for c, teal_h in zip(bar_cols, teal_heights):
        _place_upper_segment(grid, int(c), red_row, int(teal_h), upper_floating)
    _place_lower_segments(grid, bar_cols, grey_row, blue_heights, floating=lower_floating)
    return grid


def _sample_layout() -> dict | None:
    n_bars = int(_RNG.choice(np.array(_VALID_BAR_COUNTS, dtype=int)))
    start_col = int(_RNG.choice(np.array(_START_COL_OPTIONS, dtype=int)))
    marker_cols = [int(start_col + 2 * i) for i in range(n_bars + 1)]
    bar_cols = [c + 1 for c in marker_cols[:-1]]

    right_pad = int(_RNG.integers(_RIGHT_PAD_RANGE[0], _RIGHT_PAD_RANGE[1] + 1))
    W = int(marker_cols[-1] + 1 + right_pad)

    max_teal = int(_RNG.integers(_MAX_TEAL_RANGE[0], _MAX_TEAL_RANGE[1] + 1))
    teal_heights = _sample_teal_heights(n_bars, max_teal)
    blue_heights = _sample_blue_heights(n_bars)
    max_blue = int(np.max(blue_heights))

    top_pad = int(_RNG.integers(_TOP_PAD_RANGE[0], _TOP_PAD_RANGE[1] + 1))
    red_row = int(top_pad + max_teal)
    grey_row = red_row + 1

    bottom_pad = int(_RNG.integers(_BOTTOM_PAD_RANGE[0], _BOTTOM_PAD_RANGE[1] + 1))
    H = int(red_row + 2 + max_blue + bottom_pad)

    if H > 30 or W > 30:
        return None

    return {
        "H": H,
        "W": W,
        "red_row": red_row,
        "grey_row": grey_row,
        "marker_cols": marker_cols,
        "bar_cols": bar_cols,
        "teal_heights": teal_heights.tolist(),
        "blue_heights": blue_heights.tolist(),
        "upper_floating": bool(_RNG.random() < 0.35),
        "lower_floating": bool(_RNG.random() < 0.35),
    }


def _direct_expected(inp: np.ndarray) -> np.ndarray:
    structure = _extract_structure(inp)
    return _render_output(
        inp,
        structure,
        normalize_upper=True,
        select_from_right=True,
        recolor_selected_only=True,
    )


def _wrong_without_normalization(inp: np.ndarray) -> np.ndarray:
    structure = _extract_structure(inp)
    return _render_output(
        inp,
        structure,
        normalize_upper=False,
        select_from_right=True,
        recolor_selected_only=True,
    )


def _wrong_from_left(inp: np.ndarray) -> np.ndarray:
    structure = _extract_structure(inp)
    return _render_output(
        inp,
        structure,
        normalize_upper=True,
        select_from_right=False,
        recolor_selected_only=True,
    )


def _wrong_recolor_all(inp: np.ndarray) -> np.ndarray:
    structure = _extract_structure(inp)
    out = _render_output(
        inp,
        structure,
        normalize_upper=True,
        select_from_right=True,
        recolor_selected_only=False,
    )
    for c, blue_h in zip(structure["bar_cols"], structure["blue_heights"]):
        out[structure["grey_row"] + 1 : structure["grey_row"] + 1 + blue_h, c] = TEAL
    return out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    if np.array_equal(inp, expected):
        return False

    structure = _extract_structure(inp)
    teal_heights = structure["teal_heights"]
    if len(set(teal_heights)) < 2:
        return False
    if max(teal_heights) <= min(teal_heights):
        return False

    if not any(start > structure["grey_row"] + 1 for start in structure["blue_starts"]):
        return False

    wrong_a = _wrong_without_normalization(inp)
    wrong_b = _wrong_from_left(inp)
    wrong_c = _wrong_recolor_all(inp)
    return (
        not np.array_equal(expected, wrong_a)
        and not np.array_equal(expected, wrong_b)
        and not np.array_equal(expected, wrong_c)
    )


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None

    inp = _build_input(layout)
    expected = _direct_expected(inp)

    try:
        actual = main(inp)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    if not _semantic_checks(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
