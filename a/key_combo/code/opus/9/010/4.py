from __future__ import annotations

import importlib.util
import itertools
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    utils_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_010_4", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load arc_utils from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_arc_utils = _load_arc_utils()
colors_present = _arc_utils.colors_present
draw_line = _arc_utils.draw_line
get_bg = _arc_utils.get_bg


LEGEND_COLS = 2
GAP_COL = 2
MAIN_START_COL = 3
PALETTE = tuple(range(1, 10))
LAYOUT_CHOICES = ((2, 1), (2, 2), (3, 1), (3, 2), (4, 1))
GEN_RETRIES = 500


def _legend_pairs(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        src = int(grid[r, 0])
        dst = int(grid[r, 1])
        if src == bg and dst == bg:
            break
        assert src != bg and dst != bg, "legend rows must be fully populated before the first blank row"
        pairs.append((src, dst))

    assert 2 <= len(pairs) <= 4, "expected a 2-4 row legend"
    assert len({src for src, _dst in pairs}) == len(pairs), "legend source colors must be unique"
    assert np.all(grid[:, GAP_COL] == bg), "column 2 must be the black separator"
    assert np.all(grid[len(pairs) :, :LEGEND_COLS] == bg), "legend area below the entries must be black"
    return pairs


def _analyze(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    assert h >= 8 and w >= MAIN_START_COL + 6, "grid is too small for the puzzle structure"

    bg = get_bg(g)
    assert bg == 0, "background must be black"
    pairs = _legend_pairs(g, bg)
    mapping = {src: dst for src, dst in pairs}

    main = g[:, MAIN_START_COL:]
    main_colors = colors_present(main, exclude={bg})
    assert 3 <= len(main_colors) <= 5, "main area must contain 3-5 distinct endpoint colors"

    horizontal_owner = np.zeros_like(g, dtype=int)
    vertical_owner = np.zeros_like(g, dtype=int)
    segments: list[tuple[str, int, int, int, int, int]] = []

    for color in main_colors:
        coords = np.argwhere(main == int(color))
        assert coords.shape == (2, 2), "each main-area color must appear exactly twice"
        coords = coords.astype(int, copy=True)
        coords[:, 1] += MAIN_START_COL
        (r0, c0), (r1, c1) = coords.tolist()
        assert c0 >= MAIN_START_COL and c1 >= MAIN_START_COL

        if r0 == r1:
            left, right = sorted((c0, c1))
            assert right - left >= 2, "horizontal endpoints should leave room for interior crossings"
            for c in range(left, right + 1):
                assert horizontal_owner[r0, c] == 0, "horizontal segments must not overlap each other"
                horizontal_owner[r0, c] = int(color)
            segments.append(("horizontal", int(color), int(r0), int(left), int(r1), int(right)))
        else:
            assert c0 == c1, "same-color endpoints must share either a row or a column"
            top, bottom = sorted((r0, r1))
            assert bottom - top >= 2, "vertical endpoints should leave room for interior crossings"
            for r in range(top, bottom + 1):
                assert vertical_owner[r, c0] == 0, "vertical segments must not overlap each other"
                vertical_owner[r, c0] = int(color)
            segments.append(("vertical", int(color), int(top), int(c0), int(bottom), int(c1)))

    vertical_colors = {color for orient, color, *_rest in segments if orient == "vertical"}
    horizontal_colors = {color for orient, color, *_rest in segments if orient == "horizontal"}
    assert vertical_colors and horizontal_colors, "need both horizontal and vertical segments"
    assert vertical_colors.issubset(mapping), "every vertical segment color must be present in the legend"

    cross_mask = (horizontal_owner != 0) & (vertical_owner != 0)
    crossings = [tuple(map(int, rc)) for rc in np.argwhere(cross_mask)]

    return {
        "grid": g,
        "bg": bg,
        "pairs": pairs,
        "mapping": mapping,
        "segments": segments,
        "horizontal_owner": horizontal_owner,
        "vertical_owner": vertical_owner,
        "crossings": crossings,
        "horizontal_colors": horizontal_colors,
        "vertical_colors": vertical_colors,
    }


def _expected_output_from_analysis(scene: dict[str, object]) -> np.ndarray:
    g = np.asarray(scene["grid"], dtype=int)
    out = g.copy()
    segments = scene["segments"]
    mapping = scene["mapping"]
    horizontal_owner = np.asarray(scene["horizontal_owner"], dtype=int)
    vertical_owner = np.asarray(scene["vertical_owner"], dtype=int)

    for orient, color, r0, c0, r1, c1 in segments:
        if orient == "horizontal":
            draw_line(out, r0, c0, end_r=r1, end_c=c1, color=color)
    for orient, color, r0, c0, r1, c1 in segments:
        if orient == "vertical":
            draw_line(out, r0, c0, end_r=r1, end_c=c1, color=color)

    for r, c in np.argwhere((horizontal_owner != 0) & (vertical_owner != 0)):
        vertical_color = int(vertical_owner[r, c])
        out[r, c] = int(mapping.get(vertical_color, vertical_color))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw same-color row/column segments, then recolor their crossings by consulting the legend with the vertical color.

    Input structure:
    - The background is black.
    - Columns 0 and 1 contain a top-aligned two-column legend: source color in column 0, mapped color in column 1; column 2 is a black separator.
    - In the main area starting at column 3, each non-black color appears exactly twice and the two pixels share a row or a column.

    Rule:
    - Connect each same-color pair with its inclusive horizontal or vertical segment.
    - Crossings occur where one horizontal and one vertical segment pass through the same cell.
    - At each crossing, keep the crossing position but recolor it to the legend-mapped color of the vertical segment's color; if that color is absent from the legend, keep the vertical color.
    - All non-crossing segment cells keep their segment color, and the legend area stays unchanged.

    Key insight:
    - The legend is not a global recoloring table; it is only activated at line-crossing events, and the lookup key is the vertical segment color.
    """
    scene = _analyze(grid)
    return _expected_output_from_analysis(scene)


def _spaced_positions(count: int, low: int, high: int, min_gap: int) -> list[int]:
    options = [
        combo
        for combo in itertools.combinations(range(low, high + 1), count)
        if all(combo[i + 1] - combo[i] >= min_gap for i in range(len(combo) - 1))
    ]
    if not options:
        raise ValueError("no valid spaced positions")
    return list(random.choice(options))


def _sample_color_roles(n_vertical: int, n_horizontal: int) -> tuple[list[int], list[int], dict[int, int]]:
    chosen = random.sample(PALETTE, n_vertical + n_horizontal)
    vertical_colors = chosen[:n_vertical]
    horizontal_colors = chosen[n_vertical:]
    remaining = [color for color in PALETTE if color not in chosen]

    if n_vertical == 1:
        non_identity_count = 1
    elif random.random() < 0.5:
        non_identity_count = random.randint(1, n_vertical - 1)
    else:
        non_identity_count = random.randint(1, n_vertical)

    non_identity_indices = set(random.sample(range(n_vertical), non_identity_count))
    random.shuffle(remaining)

    mapping: dict[int, int] = {}
    for idx, color in enumerate(vertical_colors):
        if idx in non_identity_indices:
            mapping[color] = remaining.pop()
        else:
            mapping[color] = color

    return vertical_colors, horizontal_colors, mapping


def _render_input(
    height: int,
    main_width: int,
    legend_pairs: list[tuple[int, int]],
    h_segments: list[tuple[int, int, int, int]],
    v_segments: list[tuple[int, int, int, int]],
) -> np.ndarray:
    grid = np.zeros((height, main_width + MAIN_START_COL), dtype=int)

    for r, (src, dst) in enumerate(legend_pairs):
        grid[r, 0] = int(src)
        grid[r, 1] = int(dst)

    for color, row, left, right in h_segments:
        grid[row, MAIN_START_COL + left] = int(color)
        grid[row, MAIN_START_COL + right] = int(color)

    for color, col, top, bottom in v_segments:
        grid[top, MAIN_START_COL + col] = int(color)
        grid[bottom, MAIN_START_COL + col] = int(color)

    return grid


def _make_attempt() -> np.ndarray:
    n_vertical, n_horizontal = random.choice(LAYOUT_CHOICES)
    min_height = max(11, n_vertical + 2 * n_horizontal + 2)
    min_main_width = max(11, 2 * n_vertical + 3)
    height = random.randint(min_height, 18)
    main_width = random.randint(min_main_width, 16)

    vertical_colors, horizontal_colors, mapping = _sample_color_roles(n_vertical, n_horizontal)
    h_rows = _spaced_positions(n_horizontal, n_vertical + 1, height - 3, 2)
    v_cols = _spaced_positions(n_vertical, 2, main_width - 3, 2)

    h_segments: list[tuple[int, int, int, int]] = []
    v_segments: list[tuple[int, int, int, int]] = []

    for color, row in zip(horizontal_colors, h_rows):
        left = random.randint(0, v_cols[0] - 1)
        right = random.randint(v_cols[-1] + 1, main_width - 1)
        h_segments.append((int(color), int(row), int(left), int(right)))

    for color, col in zip(vertical_colors, v_cols):
        top = random.randint(0, h_rows[0] - 1)
        bottom = random.randint(h_rows[-1] + 1, height - 1)
        v_segments.append((int(color), int(col), int(top), int(bottom)))

    legend_pairs = list(mapping.items())
    random.shuffle(legend_pairs)
    return _render_input(height, main_width, legend_pairs, h_segments, v_segments)


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        scene = _analyze(grid)
    except AssertionError:
        return False

    pairs = scene["pairs"]
    mapping = scene["mapping"]
    segments = scene["segments"]
    crossings = scene["crossings"]
    horizontal_owner = np.asarray(scene["horizontal_owner"], dtype=int)
    vertical_owner = np.asarray(scene["vertical_owner"], dtype=int)

    if len(pairs) != len(scene["vertical_colors"]):
        return False
    if len(segments) != len(scene["horizontal_colors"]) + len(scene["vertical_colors"]):
        return False
    if len(crossings) < 2:
        return False

    changed_crossings = 0
    for r, c in crossings:
        vcol = int(vertical_owner[r, c])
        hcol = int(horizontal_owner[r, c])
        mapped = int(mapping.get(vcol, vcol))
        if mapped != vcol:
            changed_crossings += 1
            if mapped == hcol:
                return False

    if changed_crossings == 0:
        return False

    used_h = {int(horizontal_owner[r, c]) for r, c in crossings}
    used_v = {int(vertical_owner[r, c]) for r, c in crossings}
    if used_h != scene["horizontal_colors"]:
        return False
    if used_v != scene["vertical_colors"]:
        return False

    expected = _expected_output_from_analysis(scene)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _make_attempt()
        if _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid legend-crossing recolor puzzle")
