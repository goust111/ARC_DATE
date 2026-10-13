from __future__ import annotations

import random
from collections import Counter

import numpy as np


MAX_GEN_ATTEMPTS = 400
BG = 0
PALETTE = tuple(range(1, 10))
SIDE_CHOICES = (4, 5, 6, 7, 8)
ENTRY_COUNT_CHOICES = (4, 5)

# Connected 3x3 cluster shapes keyed by pixel count.
CLUSTER_SHAPES: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
    4: (
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (1, 1), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
    ),
    5: (
        ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
        ((0, 1), (0, 2), (1, 0), (1, 1), (2, 1)),
    ),
    6: (
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 1), (2, 2)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
    ),
    7: (
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)),
        ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
        ((0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)),
    ),
    8: (
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
        ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
    ),
}

# 3x3 glyph masks; every row and column is represented so the glyph is visually stable.
GLYPH_MASKS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 2), (1, 1), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (1, 2), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 0)),
    ((0, 2), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 1), (1, 0), (1, 2), (2, 0), (2, 2)),
)


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(np.asarray(mask_1d, dtype=bool).tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask_1d) - 1))
    return runs


def _coords_to_patch(coords: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    patch = np.zeros((3, 3), dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _extract_square_info(main_zone: np.ndarray, bg: int) -> tuple[int, tuple[int, int], np.ndarray]:
    coords = np.argwhere(main_zone != bg)
    assert coords.shape[0] == 3, "expected exactly three main-area markers"
    colors = {int(main_zone[int(r), int(c)]) for r, c in coords}
    assert len(colors) == 1, "main-area markers must share one color"

    rows = sorted(set(map(int, coords[:, 0])))
    cols = sorted(set(map(int, coords[:, 1])))
    assert len(rows) == 2 and len(cols) == 2, "markers must occupy two rows and two columns"
    side = int(rows[1] - rows[0])
    assert side == int(cols[1] - cols[0]) and side > 0, "markers must define a non-degenerate square"

    present = {tuple(map(int, rc)) for rc in coords}
    all_corners = {(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])}
    missing = list(all_corners - present)
    assert len(missing) == 1, "expected exactly one missing corner"
    return side, missing[0], coords


def _parse_dictionary(dict_zone: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    row_runs = _runs(np.any(dict_zone != bg, axis=1))
    assert row_runs, "dictionary must contain at least one entry"

    mapping: dict[int, np.ndarray] = {}
    for r0, r1 in row_runs:
        sub = dict_zone[r0 : r1 + 1, :]
        assert sub.shape[0] == 3, "each dictionary entry must occupy a 3-row band"
        col_runs = _runs(np.any(sub != bg, axis=0))
        assert len(col_runs) == 2, "each dictionary entry must contain one cluster and one glyph"
        (c0a, c1a), (c0b, c1b) = col_runs
        assert c0a < c0b, "cluster must appear left of glyph"

        cluster = sub[:, c0a : c1a + 1]
        glyph = sub[:, c0b : c1b + 1]
        assert glyph.shape == (3, 3), "glyph must be 3x3"

        count = int(np.count_nonzero(cluster != bg))
        assert count not in mapping, "cluster counts must be unique across dictionary entries"
        mapping[count] = glyph.copy()

    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the missing square corner, use the square side length as a dictionary key, and stamp the matched glyph there.

    Input structure:
    - A full-height separator line splits the grid into a black-background main zone and a dictionary zone.
    - In the main zone, exactly three same-colored pixels occupy three corners of one axis-aligned square; the square edges and fourth corner are absent.
    - In the dictionary zone, each entry is a pixel cluster beside a 3x3 glyph, and different entries have different cluster pixel counts.

    Rule:
    - Reconstruct the implied square from the three corner markers, determine the missing fourth corner, and measure the square side length in grid steps.
    - In the dictionary zone, find the unique entry whose cluster has exactly that many pixels.
    - Output only the main zone, preserving the three corner markers and stamping the matched 3x3 glyph centered on the missing corner.

    Key insight:
    - The lookup key is not a visible object property such as color or marker count; it is the unseen square side length implied by the three observed corners.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)

    sep_cols = [
        c
        for c in range(g.shape[1])
        if int(g[0, c]) != bg and bool(np.all(g[:, c] == g[0, c]))
    ]
    assert len(sep_cols) == 1, "expected exactly one full-height separator line"
    sep_c = int(sep_cols[0])
    assert 0 < sep_c < g.shape[1] - 1, "separator must split non-empty main and dictionary zones"

    main_zone = g[:, :sep_c].copy()
    dict_zone = g[:, sep_c + 1 :]

    side, missing_corner, _coords = _extract_square_info(main_zone, bg)
    entries = _parse_dictionary(dict_zone, bg)
    assert side in entries, "square side must match one dictionary cluster count"

    glyph = entries[side]
    mr, mc = map(int, missing_corner)
    r0 = mr - 1
    c0 = mc - 1
    assert 0 <= r0 and 0 <= c0 and r0 + 3 <= main_zone.shape[0] and c0 + 3 <= main_zone.shape[1], (
        "glyph stamp must fit inside the main zone"
    )

    out = main_zone
    patch = out[r0 : r0 + 3, c0 : c0 + 3]
    mask = glyph != bg
    patch[mask] = glyph[mask]
    return out


def _expected_from_case(
    height: int,
    main_w: int,
    marker_color: int,
    present_corners: list[tuple[int, int]],
    missing_corner: tuple[int, int],
    glyph: np.ndarray,
) -> np.ndarray:
    out = np.zeros((height, main_w), dtype=int)
    for r, c in present_corners:
        out[int(r), int(c)] = int(marker_color)
    mr, mc = missing_corner
    r0 = int(mr) - 1
    c0 = int(mc) - 1
    mask = glyph != BG
    out[r0 : r0 + 3, c0 : c0 + 3][mask] = glyph[mask]
    return out


def _sample_counts(target_side: int, n_entries: int) -> list[int]:
    others = [side for side in SIDE_CHOICES if side != target_side]
    picks = random.sample(others, n_entries - 1)
    counts = [target_side, *picks]
    random.shuffle(counts)
    return counts


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    target_side = int(random.choice(SIDE_CHOICES))
    n_entries = int(random.choice(ENTRY_COUNT_CHOICES))
    entry_counts = _sample_counts(target_side, n_entries)

    dict_h = 4 * n_entries
    height = random.randint(max(target_side + 3, dict_h), min(24, dict_h + 4))
    main_w = random.randint(target_side + 4, min(19, target_side + 9))
    dict_w = 9
    width = main_w + 1 + dict_w
    assert width <= 30 and height <= 30

    grid = np.zeros((height, width), dtype=int)
    sep_c = main_w
    sep_color = int(random.choice(PALETTE))
    grid[:, sep_c] = sep_color

    marker_color = int(random.choice(PALETTE))
    top = random.randint(1, height - target_side - 2)
    left = random.randint(1, main_w - target_side - 2)
    corners = [
        (top, left),
        (top, left + target_side),
        (top + target_side, left),
        (top + target_side, left + target_side),
    ]
    missing_index = random.randrange(4)
    missing_corner = corners[missing_index]
    present_corners = [pos for i, pos in enumerate(corners) if i != missing_index]
    for r, c in present_corners:
        grid[int(r), int(c)] = marker_color

    glyph_masks = random.sample(GLYPH_MASKS, n_entries)
    glyph_colors = [int(random.choice(PALETTE)) for _ in range(n_entries)]
    cluster_colors = [int(random.choice(PALETTE)) for _ in range(n_entries)]
    target_glyph: np.ndarray | None = None

    for i, count in enumerate(entry_counts):
        row0 = 1 + 4 * i
        cluster_patch = _coords_to_patch(random.choice(CLUSTER_SHAPES[int(count)]), cluster_colors[i])
        glyph_patch = _coords_to_patch(glyph_masks[i], glyph_colors[i])
        if count == target_side:
            target_glyph = glyph_patch.copy()

        dict_col = sep_c + 1
        grid[row0 : row0 + 3, dict_col + 1 : dict_col + 4][cluster_patch != BG] = cluster_patch[cluster_patch != BG]
        grid[row0 : row0 + 3, dict_col + 5 : dict_col + 8][glyph_patch != BG] = glyph_patch[glyph_patch != BG]

    assert target_glyph is not None
    expected = _expected_from_case(height, main_w, marker_color, present_corners, missing_corner, target_glyph)
    return grid, expected


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    bg = get_bg(inp)
    sep_cols = [
        c
        for c in range(inp.shape[1])
        if int(inp[0, c]) != bg and bool(np.all(inp[:, c] == inp[0, c]))
    ]
    if len(sep_cols) != 1:
        return False

    sep_c = int(sep_cols[0])
    main_zone = inp[:, :sep_c]
    dict_zone = inp[:, sep_c + 1 :]

    try:
        side, missing_corner, coords = _extract_square_info(main_zone, bg)
        mapping = _parse_dictionary(dict_zone, bg)
    except AssertionError:
        return False

    if side not in mapping:
        return False
    if not np.array_equal(main(inp), expected):
        return False

    # Verify that the glyph actually contributes non-corner pixels at the inferred missing corner.
    mr, mc = missing_corner
    glyph = mapping[side]
    placed = np.zeros_like(expected)
    placed[mr - 1 : mr + 2, mc - 1 : mc + 2][glyph != bg] = glyph[glyph != bg]
    if np.count_nonzero(placed) == 0:
        return False

    marker_positions = {tuple(map(int, rc)) for rc in coords}
    if marker_positions & {
        (mr - 1, mc - 1),
        (mr - 1, mc),
        (mr - 1, mc + 1),
        (mr, mc - 1),
        (mr, mc),
        (mr, mc + 1),
        (mr + 1, mc - 1),
        (mr + 1, mc),
        (mr + 1, mc + 1),
    }:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        inp, expected = _build_case()
        if _semantic_checks(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid invisible-side-key input")
