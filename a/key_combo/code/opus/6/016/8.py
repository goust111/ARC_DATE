from __future__ import annotations

from importlib.machinery import SourceFileLoader

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
tile_fill = _ARC_UTILS.tile_fill


BG = Color.BLACK
SEPARATOR = Color.YELLOW
PHASE = Color.ORANGE
CENTER = Color.MAROON
OUTPUT_SHAPE = (15, 15)
MAX_RETRIES = 500
STRIPE_PALETTE = tuple(range(1, 10))


def _separator_columns(grid: np.ndarray) -> tuple[int, int]:
    cols = [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == SEPARATOR))]
    assert len(cols) == 2, "expected exactly two full-height separator columns"
    s0, s1 = map(int, cols)
    assert 0 < s0 < s1 < grid.shape[1] - 1
    return s0, s1


def _panel_slices(grid: np.ndarray) -> list[slice]:
    s0, s1 = _separator_columns(grid)
    panel_w = int(s0)
    assert int(s1 - s0 - 1) == panel_w
    assert int(grid.shape[1] - s1 - 1) == panel_w
    return [slice(0, s0), slice(s0 + 1, s1), slice(s1 + 1, grid.shape[1])]


def _row_color(row: np.ndarray) -> int | None:
    vals = sorted(set(map(int, row)))
    if vals == [BG]:
        return None
    assert BG not in vals, "non-empty stripe rows must fill the full panel width"
    assert len(vals) == 1, "each stripe row must be monochrome"
    return int(vals[0])


def _extract_sequence(grid: np.ndarray) -> tuple[list[int], list[list[int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert int(np.min(g)) >= 0 and int(np.max(g)) <= 9

    panel_views = [g[:, panel_slice] for panel_slice in _panel_slices(g)]
    per_panel: list[list[int]] = []
    for panel in panel_views:
        colors: list[int] = []
        for r in range(panel.shape[0]):
            color = _row_color(panel[r, :])
            if color is not None:
                colors.append(color)
        assert colors, "each panel must contain at least one stripe"
        per_panel.append(colors)

    sequence: list[int] = []
    for colors in reversed(per_panel):
        sequence.extend(colors)

    assert 4 <= len(sequence) <= 9
    assert len(set(sequence)) == len(sequence), "each stripe color must appear exactly once"
    return sequence, per_panel


def _glyph_from_sequence_length(length: int) -> np.ndarray:
    assert 4 <= int(length) <= 9
    glyph = np.zeros((3, 3), dtype=bool)
    glyph.reshape(-1)[: int(length)] = True
    return glyph


def _build_output(length: int) -> np.ndarray:
    glyph = _glyph_from_sequence_length(length)
    phase_tile = np.where(glyph, PHASE, BG).astype(int)
    out = tile_fill(OUTPUT_SHAPE, phase_tile).astype(int)

    center_patch = np.tile(glyph, (2, 2))
    r0 = (out.shape[0] - center_patch.shape[0]) // 2
    c0 = (out.shape[1] - center_patch.shape[1]) // 2
    out[r0 : r0 + center_patch.shape[0], c0 : c0 + center_patch.shape[1]][center_patch] = CENTER
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress three separator-split stripe panels into a reverse-scanned code and decode that code as a periodic 3x3 phase lattice.

    Input structure:
    - The background is black, and two full-height yellow columns split the input into three equal-width panels.
    - Inside each panel, each occupied row is a one-pixel-thick full-width monochrome stripe and each empty row is all black.
    - Across all three panels, every stripe color appears exactly once, each panel contributes at least one stripe, and the reverse-scanned stripe sequence has length 4-9.

    Rule:
    - Scan panels from right to left, and within each panel scan rows from top to bottom.
    - Discard empty rows and record the colors of the occupied rows in that order.
    - Decode the resulting sequence length N as a 3x3 row-major binary glyph whose first N positions are foreground and whose remaining positions are background.
    - Use that 3x3 glyph as an orange-on-black phase tile repeated across a 15x15 output, then overwrite the centered 6x6 region with a 2x2 tiling of the same glyph recolored to maroon.

    Key insight:
    - The panel stripes are only an intermediate code: the useful information is the compressed reverse-scan occupancy count, which is reinterpreted as spatial phase and then expanded back into a periodic lattice.
    """
    sequence, _per_panel = _extract_sequence(np.asarray(grid, dtype=int))
    return _build_output(len(sequence))


def _compositions(total: int, per_panel_cap: int) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for right in range(1, per_panel_cap + 1):
        for middle in range(1, per_panel_cap + 1):
            left = total - right - middle
            if 1 <= left <= per_panel_cap:
                out.append((right, middle, left))
    return out


def _rows_have_gap(rows: list[int]) -> bool:
    return rows[0] > 0 or any(b - a > 1 for a, b in zip(rows, rows[1:]))


def _choose_rows(
    rng: np.random.Generator,
    height: int,
    count: int,
    *,
    require_gap: bool,
) -> list[int]:
    for _ in range(120):
        rows = sorted(int(v) for v in rng.choice(height, size=count, replace=False))
        if require_gap and not _rows_have_gap(rows):
            continue
        return rows
    raise RuntimeError("failed to choose stripe rows")


def _build_input(
    height: int,
    panel_w: int,
    groups: tuple[int, int, int],
    sequence: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    grid = np.full((height, 3 * panel_w + 2), BG, dtype=int)
    grid[:, panel_w] = SEPARATOR
    grid[:, 2 * panel_w + 1] = SEPARATOR

    right_n, middle_n, left_n = map(int, groups)
    cut0 = right_n
    cut1 = right_n + middle_n
    panel_sequences = {
        "right": sequence[:cut0],
        "middle": sequence[cut0:cut1],
        "left": sequence[cut1:],
    }

    gap_candidates = [name for name, count in zip(("left", "middle", "right"), (left_n, middle_n, right_n)) if count < height]
    assert gap_candidates
    gap_panel = gap_candidates[int(rng.integers(len(gap_candidates)))]

    panel_rows = {
        "left": _choose_rows(rng, height, left_n, require_gap=(gap_panel == "left")),
        "middle": _choose_rows(rng, height, middle_n, require_gap=(gap_panel == "middle")),
        "right": _choose_rows(rng, height, right_n, require_gap=(gap_panel == "right")),
    }
    if not any(_rows_have_gap(rows) for rows in panel_rows.values()):
        name = gap_candidates[int(rng.integers(len(gap_candidates)))]
        count = {"left": left_n, "middle": middle_n, "right": right_n}[name]
        panel_rows[name] = _choose_rows(rng, height, count, require_gap=True)

    assignments = [
        (slice(0, panel_w), "left"),
        (slice(panel_w + 1, 2 * panel_w + 1), "middle"),
        (slice(2 * panel_w + 2, 3 * panel_w + 2), "right"),
    ]
    for cols, name in assignments:
        rows = panel_rows[name]
        colors = panel_sequences[name]
        assert len(rows) == len(colors)
        for r, color in zip(rows, colors):
            grid[int(r), cols] = int(color)

    return grid


def _panel_row_indices(grid: np.ndarray) -> list[list[int]]:
    out: list[list[int]] = []
    for panel_slice in _panel_slices(grid):
        panel = grid[:, panel_slice]
        rows = [int(r) for r in range(panel.shape[0]) if bool(np.any(panel[r, :] != BG))]
        out.append(rows)
    return out


def _validate_generated(grid: np.ndarray, expected_length: int) -> bool:
    try:
        sequence, per_panel = _extract_sequence(grid)
    except AssertionError:
        return False

    if len(sequence) != expected_length:
        return False
    if not any(len(colors) >= 2 for colors in per_panel):
        return False
    if not any(_rows_have_gap(rows) for rows in _panel_row_indices(grid)):
        return False

    out = main(grid)
    if not np.array_equal(out, _build_output(expected_length)):
        return False

    center = out[4:10, 4:10]
    if int(np.count_nonzero(center == CENTER)) != 4 * expected_length:
        return False
    if out.shape != OUTPUT_SHAPE:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    length = int(rng.choice(np.array([4, 5, 6, 7, 8, 9], dtype=int)))
    min_height = max(4, (length + 2) // 3)
    height = int(rng.integers(max(5, min_height), 9))
    panel_w = int(rng.integers(4, 8))

    group_options = [g for g in _compositions(length, height) if max(g) < height or min(g) < max(g)]
    if not group_options:
        return None
    groups = group_options[int(rng.integers(len(group_options)))]

    colors = [int(v) for v in rng.choice(np.array(STRIPE_PALETTE, dtype=int), size=length, replace=False)]

    try:
        grid = _build_input(height, panel_w, groups, colors, rng)
    except (AssertionError, RuntimeError):
        return None

    if not _validate_generated(grid, length):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
