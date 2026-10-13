from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("arc_utils_037_5", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_arc_utils = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_arc_utils)

Color = _arc_utils.Color
tile_fill = _arc_utils.tile_fill


CONTROL_FILL = int(Color.PINK)
CONTROL_MARKER = int(Color.RED)
PATTERN_COLORS = [
    int(Color.BLUE),
    int(Color.GREEN),
    int(Color.YELLOW),
    int(Color.GREY),
    int(Color.ORANGE),
    int(Color.TEAL),
    int(Color.MAROON),
]

PERIOD_CHOICES = (2, 3, 4, 5, 6)
REPEAT_CHOICES = (3, 4, 5)
DATA_WIDTH_RANGE = (7, 22)
MAX_ATTEMPTS = 300

_GEN_COUNTER = 0


def _marker_row(control: np.ndarray) -> int:
    marker_rows = np.flatnonzero(control == CONTROL_MARKER)
    assert marker_rows.size == 1, "expected exactly one red marker"
    assert np.all((control == CONTROL_FILL) | (control == CONTROL_MARKER)), "control column must be pink/red only"
    return int(marker_rows[0])


def _mode_row(rows: np.ndarray) -> np.ndarray | None:
    counts = Counter(tuple(map(int, row)) for row in rows)
    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return np.array(ranked[0][0], dtype=int)


def _find_period_and_defect(image: np.ndarray) -> tuple[int, int]:
    g = np.asarray(image, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 1
    height = int(g.shape[0])

    for period in range(1, height):
        if 2 * period > height:
            break

        mismatches = np.zeros(height, dtype=int)
        valid = True
        for phase in range(period):
            rows = g[phase:height:period]
            if rows.shape[0] < 2:
                valid = False
                break
            prototype = _mode_row(rows)
            if prototype is None:
                valid = False
                break
            diff = np.count_nonzero(rows != prototype, axis=1)
            if np.any(diff > 1):
                valid = False
                break
            mismatches[phase:height:period] = diff

        if not valid or int(mismatches.sum()) != 1:
            continue

        defect_rows = np.flatnonzero(mismatches == 1)
        if defect_rows.size == 1:
            return period, int(defect_rows[0])

    raise AssertionError("failed to infer a unique defect row from the periodic pattern")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the left control column and cyclically roll the striped image so the defect row lands on the red marker row.

    Input structure:
    - The leftmost column is a control strip containing pink cells and exactly one red marker cell.
    - The remaining columns form a vertically periodic colored pattern.
    - Exactly one cell in that periodic area violates the repeated row pattern.

    Rule:
    - Delete the control column.
    - Infer the periodic row cycle of the remaining image and locate the unique defect row d.
    - Let r be the row of the red control marker, then roll the remaining image downward by (r - d) mod height.

    Key insight:
    - The red marker gives the destination row, not a fixed shift amount; the true roll depends on first detecting which repeated row class was broken by the lone bad cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 2, "expected a 2D grid with a control column"

    target_row = _marker_row(g[:, 0])
    image = g[:, 1:].copy()
    _period, defect_row = _find_period_and_defect(image)
    shift = (target_row - defect_row) % int(image.shape[0])
    return np.roll(image, shift=shift, axis=0)


def _partition_total(total: int, parts: int, *, min_part: int) -> list[int]:
    remaining = int(total)
    out: list[int] = []
    for idx in range(parts - 1):
        min_rest = min_part * (parts - idx - 1)
        piece = random.randint(min_part, remaining - min_rest)
        out.append(piece)
        remaining -= piece
    out.append(remaining)
    random.shuffle(out)
    return out


def _sample_band_row(band_widths: list[int], palette: list[int]) -> np.ndarray:
    colors: list[int] = []
    for _ in band_widths:
        options = palette if not colors else [c for c in palette if c != colors[-1]]
        colors.append(random.choice(options))
    return np.repeat(np.array(colors, dtype=int), np.array(band_widths, dtype=int))


def _build_banded_rows(period: int, width: int) -> np.ndarray:
    band_count = random.randint(2, min(4, width // 2))
    band_widths = _partition_total(width, band_count, min_part=2)
    palette_size = random.randint(max(3, band_count), min(len(PATTERN_COLORS), period + 3))
    palette = random.sample(PATTERN_COLORS, palette_size)

    rows: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for _ in range(period):
        for _attempt in range(100):
            row = _sample_band_row(band_widths, palette)
            sig = tuple(map(int, row))
            if sig not in seen:
                seen.add(sig)
                rows.append(row)
                break
        else:
            raise RuntimeError("failed to build distinct base rows")
    return np.vstack(rows).astype(int)


def _build_repeating_motif_rows(period: int, width: int) -> np.ndarray:
    motif_width = random.randint(3, min(6, width))
    band_count = random.randint(2, min(3, motif_width))
    band_widths = _partition_total(motif_width, band_count, min_part=1)
    palette_size = random.randint(max(3, band_count), min(len(PATTERN_COLORS), period + 2))
    palette = random.sample(PATTERN_COLORS, palette_size)

    motif_rows: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for _ in range(period):
        for _attempt in range(100):
            row = _sample_band_row(band_widths, palette)
            sig = tuple(map(int, row))
            if sig not in seen:
                seen.add(sig)
                motif_rows.append(row)
                break
        else:
            raise RuntimeError("failed to build distinct motif rows")

    motif = np.vstack(motif_rows).astype(int)
    return tile_fill((period, width), motif).astype(int)


def _choose_rows(height: int, branch: int) -> tuple[int, int]:
    if branch == 0:
        defect_row = random.randrange(height)
        return defect_row, defect_row
    if branch == 1:
        defect_row = random.randrange(height - 1)
        marker_row = random.randrange(defect_row + 1, height)
        return defect_row, marker_row
    defect_row = random.randrange(1, height)
    marker_row = random.randrange(defect_row)
    return defect_row, marker_row


def _gen_attempt(call_index: int) -> tuple[np.ndarray, int, int, np.ndarray]:
    style = (call_index // 3) % 2
    period = random.choice(PERIOD_CHOICES)
    repeats = random.choice([rep for rep in REPEAT_CHOICES if period * rep <= 30])
    height = period * repeats
    width = random.randint(*DATA_WIDTH_RANGE)

    if style == 0:
        base_rows = _build_banded_rows(period, width)
    else:
        base_rows = _build_repeating_motif_rows(period, width)

    image = tile_fill((height, width), base_rows).astype(int)
    defect_row, marker_row = _choose_rows(height, call_index % 3)
    defect_col = random.randrange(width)
    expected = int(image[defect_row, defect_col])
    alternatives = sorted(set(map(int, image.flatten())) - {expected})
    defect_color = random.choice(alternatives)
    image[defect_row, defect_col] = defect_color

    grid = np.full((height, width + 1), CONTROL_FILL, dtype=int)
    grid[marker_row, 0] = CONTROL_MARKER
    grid[:, 1:] = image
    return grid, period, defect_row, image


def _is_valid_instance(grid: np.ndarray, period: int, defect_row: int, image: np.ndarray) -> bool:
    detected_period, detected_defect = _find_period_and_defect(image)
    if detected_period != period or detected_defect != defect_row:
        return False

    marker_row = _marker_row(grid[:, 0])
    expected = np.roll(image, shift=(marker_row - defect_row) % int(image.shape[0]), axis=0)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    out_period, out_defect = _find_period_and_defect(actual)
    return out_period == period and out_defect == marker_row


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    for _ in range(MAX_ATTEMPTS):
        grid, period, defect_row, image = _gen_attempt(_GEN_COUNTER)
        if _is_valid_instance(grid, period, defect_row, image):
            _GEN_COUNTER += 1
            return grid
    raise RuntimeError("failed to generate a valid rolling-defect input")
