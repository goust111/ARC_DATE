from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import colors_present, get_bg


BAR_ORIENTATIONS = ("H", "V")
BACKGROUND = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

GRID_HEIGHT_RANGE = (13, 18)
GRID_WIDTH_RANGE = (13, 18)
BAR_COUNT_OPTIONS = (4, 5)
BAR_LENGTHS = (3, 5, 7, 9)
GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 120


def _is_consecutive(values: np.ndarray) -> bool:
    ordered = np.sort(values)
    return bool(ordered.size <= 1 or np.all(np.diff(ordered) == 1))


def _analyze_bar(coords: np.ndarray) -> dict[str, int | str]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    area = int(coords.shape[0])

    assert height == 1 or width == 1, "each object must be a one-cell-thick bar"
    length = max(height, width)
    assert area == length, "bar pixels must form one solid contiguous segment"
    assert length >= 3 and length % 2 == 1, "bars must have odd length 3-9"

    if height == 1:
        assert np.unique(rows).size == 1 and _is_consecutive(cols)
        center_row = r0
        top_after_rotation = center_row - length // 2
        orientation = "H"
    else:
        assert np.unique(cols).size == 1 and _is_consecutive(rows)
        center_row = (r0 + r1) // 2
        top_after_rotation = center_row
        orientation = "V"

    return {
        "orientation": orientation,
        "length": length,
        "top_before": r0,
        "top_after": int(top_after_rotation),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each colored bar mentally, sort colors by the rotated bars' top rows, and emit a compact stripe stack.

    Input structure:
    - The grid has a dominant black background.
    - Each non-black color forms exactly one contiguous one-cell-thick horizontal or vertical bar.
    - Every bar has odd length, and different bars use different colors.

    Rule:
    - For each colored bar, rotate it 90 degrees around its center while preserving its length and color.
    - Compute the minimum row occupied by that rotated bar.
    - Sort the bar colors from smallest rotated top row to largest, then output a Kx3 grid whose rows are solid 1x3 stripes in that order.

    Key insight:
    - The output order is not determined by the bars' visible top rows in the input; horizontal bars move upward after rotation, while vertical bars rank by their center rows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    ordered: list[tuple[int, int]] = []
    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == int(color))
        assert coords.size > 0, "every foreground color must occupy pixels"
        info = _analyze_bar(coords)
        ordered.append((int(info["top_after"]), int(color)))

    assert ordered, "expected at least one foreground bar"
    top_rows = [top for top, _color in ordered]
    assert len(set(top_rows)) == len(top_rows), "rotated top rows must be unique"

    ordered.sort()
    out = np.full((len(ordered), 3), bg, dtype=int)
    for r, (_top, color) in enumerate(ordered):
        out[r, :] = color
    return out


def _bar_pixels(center_row: int, center_col: int, length: int, orientation: str) -> set[tuple[int, int]]:
    half = length // 2
    if orientation == "H":
        return {(center_row, center_col + dc) for dc in range(-half, half + 1)}
    return {(center_row + dr, center_col) for dr in range(-half, half + 1)}


def _halo(pixels: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    height, width = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    expanded.add((nr, nc))
    return expanded


def _sample_orientations(count: int) -> list[str]:
    while True:
        orientations = [random.choice(BAR_ORIENTATIONS) for _ in range(count)]
        if "H" in orientations and "V" in orientations:
            return orientations


def _sample_lengths(count: int, max_length: int) -> list[int]:
    choices = [length for length in BAR_LENGTHS if length <= max_length]
    weights = [6, 5, 3, 2][: len(choices)]
    while True:
        lengths = random.choices(choices, weights=weights, k=count)
        if len(set(lengths)) >= 2:
            return lengths


def _place_bars(height: int, width: int, specs: list[dict[str, int | str]]) -> list[dict[str, int | str]] | None:
    occupied_halo: set[tuple[int, int]] = set()
    placed: list[dict[str, int | str]] = []

    for spec in sorted(specs, key=lambda item: int(item["length"]), reverse=True):
        length = int(spec["length"])
        half = length // 2
        row_lo, row_hi = half, height - 1 - half
        col_lo, col_hi = half, width - 1 - half
        centers = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
        random.shuffle(centers)

        for center_row, center_col in centers[:PLACEMENT_ATTEMPTS]:
            pixels = _bar_pixels(center_row, center_col, length, str(spec["orientation"]))
            if pixels & occupied_halo:
                continue

            placed_spec = {
                **spec,
                "center_row": center_row,
                "center_col": center_col,
                "pixels": pixels,
                "top_before": center_row if spec["orientation"] == "H" else center_row - half,
                "top_after": center_row - half if spec["orientation"] == "H" else center_row,
            }
            placed.append(placed_spec)
            occupied_halo |= _halo(pixels, (height, width))
            break
        else:
            return None

    return placed


def _is_good_example(bars: list[dict[str, int | str]]) -> bool:
    before = [int(bar["top_before"]) for bar in bars]
    after = [int(bar["top_after"]) for bar in bars]
    if len(set(before)) != len(before) or len(set(after)) != len(after):
        return False

    original_order = [int(bar["color"]) for bar in sorted(bars, key=lambda bar: int(bar["top_before"]))]
    rotated_order = [int(bar["color"]) for bar in sorted(bars, key=lambda bar: int(bar["top_after"]))]
    if original_order == rotated_order:
        return False

    shifts = [int(bar["top_before"]) - int(bar["top_after"]) for bar in bars]
    if not any(shift > 0 for shift in shifts):
        return False
    if not any(shift < 0 for shift in shifts):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    count = random.choice(BAR_COUNT_OPTIONS)
    max_length = min(max(BAR_LENGTHS), height, width)
    if max_length < 3:
        return None

    specs: list[dict[str, int | str]] = []
    for color, orientation, length in zip(
        random.sample(PALETTE, count),
        _sample_orientations(count),
        _sample_lengths(count, max_length),
    ):
        specs.append(
            {
                "color": int(color),
                "orientation": orientation,
                "length": int(length),
            }
        )

    bars = _place_bars(height, width, specs)
    if bars is None or not _is_good_example(bars):
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for bar in bars:
        color = int(bar["color"])
        for r, c in bar["pixels"]:
            grid[r, c] = color

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != (count, 3):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
