from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, colors_present, find_components, get_bg, paste, random_free_location
import random


def _is_solid_vertical_bar(component: dict) -> bool:
    return bbox_w(component["bbox"]) == 1 and int(component["area"]) == bbox_h(component["bbox"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Advance the marker from one half of a vertical bar to the same half of the next longer bar.

    Input structure
    - The grid has one dominant background color and exactly two foreground colors.
    - One foreground color forms many vertical one-cell-wide bars; the other appears as a single vertical marker segment.
    - The marker segment occupies exactly the top or bottom half of its containing vertical bar.
    Rule
    - Recolor the current marker segment back to the bar color.
    - Find the unique all-bar-color vertical bar whose height is two pixels greater than the marked source bar.
    - Recolor the same half of that next longer bar to the marker color.
    Key insight
    - Each bar's total height encodes an ordered size, and the marker advances from a 2k bar to the corresponding 2(k+1) bar while preserving whether it marks the top or bottom half.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, f"expected exactly two foreground colors, got {colors}"

    mono_component_counts = {
        int(color): len(
            find_components(
                g,
                bg=bg,
                colors=[int(color)],
                connectivity=4,
                monochromatic=True,
            )
        )
        for color in colors
    }
    marker_colors = [color for color, count in mono_component_counts.items() if count == 1]
    assert len(marker_colors) == 1, f"expected one single-component marker color, got {mono_component_counts}"
    marker_color = int(marker_colors[0])

    bar_colors = [int(color) for color in colors if int(color) != marker_color]
    assert len(bar_colors) == 1
    bar_color = int(bar_colors[0])

    marker_components = find_components(
        g,
        bg=bg,
        colors=[marker_color],
        connectivity=4,
        monochromatic=True,
    )
    assert len(marker_components) == 1
    marker = marker_components[0]
    marker_area = int(marker["area"])
    mr0, mc0, mr1, _mc1 = marker["bbox"]
    assert _is_solid_vertical_bar(marker)

    bars = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    source_bars = [
        bar
        for bar in bars
        if np.any(g[bar["coords"][:, 0], bar["coords"][:, 1]] == marker_color)
    ]
    assert len(source_bars) == 1, f"expected one bar containing the marker, got {len(source_bars)}"
    source = source_bars[0]
    sr0, sc0, sr1, sc1 = source["bbox"]
    source_h = bbox_h(source["bbox"])
    assert _is_solid_vertical_bar(source)
    assert source_h == 2 * marker_area
    assert mc0 == sc0 == sc1

    if mr0 == sr0 and mr1 == sr0 + marker_area - 1:
        marked_end = "top"
    elif mr1 == sr1 and mr0 == sr1 - marker_area + 1:
        marked_end = "bottom"
    else:
        raise AssertionError(f"marker is not exactly one end-half: marker={marker['bbox']} source={source['bbox']}")

    target_h = source_h + 2
    target_bars = []
    for bar in bars:
        values = g[bar["coords"][:, 0], bar["coords"][:, 1]]
        if set(map(int, values)) != {bar_color}:
            continue
        if not _is_solid_vertical_bar(bar):
            continue
        if bbox_h(bar["bbox"]) == target_h:
            target_bars.append(bar)

    assert len(target_bars) == 1, f"expected one next-longer target bar, got {[bar['bbox'] for bar in target_bars]}"
    target = target_bars[0]
    tr0, tc0, tr1, _tc1 = target["bbox"]
    target_half = target_h // 2

    out = g.copy()
    out[g == marker_color] = bar_color
    if marked_end == "top":
        out[tr0 : tr0 + target_half, tc0] = marker_color
    else:
        out[tr1 - target_half + 1 : tr1 + 1, tc0] = marker_color

    return out

# --------------- generator (att2, o5x) ---------------

# Fixed palette observed in all examples.
BG = 7
BAR_COLOR = 8
MARKER_COLOR = 9
# Grid geometry and vertical bar vocabulary.
GRID_SHAPE = (16, 16)
PROGRESSION_HEIGHTS = (2, 4, 6, 8, 10)
SOURCE_HEIGHTS = PROGRESSION_HEIGHTS[:-1]
DISTRACTOR_HEIGHT = 2
HEIGHT_STEP = 2
# Placement constraints and structural sampling.
PLACEMENT_PADDING = 1
PLACEMENT_CONNECTIVITY = 4
DISTRACTOR_COUNT_RANGE = (5, 11)
MARKED_HALVES = ("top", "bottom")
# Rejection-sampling retry budget.
MAX_GENERATION_ATTEMPTS = 300


def _paint_bar_half(grid: np.ndarray, row: int, col: int, height: int, marked_half: str, color: int) -> None:
    """Paint the selected top or bottom half in place."""
    half = height // 2
    if marked_half == "top":
        grid[row : row + half, col] = color
    else:
        grid[row + height - half : row + height, col] = color


def _bar_patch(height: int, *, marked_half: str | None = None) -> np.ndarray:
    patch = np.full((height, 1), BAR_COLOR, dtype=int)
    if marked_half is not None:
        _paint_bar_half(patch, 0, 0, height, marked_half, MARKER_COLOR)
    return patch


def _place_bar(canvas: np.ndarray, patch: np.ndarray) -> tuple[int, int]:
    r, c = random_free_location(
        canvas,
        patch,
        transparent=BG,
        padding=PLACEMENT_PADDING,
        padding_connectivity=PLACEMENT_CONNECTIVITY,
    )
    paste(canvas, patch, r, c)
    return r, c


def _place_progression_bars(canvas: np.ndarray, target_h: int, marked_half: str) -> dict[int, tuple[int, int]]:
    """Place one bar for each progression height and return height to top-left."""
    bar_tops: dict[int, tuple[int, int]] = {}
    for height in random.sample(PROGRESSION_HEIGHTS, len(PROGRESSION_HEIGHTS)):
        marker = marked_half if height == target_h else None
        bar_tops[height] = _place_bar(canvas, _bar_patch(height, marked_half=marker))
    return bar_tops


def _derive_input(out: np.ndarray, bar_tops: dict[int, tuple[int, int]], source_h: int, target_h: int, marked_half: str) -> np.ndarray:
    """Move the output marker back from target to source."""
    inp = out.copy()
    tr, tc = bar_tops[target_h]
    _paint_bar_half(inp, tr, tc, target_h, marked_half, BAR_COLOR)

    sr, sc = bar_tops[source_h]
    _paint_bar_half(inp, sr, sc, source_h, marked_half, MARKER_COLOR)
    return inp


def _gen_attempt() -> np.ndarray | None:
    source_h = random.choice(SOURCE_HEIGHTS)
    target_h = source_h + HEIGHT_STEP
    marked_half = random.choice(MARKED_HALVES)
    out = np.full(GRID_SHAPE, BG, dtype=int)

    try:
        bar_tops = _place_progression_bars(out, target_h, marked_half)
        for _ in range(random.randint(*DISTRACTOR_COUNT_RANGE)):
            _place_bar(out, _bar_patch(DISTRACTOR_HEIGHT))
    except ValueError:
        return None

    inp = _derive_input(out, bar_tops, source_h, target_h, marked_half)
    try:
        if np.array_equal(main(inp), out):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 48634b99 input")
