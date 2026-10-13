from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import get_bg


def _runs(rows: np.ndarray) -> list[tuple[int, int]]:
    rows = np.array(sorted(int(r) for r in rows), dtype=int)
    if rows.size == 0:
        return []
    out: list[tuple[int, int]] = []
    start = prev = int(rows[0])
    for value in rows[1:]:
        value = int(value)
        if value != prev + 1:
            out.append((start, prev))
            start = value
        prev = value
    out.append((start, prev))
    return out


def _find_stain(grid: np.ndarray, bg: int) -> tuple[str, int, int, int]:
    H, W = grid.shape
    candidates: list[tuple[str, int, int, int]] = []
    for side, col in (("left", 0), ("right", W - 1)):
        edge = grid[:, col]
        rows = np.where(edge != bg)[0]
        if rows.size == 0:
            continue
        colors = {int(edge[r]) for r in rows}
        assert len(colors) == 1, "edge stain must be monochrome"
        color = next(iter(colors))
        runs = _runs(rows)
        assert len(runs) == 1, "edge stain must be one contiguous run"
        r0, r1 = runs[0]
        coords = np.argwhere(grid == color)
        assert np.all(coords[:, 1] == col), "slider color may only appear in the stain"
        assert int(coords.shape[0]) == r1 - r0 + 1, "stain color must appear exactly once"
        candidates.append((side, color, r0, r1))
    assert len(candidates) == 1, "expected exactly one boundary stain"
    return candidates[0]


def _find_bars(
    grid: np.ndarray, bg: int, slider_color: int
) -> tuple[list[tuple[int, int]], int, int]:
    H, _W = grid.shape
    bar_colors = sorted(set(map(int, np.unique(grid))) - {bg, slider_color})
    assert len(bar_colors) >= 3, "expected multiple split bars"

    bars: list[tuple[int, int]] = []
    gap_tops: list[int] = []
    gap_bottoms: list[int] = []
    for color in bar_colors:
        coords = np.argwhere(grid == color)
        cols = sorted({int(c) for _r, c in coords})
        assert len(cols) == 1, "each bar must stay in one column"
        col = cols[0]
        runs = _runs(coords[:, 0])
        assert len(runs) == 2, "each bar must have exactly two fragments"
        (top0, top1), (bot0, bot1) = runs
        assert top0 == 0 and bot1 == H - 1, "split bars must be shifted outward to the canvas boundary"
        gap_tops.append(top1 + 1)
        gap_bottoms.append(bot0 - 1)
        bars.append((color, col))

    assert len(set(gap_tops)) == 1 and len(set(gap_bottoms)) == 1, "all bars must share one corridor"
    gap_top = gap_tops[0]
    gap_bottom = gap_bottoms[0]
    assert gap_bottom - gap_top + 1 >= 4, "corridor must allow a nontrivial square"
    return sorted(bars, key=lambda item: item[1]), gap_top, gap_bottom


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the missing edge-launched square by undoing the shared split residue on vertical bars.

    Input structure:
    - The background is dominant.
    - Several different-colored vertical bars each occupy one column and are broken in the
      same row interval, leaving an upper fragment shifted to the top edge and a lower fragment
      shifted to the bottom edge.
    - Exactly one left or right boundary column contains a contiguous vertical paint stain in a
      color not used by the bars; the square itself is absent.

    Rule:
    - Use the common gap between the bar fragments to recover the horizontal travel corridor.
    - Use the boundary stain to identify the square's color, side, and side length, and verify
      that its rows are centered inside that corridor.
    - Move every bar fragment one cell back inward and fill the corridor rows so each bar becomes
      a continuous vertical bar again.
    - Draw the solid square flush against its stained boundary side, covering the stain.

    Key insight:
    - The broken bars reveal where the square passed, while the lone boundary stain supplies the
      missing square's identity and launch side; both clues together restore the pre-event scene.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"
    bg = int(get_bg(g))
    H, W = g.shape

    side, slider_color, stain_r0, stain_r1 = _find_stain(g, bg)
    size = stain_r1 - stain_r0 + 1
    assert size >= 2, "square side must be at least two"

    bars, gap_top, gap_bottom = _find_bars(g, bg, slider_color)
    assert stain_r0 == gap_top + 1 and stain_r1 == gap_bottom - 1, "stain must be centered in the corridor"

    out = np.full_like(g, bg)
    for color, col in bars:
        out[1 : H - 1, col] = int(color)

    if side == "left":
        out[stain_r0 : stain_r1 + 1, :size] = slider_color
    else:
        out[stain_r0 : stain_r1 + 1, W - size : W] = slider_color
    return out


GRID_HEIGHT_RANGE = (10, 22)
GRID_WIDTH_RANGE = (14, 26)
BAR_COUNT_RANGE = (3, 6)
SQUARE_SIZE_RANGE = (2, 5)
PALETTE = tuple(range(10))
MAX_ATTEMPTS = 400


def _choose_spaced_columns(rng: np.random.Generator, candidates: list[int], count: int) -> list[int] | None:
    if count <= 0:
        return []
    order = candidates.copy()
    rng.shuffle(order)
    chosen: list[int] = []
    for col in order:
        if all(abs(col - other) >= 2 for other in chosen):
            chosen.append(int(col))
            if len(chosen) == count:
                return sorted(chosen)
    return None


def _build_output(
    H: int,
    W: int,
    bg: int,
    bars: list[tuple[int, int]],
    side: str,
    slider_color: int,
    square_r0: int,
    size: int,
) -> np.ndarray:
    out = np.full((H, W), bg, dtype=int)
    for color, col in bars:
        out[1 : H - 1, col] = int(color)
    if side == "left":
        out[square_r0 : square_r0 + size, :size] = int(slider_color)
    else:
        out[square_r0 : square_r0 + size, W - size : W] = int(slider_color)
    return out


def _forward_residue(
    H: int,
    W: int,
    bg: int,
    bars: list[tuple[int, int]],
    side: str,
    slider_color: int,
    square_r0: int,
    size: int,
) -> np.ndarray:
    inp = np.full((H, W), bg, dtype=int)
    top_end = square_r0 - 2
    bottom_start = square_r0 + size + 1
    for color, col in bars:
        inp[: top_end + 1, col] = int(color)
        inp[bottom_start:H, col] = int(color)
    stain_col = 0 if side == "left" else W - 1
    inp[square_r0 : square_r0 + size, stain_col] = int(slider_color)
    return inp


def _validate_scene(
    inp: np.ndarray,
    out: np.ndarray,
    side: str,
    slider_color: int,
    square_r0: int,
    size: int,
    bars: list[tuple[int, int]],
) -> bool:
    bg = int(get_bg(inp))
    try:
        parsed_side, parsed_color, stain_r0, stain_r1 = _find_stain(inp, bg)
        parsed_bars, gap_top, gap_bottom = _find_bars(inp, bg, parsed_color)
    except AssertionError:
        return False

    if parsed_side != side or parsed_color != slider_color:
        return False
    if stain_r0 != square_r0 or stain_r1 != square_r0 + size - 1:
        return False
    if gap_top != square_r0 - 1 or gap_bottom != square_r0 + size:
        return False
    if parsed_bars != sorted(bars, key=lambda item: item[1]):
        return False
    return np.array_equal(main(inp), out)


def _attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1] + 1))
    W = int(rng.integers(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1] + 1))

    max_size = min(SQUARE_SIZE_RANGE[1], H - 4, W - 6)
    if max_size < SQUARE_SIZE_RANGE[0]:
        return None
    size = int(rng.integers(SQUARE_SIZE_RANGE[0], max_size + 1))
    square_r0 = int(rng.integers(2, H - size - 1))
    side = str(rng.choice(["left", "right"]))

    if side == "left":
        candidate_cols = list(range(size + 1, W - 1))
    else:
        candidate_cols = list(range(1, W - size - 1))
    max_bars = min(BAR_COUNT_RANGE[1], (len(candidate_cols) + 1) // 2)
    if max_bars < BAR_COUNT_RANGE[0]:
        return None
    bar_count = int(rng.integers(BAR_COUNT_RANGE[0], max_bars + 1))
    cols = _choose_spaced_columns(rng, candidate_cols, bar_count)
    if cols is None:
        return None

    colors = list(PALETTE)
    rng.shuffle(colors)
    bg = int(colors[0])
    slider_color = int(colors[1])
    bar_colors = [int(c) for c in colors[2 : 2 + bar_count]]
    bars = list(zip(bar_colors, cols))

    out = _build_output(H, W, bg, bars, side, slider_color, square_r0, size)
    inp = _forward_residue(H, W, bg, bars, side, slider_color, square_r0, size)

    if not _validate_scene(inp, out, side, slider_color, square_r0, size, bars):
        return None
    if np.sum(inp == bg) <= (inp.size // 2):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid disappearing-pusher input")
