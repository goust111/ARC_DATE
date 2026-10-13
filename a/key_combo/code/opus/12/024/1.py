from __future__ import annotations

from collections import deque
import random

import numpy as np


BLUE = 1
GREEN = 3
YELLOW = 4
BLACK = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_HEIGHTS = (15, 17, 19)
GRID_WIDTHS = (13, 15, 17)
MAX_GENERATION_ATTEMPTS = 500
CONFIGS = (
    (1, 1),  # 3 cavities: 1 stable band + 1 leaking band
    (1, 2),  # 4 cavities
    (2, 1),  # 5 cavities
)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as (N, 2) coord arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _reachable_black(grid: np.ndarray) -> np.ndarray:
    """Mark black cells reachable from the border through 4-connected black paths."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == BLACK and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == BLACK and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        rr, cc = q.popleft()
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BLACK and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _original_cavities(grid: np.ndarray) -> list[np.ndarray]:
    """Return black components that are not border-reachable in the original grid."""
    h, w = grid.shape
    cavities: list[np.ndarray] = []
    for comp in _components(grid == BLACK):
        touches_border = np.any(
            (comp[:, 0] == 0) | (comp[:, 0] == h - 1) | (comp[:, 1] == 0) | (comp[:, 1] == w - 1)
        )
        if not touches_border:
            cavities.append(comp)
    return cavities


def _eroded_foreground(grid: np.ndarray) -> np.ndarray:
    """Keep only blue pixels whose left-right mirror is also blue."""
    fg = grid == BLUE
    symmetric = fg & fg[:, ::-1]
    out = np.zeros_like(grid)
    out[symmetric] = BLUE
    return out


def _analyze(grid: np.ndarray) -> dict[str, object]:
    """Compute the symmetry erosion and cavity-state comparison used by both solver and generator."""
    g = np.asarray(grid, dtype=int)
    eroded = _eroded_foreground(g)
    cavities = _original_cavities(g)
    reachable_after = _reachable_black(eroded)
    opened: list[bool] = []
    for cavity in cavities:
        opened.append(bool(np.any(reachable_after[cavity[:, 0], cavity[:, 1]])))

    fg = g == BLUE
    symmetric = fg & fg[:, ::-1]
    return {
        "eroded": eroded,
        "cavities": cavities,
        "opened": opened,
        "opened_count": sum(opened),
        "closed_count": len(opened) - sum(opened),
        "fg_count": int(np.count_nonzero(fg)),
        "sym_count": int(np.count_nonzero(symmetric)),
        "asym_count": int(np.count_nonzero(fg & ~symmetric)),
        "fg_components": len(_components(fg)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase vertically asymmetric foreground pixels, then color each original hole by whether that erosion opens it.

    Input structure:
    - The grid uses black background and blue foreground pixels.
    - The blue pixels form one connected shape that is almost, but not perfectly, mirror-symmetric left-to-right.
    - The shape encloses several black cavities, while other black cells remain connected to the border.

    Rule:
    - Reflect the foreground across the grid's vertical centerline and keep only blue pixels whose mirror is also blue.
    - Treat the removed blue pixels as erased to black, then flood-fill black cells reachable from the border in this eroded grid.
    - For every cavity that was enclosed in the original grid, fill it yellow if it is still unreachable after erosion, or green if erosion makes any part of it border-reachable.

    Key insight:
    - The output color of a cavity is determined by a comparison between topology before and after symmetry-based erosion, not by the cavity's shape alone.
    """
    analysis = _analyze(np.asarray(grid, dtype=int))
    out = np.array(analysis["eroded"], copy=True)
    for cavity, opened in zip(analysis["cavities"], analysis["opened"]):
        out[cavity[:, 0], cavity[:, 1]] = GREEN if opened else YELLOW
    return out


def _draw_double_room(
    grid: np.ndarray,
    top: int,
    height: int,
    half_width: int,
    axis: int,
    band_type: str,
    plug_side: str | None,
    gap_offset: int | None,
) -> None:
    """Draw one horizontal band containing a left room and a right room sharing the center wall."""
    left = axis - half_width + 1
    right = axis + half_width - 1
    bottom = top + height - 1

    grid[top, left : right + 1] = BLUE
    grid[bottom, left : right + 1] = BLUE
    grid[top : bottom + 1, left] = BLUE
    grid[top : bottom + 1, axis] = BLUE
    grid[top : bottom + 1, right] = BLUE

    if band_type == "gap":
        assert gap_offset is not None and plug_side in {"left", "right"}
        gap_row = top + gap_offset
        grid[gap_row, left] = BLACK
        grid[gap_row, right] = BLACK
        if plug_side == "left":
            grid[gap_row, left] = BLUE
        else:
            grid[gap_row, right] = BLUE


def _build_candidate(rng: random.Random) -> tuple[np.ndarray, int, int] | None:
    n_closed, n_gap = rng.choice(CONFIGS)
    band_types = ["closed"] * n_closed + ["gap"] * n_gap
    rng.shuffle(band_types)
    n_bands = len(band_types)

    heights: list[int] = []
    half_widths: list[int] = []
    plug_sides: list[str | None] = []
    gap_offsets: list[int | None] = []
    for band_type in band_types:
        if n_bands == 3:
            height = rng.choice((3, 4))
            half_width = rng.choice((3, 4))
        else:
            height = rng.choice((3, 4, 5))
            half_width = rng.choice((3, 4, 5))
        heights.append(height)
        half_widths.append(half_width)
        if band_type == "gap":
            plug_sides.append(rng.choice(("left", "right")))
            gap_offsets.append(rng.randint(1, height - 2))
        else:
            plug_sides.append(None)
            gap_offsets.append(None)

    separators = [rng.choice((1, 2)) for _ in range(n_bands - 1)]
    ext_top = rng.choice((0, 1, 2))
    ext_bottom = rng.choice((0, 1, 2))
    total_height = ext_top + sum(heights) + sum(separators) + ext_bottom
    max_half_width = max(half_widths)

    viable_sizes = [
        (h, w)
        for h in GRID_HEIGHTS
        for w in GRID_WIDTHS
        if total_height <= h - 2 and (2 * max_half_width - 1) <= w - 4
    ]
    if not viable_sizes:
        return None

    grid_h, grid_w = rng.choice(viable_sizes)
    axis = grid_w // 2
    grid = np.zeros((grid_h, grid_w), dtype=int)

    top_min = 1
    top_max = grid_h - total_height - 1
    if top_max < top_min:
        return None
    y = rng.randint(top_min, top_max) + ext_top
    first_axis_row = y - ext_top

    band_rows: list[tuple[int, int]] = []
    for idx, band_type in enumerate(band_types):
        _draw_double_room(
            grid,
            y,
            heights[idx],
            half_widths[idx],
            axis,
            band_type,
            plug_sides[idx],
            gap_offsets[idx],
        )
        band_rows.append((y, y + heights[idx] - 1))
        y += heights[idx]
        if idx < n_bands - 1:
            next_y = y + separators[idx]
            grid[y:next_y, axis] = BLUE
            y = next_y

    last_axis_row = band_rows[-1][1] + ext_bottom
    grid[first_axis_row : last_axis_row + 1, axis] = BLUE
    return grid, n_closed, n_gap


def _generator_checks(grid: np.ndarray, expected_closed_bands: int, expected_gap_bands: int) -> bool:
    analysis = _analyze(grid)
    expected_cavities = 2 * expected_closed_bands + expected_gap_bands

    if analysis["fg_components"] != 1:
        return False
    if not (40 <= int(analysis["fg_count"]) <= 60):
        return False
    if len(analysis["cavities"]) != expected_cavities:
        return False
    if int(analysis["opened_count"]) != expected_gap_bands:
        return False
    if int(analysis["closed_count"]) != 2 * expected_closed_bands:
        return False
    if int(analysis["asym_count"]) != expected_gap_bands:
        return False
    if int(analysis["sym_count"]) <= int(analysis["asym_count"]):
        return False
    if any(len(cavity) == 0 for cavity in analysis["cavities"]):
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.any(out == GREEN) or not np.any(out == YELLOW):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_candidate(rng)
        if built is None:
            continue
        grid, expected_closed_bands, expected_gap_bands = built
        if _generator_checks(grid, expected_closed_bands, expected_gap_bands):
            return grid
    raise RuntimeError("generate_input: failed to construct a valid symmetric-erosion cavity puzzle")
