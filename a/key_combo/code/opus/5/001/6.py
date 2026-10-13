from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_UTILS = SourceFileLoader(
    "_arc_utils_task6",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _UTILS.Color
colors_present = _UTILS.colors_present


BG = int(Color.BLACK)
BLUE = int(Color.BLUE)
MARKER_COLORS = [int(c) for c in colors_present(np.arange(10), exclude={BG, BLUE})]
GEN_ATTEMPTS = 400


def _candidate_axes2(width: int) -> list[int]:
    return [2 * c + 1 for c in range(width - 1)]


def _mirror_col(axis2: int, col: int) -> int:
    return int(axis2 - col)


def _blue_set(grid: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(grid == BLUE)}


def _axis_score(blue: set[tuple[int, int]], axis2: int) -> int:
    return sum((r, _mirror_col(axis2, c)) in blue for r, c in blue)


def _best_axis2(blue: set[tuple[int, int]], width: int) -> int:
    scores = {axis2: _axis_score(blue, axis2) for axis2 in _candidate_axes2(width)}
    best_score = max(scores.values())
    winners = [axis2 for axis2, score in scores.items() if score == best_score]
    assert len(winners) == 1, "expected a unique best half-integer symmetry axis"
    return int(winners[0])


def _clean_blue(blue: set[tuple[int, int]], axis2: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r, c in blue
        if (r, _mirror_col(axis2, c)) in blue
    }


def _top_markers(grid: np.ndarray) -> list[tuple[int, int]]:
    markers = []
    for c in range(grid.shape[1]):
        color = int(grid[0, c])
        if color not in (BG, BLUE):
            markers.append((c, color))
    return markers


def _highest_blue_rows(blue: set[tuple[int, int]], width: int) -> dict[int, int]:
    highest: dict[int, int] = {}
    for r, c in blue:
        highest[c] = min(r, highest.get(c, r))
    return highest


def _apply_rays(grid: np.ndarray, clean_blue: set[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    raw_blue = _blue_set(out)
    for r, c in raw_blue - clean_blue:
        out[r, c] = BG

    for c, color in _top_markers(out):
        for r in range(1, out.shape[0]):
            if (r, c) in clean_blue:
                break
            if int(out[r, c]) == BG:
                out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean a noisy vertically symmetric blue pattern and use the surviving blue cells as blockers for top-down color rays.

    Input structure:
    - Black is the background and blue pixels form one lower-half pattern that is almost symmetric about a unique half-integer vertical axis.
    - A few blue noise pixels lack mirrored blue partners on that best axis.
    - The top row contains 3 to 5 single-pixel non-black, non-blue markers in distinct columns.

    Rule:
    - Evaluate every half-integer vertical axis between neighboring columns and pick the unique one that preserves the most mirrored blue pairings.
    - Remove every blue pixel whose mirrored position across that axis is not also blue.
    - Keep the surviving blue shape and all top markers, then from row 1 downward cast a same-color vertical ray below each marker.
    - Each ray paints only black cells and stops immediately before the first surviving blue blocker in its column, or at the bottom edge if no blocker exists.

    Key insight:
    - The symmetry step does not produce the final answer directly; it changes which blue pixels remain to block the later rays, so incorrect denoising yields incorrect ray lengths.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    blue = _blue_set(g)
    assert blue, "expected blue foreground pixels"
    axis2 = _best_axis2(blue, g.shape[1])
    clean_blue = _clean_blue(blue, axis2)
    assert clean_blue, "cleaning removed the whole pattern"
    return _apply_rays(g, clean_blue)


def _sample_extents(rng: np.random.Generator, rows: int, max_extent: int) -> list[int] | None:
    for _ in range(200):
        extents = [int(rng.integers(1, max_extent + 1))]
        for _ in range(rows - 1):
            delta = int(rng.choice([-1, 0, 1]))
            nxt = min(max_extent, max(1, extents[-1] + delta))
            extents.append(int(nxt))
        if len(set(extents)) < 2:
            continue
        total = sum(extents)
        if 9 <= total <= 15 and max(extents) >= 2:
            return extents
    return None


def _build_clean_shape(
    rng: np.random.Generator, height: int, width: int
) -> tuple[set[tuple[int, int]], int] | None:
    for _ in range(200):
        axis_col = int(rng.integers(4, width - 5))
        axis2 = 2 * axis_col + 1
        max_extent = min(axis_col, width - axis_col - 2, 4)
        if max_extent < 2:
            continue
        rows = int(rng.integers(5, 8))
        start_row_max = height - rows - 2
        if start_row_max < height // 2:
            continue
        start_row = int(rng.integers(height // 2, start_row_max + 1))
        extents = _sample_extents(rng, rows, max_extent)
        if extents is None:
            continue

        clean_blue: set[tuple[int, int]] = set()
        for dr, extent in enumerate(extents):
            r = start_row + dr
            for c in range(axis_col - extent + 1, axis_col + 1):
                clean_blue.add((r, c))
                clean_blue.add((r, _mirror_col(axis2, c)))

        if 18 <= len(clean_blue) <= 30:
            return clean_blue, axis2
    return None


def _pick_noise_positions(
    rng: np.random.Generator,
    clean_blue: set[tuple[int, int]],
    axis2: int,
    width: int,
) -> tuple[list[tuple[int, int]], list[int], list[int]] | None:
    highest = _highest_blue_rows(clean_blue, width)
    eligible = [c for c, r in highest.items() if r >= 3]
    if len(eligible) < 3:
        return None

    rng.shuffle(eligible)
    affected_cols = [int(eligible[0])]
    if len(eligible) >= 5 and rng.random() < 0.5:
        affected_cols.append(int(eligible[1]))

    noise_total = int(rng.integers(3, 6))
    noise_positions: list[tuple[int, int]] = []
    used = set()

    def try_add(col: int, row_choices: list[int]) -> bool:
        rng.shuffle(row_choices)
        for row in row_choices:
            pos = (int(row), int(col))
            mirror = (int(row), _mirror_col(axis2, col))
            if pos in used or mirror in clean_blue or mirror in used:
                continue
            noise_positions.append(pos)
            used.add(pos)
            return True
        return False

    for col in affected_cols:
        if not try_add(col, list(range(1, highest[col]))):
            return None

    remaining_cols = [c for c in eligible if c not in affected_cols]
    rng.shuffle(remaining_cols)
    for col in remaining_cols:
        if len(noise_positions) >= noise_total:
            break
        try_add(int(col), list(range(1, highest[col])))
    if len(noise_positions) < noise_total:
        return None

    unaffected_choices = [c for c in eligible if c not in {pos[1] for pos in noise_positions}]
    if not unaffected_choices:
        return None
    unaffected_col = int(rng.choice(unaffected_choices))
    return noise_positions, affected_cols, [unaffected_col]


def _pick_marker_columns(
    rng: np.random.Generator,
    width: int,
    clean_blue: set[tuple[int, int]],
    affected_cols: list[int],
    unaffected_cols: list[int],
) -> list[int]:
    clean_cols = sorted({c for _r, c in clean_blue})
    used = set(affected_cols + unaffected_cols)
    markers = list(affected_cols + unaffected_cols)
    target = int(rng.integers(3, 6))

    no_blue_cols = [c for c in range(width) if c not in clean_cols and c not in used]
    rng.shuffle(no_blue_cols)
    clean_extra = [c for c in clean_cols if c not in used]
    rng.shuffle(clean_extra)

    pools = [no_blue_cols, clean_extra, [c for c in range(width) if c not in used]]
    for pool in pools:
        for col in pool:
            if len(markers) >= target:
                break
            if col in used:
                continue
            markers.append(int(col))
            used.add(int(col))
        if len(markers) >= target:
            break
    return markers


def _build_grid(
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, object]] | None:
    height = int(rng.integers(13, 19))
    width = int(rng.integers(13, 19))
    built = _build_clean_shape(rng, height, width)
    if built is None:
        return None
    clean_blue, axis2 = built

    noise_info = _pick_noise_positions(rng, clean_blue, axis2, width)
    if noise_info is None:
        return None
    noise_positions, affected_cols, unaffected_cols = noise_info
    full_blue = set(clean_blue) | set(noise_positions)

    if not 21 <= len(full_blue) <= 35:
        return None

    marker_cols = _pick_marker_columns(rng, width, clean_blue, affected_cols, unaffected_cols)
    marker_colors = list(rng.choice(MARKER_COLORS, size=len(marker_cols), replace=False))

    grid = np.zeros((height, width), dtype=int)
    for r, c in full_blue:
        grid[r, c] = BLUE
    for col, color in zip(marker_cols, marker_colors):
        grid[0, int(col)] = int(color)

    meta = {
        "axis2": axis2,
        "clean_blue": clean_blue,
        "noise_positions": noise_positions,
        "affected_cols": affected_cols,
        "unaffected_cols": unaffected_cols,
        "marker_cols": marker_cols,
    }
    return grid, meta


def _validate_example(grid: np.ndarray) -> bool:
    blue = _blue_set(grid)
    if not blue:
        return False

    markers = _top_markers(grid)
    if not (3 <= len(markers) <= 5):
        return False
    if len({c for c, _color in markers}) != len(markers):
        return False
    if len({color for _c, color in markers}) != len(markers):
        return False

    try:
        axis2 = _best_axis2(blue, grid.shape[1])
    except AssertionError:
        return False
    clean_blue = _clean_blue(blue, axis2)
    removed = blue - clean_blue
    if not (3 <= len(removed) <= 5):
        return False
    if not clean_blue:
        return False
    if any((r, _mirror_col(axis2, c)) not in clean_blue for r, c in clean_blue):
        return False

    before = _highest_blue_rows(blue, grid.shape[1])
    after = _highest_blue_rows(clean_blue, grid.shape[1])
    affected = []
    unaffected = []
    for col, _color in markers:
        if before.get(col) != after.get(col):
            affected.append(col)
        else:
            unaffected.append(col)
    if not affected or not unaffected:
        return False
    if not any(any(c == col for _r, c in removed) for col in affected):
        return False

    out = main(grid)
    no_clean_out = _apply_rays(grid, blue)
    clean_only = np.array(grid, copy=True)
    for r, c in removed:
        clean_only[r, c] = BG
    if np.array_equal(out, no_clean_out):
        return False
    if np.array_equal(out, clean_only):
        return False

    for col in unaffected:
        ray_color = int(grid[0, col])
        top_after = after.get(col, grid.shape[0])
        expected_len = max(0, top_after - 1)
        painted = sum(int(out[r, col]) == ray_color for r in range(1, grid.shape[0]))
        if painted != expected_len:
            return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        built = _build_grid(rng)
        if built is None:
            continue
        grid, _meta = built
        if _validate_example(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
