from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
MARKER = 7
REGION_COLORS = (1, 2, 3, 4, 6, 8, 9)

MAX_TILE_TRIES = 240
MAX_REGION_TRIES = 240
MAX_GENERATION_TRIES = 320


def _full_separator_indices(grid: np.ndarray, color: int) -> tuple[list[int], list[int]]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]
    return rows, cols


def _intervals(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1, *sorted(separators), length]
    return [
        (bounds[i] + 1, bounds[i + 1])
        for i in range(len(bounds) - 1)
        if bounds[i] + 1 < bounds[i + 1]
    ]


def _majority_color(values: list[int]) -> int:
    return int(Counter(values).most_common(1)[0][0])


def _repair_region(region: np.ndarray) -> np.ndarray:
    markers = np.argwhere(region == MARKER)
    assert markers.shape == (2, 2), "each partition must contain exactly two orange markers"

    (r0, c0), (r1, c1) = markers
    period = abs(int(r0) - int(r1)) + abs(int(c0) - int(c1))
    h, w = map(int, region.shape)
    assert 1 <= period <= min(h, w), "marker Manhattan distance must be a valid square tile size"

    residues: dict[tuple[int, int], list[int]] = {}
    for r in range(h):
        for c in range(w):
            color = int(region[r, c])
            if color in (SEPARATOR, MARKER):
                continue
            residues.setdefault((r % period, c % period), []).append(color)

    tile = np.full((period, period), BACKGROUND, dtype=int)
    for key, values in residues.items():
        tile[key] = _majority_color(values)

    repaired = np.empty_like(region)
    for r in range(h):
        for c in range(w):
            repaired[r, c] = tile[r % period, c % period]

    repaired[markers[:, 0], markers[:, 1]] = BACKGROUND
    return repaired


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each partition's tile period from the orange-marker Manhattan distance, repair the noisy periodic pattern, and erase the markers.

    Input structure:
    - The background is black, and full gray rows and/or columns split the canvas into 3 or 4 rectangular partitions.
    - Inside each partition, one non-gray non-orange foreground color forms a square-periodic pattern over black, but a few cells are corrupted by noise.
    - Each partition also contains exactly two orange marker pixels on top of background cells, and their Manhattan distance encodes that partition's tile size.

    Rule:
    - Keep the gray separator lines fixed and process each partition independently in its own local coordinate system.
    - For one partition, compute D from the two orange markers' Manhattan distance, then group all non-gray non-orange cells by (row mod D, col mod D) and take a majority vote per residue class.
    - Re-tile the whole partition with the recovered D x D unit, replacing noisy cells, and finally erase the two orange markers to black.

    Key insight:
    - The orange pixels are not picture content: their spacing is the only reliable source of the tile period, and that period is required before majority-vote repair becomes well-defined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    out = g.copy()
    sep_rows, sep_cols = _full_separator_indices(g, SEPARATOR)
    row_ranges = _intervals(g.shape[0], sep_rows)
    col_ranges = _intervals(g.shape[1], sep_cols)

    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            out[r0:r1, c0:c1] = _repair_region(g[r0:r1, c0:c1])

    return out


def _axis_period(tile: np.ndarray, axis: int) -> int:
    length = tile.shape[axis]
    for period in range(1, length):
        if axis == 0:
            if np.array_equal(tile[:-period, :], tile[period:, :]):
                return period
        else:
            if np.array_equal(tile[:, :-period], tile[:, period:]):
                return period
    return length


def _render_tile(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    rh = (height + tile.shape[0] - 1) // tile.shape[0]
    rw = (width + tile.shape[1] - 1) // tile.shape[1]
    return np.tile(tile, (rh, rw))[:height, :width].astype(int)


def _residue_counts(height: int, width: int, period: int) -> np.ndarray:
    counts = np.zeros((period, period), dtype=int)
    for r in range(height):
        for c in range(width):
            counts[r % period, c % period] += 1
    return counts


def _region_shapes() -> tuple[list[int], list[int]]:
    layout = random.choice(("1x3", "3x1", "2x2"))
    long_sizes = [random.randint(8, 11) for _ in range(3)]
    short_sizes = [random.randint(5, 7) for _ in range(3)]

    if layout == "1x3":
        row_heights = [random.choice(long_sizes)]
        col_widths = short_sizes
    elif layout == "3x1":
        row_heights = short_sizes
        col_widths = [random.choice(long_sizes)]
    else:
        if random.choice((True, False)):
            row_heights = long_sizes[:2]
            col_widths = short_sizes[:2]
        else:
            row_heights = short_sizes[:2]
            col_widths = long_sizes[:2]
    return row_heights, col_widths


def _period_options(height: int, width: int) -> list[int]:
    options = []
    for period in range(3, min(5, height, width) + 1):
        if period > height + width - 6:
            continue
        if min(height, width) < 2 * period <= max(height, width):
            options.append(period)
    return options


def _choose_period_plan(shapes: list[tuple[int, int]]) -> list[int] | None:
    choices = [_period_options(height, width) for height, width in shapes]
    if any(not opts for opts in choices):
        return None

    for _ in range(80):
        plan = [random.choice(opts) for opts in choices]
        if len(plan) == 1 or len(set(plan)) >= 2:
            return plan
    return [random.choice(opts) for opts in choices]


def _random_tile(period: int, color: int) -> np.ndarray | None:
    area = period * period
    min_fg = max(period, area // 3)
    max_fg = min(area - period, (2 * area) // 3)
    if min_fg >= max_fg:
        min_fg = max(1, min_fg - 1)
        max_fg = min(area - 1, max_fg + 1)

    for _ in range(MAX_TILE_TRIES):
        fg_count = random.randint(min_fg, max_fg)
        indices = set(random.sample(range(area), fg_count))
        tile = np.full((period, period), BACKGROUND, dtype=int)
        for idx in indices:
            tile[idx // period, idx % period] = color

        if _axis_period(tile, 0) != period or _axis_period(tile, 1) != period:
            continue
        if np.count_nonzero(tile == BACKGROUND) < period:
            continue
        if np.count_nonzero(tile == color) < period:
            continue
        if any(np.all(tile[r, :] == tile[r, 0]) for r in range(period)):
            continue
        if any(np.all(tile[:, c] == tile[0, c]) for c in range(period)):
            continue
        return tile
    return None


def _candidate_marker_pairs(
    clean: np.ndarray,
    period: int,
    residue_counts: np.ndarray,
    kind: str,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = clean.shape
    cells = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(clean[r, c]) != BACKGROUND:
                continue
            if int(residue_counts[r % period, c % period]) < 3:
                continue
            cells.append((r, c))

    pairs = []
    for i, a in enumerate(cells):
        ra, ca = a
        residue_a = (ra % period, ca % period)
        for rb, cb in cells[i + 1 :]:
            dr = abs(ra - rb)
            dc = abs(ca - cb)
            if dr + dc != period:
                continue
            if kind == "row" and dr != 0:
                continue
            if kind == "col" and dc != 0:
                continue
            if kind == "diag" and (dr == 0 or dc == 0):
                continue
            residue_b = (rb % period, cb % period)
            if residue_a == residue_b:
                continue
            pairs.append((a, (rb, cb)))
    random.shuffle(pairs)
    return pairs


def _choose_markers(clean: np.ndarray, period: int, preferred_kind: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    residue_counts = _residue_counts(clean.shape[0], clean.shape[1], period)
    order = [preferred_kind, "diag", "row", "col"]
    seen: set[str] = set()
    for kind in order:
        if kind in seen:
            continue
        seen.add(kind)
        pairs = _candidate_marker_pairs(clean, period, residue_counts, kind)
        if pairs:
            return pairs[0]
    return None


def _choose_noise_positions(
    clean: np.ndarray,
    period: int,
    markers: tuple[tuple[int, int], tuple[int, int]],
    color: int,
) -> list[tuple[int, int]] | None:
    h, w = clean.shape
    residue_counts = _residue_counts(h, w, period)
    forbidden = {
        markers[0],
        markers[1],
    }
    forbidden_residues = {
        (markers[0][0] % period, markers[0][1] % period),
        (markers[1][0] % period, markers[1][1] % period),
    }

    bg_candidates = []
    fg_candidates = []
    all_candidates = []
    for r in range(h):
        for c in range(w):
            if (r, c) in forbidden:
                continue
            residue = (r % period, c % period)
            if residue in forbidden_residues:
                continue
            if int(residue_counts[residue]) < 3:
                continue
            all_candidates.append((r, c))
            if int(clean[r, c]) == BACKGROUND:
                bg_candidates.append((r, c))
            elif int(clean[r, c]) == color:
                fg_candidates.append((r, c))

    if not bg_candidates or not fg_candidates:
        return None

    random.shuffle(bg_candidates)
    random.shuffle(fg_candidates)
    used_residues: set[tuple[int, int]] = set()
    picked: list[tuple[int, int]] = []
    for pool in (bg_candidates, fg_candidates):
        for pos in pool:
            residue = (pos[0] % period, pos[1] % period)
            if residue in used_residues:
                continue
            picked.append(pos)
            used_residues.add(residue)
            break

    remaining = [pos for pos in all_candidates if (pos[0] % period, pos[1] % period) not in used_residues]
    random.shuffle(remaining)
    target = random.randint(2, min(4, len(all_candidates)))
    for pos in remaining:
        if len(picked) >= target:
            break
        residue = (pos[0] % period, pos[1] % period)
        if residue in used_residues:
            continue
        picked.append(pos)
        used_residues.add(residue)

    return picked if len(picked) >= 2 else None


def _apply_noise(region: np.ndarray, positions: list[tuple[int, int]], color: int) -> None:
    for r, c in positions:
        true_color = int(region[r, c])
        if true_color == BACKGROUND:
            choices = [v for v in REGION_COLORS if v not in (color,)]
        else:
            choices = [BACKGROUND, *[v for v in REGION_COLORS if v != color]]
        region[r, c] = random.choice(choices)


def _build_region(height: int, width: int, color: int, preferred_kind: str, period: int) -> np.ndarray:
    for _ in range(MAX_REGION_TRIES):
        tile = _random_tile(period, color)
        if tile is None:
            continue

        clean = _render_tile(tile, height, width)
        markers = _choose_markers(clean, period, preferred_kind)
        if markers is None:
            continue

        noise_positions = _choose_noise_positions(clean, period, markers, color)
        if noise_positions is None:
            continue

        region = clean.copy()
        _apply_noise(region, noise_positions, color)
        for r, c in markers:
            region[r, c] = MARKER
        return region

    raise RuntimeError("failed to build a non-degenerate partition")


def _period_preferences(region_count: int) -> list[str]:
    base = ["diag", "row", "col"]
    random.shuffle(base)
    prefs = []
    for i in range(region_count):
        if i < len(base):
            prefs.append(base[i])
        else:
            prefs.append(random.choice(base))
    return prefs


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_TRIES):
        row_heights, col_widths = _region_shapes()
        height = sum(row_heights) + len(row_heights) - 1
        width = sum(col_widths) + len(col_widths) - 1
        if height > 30 or width > 30:
            continue

        shapes = [(rh, cw) for rh in row_heights for cw in col_widths]
        period_plan = _choose_period_plan(shapes)
        if period_plan is None:
            continue

        region_count = len(shapes)
        colors = random.sample(REGION_COLORS, region_count)
        preferences = _period_preferences(region_count)
        grid = np.full((height, width), SEPARATOR, dtype=int)

        region_idx = 0
        r0 = 0
        for rh in row_heights:
            c0 = 0
            for cw in col_widths:
                try:
                    region = _build_region(
                        rh,
                        cw,
                        colors[region_idx],
                        preferences[region_idx],
                        period_plan[region_idx],
                    )
                except RuntimeError:
                    grid = None
                    break
                grid[r0 : r0 + rh, c0 : c0 + cw] = region
                region_idx += 1
                c0 += cw + 1
            if grid is None:
                break
            r0 += rh + 1

        if grid is not None:
            return grid

    raise RuntimeError("generate_input: failed to construct a valid grid")
