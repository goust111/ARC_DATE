from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _background(grid: np.ndarray) -> int:
    """Return the unique most frequent color."""
    counts = Counter(map(int, grid.ravel()))
    largest = max(counts.values())
    winners = [color for color, count in counts.items() if count == largest]
    assert len(winners) == 1, "expected a unique dominant background"
    return winners[0]


def _runs(values: np.ndarray) -> list[tuple[int, int, int]]:
    """Run-length encode a one-dimensional color sequence as [start, end)."""
    assert values.ndim == 1 and len(values) > 0
    result: list[tuple[int, int, int]] = []
    start = 0
    for end in range(1, len(values) + 1):
        if end == len(values) or int(values[end]) != int(values[start]):
            result.append((start, end, int(values[start])))
            start = end
    return result


def _parse_scene(grid: np.ndarray):
    """Identify markers, bottom-grown terrain, and the side exemplar structurally."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    height, width = g.shape
    bg = _background(g)

    top_cols = np.flatnonzero(g[0] != bg)
    assert len(top_cols) >= 2, "expected several top-edge target markers"
    marker_colors = set(map(int, g[0, top_cols]))
    assert len(marker_colors) == 1, "all edge markers must share one color"
    marker_color = next(iter(marker_colors))

    bottom_colors = set(map(int, g[-1, g[-1] != bg]))
    assert len(bottom_colors) == 1, "expected one bottom-connected terrain color"
    terrain_color = next(iter(bottom_colors))
    assert terrain_color != marker_color
    terrain = g == terrain_color

    # Every terrain column is a solid suffix.  Only marked columns are terrain-bearing,
    # so the marker-to-clearance correspondence is visible and one-to-one.
    terrain_cols = set(map(int, np.flatnonzero(np.any(terrain, axis=0))))
    assert terrain_cols == set(map(int, top_cols))
    terrain_tops: dict[int, int] = {}
    for c in top_cols:
        rows = np.flatnonzero(terrain[:, c])
        assert len(rows) > 0 and np.array_equal(rows, np.arange(rows[0], height))
        terrain_tops[int(c)] = int(rows[0])

    excluded = terrain.copy()
    excluded[0, top_cols] = True
    exemplar_mask = (g != bg) & ~excluded
    coords = np.argwhere(exemplar_mask)
    assert len(coords) > 0, "expected one multicolor exemplar strip"
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    assert c0 == 0 or c1 == width - 1, "exemplar must stand at a side"
    patch_mask = exemplar_mask[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(patch_mask), "exemplar must be a filled rectangle"
    assert int(exemplar_mask.sum()) == patch_mask.size
    exemplar = g[r0 : r1 + 1, c0 : c1 + 1]
    assert exemplar.shape[1] >= 2, "exemplar must be wider than target markers"
    assert np.all(exemplar == exemplar[:, :1]), "each exemplar band must be horizontal"

    bands = _runs(exemplar[:, 0])
    assert len(bands) == 3, "expected two fixed end bands and one internal travel band"
    assert bands[0][2] != bands[1][2] and bands[1][2] != bands[2][2]
    fixed_height = (bands[0][1] - bands[0][0]) + (bands[2][1] - bands[2][0])
    assert all(top > fixed_height for top in terrain_tops.values())
    return bg, top_cols, terrain_tops, (r0, c0, r1, c1), bands


def main(grid: np.ndarray) -> np.ndarray:
    """Stretch the exemplar's internal band to erect a clearance-filling pillar in every marked column.

    Input structure:
    - A dominant background surrounds one multicolor, horizontally banded vertical
      exemplar rectangle standing at a side without touching the top or bottom.
    - Equal-colored single-pixel markers occupy several top-edge columns, and a
      different-colored solid terrain suffix rises from the bottom of each such column.
    - The exemplar has exactly three row runs; its middle run differs from both neighbors.

    Rule:
    - In each marked column, measure clearance in grid coordinates from row zero through
      the row immediately above that column's highest terrain cell.
    - Copy the exemplar's first and last color bands at their original heights, and
      resize only its middle band so the resulting pillar exactly fills that clearance.
    - Preserve the terrain, exemplar, and all other cells; the pillar replaces its marker.

    Key insight:
    - This is run-level elastic sizing, not geometric scaling: column-specific terrain
      fixes the total height while the sole internal run absorbs the entire difference.
    """
    g = np.asarray(grid, dtype=int)
    bg, marker_cols, terrain_tops, _box, bands = _parse_scene(g)
    out = g.copy()

    top_len = bands[0][1] - bands[0][0]
    bottom_len = bands[2][1] - bands[2][0]
    top_color, middle_color, bottom_color = (band[2] for band in bands)
    for c_value in marker_cols:
        c = int(c_value)
        clear_height = terrain_tops[c]
        middle_len = clear_height - top_len - bottom_len
        assert middle_len >= 1
        pillar = np.array(
            [top_color] * top_len
            + [middle_color] * middle_len
            + [bottom_color] * bottom_len,
            dtype=out.dtype,
        )
        assert len(pillar) == clear_height
        assert np.all(g[1:clear_height, c] == bg), "clearance must contain only background"
        out[:clear_height, c] = pillar
    return out


def _nearest_neighbor_resize(sequence: np.ndarray, target: int) -> np.ndarray:
    """A deliberately wrong whole-strip scaling used by generator validation."""
    indices = np.floor(np.arange(target) * len(sequence) / target).astype(int)
    indices = np.minimum(indices, len(sequence) - 1)
    return sequence[indices]


def _validate_generated(
    grid: np.ndarray,
    expected: np.ndarray,
    exemplar_box: tuple[int, int, int, int],
    marker_cols: list[int],
    terrain_tops: dict[int, int],
    band_lengths: tuple[int, int, int],
) -> bool:
    """Check puzzle semantics independently, including common shortcut counterexamples."""
    try:
        solved = main(grid)
        if not np.array_equal(solved, expected) or np.array_equal(solved, grid):
            return False
        bg, parsed_cols, parsed_tops, parsed_box, parsed_bands = _parse_scene(grid)
        if list(map(int, parsed_cols)) != marker_cols or parsed_tops != terrain_tops:
            return False
        if parsed_box != exemplar_box:
            return False

        r0, c0, r1, c1 = exemplar_box
        if not np.array_equal(solved[r0 : r1 + 1, c0 : c1 + 1], grid[r0 : r1 + 1, c0 : c1 + 1]):
            return False
        terrain_mask = np.zeros_like(grid, dtype=bool)
        for c, top in terrain_tops.items():
            terrain_mask[top:, c] = True
        if not np.array_equal(solved[terrain_mask], grid[terrain_mask]):
            return False

        a, original_middle, b = band_lengths
        new_middles = [terrain_tops[c] - a - b for c in marker_cols]
        if not (any(n < original_middle for n in new_middles) and any(n > original_middle for n in new_middles)):
            return False
        if len(set(terrain_tops.values())) != len(marker_cols):
            return False

        exemplar_seq = grid[r0 : r1 + 1, c0]
        differs_from_geometric_scaling = False
        for c in marker_cols:
            top = terrain_tops[c]
            actual = solved[:top, c]
            actual_runs = _runs(actual)
            if len(actual_runs) != 3:
                return False
            if (actual_runs[0][1], actual_runs[2][1] - actual_runs[2][0]) != (a, b):
                return False
            if not np.array_equal(actual, _nearest_neighbor_resize(exemplar_seq, top)):
                differs_from_geometric_scaling = True
        if not differs_from_geometric_scaling:
            return False

        # Outside target clearances nothing may change.  This also ensures distinct
        # inputs cannot collapse to a single canned output because their scene remains.
        allowed = np.zeros_like(grid, dtype=bool)
        for c, top in terrain_tops.items():
            allowed[:top, c] = True
        if np.any((solved != grid) & ~allowed):
            return False
        if int(np.sum(grid == bg)) <= grid.size // 2:
            return False
        if [end - start for start, end, _ in parsed_bands] != list(band_lengths):
            return False
        return True
    except (AssertionError, ValueError, IndexError):
        return False


def _generation_attempt() -> np.ndarray | None:
    bg = 0
    palette = list(range(1, 10))
    random.shuffle(palette)
    marker_color, terrain_color, top_color, middle_color = palette[:4]
    # Half the time equal end colors make the stable caps especially visible; the
    # remaining cases prevent a solver from assuming equality is part of the rule.
    bottom_color = top_color if random.random() < 0.5 else palette[4]

    top_len = random.randint(1, 3)
    middle_len = random.randint(2, 5)
    bottom_len = random.randint(1, 3)
    exemplar_height = top_len + middle_len + bottom_len
    exemplar_width = random.randint(2, 3)

    height = random.randint(max(15, exemplar_height + 6), 25)
    max_new_middle = height - 2 - top_len - bottom_len
    if max_new_middle <= middle_len:
        return None
    possible = list(range(1, max_new_middle + 1))

    width = random.randint(16, 26)
    exemplar_on_left = bool(random.getrandbits(1))
    if exemplar_on_left:
        exemplar_c0 = 0
        candidates = list(range(exemplar_width + 3, width - 1, 2))
    else:
        exemplar_c0 = width - exemplar_width
        candidates = list(range(1, width - exemplar_width - 2, 2))
    max_targets = min(5, len(candidates), len(possible))
    if max_targets < 3:
        return None
    target_count = random.randint(3, max_targets)
    marker_cols = sorted(random.sample(candidates, target_count))

    compressed = random.randint(1, middle_len - 1)
    stretched = random.randint(middle_len + 1, max_new_middle)
    chosen_middles = [compressed, stretched]
    remaining = [n for n in possible if n not in chosen_middles]
    if target_count >= 3:
        # Give the unchanged-height branch a high chance, while retaining broad sizes.
        third = middle_len if random.random() < 0.7 else random.choice(remaining)
        chosen_middles.append(third)
        remaining = [n for n in remaining if n != third]
    if target_count > 3:
        chosen_middles.extend(random.sample(remaining, target_count - 3))
    random.shuffle(chosen_middles)

    grid = np.full((height, width), bg, dtype=int)
    exemplar_r0 = random.randint(2, height - exemplar_height - 2)
    exemplar_r1 = exemplar_r0 + exemplar_height - 1
    exemplar_c1 = exemplar_c0 + exemplar_width - 1
    exemplar_sequence = np.array(
        [top_color] * top_len + [middle_color] * middle_len + [bottom_color] * bottom_len,
        dtype=int,
    )
    grid[exemplar_r0 : exemplar_r1 + 1, exemplar_c0 : exemplar_c1 + 1] = exemplar_sequence[:, None]

    terrain_tops: dict[int, int] = {}
    expected = grid.copy()
    for c, new_middle in zip(marker_cols, chosen_middles):
        terrain_top = top_len + new_middle + bottom_len
        terrain_tops[c] = terrain_top
        grid[0, c] = marker_color
        grid[terrain_top:, c] = terrain_color
        expected[0, c] = marker_color
        expected[terrain_top:, c] = terrain_color
        expected[:terrain_top, c] = np.array(
            [top_color] * top_len + [middle_color] * new_middle + [bottom_color] * bottom_len,
            dtype=int,
        )

    box = (exemplar_r0, exemplar_c0, exemplar_r1, exemplar_c1)
    lengths = (top_len, middle_len, bottom_len)
    if _validate_generated(grid, expected, box, marker_cols, terrain_tops, lengths):
        return grid
    return None


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate elastic-pillar scene."""
    for _ in range(300):
        result = _generation_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid elastic-pillar input")
