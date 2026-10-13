from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np


BACKGROUND = 0
CELL_SIZE = 3
ARC_COLORS = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 800


@dataclass(frozen=True)
class Seed:
    color: int
    period: int
    cell: tuple[int, int]
    offsets: tuple[tuple[int, int], ...]


def _background_color(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _colors_present(g: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = {int(c) for c in np.unique(g)}
    if exclude:
        colors -= {int(c) for c in exclude}
    return sorted(colors)


def _open_runs(length: int, separators: np.ndarray) -> list[tuple[int, int]]:
    sep = {int(x) for x in separators}
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx in range(int(length)):
        if idx in sep:
            if start is not None:
                runs.append((start, idx - 1))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        runs.append((start, int(length) - 1))
    return runs


def _run_index(runs: list[tuple[int, int]], value: int) -> int:
    matches = [idx for idx, (lo, hi) in enumerate(runs) if lo <= int(value) <= hi]
    assert len(matches) == 1, "foreground pixels must lie inside exactly one open cell"
    return matches[0]


def _find_scaffold(g: np.ndarray, bg: int) -> tuple[int, np.ndarray, np.ndarray]:
    candidates: list[tuple[int, np.ndarray, np.ndarray]] = []
    for color in _colors_present(g, exclude={bg}):
        row_lines = np.flatnonzero(np.all(g == int(color), axis=1))
        col_lines = np.flatnonzero(np.all(g == int(color), axis=0))
        if row_lines.size and col_lines.size:
            candidates.append((int(color), row_lines, col_lines))
    assert len(candidates) == 1, "expected one full-line scaffold color"
    return candidates[0]


def _parse_input(g: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]], list[Seed]]:
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30x30"

    bg = _background_color(g)
    scaffold, row_lines, col_lines = _find_scaffold(g, bg)
    row_cells = _open_runs(g.shape[0], row_lines)
    col_cells = _open_runs(g.shape[1], col_lines)
    assert row_cells and col_cells, "scaffold must leave open cells"

    seed_colors = _colors_present(g, exclude={bg, scaffold})
    assert len(seed_colors) == 2, "expected exactly two seed colors"

    seeds: list[Seed] = []
    occupied_cells: set[tuple[int, int]] = set()
    for color in seed_colors:
        coords = np.argwhere(g == int(color))
        assert coords.size > 0, "each seed color must be present"
        row_indices = {_run_index(row_cells, int(r)) for r, _c in coords}
        col_indices = {_run_index(col_cells, int(c)) for _r, c in coords}
        assert len(row_indices) == 1 and len(col_indices) == 1, "each seed must occupy one open cell"
        cell = (row_indices.pop(), col_indices.pop())
        assert cell not in occupied_cells, "the two seeds must be in different cells"
        occupied_cells.add(cell)

        r0 = row_cells[cell[0]][0]
        c0 = col_cells[cell[1]][0]
        offsets = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
        period = len(offsets)
        seeds.append(Seed(int(color), int(period), cell, offsets))

    assert seeds[0].period != seeds[1].period, "seed pixel counts must be different"
    return bg, scaffold, row_cells, col_cells, seeds


def _cell_hits(cell: tuple[int, int], seeds: list[Seed]) -> list[Seed]:
    row, col = cell
    hits: list[Seed] = []
    for seed in seeds:
        sr, sc = seed.cell
        if (row - sr) % seed.period == 0 and (col - sc) % seed.period == 0:
            hits.append(seed)
    return hits


def _stamp(out: np.ndarray, row_cell: tuple[int, int], col_cell: tuple[int, int], seed: Seed) -> None:
    r0, r1 = row_cell
    c0, c1 = col_cell
    for dr, dc in seed.offsets:
        r = r0 + int(dr)
        c = c0 + int(dc)
        if r <= r1 and c <= c1:
            out[r, c] = int(seed.color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp two seed motifs on their own pixel-count lattices, leaving non-source collisions blank.

    Input structure:
    - A single scaffold color forms complete horizontal and vertical separator lines, splitting the canvas into open cells whose outer edge cells may be truncated.
    - Exactly two open cells contain seed motifs in different colors; their pixel counts are different, and every other open cell starts empty.
    - The seed offsets are measured from the upper-left corner of the seed's own open cell.

    Rule:
    - Count each seed's pixels and use that count as the seed's row-and-column period over the open-cell array.
    - For every non-source open cell, stamp the seed motif if exactly one seed lattice reaches that cell, clipping offsets that fall outside truncated edge cells.
    - If both seed lattices reach the same non-source cell, draw nothing there; the two original source cells are always preserved, and scaffold lines never change.

    Key insight:
    - Each motif is both the stamp and the beat: its pixel count defines a self-referential lattice, and common lattice cells are deliberate annihilation gaps.
    """
    g = np.asarray(grid, dtype=int)
    bg, _scaffold, row_cells, col_cells, seeds = _parse_input(g)

    out = g.copy()
    source_cells = {seed.cell for seed in seeds}
    for row_idx, row_cell in enumerate(row_cells):
        for col_idx, col_cell in enumerate(col_cells):
            cell = (row_idx, col_idx)
            if cell in source_cells:
                continue

            r0, r1 = row_cell
            c0, c1 = col_cell
            assert np.all(out[r0 : r1 + 1, c0 : c1 + 1] == bg), "non-source open cells must start empty"
            hits = _cell_hits(cell, seeds)
            assert len(hits) <= 2
            if len(hits) == 1:
                _stamp(out, row_cell, col_cell, hits[0])

    return out


# --------------- generator ---------------


def _lcm(a: int, b: int) -> int:
    return abs(int(a) * int(b)) // math.gcd(int(a), int(b))


def _bands_from_sizes(sizes: list[int]) -> tuple[int, list[tuple[int, int]], list[int]]:
    bands: list[tuple[int, int]] = []
    separators: list[int] = []
    pos = 0
    for idx, size in enumerate(sizes):
        assert int(size) > 0
        bands.append((pos, pos + int(size) - 1))
        pos += int(size)
        if idx != len(sizes) - 1:
            separators.append(pos)
            pos += 1
    return pos, bands, separators


def _side_sizes(cell_count: int, *, prefer_truncated: bool) -> list[int]:
    cell_count = int(cell_count)
    if cell_count == 8:
        last = random.choice([1, 2])
    elif prefer_truncated:
        last = random.choice([1, 2])
    else:
        last = random.choice([1, 2, 3, 3])
    return [CELL_SIZE] * (cell_count - 1) + [last]


def _gen_offsets(count: int) -> tuple[tuple[int, int], ...]:
    count = int(count)
    mandatory = {(0, 0), (CELL_SIZE - 1, CELL_SIZE - 1)}
    assert count >= len(mandatory)
    rest = [(r, c) for r in range(CELL_SIZE) for c in range(CELL_SIZE) if (r, c) not in mandatory]
    random.shuffle(rest)
    coords = set(mandatory)
    coords.update(rest[: count - len(coords)])
    return tuple(sorted(coords))


def _source_collision_count(
    source1: tuple[int, int],
    source2: tuple[int, int],
    period1: int,
    period2: int,
) -> int:
    seeds = [
        Seed(1, period1, source1, ()),
        Seed(2, period2, source2, ()),
    ]
    return sum(1 for cell in (source1, source2) if len(_cell_hits(cell, seeds)) == 2)


def _layout_stats(
    row_sizes: list[int],
    col_sizes: list[int],
    source1: tuple[int, int],
    source2: tuple[int, int],
    period1: int,
    period2: int,
) -> dict[str, int]:
    seeds = [
        Seed(1, period1, source1, ()),
        Seed(2, period2, source2, ()),
    ]
    source_cells = {source1, source2}
    stats = {
        "only1": 0,
        "only2": 0,
        "full_only1": 0,
        "full_only2": 0,
        "clipped_single": 0,
        "collision": 0,
    }
    for rr, row_size in enumerate(row_sizes):
        for cc, col_size in enumerate(col_sizes):
            cell = (rr, cc)
            if cell in source_cells:
                continue
            hits = _cell_hits(cell, seeds)
            if len(hits) == 2:
                stats["collision"] += 1
            elif len(hits) == 1:
                hit_idx = 1 if hits[0].color == 1 else 2
                stats[f"only{hit_idx}"] += 1
                if row_size == CELL_SIZE and col_size == CELL_SIZE:
                    stats[f"full_only{hit_idx}"] += 1
                else:
                    stats["clipped_single"] += 1
    return stats


def _candidate_sources(
    row_sizes: list[int],
    col_sizes: list[int],
    period1: int,
    period2: int,
    want_source_collision: bool,
) -> list[tuple[tuple[int, int], tuple[int, int], dict[str, int]]]:
    full_rows = [idx for idx, size in enumerate(row_sizes) if size == CELL_SIZE]
    full_cols = [idx for idx, size in enumerate(col_sizes) if size == CELL_SIZE]
    full_cells = [(r, c) for r in full_rows for c in full_cols]
    candidates: list[tuple[tuple[int, int], tuple[int, int], dict[str, int]]] = []
    for source1 in full_cells:
        for source2 in full_cells:
            if source1 == source2:
                continue
            source_collisions = _source_collision_count(source1, source2, period1, period2)
            if want_source_collision and source_collisions == 0:
                continue
            if not want_source_collision and source_collisions != 0:
                continue

            stats = _layout_stats(row_sizes, col_sizes, source1, source2, period1, period2)
            if stats["only1"] < 2 or stats["only2"] < 2:
                continue
            if stats["full_only1"] < 1 or stats["full_only2"] < 1:
                continue
            if stats["collision"] < 1 or stats["clipped_single"] < 1:
                continue
            candidates.append((source1, source2, stats))
    return candidates


def _build_grid(
    row_sizes: list[int],
    col_sizes: list[int],
    scaffold: int,
    seeds: list[Seed],
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height, row_bands, row_seps = _bands_from_sizes(row_sizes)
    width, col_bands, col_seps = _bands_from_sizes(col_sizes)
    assert height <= 30 and width <= 30

    grid = np.full((height, width), BACKGROUND, dtype=int)
    if row_seps:
        grid[row_seps, :] = int(scaffold)
    if col_seps:
        grid[:, col_seps] = int(scaffold)

    for seed in seeds:
        row_band = row_bands[seed.cell[0]]
        col_band = col_bands[seed.cell[1]]
        _stamp(grid, row_band, col_band, seed)
    return grid, row_bands, col_bands


def _render_expected(
    inp: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    seeds: list[Seed],
) -> np.ndarray:
    out = inp.copy()
    source_cells = {seed.cell for seed in seeds}
    for row_idx, row_band in enumerate(row_bands):
        for col_idx, col_band in enumerate(col_bands):
            cell = (row_idx, col_idx)
            if cell in source_cells:
                continue
            hits = _cell_hits(cell, seeds)
            if len(hits) == 1:
                _stamp(out, row_band, col_band, hits[0])
    return out


def _relative_color_coords(
    grid: np.ndarray,
    row_band: tuple[int, int],
    col_band: tuple[int, int],
    color: int,
) -> set[tuple[int, int]]:
    r0, r1 = row_band
    c0, c1 = col_band
    coords = np.argwhere(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color))
    return {(int(r), int(c)) for r, c in coords}


def _clipped_offsets(
    row_band: tuple[int, int],
    col_band: tuple[int, int],
    seed: Seed,
) -> set[tuple[int, int]]:
    height = row_band[1] - row_band[0] + 1
    width = col_band[1] - col_band[0] + 1
    return {(dr, dc) for dr, dc in seed.offsets if dr < height and dc < width}


def _semantic_validate(
    inp: np.ndarray,
    out: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    seeds: list[Seed],
    stats: dict[str, int],
) -> bool:
    if inp.shape != out.shape or max(inp.shape) > 30:
        return False
    if not np.array_equal(out[inp != BACKGROUND], inp[inp != BACKGROUND]):
        return False

    source_cells = {seed.cell for seed in seeds}
    full_single_seen = {seed.color: False for seed in seeds}
    clipped_seen = False
    collision_seen = False

    for row_idx, row_band in enumerate(row_bands):
        for col_idx, col_band in enumerate(col_bands):
            cell = (row_idx, col_idx)
            hits = _cell_hits(cell, seeds)
            r0, r1 = row_band
            c0, c1 = col_band
            patch = out[r0 : r1 + 1, c0 : c1 + 1]

            if cell in source_cells:
                assert np.array_equal(patch, inp[r0 : r1 + 1, c0 : c1 + 1])
                continue

            if len(hits) == 0 or len(hits) == 2:
                assert np.all(patch == BACKGROUND)
                if len(hits) == 2:
                    collision_seen = True
                continue

            seed = hits[0]
            expected = _clipped_offsets(row_band, col_band, seed)
            actual = _relative_color_coords(out, row_band, col_band, seed.color)
            assert actual == expected
            other_seed_colors = [other.color for other in seeds if other.color != seed.color]
            assert all(not _relative_color_coords(out, row_band, col_band, color) for color in other_seed_colors)
            if len(expected) == seed.period:
                full_single_seen[seed.color] = True
            elif 0 < len(expected) < seed.period:
                clipped_seen = True

    return (
        collision_seen
        and clipped_seen
        and all(full_single_seen.values())
        and stats["only1"] >= 2
        and stats["only2"] >= 2
    )


def _gen_attempt() -> np.ndarray | None:
    want_source_collision = random.random() < 0.55
    # Within 30x30, the 3x3 seed cells allow the coprime 2/3 periods to show
    # both exclusive lattice cells and true collision cells without degeneracy.
    period_choices = [(2, 3)]
    period1, period2 = random.choice(period_choices)
    if random.random() < 0.5:
        period1, period2 = period2, period1
    lcm = _lcm(period1, period2)
    min_cells = max(5, lcm + 1)
    if min_cells > 8:
        return None

    row_count = random.randint(min_cells, 8)
    col_count = random.randint(min_cells, 8)
    row_sizes = _side_sizes(row_count, prefer_truncated=True)
    col_sizes = _side_sizes(col_count, prefer_truncated=True)
    if row_sizes[-1] == CELL_SIZE and col_sizes[-1] == CELL_SIZE:
        if random.random() < 0.5:
            row_sizes[-1] = random.choice([1, 2])
        else:
            col_sizes[-1] = random.choice([1, 2])

    candidates = _candidate_sources(row_sizes, col_sizes, period1, period2, want_source_collision)
    if not candidates:
        return None

    source1, source2, stats = random.choice(candidates)
    colors = random.sample([c for c in ARC_COLORS if c != BACKGROUND], 3)
    scaffold, color1, color2 = colors
    seeds = [
        Seed(color1, period1, source1, _gen_offsets(period1)),
        Seed(color2, period2, source2, _gen_offsets(period2)),
    ]
    inp, row_bands, col_bands = _build_grid(row_sizes, col_sizes, scaffold, seeds)
    expected = _render_expected(inp, row_bands, col_bands, seeds)

    try:
        actual = main(inp)
        if not np.array_equal(actual, expected):
            return None
        if not _semantic_validate(inp, expected, row_bands, col_bands, seeds, stats):
            return None
    except Exception:
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-seed lattice input")
