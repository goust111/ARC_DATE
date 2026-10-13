from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import Color, get_bg


PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.TEAL]
MAX_GEN_ATTEMPTS = 400


def _motif_row_count(grid: np.ndarray) -> int:
    count = 0
    while count < grid.shape[0] and int(grid[count, 0]) == Color.GREY:
        count += 1
    assert count >= 3, "expected at least three grey-delimited motif rows"
    return count


def _extract_motif(grid: np.ndarray, motif_rows: int) -> np.ndarray:
    motif = np.asarray(grid[:motif_rows], dtype=int).copy()
    motif[:, 0] = Color.BLACK
    return motif


def _last_non_obstacle_foreground_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.any((grid != Color.BLACK) & (grid != Color.ORANGE), axis=1))
    assert rows.size > 0, "expected some non-obstacle foreground content"
    return int(rows[-1])


def _fill_suffix(
    grid: np.ndarray,
    motif: np.ndarray,
    start_row: int,
    *,
    reset_on_obstacle: bool,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    motif_rows = int(motif.shape[0])
    phase = 0
    for r in range(int(start_row), out.shape[0]):
        src = motif[phase]
        fill_mask = out[r] == Color.BLACK
        out[r, fill_mask] = src[fill_mask]
        if np.any(out[r] == Color.ORANGE):
            phase = 0 if reset_on_obstacle else (phase + 1) % motif_rows
        else:
            phase = (phase + 1) % motif_rows
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the empty lower suffix by cycling motif rows, with orange obstacles resetting the row phase.

    Input structure:
    - The grid has a black background and starts with 3-5 top rows whose leftmost cells are grey markers.
    - Those grey-marked rows define the motif; after turning the grey cells black, the remaining row contents are the cyclic template.
    - Below the motif there may be a few preserved foreground rows, and then a large black suffix containing isolated orange obstacle pixels.

    Rule:
    - Find the consecutive grey run in the first column starting from the top; those rows are the motif rows.
    - Replace the grey marker cells in those rows with black to obtain an M-row motif, then find the last row containing any non-black, non-orange foreground cell; filling starts on the next row.
    - From that start row downward, copy motif row 1, 2, ..., M cyclically into black cells only; rows with orange cells keep the orange cells but still use the current motif row for the remaining black cells.
    - After any row containing orange, reset the cycle so the next processed row uses motif row 1 again; rows above the fill start stay unchanged.

    Key insight:
    - Orange cells are not just local blockers: each obstacle row resets the vertical cycle phase, so one obstacle changes the alignment of every later filled row until the next reset.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected a 2D ARC-sized grid"
    assert int(get_bg(g)) == Color.BLACK, "expected black to be the background"

    motif_rows = _motif_row_count(g)
    motif = _extract_motif(g, motif_rows)
    start_row = _last_non_obstacle_foreground_row(g) + 1
    assert start_row <= g.shape[0], "fill start must stay inside the grid"
    return _fill_suffix(g, motif, start_row, reset_on_obstacle=True)


def _minimal_period(motif: np.ndarray) -> int:
    rows = motif.shape[0]
    for p in range(1, rows):
        if all(np.array_equal(motif[r], motif[r % p]) for r in range(rows)):
            return p
    return rows


def _valid_track_start(width: int, slope: int, motif_rows: int, grid_width: int) -> tuple[int, int] | None:
    if slope >= 0:
        lo = 1
        hi = grid_width - width - slope * (motif_rows - 1)
    else:
        lo = 1 - slope * (motif_rows - 1)
        hi = grid_width - width
    if lo > hi:
        return None
    return int(lo), int(hi)


def _build_motif(rng: np.random.Generator, motif_rows: int, grid_width: int, palette: list[int]) -> np.ndarray | None:
    for _ in range(120):
        motif = np.zeros((motif_rows, grid_width), dtype=int)

        track_specs: list[tuple[int, int, int, int]] = []
        track_count = int(rng.integers(2, 5))
        for track_idx in range(track_count):
            width = int(rng.integers(2, min(6, grid_width - 1)))
            slopes = [-2, -1, 0, 1, 2] if track_idx else [-2, -1, 1, 2]
            slope = int(slopes[int(rng.integers(len(slopes)))])
            bounds = _valid_track_start(width, slope, motif_rows, grid_width)
            if bounds is None:
                continue
            lo, hi = bounds
            start = int(rng.integers(lo, hi + 1))
            color = int(palette[int(rng.integers(len(palette)))])
            track_specs.append((start, slope, width, color))

        if len(track_specs) < 2:
            continue

        for r in range(motif_rows):
            for start, slope, width, color in track_specs:
                c0 = start + slope * r
                motif[r, c0 : c0 + width] = color

        accent_cols = rng.choice(np.arange(1, grid_width - 1), size=motif_rows, replace=False)
        for r, col in enumerate(accent_cols):
            accent_color = int(palette[(r + int(rng.integers(len(palette)))) % len(palette)])
            motif[r, int(col)] = accent_color
            if rng.random() < 0.35 and int(col) + 1 < grid_width:
                motif[r, int(col) + 1] = accent_color

        for r in range(motif_rows):
            filled = np.flatnonzero(motif[r] != Color.BLACK)
            if filled.size > 5 and rng.random() < 0.4:
                motif[r, int(filled[int(rng.integers(filled.size))])] = Color.BLACK

        row_counts = [int(np.count_nonzero(motif[r])) for r in range(motif_rows)]
        if min(row_counts) < 3 or max(row_counts) > max(8, grid_width // 2):
            continue
        if len({tuple(row.tolist()) for row in motif}) != motif_rows:
            continue
        if _minimal_period(motif) != motif_rows:
            continue
        if len(set(map(int, motif[:, 1:].ravel())) - {Color.BLACK}) == 0:
            continue
        return motif
    return None


def _choose_segment_lengths(
    rng: np.random.Generator,
    suffix_height: int,
    motif_rows: int,
    obstacle_count: int,
) -> tuple[list[int], int] | None:
    usable = suffix_height - 1
    if usable < obstacle_count * 3:
        return None
    for _ in range(120):
        lengths: list[int] = []
        remaining = usable
        cumulative = 0
        ok = True
        for idx in range(obstacle_count):
            segments_left = obstacle_count - idx - 1
            max_len = remaining - segments_left * 3
            options = [
                length
                for length in range(3, max_len + 1)
                if (cumulative + length) % motif_rows != 0
            ]
            if not options:
                ok = False
                break
            length = int(options[int(rng.integers(len(options)))])
            lengths.append(length)
            remaining -= length
            cumulative += length
        if ok and remaining >= 1:
            return lengths, int(remaining)
    return None


def _obstacle_rows_from_lengths(start_row: int, lengths: list[int]) -> tuple[list[int], list[int]]:
    rows: list[int] = []
    phase_rows: list[int] = []
    offset = 0
    for length in lengths:
        offset += int(length)
        rows.append(int(start_row + offset - 1))
        phase_rows.append((int(length) - 1))
    return rows, phase_rows


def _render_input(
    motif: np.ndarray,
    extension_rows: int,
    suffix_height: int,
    obstacle_rows: list[int],
    obstacle_cols: list[int],
) -> np.ndarray:
    motif_rows, grid_width = motif.shape
    grid_height = motif_rows + extension_rows + suffix_height
    grid = np.zeros((grid_height, grid_width), dtype=int)
    grid[:motif_rows] = motif
    grid[:motif_rows, 0] = Color.GREY
    for k in range(extension_rows):
        grid[motif_rows + k] = motif[k % motif_rows]
    for r, c in zip(obstacle_rows, obstacle_cols):
        grid[int(r), int(c)] = Color.ORANGE
    return grid


def _semantic_checks(
    grid: np.ndarray,
    motif: np.ndarray,
    start_row: int,
    obstacle_rows: list[int],
) -> bool:
    motif_rows = int(motif.shape[0])
    assert _motif_row_count(grid) == motif_rows
    assert np.array_equal(_extract_motif(grid, motif_rows), motif)
    assert _last_non_obstacle_foreground_row(grid) + 1 == start_row
    assert _minimal_period(motif) == motif_rows
    assert len(obstacle_rows) in {2, 3, 4}
    assert all(b - a >= 3 for a, b in zip(obstacle_rows, obstacle_rows[1:]))
    assert grid.shape[0] - start_row >= 3 * motif_rows

    expected = _fill_suffix(grid, motif, start_row, reset_on_obstacle=True)
    wrong = _fill_suffix(grid, motif, start_row, reset_on_obstacle=False)
    actual = main(grid)

    assert np.array_equal(actual, expected)
    assert not np.array_equal(actual, wrong)
    assert np.array_equal(actual[:start_row], grid[:start_row])
    assert np.count_nonzero(actual[start_row:] != grid[start_row:]) > len(obstacle_rows)

    for idx, obstacle_row in enumerate(obstacle_rows):
        end_row = obstacle_rows[idx + 1] - 1 if idx + 1 < len(obstacle_rows) else grid.shape[0] - 1
        assert obstacle_row < end_row
        assert np.any(actual[obstacle_row + 1 : end_row + 1] != wrong[obstacle_row + 1 : end_row + 1])

    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    motif_rows = int(rng.integers(3, 6))
    extension_rows = int(rng.choice(np.array([0, 0, 1, 1, 2])))
    suffix_min = 3 * motif_rows
    suffix_max = min(18, 30 - motif_rows - extension_rows)
    if suffix_max < suffix_min:
        return None
    suffix_height = int(rng.integers(suffix_min, suffix_max + 1))
    grid_width = int(rng.integers(20, 29))

    palette_size = int(rng.integers(1, 4))
    palette = [int(c) for c in rng.choice(np.array(PALETTE), size=palette_size, replace=False)]
    motif = _build_motif(rng, motif_rows, grid_width, palette)
    if motif is None:
        return None

    obstacle_count = int(rng.integers(2, 5))
    lengths_and_tail = _choose_segment_lengths(rng, suffix_height, motif_rows, obstacle_count)
    if lengths_and_tail is None:
        return None
    lengths, _tail = lengths_and_tail
    start_row = motif_rows + extension_rows
    obstacle_rows, phase_offsets = _obstacle_rows_from_lengths(start_row, lengths)

    obstacle_cols: list[int] = []
    used_cols: set[int] = set()
    for phase_offset in phase_offsets:
        motif_row = motif[phase_offset % motif_rows]
        preferred = [int(c) for c in np.flatnonzero(motif_row != Color.BLACK) if int(c) not in used_cols]
        fallback = [c for c in range(grid_width) if c not in used_cols]
        candidates = preferred or fallback or list(range(grid_width))
        col = int(candidates[int(rng.integers(len(candidates)))])
        obstacle_cols.append(col)
        used_cols.add(col)

    grid = _render_input(motif, extension_rows, suffix_height, obstacle_rows, obstacle_cols)
    if not _semantic_checks(grid, motif, start_row, obstacle_rows):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid obstacle-reset cyclic-fill input")
