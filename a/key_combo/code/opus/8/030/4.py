from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np


BG = 0
TILE_SIDE = 3
GRID_SIDE = 13
PRESENT_COUNT_CHOICES = (5, 6)
PALETTE_COLORS = (1, 2, 3)
MAX_GENERATION_ATTEMPTS = 500
_GEN_RNG = np.random.default_rng()


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive index runs where a 1D boolean mask is True."""
    out: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            out.append((start, idx - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def _tile_spans(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Infer the three tile-row spans and three tile-column spans from visible non-bg runs."""
    row_spans = _runs(np.any(grid != bg, axis=1))
    col_spans = _runs(np.any(grid != bg, axis=0))
    assert len(row_spans) == 3 and len(col_spans) == 3, "expected a 3x3 separated tile array"
    assert all(r1 - r0 + 1 == TILE_SIDE for r0, r1 in row_spans), "tile rows must be 3 cells tall"
    assert all(c1 - c0 + 1 == TILE_SIDE for c0, c1 in col_spans), "tile cols must be 3 cells wide"
    return row_spans, col_spans


def _tile_records(
    grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]], bg: int
) -> tuple[dict[tuple[int, int], int], set[tuple[int, int]]]:
    """Parse visible tiles as {(tile_r, tile_c): fill_color} and return missing positions."""
    observed: dict[tuple[int, int], int] = {}
    missing: set[tuple[int, int]] = set()
    for tile_r, (r0, r1) in enumerate(row_spans):
        for tile_c, (c0, c1) in enumerate(col_spans):
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if np.all(patch == bg):
                missing.add((tile_r, tile_c))
                continue
            holes = np.argwhere(patch == bg)
            colors = {int(v) for v in np.unique(patch) if int(v) != bg}
            assert len(holes) == 1, "present tile must have exactly one background hole"
            assert len(colors) == 1, "present tile must use exactly one fill color"
            hole_r, hole_c = map(int, holes[0])
            assert (hole_r, hole_c) == (tile_r, tile_c), "hole must encode the tile coordinates"
            observed[(tile_r, tile_c)] = next(iter(colors))
    assert observed and missing, "need both visible tiles and missing tiles"
    return observed, missing


def _candidate_models(observed: dict[tuple[int, int], int]) -> list[tuple[tuple[int, int, int], int]]:
    """
    Return every cycle-order / row-step model consistent with the visible tiles.

    Model:
      color(r, c) = cycle[(c + row_step * r) mod 3]
    where row_step is 1 or 2, encoding the diagonal phase direction.
    """
    models: list[tuple[tuple[int, int, int], int]] = []
    for cycle in itertools.permutations(PALETTE_COLORS):
        for row_step in (1, 2):
            if all(cycle[(c + row_step * r) % 3] == color for (r, c), color in observed.items()):
                models.append((tuple(map(int, cycle)), int(row_step)))
    return models


def _render_output(
    shape: tuple[int, int],
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    cycle: tuple[int, int, int],
    row_step: int,
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for tile_r, (r0, r1) in enumerate(row_spans):
        for tile_c, (c0, c1) in enumerate(col_spans):
            color = cycle[(tile_c + row_step * tile_r) % 3]
            out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
            out[r0 + tile_r, c0 + tile_c] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the separated 3x3 tile array by inferring the cyclic color phase.

    Input structure:
    - The grid is a 13x13 black background with a 3x3 array of 3x3 tiles.
    - Tile rows and tile columns are separated by one-cell black gaps, with a
      one-cell black border around the whole array.
    - Each visible tile is monochrome except for one black hole at local
      coordinate (tile_row, tile_col), which labels that tile's array position.
    - Some tile slots are completely blank.

    Rule:
    - Infer the three tile-row bands and three tile-column bands from the
      separated layout.
    - Read each visible tile's array coordinates from its hole position and
      record its fill color.
    - Infer the unique global three-color cycle and the unique diagonal row
      phase direction such that tile (r, c) has color cycle[(c + row_step*r) mod 3].
    - Fill every missing tile with its inferred color, preserving the black hole
      at local coordinate (r, c); keep separators and border black.

    Key insight:
    - The hole inside each tile is both a coordinate label and the anchor that
      lets visible samples reveal one global column-periodic color rule with a
      row-dependent diagonal phase shift.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape == (GRID_SIDE, GRID_SIDE), "expected a 13x13 grid"
    assert set(map(int, np.unique(g))).issubset({BG, *PALETTE_COLORS}), "unexpected colors in grid"
    bg = BG

    row_spans, col_spans = _tile_spans(g, bg)
    observed, missing = _tile_records(g, row_spans, col_spans, bg)

    present_mask = np.zeros((3, 3), dtype=bool)
    for r, c in observed:
        present_mask[r, c] = True
    assert np.all(np.any(present_mask, axis=1)), "each tile row must contain a visible tile"
    assert np.all(np.any(present_mask, axis=0)), "each tile col must contain a visible tile"
    assert len(observed) in PRESENT_COUNT_CHOICES, "expected 5 or 6 visible tiles"
    assert len(missing) == 9 - len(observed)

    models = _candidate_models(observed)
    assert len(models) == 1, "visible tiles must determine a unique cyclic phase model"
    cycle, row_step = models[0]

    out = _render_output(g.shape, row_spans, col_spans, cycle, row_step)
    for (r, c), color in observed.items():
        r0, r1 = row_spans[r]
        c0, c1 = col_spans[c]
        assert out[r0 : r1 + 1, c0 : c1 + 1][r, c] == BG
        expected_color = cycle[(c + row_step * r) % 3]
        assert color == expected_color, "visible tile color must match the inferred phase rule"
    return out


def _default_spans() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_spans = [(1, 3), (5, 7), (9, 11)]
    col_spans = [(1, 3), (5, 7), (9, 11)]
    return row_spans, col_spans


def _colors_by_position(cycle: tuple[int, int, int], row_step: int) -> dict[tuple[int, int], int]:
    return {(r, c): int(cycle[(c + row_step * r) % 3]) for r in range(3) for c in range(3)}


def _valid_present_sets(cycle: tuple[int, int, int], row_step: int) -> list[tuple[tuple[int, int], ...]]:
    colors = _colors_by_position(cycle, row_step)
    positions = [(r, c) for r in range(3) for c in range(3)]
    valid: list[tuple[tuple[int, int], ...]] = []
    for count in PRESENT_COUNT_CHOICES:
        for present in itertools.combinations(positions, count):
            present_set = set(present)
            if any(not any((r, c) in present_set for c in range(3)) for r in range(3)):
                continue
            if any(not any((r, c) in present_set for r in range(3)) for c in range(3)):
                continue
            observed = {pos: colors[pos] for pos in present}
            if len(set(observed.values())) != 3:
                continue
            models = _candidate_models(observed)
            if len(models) != 1:
                continue
            missing_colors = {colors[pos] for pos in positions if pos not in present_set}
            if len(missing_colors) < 2:
                continue
            valid.append(tuple(sorted(present)))
    return valid


def _build_input(cycle: tuple[int, int, int], row_step: int, present: Iterable[tuple[int, int]]) -> np.ndarray:
    row_spans, col_spans = _default_spans()
    full = _render_output((GRID_SIDE, GRID_SIDE), row_spans, col_spans, cycle, row_step)
    inp = full.copy()
    present_set = set(present)
    for r in range(3):
        for c in range(3):
            if (r, c) in present_set:
                continue
            r0, r1 = row_spans[r]
            c0, c1 = col_spans[c]
            inp[r0 : r1 + 1, c0 : c1 + 1] = BG
    return inp


def _gen_attempt() -> np.ndarray | None:
    cycle = tuple(int(x) for x in _GEN_RNG.permutation(PALETTE_COLORS))
    row_step = 1
    valid_present = _valid_present_sets(cycle, row_step)
    if not valid_present:
        return None
    present = valid_present[int(_GEN_RNG.integers(len(valid_present)))]
    inp = _build_input(cycle, row_step, present)
    out = _build_input(cycle, row_step, [(r, c) for r in range(3) for c in range(3)])

    observed, _ = _tile_records(inp, *_default_spans(), BG)
    models = _candidate_models(observed)
    if len(models) != 1:
        return None
    inferred_cycle, inferred_step = models[0]
    if inferred_cycle != cycle or inferred_step != row_step:
        return None
    if np.array_equal(inp, out):
        return None
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tile-phase completion sample")
