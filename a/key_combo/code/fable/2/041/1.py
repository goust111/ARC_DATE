from __future__ import annotations

import random

import numpy as np


BLACK = 0
NON_BLACK_COLORS = tuple(range(1, 10))

TILE_H_RANGE = (3, 7)
TILE_W_RANGE = (3, 7)
TILE_COLOR_COUNT_RANGE = (3, 5)
MAX_OUTPUT_HITS = 10
GEN_RETRY_LIMIT = 500
INNER_RETRY_LIMIT = 200


def _minimal_period(tile: np.ndarray) -> tuple[int, int]:
    h, w = map(int, tile.shape)
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    for tile_h in range(1, h + 1):
        for tile_w in range(1, w + 1):
            if np.array_equal(tile, tile[rows % tile_h, cols % tile_w]):
                return tile_h, tile_w
    return h, w


def _candidate_tile_from_known(
    known_rows: np.ndarray,
    known_cols: np.ndarray,
    known_vals: np.ndarray,
    tile_h: int,
    tile_w: int,
) -> np.ndarray | None:
    if known_vals.size == 0:
        return None

    phase = (known_rows % tile_h) * tile_w + (known_cols % tile_w)
    order = np.argsort(phase, kind="mergesort")
    phase = phase[order]
    vals = known_vals[order]
    if np.any((phase[1:] == phase[:-1]) & (vals[1:] != vals[:-1])):
        return None

    first = np.ones(phase.shape[0], dtype=bool)
    first[1:] = phase[1:] != phase[:-1]
    unique_phase = phase[first]
    unique_vals = vals[first]
    if unique_phase.size != tile_h * tile_w:
        return None

    tile = np.empty(tile_h * tile_w, dtype=int)
    tile[unique_phase] = unique_vals
    return tile.reshape(tile_h, tile_w)


def _minimal_candidate_tiles(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    known = g != BLACK
    known_rows, known_cols = np.nonzero(known)
    known_vals = g[known]
    candidates: list[tuple[int, np.ndarray]] = []
    for tile_h in range(1, g.shape[0] + 1):
        for tile_w in range(1, g.shape[1] + 1):
            tile = _candidate_tile_from_known(known_rows, known_cols, known_vals, tile_h, tile_w)
            if tile is not None:
                candidates.append((tile_h * tile_w, tile))

    assert candidates, "expected at least one repeating tile candidate"
    best_area = min(area for area, _tile in candidates)
    return [tile for area, tile in candidates if area == best_area]


def _infer_minimal_tile(grid: np.ndarray) -> np.ndarray:
    best = _minimal_candidate_tiles(grid)
    assert len(best) == 1, "expected a unique minimum-area repeating tile"
    return best[0]


def _tile_consistent_when_black_counts_as_color(grid: np.ndarray, tile_h: int, tile_w: int) -> bool:
    rows, cols = np.indices(grid.shape)
    phase = (rows.ravel() % tile_h) * tile_w + (cols.ravel() % tile_w)
    vals = grid.ravel()
    order = np.argsort(phase, kind="mergesort")
    phase = phase[order]
    vals = vals[order]
    return not np.any((phase[1:] == phase[:-1]) & (vals[1:] != vals[:-1]))


def _wallpaper_from_tile(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    tile_h, tile_w = map(int, tile.shape)
    rows = np.arange(shape[0])[:, None] % tile_h
    cols = np.arange(shape[1])[None, :] % tile_w
    return tile[rows, cols]


def _phase_occurrences(shape: tuple[int, int], tile_shape: tuple[int, int]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    tile_h, tile_w = map(int, tile_shape)
    out = {(r, c): [] for r in range(tile_h) for c in range(tile_w)}
    for r in range(shape[0]):
        for c in range(shape[1]):
            out[(r % tile_h, c % tile_w)].append((r, c))
    return out


def _sample_tile() -> np.ndarray | None:
    for _ in range(INNER_RETRY_LIMIT):
        tile_h = random.randint(*TILE_H_RANGE)
        tile_w = random.randint(*TILE_W_RANGE)
        area = tile_h * tile_w
        color_count = random.randint(
            TILE_COLOR_COUNT_RANGE[0],
            min(TILE_COLOR_COUNT_RANGE[1], area - 1, len(NON_BLACK_COLORS)),
        )
        palette = random.sample(NON_BLACK_COLORS, color_count)
        flat = palette + [random.choice(palette) for _ in range(area - color_count)]
        random.shuffle(flat)
        tile = np.array(flat, dtype=int).reshape(tile_h, tile_w)
        if _minimal_period(tile) != (tile_h, tile_w):
            continue
        _, counts = np.unique(tile, return_counts=True)
        if counts.max() < 2:
            continue
        return tile
    return None


def _sample_grid_shape(tile_shape: tuple[int, int]) -> tuple[int, int]:
    tile_h, tile_w = map(int, tile_shape)
    row_repeats = random.randint(2, min(4, 30 // tile_h))
    col_repeats = random.randint(2, min(4, 30 // tile_w))
    extra_h = random.randint(0, min(tile_h - 1, 30 - row_repeats * tile_h))
    extra_w = random.randint(0, min(tile_w - 1, 30 - col_repeats * tile_w))
    return row_repeats * tile_h + extra_h, col_repeats * tile_w + extra_w


def _sample_hit_phases(tile: np.ndarray) -> list[tuple[int, int]] | None:
    tile_h, tile_w = map(int, tile.shape)
    all_phases = [(r, c) for r in range(tile_h) for c in range(tile_w)]
    max_hits = min(len(all_phases) - 1, MAX_OUTPUT_HITS)
    for _ in range(INNER_RETRY_LIMIT):
        hit_count = random.randint(3, max_hits)
        chosen = set(random.sample(all_phases, hit_count))
        chosen_rows = {r for r, _c in chosen}
        chosen_cols = {c for _r, c in chosen}
        if len(chosen_rows) < 2 or len(chosen_cols) < 2:
            continue

        chosen_colors = {int(tile[phase]) for phase in chosen}
        if len(chosen_colors) < 2:
            continue

        unchosen = [phase for phase in all_phases if phase not in chosen]
        if not any(int(tile[p]) == int(tile[q]) for p in chosen for q in unchosen):
            continue
        return sorted(chosen)
    return None


def _render_revealed_phases(tile: np.ndarray, hit_phases: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros(tile.shape, dtype=int)
    for phase in hit_phases:
        out[phase] = int(tile[phase])
    return out


def _sample_missing_wallpaper(
    tile: np.ndarray,
    full: np.ndarray,
    hit_phases: list[tuple[int, int]],
) -> np.ndarray | None:
    inp = full.copy()
    occurrences = _phase_occurrences(inp.shape, tile.shape)
    black_counts: dict[tuple[int, int], int] = {}
    for phase in hit_phases:
        n_occ = len(occurrences[phase])
        upper = min(3, n_occ - 1)
        black_counts[phase] = random.randint(1, upper)

    if max(black_counts.values()) < 2:
        multi_choices = [phase for phase in hit_phases if len(occurrences[phase]) >= 3]
        if not multi_choices:
            return None
        black_counts[random.choice(multi_choices)] = 2

    for phase in hit_phases:
        for r, c in random.sample(occurrences[phase], black_counts[phase]):
            inp[r, c] = BLACK
    return inp


def _semantic_checks(inp: np.ndarray, tile: np.ndarray, hit_phases: list[tuple[int, int]]) -> bool:
    tile_h, tile_w = map(int, tile.shape)
    expected = _render_revealed_phases(tile, hit_phases)
    if np.array_equal(expected, tile):
        return False
    if not np.any(expected == BLACK) or not np.any(expected != BLACK):
        return False

    try:
        best = _minimal_candidate_tiles(inp)
        if len(best) != 1 or not np.array_equal(best[0], tile):
            return False
        if not np.array_equal(main(inp), expected):
            return False
    except Exception:
        return False

    if _tile_consistent_when_black_counts_as_color(inp, tile_h, tile_w):
        return False

    hit_set = set(hit_phases)
    occurrences = _phase_occurrences(inp.shape, tile.shape)
    black_per_phase: dict[tuple[int, int], int] = {}
    for phase, coords in occurrences.items():
        values = np.array([int(inp[r, c]) for r, c in coords], dtype=int)
        black_count = int(np.count_nonzero(values == BLACK))
        visible = values[values != BLACK]
        if not np.all(visible == int(tile[phase])):
            return False
        if phase in hit_set:
            if black_count == 0 or visible.size == 0:
                return False
        elif black_count != 0:
            return False
        black_per_phase[phase] = black_count

    if max(black_per_phase[phase] for phase in hit_set) < 2:
        return False
    if len({r for r, _c in hit_set}) < 2 or len({c for _r, c in hit_set}) < 2:
        return False

    black_coords = [tuple(map(int, rc)) for rc in np.argwhere(inp == BLACK)]
    if len({r // tile_h for r, _c in black_coords}) < 2:
        return False
    if len({_c // tile_w for _r, _c in black_coords}) < 2:
        return False

    all_phases = [(r, c) for r in range(tile_h) for c in range(tile_w)]
    unhit = [phase for phase in all_phases if phase not in hit_set]
    if not any(int(tile[p]) == int(tile[q]) for p in hit_set for q in unhit):
        return False
    return True


def _attempt_instance() -> np.ndarray | None:
    tile = _sample_tile()
    if tile is None:
        return None

    hit_phases = _sample_hit_phases(tile)
    if hit_phases is None:
        return None

    shape = _sample_grid_shape(tile.shape)
    full = _wallpaper_from_tile(tile, shape)
    for _ in range(INNER_RETRY_LIMIT):
        inp = _sample_missing_wallpaper(tile, full, hit_phases)
        if inp is not None and _semantic_checks(inp, tile, hit_phases):
            return inp
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold black holes from a repeated wallpaper onto one latent tile and reveal the hit phases' true colors.

    Input structure:
    - The grid is a finite crop of one non-black tile repeated by `(r, c)` phase from the top-left origin.
    - Some wallpaper cells are replaced by black missing cells, but the underlying tile itself contains no black.
    - Ignoring black cells, the unique minimum-area repeating tile is fully determined.

    Rule:
    - Ignore black cells and infer the unique minimum-area tile using phases `(r mod tile_h, c mod tile_w)` with origin at the grid's top-left corner.
    - Output a grid with exactly that tile size.
    - For every black input cell, mark its tile phase in the output with the true tile color of that phase; leave phases never hit by black as black.

    Key insight:
    - The black cells are not repaired in place; they are phase markers that must be folded back onto a single period tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    tile = _infer_minimal_tile(g)
    tile_h, tile_w = map(int, tile.shape)

    out = np.zeros(tile.shape, dtype=int)
    black_rows, black_cols = np.nonzero(g == BLACK)
    if black_rows.size:
        phase_rows = black_rows % tile_h
        phase_cols = black_cols % tile_w
        out[phase_rows, phase_cols] = tile[phase_rows, phase_cols]
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        inp = _attempt_instance()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid hole-code wallpaper input")
