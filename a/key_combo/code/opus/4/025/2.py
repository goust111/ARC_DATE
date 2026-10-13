from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4, 7)
TILE_SIDE = 3
LATTICE_STEP = TILE_SIDE + 1
MAX_GENERATION_ATTEMPTS = 500


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tile_bbox(row_seps: list[int], col_seps: list[int], tile_r: int, tile_c: int) -> tuple[int, int, int, int]:
    return (
        row_seps[tile_r] + 1,
        col_seps[tile_c] + 1,
        row_seps[tile_r + 1] - 1,
        col_seps[tile_c + 1] - 1,
    )


def _detect_lattice(grid: np.ndarray) -> dict:
    grey_coords = np.argwhere(grid == SEP)
    assert grey_coords.size > 0, "expected grey lattice separators"
    r0, c0, r1, c1 = _bbox_coords(grey_coords)
    lattice = grid[r0 : r1 + 1, c0 : c1 + 1]

    row_seps_rel = [r for r in range(lattice.shape[0]) if np.all(lattice[r, :] == SEP)]
    col_seps_rel = [c for c in range(lattice.shape[1]) if np.all(lattice[:, c] == SEP)]
    assert len(row_seps_rel) >= 4 and len(col_seps_rel) >= 4, "expected a multi-tile lattice"
    assert row_seps_rel[0] == 0 and row_seps_rel[-1] == lattice.shape[0] - 1
    assert col_seps_rel[0] == 0 and col_seps_rel[-1] == lattice.shape[1] - 1
    assert all(b - a == LATTICE_STEP for a, b in zip(row_seps_rel, row_seps_rel[1:]))
    assert all(b - a == LATTICE_STEP for a, b in zip(col_seps_rel, col_seps_rel[1:]))

    row_seps = [r0 + r for r in row_seps_rel]
    col_seps = [c0 + c for c in col_seps_rel]
    tile_rows = len(row_seps) - 1
    tile_cols = len(col_seps) - 1

    for rel_r in range(lattice.shape[0]):
        abs_r = r0 + rel_r
        if rel_r in row_seps_rel:
            assert np.all(grid[abs_r, c0 : c1 + 1] == SEP)
        else:
            for abs_c in col_seps:
                assert int(grid[abs_r, abs_c]) == SEP
            non_sep_cols = [c for c in range(c0, c1 + 1) if c not in col_seps]
            assert np.all(grid[abs_r, non_sep_cols] != SEP)

    for rel_c in range(lattice.shape[1]):
        abs_c = c0 + rel_c
        if rel_c in col_seps_rel:
            assert np.all(grid[r0 : r1 + 1, abs_c] == SEP)

    return {
        "bbox": (r0, c0, r1, c1),
        "row_seps": row_seps,
        "col_seps": col_seps,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
    }


def _parse_tiles(grid: np.ndarray) -> tuple[dict, dict[tuple[int, int], np.ndarray], dict[tuple[int, int], Counter], Counter]:
    info = _detect_lattice(grid)
    row_seps = info["row_seps"]
    col_seps = info["col_seps"]
    tile_arrays: dict[tuple[int, int], np.ndarray] = {}
    tile_counts: dict[tuple[int, int], Counter] = {}
    global_freq: Counter = Counter()

    for tile_r in range(info["tile_rows"]):
        for tile_c in range(info["tile_cols"]):
            r0, c0, r1, c1 = _tile_bbox(row_seps, col_seps, tile_r, tile_c)
            tile = grid[r0 : r1 + 1, c0 : c1 + 1]
            assert tile.shape == (TILE_SIDE, TILE_SIDE)
            colors = set(map(int, np.unique(tile)))
            assert SEP not in colors and BG not in colors, "tile interiors must be colored"
            assert colors.issubset(set(PALETTE))
            tile_arrays[(tile_r, tile_c)] = tile.copy()
            counts = Counter(map(int, tile.reshape(-1)))
            tile_counts[(tile_r, tile_c)] = counts
            global_freq.update(counts)

    return info, tile_arrays, tile_counts, global_freq


def _vote_from_tiles(
    tile_keys: list[tuple[int, int]],
    tile_counts: dict[tuple[int, int], Counter],
    global_freq: Counter,
) -> tuple[int, Counter, bool]:
    local = Counter()
    for key in tile_keys:
        local.update(tile_counts[key])
    max_count = max(local.values())
    tied = [color for color, count in local.items() if count == max_count]
    if len(tied) == 1:
        return int(tied[0]), local, False
    best_global = max(global_freq[color] for color in tied)
    winners = [color for color in tied if global_freq[color] == best_global]
    assert len(winners) == 1, "global frequency tiebreak must be unique"
    return int(winners[0]), local, True


def _analyze_votes(grid: np.ndarray) -> tuple[dict, dict[tuple[int, int], np.ndarray], dict[tuple[int, int], Counter], Counter, list[dict]]:
    info, tile_arrays, tile_counts, global_freq = _parse_tiles(grid)
    row_seps = info["row_seps"]
    col_seps = info["col_seps"]
    tile_rows = info["tile_rows"]
    tile_cols = info["tile_cols"]
    votes: list[dict] = []

    for sep_r_idx, sep_r in enumerate(row_seps):
        for tile_c in range(tile_cols):
            adj: list[tuple[int, int]] = []
            if sep_r_idx > 0:
                adj.append((sep_r_idx - 1, tile_c))
            if sep_r_idx < tile_rows:
                adj.append((sep_r_idx, tile_c))
            color, local, tie = _vote_from_tiles(adj, tile_counts, global_freq)
            cells = [(sep_r, c) for c in range(col_seps[tile_c] + 1, col_seps[tile_c + 1])]
            votes.append(
                {
                    "kind": "horizontal",
                    "index": (sep_r_idx, tile_c),
                    "adj": adj,
                    "winner": color,
                    "local": local,
                    "tied": tie,
                    "cells": cells,
                }
            )

    for sep_c_idx, sep_c in enumerate(col_seps):
        for tile_r in range(tile_rows):
            adj = []
            if sep_c_idx > 0:
                adj.append((tile_r, sep_c_idx - 1))
            if sep_c_idx < tile_cols:
                adj.append((tile_r, sep_c_idx))
            color, local, tie = _vote_from_tiles(adj, tile_counts, global_freq)
            cells = [(r, sep_c) for r in range(row_seps[tile_r] + 1, row_seps[tile_r + 1])]
            votes.append(
                {
                    "kind": "vertical",
                    "index": (tile_r, sep_c_idx),
                    "adj": adj,
                    "winner": color,
                    "local": local,
                    "tied": tie,
                    "cells": cells,
                }
            )

    for sep_r_idx, sep_r in enumerate(row_seps):
        for sep_c_idx, sep_c in enumerate(col_seps):
            adj = []
            if sep_r_idx > 0 and sep_c_idx > 0:
                adj.append((sep_r_idx - 1, sep_c_idx - 1))
            if sep_r_idx > 0 and sep_c_idx < tile_cols:
                adj.append((sep_r_idx - 1, sep_c_idx))
            if sep_r_idx < tile_rows and sep_c_idx > 0:
                adj.append((sep_r_idx, sep_c_idx - 1))
            if sep_r_idx < tile_rows and sep_c_idx < tile_cols:
                adj.append((sep_r_idx, sep_c_idx))
            color, local, tie = _vote_from_tiles(adj, tile_counts, global_freq)
            votes.append(
                {
                    "kind": "intersection",
                    "index": (sep_r_idx, sep_c_idx),
                    "adj": adj,
                    "winner": color,
                    "local": local,
                    "tied": tie,
                    "cells": [(sep_r, sep_c)],
                }
            )

    return info, tile_arrays, tile_counts, global_freq, votes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each lattice separator segment by majority vote from the adjacent tile interiors.

    Input structure:
    - Black (0) background contains one grey (5) rectangular lattice with 1-pixel-wide separator rows and columns.
    - The lattice encloses a regular 3x3 or 4x4 array of tiles, each tile interior being exactly 3x3 pixels.
    - Tile interiors use 2 or 3 colors chosen from blue/red/green/yellow/orange and contain no grey.

    Rule:
    - Detect the grey lattice, its separator rows/columns, and every 3x3 tile interior between them.
    - For each horizontal or vertical separator segment between consecutive lattice crossings, collect all interior pixels from the tile(s) touching that segment and choose the locally most frequent color.
    - If a segment vote is tied, break the tie by the higher global frequency of that color across all tile interiors in the whole lattice.
    - Recolor every separator segment with its chosen color, and color each lattice crossing independently using the same vote over all touching tiles.

    Key insight:
    - The grey lattice is only the geometric scaffold; its final colors are determined by coarse tile-neighborhood voting, not by the separator pixels themselves.
    """
    g = np.asarray(grid, dtype=int)
    _info, _tile_arrays, tile_counts, global_freq, votes = _analyze_votes(g)
    out = g.copy()
    for vote in votes:
        for r, c in vote["cells"]:
            out[r, c] = vote["winner"]
    return out


def _counts_to_tile(counts: dict[int, int]) -> np.ndarray:
    cells: list[int] = []
    for color, count in counts.items():
        cells.extend([int(color)] * int(count))
    assert len(cells) == TILE_SIDE * TILE_SIDE
    random.shuffle(cells)
    return np.array(cells, dtype=int).reshape(TILE_SIDE, TILE_SIDE)


def _random_counts(colors: list[int]) -> dict[int, int]:
    if len(colors) == 2:
        first = random.randint(2, 7)
        return {colors[0]: first, colors[1]: 9 - first}

    cuts = sorted(random.sample(range(1, 9), 2))
    parts = [cuts[0], cuts[1] - cuts[0], 9 - cuts[1]]
    while max(parts) < 4:
        cuts = sorted(random.sample(range(1, 9), 2))
        parts = [cuts[0], cuts[1] - cuts[0], 9 - cuts[1]]
    random.shuffle(parts)
    return {color: count for color, count in zip(colors, parts)}


def _random_tile() -> np.ndarray:
    color_count = random.choice((2, 3))
    colors = random.sample(list(PALETTE), color_count)
    return _counts_to_tile(_random_counts(colors))


def _paint_lattice_shell(grid: np.ndarray, top: int, left: int, tile_rows: int, tile_cols: int) -> tuple[list[int], list[int]]:
    row_seps = [top + LATTICE_STEP * i for i in range(tile_rows + 1)]
    col_seps = [left + LATTICE_STEP * j for j in range(tile_cols + 1)]
    for r in row_seps:
        grid[r, left : col_seps[-1] + 1] = SEP
    for c in col_seps:
        grid[top : row_seps[-1] + 1, c] = SEP
    return row_seps, col_seps


def _majority_colors(counter: Counter) -> set[int]:
    top = max(counter.values())
    return {int(color) for color, count in counter.items() if count == top}


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        info, _tile_arrays, tile_counts, global_freq, votes = _analyze_votes(inp)
    except AssertionError:
        return False

    if not np.array_equal(main(inp), out):
        return False

    global_top = max(global_freq.values())
    if sum(1 for count in global_freq.values() if count == global_top) >= 3:
        return False

    segment_votes = [vote for vote in votes if vote["kind"] != "intersection"]
    internal_ties = [vote for vote in segment_votes if vote["tied"] and len(vote["adj"]) == 2]
    if not internal_ties:
        return False

    if not any(not vote["tied"] and len(vote["adj"]) == 2 for vote in segment_votes):
        return False

    pair_majority_split = False
    for vote in internal_ties:
        a, b = vote["adj"]
        if _majority_colors(tile_counts[a]) != _majority_colors(tile_counts[b]):
            pair_majority_split = True
            break
    if not pair_majority_split:
        return False

    separator_colors = {
        int(out[r, c])
        for vote in votes
        for r, c in vote["cells"]
    }
    if len(separator_colors) < 2:
        return False

    if info["tile_rows"] < 3 or info["tile_cols"] < 3:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    tile_rows = random.choice((3, 4))
    tile_cols = random.choice((3, 4))
    lattice_h = LATTICE_STEP * tile_rows + 1
    lattice_w = LATTICE_STEP * tile_cols + 1

    pad_top = random.randint(1, 4)
    pad_bottom = random.randint(1, 4)
    pad_left = random.randint(1, 4)
    pad_right = random.randint(1, 4)
    h = lattice_h + pad_top + pad_bottom
    w = lattice_w + pad_left + pad_right
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), BG, dtype=int)
    row_seps, col_seps = _paint_lattice_shell(grid, pad_top, pad_left, tile_rows, tile_cols)

    orientation = random.choice(("horizontal", "vertical"))
    winner_color, loser_color = random.sample(list(PALETTE), 2)
    if orientation == "horizontal":
        base_r = random.randrange(tile_rows - 1)
        base_c = random.randrange(tile_cols)
        special_tiles = {
            (base_r, base_c): _counts_to_tile({winner_color: 5, loser_color: 4}),
            (base_r + 1, base_c): _counts_to_tile({winner_color: 4, loser_color: 5}),
        }
    else:
        base_r = random.randrange(tile_rows)
        base_c = random.randrange(tile_cols - 1)
        special_tiles = {
            (base_r, base_c): _counts_to_tile({winner_color: 5, loser_color: 4}),
            (base_r, base_c + 1): _counts_to_tile({winner_color: 4, loser_color: 5}),
        }

    for tile_r in range(tile_rows):
        for tile_c in range(tile_cols):
            tile = special_tiles.get((tile_r, tile_c), _random_tile())
            r0, c0, r1, c1 = _tile_bbox(row_seps, col_seps, tile_r, tile_c)
            grid[r0 : r1 + 1, c0 : c1 + 1] = tile

    try:
        info, tile_arrays, tile_counts, global_freq, votes = _analyze_votes(grid)
    except AssertionError:
        return None

    if global_freq[winner_color] <= global_freq[loser_color]:
        return None

    if not any(vote["tied"] and len(vote["adj"]) == 2 and winner_color == vote["winner"] for vote in votes if vote["kind"] != "intersection"):
        return None

    if sum(1 for count in global_freq.values() if count == max(global_freq.values())) >= 3:
        return None

    out = main(grid)
    if not _semantic_checks(grid, out):
        return None

    if len({tuple(tile_arrays[key].reshape(-1)) for key in tile_arrays}) < 3:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lattice-voting puzzle")
