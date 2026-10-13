from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_SPEC = importlib.util.spec_from_file_location("_shared_arc_utils", _ARC_UTILS_PATH)
assert _ARC_SPEC is not None and _ARC_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_SPEC)
_ARC_SPEC.loader.exec_module(_ARC_UTILS)

crop_bbox = _ARC_UTILS.crop_bbox


GEN_RETRIES = 400
PALETTE = tuple(range(10))


def _rot180(tile: np.ndarray) -> np.ndarray:
    return np.rot90(tile, 2)


def _non_separator_runs(length: int, sep_mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    idx = 0
    while idx < length:
        while idx < length and bool(sep_mask[idx]):
            idx += 1
        if idx >= length:
            break
        start = idx
        while idx + 1 < length and not bool(sep_mask[idx + 1]):
            idx += 1
        runs.append((int(start), int(idx)))
        idx += 1
    return runs


def _tile_lattice(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for color in map(int, np.unique(g)):
        row_sep = np.all(g == color, axis=1)
        col_sep = np.all(g == color, axis=0)
        if not row_sep.any() or not col_sep.any():
            continue
        row_runs = _non_separator_runs(g.shape[0], row_sep)
        col_runs = _non_separator_runs(g.shape[1], col_sep)
        if not row_runs or not col_runs:
            continue
        row_sizes = {r1 - r0 + 1 for r0, r1 in row_runs}
        col_sizes = {c1 - c0 + 1 for c0, c1 in col_runs}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue
        if len(row_runs) * len(col_runs) < 4:
            continue
        candidates.append((int(row_sep.sum() + col_sep.sum()), color, row_runs, col_runs))

    assert candidates, "expected a separator color that forms full rows and columns"
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score = candidates[0][0]
    best = [item for item in candidates if item[0] == best_score]
    assert len(best) == 1, "separator color must be uniquely identifiable"
    _score, sep_color, row_runs, col_runs = best[0]
    return int(sep_color), row_runs, col_runs


def _extract_tiles(
    grid: np.ndarray, row_runs: list[tuple[int, int]], col_runs: list[tuple[int, int]]
) -> dict[tuple[int, int], np.ndarray]:
    tiles: dict[tuple[int, int], np.ndarray] = {}
    for tr, (r0, r1) in enumerate(row_runs):
        for tc, (c0, c1) in enumerate(col_runs):
            tiles[(tr, tc)] = crop_bbox(grid, (int(r0), int(c0), int(r1), int(c1)))
    return tiles


def _parity(pos: tuple[int, int]) -> int:
    return int((int(pos[0]) + int(pos[1])) % 2)


def _occluded_tile(grid: np.ndarray, tiles: dict[tuple[int, int], np.ndarray]) -> tuple[tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[tuple[int, int], int]] = []
    for pos, tile in tiles.items():
        colors = np.unique(tile)
        if colors.size != 1:
            continue
        color = int(colors[0])
        if int(np.count_nonzero(g == color)) == int(tile.size):
            candidates.append((pos, color))
    assert len(candidates) == 1, "expected one uniquely colored solid occluder tile"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the single missing checkerboard-oriented tile that was fully hidden by a unique solid-color cover.

    Input structure:
    - Full separator-color rows and columns partition the grid into equal-sized tile cells.
    - Every visible tile contains the same multicolor pattern; tiles with opposite row+column parity are the 180-degree rotation of each other.
    - Exactly one tile cell is instead a solid-color block whose color appears nowhere else in the grid.

    Rule:
    - Detect the separator lattice and crop each tile cell.
    - Use any visible tile as the prototype for its parity class, and use its 180-degree rotation as the prototype for the opposite parity class.
    - Find the covered tile, determine its tile-board parity, and output the prototype tile for that parity only.

    Key insight:
    - Adjacent tiles are intentionally the wrong orientation, so the hidden tile must be recovered from the checkerboard parity rule rather than by copying a neighbor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _sep_color, row_runs, col_runs = _tile_lattice(g)
    tiles = _extract_tiles(g, row_runs, col_runs)
    occluded_pos, _occluder_color = _occluded_tile(g, tiles)

    visible = {pos: tile for pos, tile in tiles.items() if pos != occluded_pos}
    assert len(visible) >= 3, "need enough visible tiles to infer the checkerboard"

    ref_pos, ref_tile = next(iter(visible.items()))
    rotated_ref = _rot180(ref_tile)
    assert not np.array_equal(ref_tile, rotated_ref), "pattern must not be 180-symmetric"

    prototypes = {_parity(ref_pos): ref_tile, 1 - _parity(ref_pos): rotated_ref}
    seen_parities = set()
    for pos, tile in visible.items():
        parity = _parity(pos)
        seen_parities.add(parity)
        assert np.array_equal(tile, prototypes[parity]), "visible tiles must follow the parity rotation rule"
    assert len(seen_parities) == 2, "both parity classes should remain visible"

    return prototypes[_parity(occluded_pos)].copy()


def _generate_pattern(
    tile_h: int, tile_w: int, colors: tuple[int, ...], rng: np.random.Generator
) -> np.ndarray:
    non_center_pairs: list[tuple[int, int, int, int]] = []
    for r in range(tile_h):
        for c in range(tile_w):
            rr = tile_h - 1 - r
            cc = tile_w - 1 - c
            if (r, c) > (rr, cc):
                continue
            if (r, c) != (rr, cc):
                non_center_pairs.append((int(r), int(c), int(rr), int(cc)))

    for _ in range(200):
        pattern = np.zeros((tile_h, tile_w), dtype=int)
        same_prob = float(rng.uniform(0.45, 0.8))
        for r in range(tile_h):
            for c in range(tile_w):
                rr = tile_h - 1 - r
                cc = tile_w - 1 - c
                if (r, c) > (rr, cc):
                    continue
                color = int(rng.choice(colors))
                if (r, c) == (rr, cc):
                    pattern[r, c] = color
                    continue
                if rng.random() < same_prob:
                    pair_color = color
                else:
                    pair_color = int(rng.choice([col for col in colors if col != color]))
                pattern[r, c] = color
                pattern[rr, cc] = pair_color

        rng.shuffle(non_center_pairs)
        n_breaks = int(rng.integers(1, min(4, len(non_center_pairs)) + 1))
        for r, c, rr, cc in non_center_pairs[:n_breaks]:
            first = int(rng.choice(colors))
            second = int(rng.choice([col for col in colors if col != first]))
            pattern[r, c] = first
            pattern[rr, cc] = second

        rotated = _rot180(pattern)
        match_ratio = float(np.mean(pattern == rotated))
        if np.array_equal(pattern, rotated):
            continue
        if not (0.35 <= match_ratio <= 0.85):
            continue
        if len(np.unique(pattern)) < min(3, len(colors)):
            continue
        if any(np.unique(pattern[r, :]).size == 1 for r in range(tile_h)):
            continue
        if any(np.unique(pattern[:, c]).size == 1 for c in range(tile_w)):
            continue
        return pattern

    raise RuntimeError("failed to build a non-degenerate tile pattern")


def _grid_shape(tile_rows: int, tile_cols: int, tile_h: int, tile_w: int, border: bool) -> tuple[int, int]:
    row_seps = tile_rows + 1 if border else tile_rows - 1
    col_seps = tile_cols + 1 if border else tile_cols - 1
    return int(tile_rows * tile_h + row_seps), int(tile_cols * tile_w + col_seps)


def _tile_starts(count: int, tile_size: int, border: bool) -> list[int]:
    start = 1 if border else 0
    return [int(start + idx * (tile_size + 1)) for idx in range(count)]


def _position_classes(tile_rows: int, tile_cols: int) -> dict[str, list[tuple[int, int]]]:
    classes = {"corner": [], "edge": [], "interior": []}
    for tr in range(tile_rows):
        for tc in range(tile_cols):
            if tr in (0, tile_rows - 1) and tc in (0, tile_cols - 1):
                classes["corner"].append((int(tr), int(tc)))
            elif tr in (0, tile_rows - 1) or tc in (0, tile_cols - 1):
                classes["edge"].append((int(tr), int(tc)))
            else:
                classes["interior"].append((int(tr), int(tc)))
    return classes


def _neighbor_positions(pos: tuple[int, int], tile_rows: int, tile_cols: int) -> list[tuple[int, int]]:
    tr, tc = pos
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = tr + dr
        nc = tc + dc
        if 0 <= nr < tile_rows and 0 <= nc < tile_cols:
            out.append((int(nr), int(nc)))
    return out


def _build_instance(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    tile_rows = int(rng.integers(2, 5))
    tile_cols = int(rng.integers(2, 5))
    tile_h = int(rng.integers(3, 7))
    tile_w = int(rng.integers(3, 7))
    border = bool(rng.integers(0, 2))

    grid_h, grid_w = _grid_shape(tile_rows, tile_cols, tile_h, tile_w, border)
    assert grid_h <= 30 and grid_w <= 30

    n_pattern_colors = int(rng.integers(3, 6))
    picked = list(map(int, rng.choice(PALETTE, size=n_pattern_colors + 2, replace=False)))
    sep_color = picked[0]
    occluder_color = picked[1]
    pattern_colors = tuple(picked[2:])

    pattern = _generate_pattern(tile_h, tile_w, pattern_colors, rng)
    rotated = _rot180(pattern)

    grid = np.full((grid_h, grid_w), sep_color, dtype=int)
    row_starts = _tile_starts(tile_rows, tile_h, border)
    col_starts = _tile_starts(tile_cols, tile_w, border)

    for tr, r0 in enumerate(row_starts):
        for tc, c0 in enumerate(col_starts):
            tile = pattern if _parity((tr, tc)) == 0 else rotated
            grid[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile

    classes = _position_classes(tile_rows, tile_cols)
    available_labels = [label for label, positions in classes.items() if positions]
    chosen_label = str(rng.choice(available_labels))
    occluded_pos = classes[chosen_label][int(rng.integers(len(classes[chosen_label])))]
    expected = pattern if _parity(occluded_pos) == 0 else rotated

    occ_r0 = row_starts[occluded_pos[0]]
    occ_c0 = col_starts[occluded_pos[1]]
    grid[occ_r0 : occ_r0 + tile_h, occ_c0 : occ_c0 + tile_w] = occluder_color

    return grid, expected


def _validate_semantics(inp: np.ndarray, expected: np.ndarray) -> None:
    sep_color, row_runs, col_runs = _tile_lattice(inp)
    tiles = _extract_tiles(inp, row_runs, col_runs)
    occluded_pos, occluder_color = _occluded_tile(inp, tiles)
    visible = {pos: tile for pos, tile in tiles.items() if pos != occluded_pos}

    assert inp.ndim == 2 and expected.ndim == 2
    assert max(inp.shape) <= 30 and max(expected.shape) <= 30
    assert len(tiles) >= 4 and len(visible) >= 3
    assert len({tile.shape for tile in tiles.values()}) == 1
    assert int(np.count_nonzero(inp == sep_color)) > 0
    assert int(np.count_nonzero(inp == occluder_color)) == int(expected.size)
    assert not np.array_equal(expected, _rot180(expected))

    same_parity = [
        pos for pos, tile in visible.items() if _parity(pos) == _parity(occluded_pos) and np.array_equal(tile, expected)
    ]
    opposite_parity = [
        pos
        for pos, tile in visible.items()
        if _parity(pos) != _parity(occluded_pos) and np.array_equal(_rot180(tile), expected)
    ]
    assert same_parity, "the hidden tile parity should still have a visible exemplar"
    assert opposite_parity, "the opposite parity must demonstrate the 180-degree rotation rule"

    for nbr in _neighbor_positions(occluded_pos, len(row_runs), len(col_runs)):
        assert not np.array_equal(tiles[nbr], expected), "copying an orthogonal neighbor must fail"

    assert np.array_equal(main(inp), expected)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        inp, expected = _build_instance(rng)
        try:
            _validate_semantics(inp, expected)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate a valid checkerboard-occlusion puzzle")
