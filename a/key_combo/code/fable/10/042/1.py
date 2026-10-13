from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np


BLACK = 0
MISSING = 6
PATTERN_COLORS = (1, 2, 3, 4, 5, 7, 8, 9)
TILE_SHAPES = ((2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4), (4, 3))
MAX_ATTEMPTS = 400


def _consistent_tile(grid: np.ndarray, row_period: int, col_period: int) -> np.ndarray | None:
    tile = np.full((row_period, col_period), -1, dtype=int)
    for r, c in np.argwhere(grid != MISSING):
        rr = int(r) % row_period
        cc = int(c) % col_period
        value = int(grid[r, c])
        if tile[rr, cc] == -1:
            tile[rr, cc] = value
        elif tile[rr, cc] != value:
            return None
    if np.any(tile == -1):
        return None
    return tile


def _infer_minimal_tile(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    h, w = map(int, grid.shape)
    visible_colors = sorted(int(v) for v in np.unique(grid) if int(v) != MISSING)
    assert len(visible_colors) == 2, "each quadrant must expose exactly two non-pink colors"

    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for row_period in range(1, h + 1):
        for col_period in range(1, w + 1):
            tile = _consistent_tile(grid, row_period, col_period)
            if tile is not None:
                candidates.append((row_period * col_period, row_period, col_period, tile))

    assert candidates, "quadrant must determine at least one periodic tile"
    min_area = min(area for area, _rh, _cw, _tile in candidates)
    minimal = [
        (row_period, col_period, tile)
        for area, row_period, col_period, tile in candidates
        if area == min_area
    ]
    assert len(minimal) == 1, "minimum-area period must be unique"

    row_period, col_period, tile = minimal[0]
    assert sorted(int(v) for v in np.unique(tile)) == visible_colors
    return row_period, col_period, tile


def _repair_quadrant(grid: np.ndarray) -> np.ndarray:
    row_period, col_period, tile = _infer_minimal_tile(grid)
    out = np.asarray(grid, dtype=int).copy()
    for r, c in np.argwhere(out == MISSING):
        out[int(r), int(c)] = tile[int(r) % row_period, int(c) % col_period]
    return out


def _separator_index(axis_lines: np.ndarray) -> int:
    idxs = [i for i, is_sep in enumerate(axis_lines) if bool(is_sep)]
    assert len(idxs) == 1, "expected exactly one full-black separator line"
    return int(idxs[0])


def _split_quadrants(grid: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sep_r = _separator_index(np.all(g == BLACK, axis=1))
    sep_c = _separator_index(np.all(g == BLACK, axis=0))
    quadrants = [
        g[:sep_r, :sep_c],
        g[:sep_r, sep_c + 1 :],
        g[sep_r + 1 :, :sep_c],
        g[sep_r + 1 :, sep_c + 1 :],
    ]
    h0, w0 = quadrants[0].shape
    assert h0 > 0 and w0 > 0
    assert all(q.shape == (h0, w0) for q in quadrants), "quadrants must be equal-sized"
    return quadrants, sep_r, sep_c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the four occluded periodic quadrants, then return the uniquely different repaired quadrant.

    Input structure:
    - One full-black separator row and one full-black separator column divide the grid into four equal quadrants.
    - Each quadrant contains a two-color periodic pattern plus pink (6) occlusion pixels.
    - Treating pink as missing, each quadrant is consistent with one unique minimum-area rectangular tile.

    Rule:
    - Split the grid into the four quadrants using the separator row and column.
    - Independently infer each quadrant's unique smallest periodic tile while ignoring pink cells, then fill every pink cell from that tile.
    - After repair, exactly three quadrants are identical and one is different; return the repaired contents of that outlier quadrant.

    Key insight:
    - The anomaly is only visible at the hidden tile level: raw quadrants differ because their occlusions differ, so comparison must happen after periodic completion.
    """
    quadrants, _sep_r, _sep_c = _split_quadrants(grid)
    repaired = [_repair_quadrant(q) for q in quadrants]
    matches = [sum(np.array_equal(repaired[i], repaired[j]) for j in range(4)) for i in range(4)]
    assert sorted(matches) == [1, 3, 3, 3], "repaired quadrants must contain exactly one outlier"
    outlier_idx = matches.index(1)
    return repaired[outlier_idx]


def _min_period_area(tile01: np.ndarray) -> int:
    h, w = map(int, tile01.shape)
    best = h * w
    for row_period in range(1, h + 1):
        for col_period in range(1, w + 1):
            ok = True
            for r in range(h):
                for c in range(w):
                    if int(tile01[r, c]) != int(tile01[r % row_period, c % col_period]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                best = min(best, row_period * col_period)
    return best


def _has_full_period(tile01: np.ndarray) -> bool:
    return _min_period_area(tile01) == int(tile01.size)


def _build_valid_binary_tiles() -> dict[tuple[int, int, int], list[np.ndarray]]:
    valid: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for h, w in TILE_SHAPES:
        area = h * w
        low = max(2, math.ceil(area * 0.25))
        high = min(area - 2, math.floor(area * 0.75))
        for bits in range(1 << area):
            count = bits.bit_count()
            if not low <= count <= high:
                continue
            flat = np.array([(bits >> i) & 1 for i in range(area)], dtype=int)
            tile01 = flat.reshape(h, w)
            if _has_full_period(tile01):
                valid.setdefault((h, w, count), []).append(tile01)
    return valid


VALID_BINARY_TILES = _build_valid_binary_tiles()
VALID_TILE_KEYS = tuple(sorted(VALID_BINARY_TILES))


def _sample_tile_pair() -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], tuple[int, int]]]:
    for _ in range(MAX_ATTEMPTS):
        h, w, _count = random.choice(VALID_TILE_KEYS)
        common = random.choice(VALID_BINARY_TILES[(h, w, _count)]).copy()
        ones = [(int(r), int(c)) for r, c in np.argwhere(common == 1)]
        zeros = [(int(r), int(c)) for r, c in np.argwhere(common == 0)]
        random.shuffle(ones)
        random.shuffle(zeros)
        for src in ones:
            for dst in zeros:
                outlier = common.copy()
                outlier[src] = 0
                outlier[dst] = 1
                if _has_full_period(outlier):
                    return common, outlier, (src, dst)
    raise RuntimeError("failed to sample a swappable full-period tile pair")


def _render_binary_tile(tile01: np.ndarray, color0: int, color1: int, rep_h: int, rep_w: int) -> np.ndarray:
    tiled = np.tile(tile01, (rep_h, rep_w))
    return np.where(tiled == 0, color0, color1).astype(int)


def _residue_occurrences(
    quadrant_shape: tuple[int, int],
    tile_shape: tuple[int, int],
    residue: tuple[int, int],
) -> list[tuple[int, int]]:
    qh, qw = quadrant_shape
    th, tw = tile_shape
    rr, cc = residue
    return [(r, c) for r in range(rr, qh, th) for c in range(cc, qw, tw)]


def _visible_color_counts(grid: np.ndarray) -> Counter:
    return Counter(int(v) for v in grid.ravel() if int(v) != MISSING)


def _marker_count(grid: np.ndarray) -> int:
    return int(np.sum(grid == MISSING))


def _pick_extra_mask_quota(
    common_full: np.ndarray,
    tile01: np.ndarray,
    diff_residues: tuple[tuple[int, int], tuple[int, int]],
    color0: int,
    color1: int,
    rep_h: int,
    rep_w: int,
) -> dict[int, int]:
    th, tw = map(int, tile01.shape)
    occurrences_per_residue = rep_h * rep_w

    available = {color0: 0, color1: 0}
    for rr in range(th):
        for cc in range(tw):
            residue = (rr, cc)
            if residue in diff_residues:
                continue
            color = int(common_full[rr, cc])
            available[color] += occurrences_per_residue - 2

    extra_total = random.randint(0, min(4, available[color0] + available[color1]))
    low = max(0, extra_total - available[color1])
    high = min(available[color0], extra_total)
    extra0 = random.randint(low, high)
    extra1 = extra_total - extra0
    return {color0: extra0, color1: extra1}


def _occlude_quadrant(
    full: np.ndarray,
    tile01: np.ndarray,
    diff_residues: tuple[tuple[int, int], tuple[int, int]],
    witness_by_residue: dict[tuple[int, int], tuple[int, int]],
    extra_mask_quota: dict[int, int],
) -> np.ndarray:
    qh, qw = map(int, full.shape)
    th, tw = map(int, tile01.shape)
    preserved: set[tuple[int, int]] = set()
    masked: set[tuple[int, int]] = set()

    for rr in range(th):
        for cc in range(tw):
            residue = (rr, cc)
            coords = _residue_occurrences((qh, qw), (th, tw), residue)
            if residue in diff_residues:
                witness = witness_by_residue[residue]
                preserved.add(witness)
                masked.update(coord for coord in coords if coord != witness)
            else:
                preserved.update(random.sample(coords, 2))

    pools: dict[int, list[tuple[int, int]]] = {}
    for r in range(qh):
        for c in range(qw):
            coord = (r, c)
            if coord in preserved or coord in masked:
                continue
            pools.setdefault(int(full[r, c]), []).append(coord)

    for color, count in extra_mask_quota.items():
        assert len(pools.get(color, [])) >= count
        masked.update(random.sample(pools[color], count))

    out = full.copy()
    if masked:
        rows, cols = zip(*sorted(masked))
        out[np.array(rows), np.array(cols)] = MISSING
    return out


def _assemble_grid(quadrants: list[np.ndarray]) -> np.ndarray:
    qh, qw = quadrants[0].shape
    grid = np.full((qh * 2 + 1, qw * 2 + 1), BLACK, dtype=int)
    grid[:qh, :qw] = quadrants[0]
    grid[:qh, qw + 1 :] = quadrants[1]
    grid[qh + 1 :, :qw] = quadrants[2]
    grid[qh + 1 :, qw + 1 :] = quadrants[3]
    return grid


def _validate_generated_instance(
    inp: np.ndarray,
    expected_outlier: np.ndarray,
    outlier_index: int,
    common_full: np.ndarray,
    outlier_full: np.ndarray,
    diff_residues: tuple[tuple[int, int], tuple[int, int]],
    witness_maps: list[dict[tuple[int, int], tuple[int, int]]],
    tile_shape: tuple[int, int],
) -> bool:
    quadrants, _sep_r, _sep_c = _split_quadrants(inp)
    if any(np.array_equal(quadrants[i], quadrants[j]) for i in range(4) for j in range(i + 1, 4)):
        return False

    marker_counts = {_marker_count(q) for q in quadrants}
    if len(marker_counts) != 1:
        return False
    visible_counts = {tuple(sorted(_visible_color_counts(q).items())) for q in quadrants}
    if len(visible_counts) != 1:
        return False

    try:
        repaired = [_repair_quadrant(q) for q in quadrants]
        solved = main(inp)
    except AssertionError:
        return False
    matches = [sum(np.array_equal(repaired[i], repaired[j]) for j in range(4)) for i in range(4)]
    if sorted(matches) != [1, 3, 3, 3]:
        return False
    if matches.index(1) != outlier_index:
        return False
    if not np.array_equal(repaired[outlier_index], expected_outlier):
        return False
    if not np.array_equal(solved, expected_outlier):
        return False

    if Counter(common_full.ravel().tolist()) != Counter(outlier_full.ravel().tolist()):
        return False
    if sum(int(common_full[rr, cc] != outlier_full[rr, cc]) for rr in range(tile_shape[0]) for cc in range(tile_shape[1])) != 2:
        return False

    for residue in diff_residues:
        if len({witness_maps[q_idx][residue] for q_idx in range(4)}) != 4:
            return False
        residue_coords = set(_residue_occurrences(common_full.shape, tile_shape, residue))
        for q_idx, quadrant in enumerate(quadrants):
            witness = witness_maps[q_idx][residue]
            if quadrant[witness] == MISSING:
                return False
            hidden_coords = residue_coords - {witness}
            if any(int(quadrant[r, c]) != MISSING for r, c in hidden_coords):
                return False
    return True


def _generate_attempt() -> np.ndarray | None:
    common01, outlier01, diff_residues = _sample_tile_pair()
    th, tw = map(int, common01.shape)
    max_rep_h = min(5, 14 // th)
    max_rep_w = min(5, 14 // tw)
    if max_rep_h < 3 or max_rep_w < 3:
        return None
    rep_h = random.randint(3, max_rep_h)
    rep_w = random.randint(3, max_rep_w)

    color0, color1 = random.sample(PATTERN_COLORS, 2)
    common_full = _render_binary_tile(common01, color0, color1, rep_h, rep_w)
    outlier_full = _render_binary_tile(outlier01, color0, color1, rep_h, rep_w)
    extra_mask_quota = _pick_extra_mask_quota(common_full, common01, diff_residues, color0, color1, rep_h, rep_w)

    witness_coords_by_residue: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for residue in diff_residues:
        coords = _residue_occurrences(common_full.shape, common01.shape, residue)
        if len(coords) < 4:
            return None
        witness_coords_by_residue[residue] = random.sample(coords, 4)

    outlier_index = random.randrange(4)
    witness_maps: list[dict[tuple[int, int], tuple[int, int]]] = []
    quadrants: list[np.ndarray] = []
    for q_idx in range(4):
        witness_map = {residue: witness_coords_by_residue[residue][q_idx] for residue in diff_residues}
        witness_maps.append(witness_map)
        full = outlier_full if q_idx == outlier_index else common_full
        quadrants.append(_occlude_quadrant(full, common01, diff_residues, witness_map, extra_mask_quota))

    inp = _assemble_grid(quadrants)
    if max(inp.shape) > 30:
        return None
    if not _validate_generated_instance(
        inp,
        outlier_full,
        outlier_index,
        common_full,
        outlier_full,
        diff_residues,
        witness_maps,
        common01.shape,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp = _generate_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input for the new ARC task")
