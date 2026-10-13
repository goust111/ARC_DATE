from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


BG = 0
SEP = 5
MASK = 6
COLOR_POOL = tuple(c for c in range(1, 10) if c not in {SEP, MASK})
OPS = ("id", "rot90", "rot180", "rot270", "flip_h", "flip_v", "diag", "anti")
GEN_ATTEMPTS = 400
MAX_TILE_ATTEMPTS = 300
MAX_LAYOUT_ATTEMPTS = 120

_N_BAG: List[int] = []
_CENTER_MASK_BAG: List[bool] = []

Pos = Tuple[int, int]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete blank or pink-masked macro tiles by propagating D4-symmetric partner tiles across a separator grid.

    Input structure:
    - The background is black, and full gray separator rows and columns split the
      canvas into an odd N x N array of equal square macro tiles.
    - Complete macro tiles contain only non-black, non-gray colors.
    - Some macro tiles are entirely black, and some otherwise-complete tiles
      contain a few pink masked pixels marking missing entries.
    - The finished macro-tile array obeys D4 symmetry: matching macro positions
      correspond after the same rotation or reflection is applied to the whole tile.

    Rule:
    - Detect the gray separator lattice to recover the macro-tile coordinates.
    - For each incomplete macro tile, search its D4 position orbit for complete
      partner tiles and transform one partner into the target orientation.
    - Fill all black or pink missing cells from that transformed partner while
      leaving visible non-masked cells unchanged.
    - If an orbit has no complete partner, it must be the self-symmetric center
      tile, whose masks are filled from its own pixel-level D4 symmetry.

    Key insight:
    The symmetry acts on macro-tile positions first and on tile pixels second, so
    the solver must infer the separator-defined macro grid before applying local
    rotations or reflections to repair each missing tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    n, tile_size, row_blocks, col_blocks = _parse_macro_layout(g)
    tiles = _extract_tiles(g, row_blocks, col_blocks)
    out = g.copy()

    for orbit in _position_orbits(n):
        complete_positions = [pos for pos in orbit if _is_complete_tile(tiles[pos])]
        for pos in orbit:
            observed = tiles[pos]
            if _is_complete_tile(observed):
                continue

            if complete_positions:
                target = _recover_tile_from_orbit(tiles, orbit, pos, complete_positions, n)
            else:
                assert len(orbit) == 1, "only the center orbit may lack a complete macro tile"
                target = _fill_by_group_symmetry(observed, _stabilizer_ops(n, pos))

            repaired = observed.copy()
            unknown = _unknown_mask(repaired)
            repaired[unknown] = target[unknown]
            assert np.array_equal(repaired[~unknown], observed[~unknown]), "known cells must stay unchanged"
            assert _is_complete_tile(repaired), "every repaired tile must become complete"

            r0, r1 = row_blocks[pos[0]]
            c0, c1 = col_blocks[pos[1]]
            out[r0:r1, c0:c1] = repaired
            tiles[pos] = repaired

    _assert_macro_d4_symmetry(_extract_tiles(out, row_blocks, col_blocks), n)
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid macro-D4 propagation input")


def _map_coord(op: str, r: int, c: int, size: int) -> Pos:
    if op == "id":
        return r, c
    if op == "rot90":
        return c, size - 1 - r
    if op == "rot180":
        return size - 1 - r, size - 1 - c
    if op == "rot270":
        return size - 1 - c, r
    if op == "flip_h":
        return size - 1 - r, c
    if op == "flip_v":
        return r, size - 1 - c
    if op == "diag":
        return c, r
    if op == "anti":
        return size - 1 - c, size - 1 - r
    raise ValueError(f"unknown operation {op}")


def _transform_tile(tile: np.ndarray, op: str) -> np.ndarray:
    arr = np.asarray(tile, dtype=int)
    h, w = arr.shape
    assert h == w, "macro tiles must be square for full D4 transport"
    out = np.empty_like(arr)
    for r in range(h):
        for c in range(w):
            rr, cc = _map_coord(op, r, c, h)
            out[rr, cc] = arr[r, c]
    return out


def _position_orbits(size: int) -> List[Tuple[Pos, ...]]:
    seen: set[Pos] = set()
    orbits: List[Tuple[Pos, ...]] = []
    for r in range(size):
        for c in range(size):
            if (r, c) in seen:
                continue
            orbit = tuple(sorted({_map_coord(op, r, c, size) for op in OPS}))
            seen.update(orbit)
            orbits.append(orbit)
    return orbits


def _stabilizer_ops(size: int, pos: Pos) -> Tuple[str, ...]:
    r, c = pos
    return tuple(op for op in OPS if _map_coord(op, r, c, size) == (r, c))


def _ops_between(src: Pos, dst: Pos, size: int) -> Tuple[str, ...]:
    r, c = src
    return tuple(op for op in OPS if _map_coord(op, r, c, size) == dst)


def _group_orbits(size: int, ops: Sequence[str]) -> List[Tuple[Pos, ...]]:
    seen: set[Pos] = set()
    out: List[Tuple[Pos, ...]] = []
    for r in range(size):
        for c in range(size):
            if (r, c) in seen:
                continue
            orbit = tuple(sorted({_map_coord(op, r, c, size) for op in ops}))
            seen.update(orbit)
            out.append(orbit)
    return out


def _symmetry_ops(tile: np.ndarray) -> Tuple[str, ...]:
    return tuple(op for op in OPS if np.array_equal(_transform_tile(tile, op), tile))


def _is_complete_tile(tile: np.ndarray) -> bool:
    arr = np.asarray(tile, dtype=int)
    return not np.any((arr == BG) | (arr == MASK) | (arr == SEP))


def _unknown_mask(tile: np.ndarray) -> np.ndarray:
    arr = np.asarray(tile, dtype=int)
    return (arr == BG) | (arr == MASK)


def _compatible_with_observed(candidate: np.ndarray, observed: np.ndarray) -> bool:
    known = ~_unknown_mask(observed)
    return bool(np.array_equal(candidate[known], observed[known]))


def _unique_tiles(candidates: Iterable[np.ndarray]) -> List[np.ndarray]:
    unique: Dict[bytes, np.ndarray] = {}
    for candidate in candidates:
        unique.setdefault(np.asarray(candidate, dtype=int).tobytes(), np.asarray(candidate, dtype=int))
    return list(unique.values())


def _recover_tile_from_orbit(
    tiles: Dict[Pos, np.ndarray],
    orbit: Sequence[Pos],
    target_pos: Pos,
    complete_positions: Sequence[Pos],
    size: int,
) -> np.ndarray:
    candidates: List[np.ndarray] = []
    observed = tiles[target_pos]

    for src in complete_positions:
        src_tile = tiles[src]
        for op in _ops_between(src, target_pos, size):
            transformed = _transform_tile(src_tile, op)
            if _compatible_with_observed(transformed, observed):
                candidates.append(transformed)

    filtered: List[np.ndarray] = []
    for candidate in _unique_tiles(candidates):
        if all(_candidate_matches_position(candidate, target_pos, pos, tiles[pos], size) for pos in orbit):
            filtered.append(candidate)

    assert len(filtered) == 1, "each incomplete tile must have one unique D4 completion"
    return filtered[0]


def _candidate_matches_position(
    candidate: np.ndarray,
    candidate_pos: Pos,
    target_pos: Pos,
    observed: np.ndarray,
    size: int,
) -> bool:
    for op in _ops_between(candidate_pos, target_pos, size):
        predicted = _transform_tile(candidate, op)
        if _compatible_with_observed(predicted, observed):
            return True
    return False


def _fill_by_group_symmetry(tile: np.ndarray, ops: Sequence[str]) -> np.ndarray:
    arr = np.asarray(tile, dtype=int).copy()
    h, w = arr.shape
    assert h == w, "pixel-level D4 completion expects square tiles"

    progress = True
    while progress and np.any(_unknown_mask(arr)):
        progress = False
        for orbit in _group_orbits(h, ops):
            values = {int(arr[r, c]) for r, c in orbit if int(arr[r, c]) not in {BG, MASK}}
            if len(values) != 1:
                continue
            color = values.pop()
            for r, c in orbit:
                if int(arr[r, c]) in {BG, MASK}:
                    arr[r, c] = color
                    progress = True

    assert not np.any(_unknown_mask(arr)), "self-symmetric repair must determine every missing cell"
    for op in ops:
        assert np.array_equal(arr, _transform_tile(arr, op)), "self-symmetric tile must respect its stabilizer"
    return arr


def _parse_macro_layout(grid: np.ndarray) -> Tuple[int, int, List[Tuple[int, int]], List[Tuple[int, int]]]:
    h, w = grid.shape
    sep_rows = [r for r in range(h) if np.all(grid[r, :] == SEP)]
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == SEP)]

    row_blocks = _blocks_from_separators(h, sep_rows)
    col_blocks = _blocks_from_separators(w, sep_cols)
    assert len(row_blocks) == len(col_blocks), "separator grid must define an N x N macro array"
    n = len(row_blocks)
    assert n in {3, 5}, "macro array must have odd size 3 or 5"

    tile_heights = {r1 - r0 for r0, r1 in row_blocks}
    tile_widths = {c1 - c0 for c0, c1 in col_blocks}
    assert len(tile_heights) == 1 and len(tile_widths) == 1, "macro tiles must be equal-sized"
    tile_h = tile_heights.pop()
    tile_w = tile_widths.pop()
    assert tile_h == tile_w, "macro tiles must be square for D4 rotations and diagonal reflections"
    return n, tile_h, row_blocks, col_blocks


def _blocks_from_separators(length: int, separators: Sequence[int]) -> List[Tuple[int, int]]:
    assert separators, "expected at least one separator line"
    assert separators[0] > 0 and separators[-1] < length - 1, "separator lines must be internal"
    for a, b in zip(separators, separators[1:]):
        assert b - a >= 2, "separator lines must have width 1 and leave non-empty blocks"

    starts = [0] + [idx + 1 for idx in separators]
    ends = list(separators) + [length]
    blocks = list(zip(starts, ends))
    sizes = {end - start for start, end in blocks}
    assert len(sizes) == 1 and next(iter(sizes)) > 0, "separator blocks must all have the same positive size"
    return blocks


def _extract_tiles(
    grid: np.ndarray,
    row_blocks: Sequence[Tuple[int, int]],
    col_blocks: Sequence[Tuple[int, int]],
) -> Dict[Pos, np.ndarray]:
    tiles: Dict[Pos, np.ndarray] = {}
    for i, (r0, r1) in enumerate(row_blocks):
        for j, (c0, c1) in enumerate(col_blocks):
            tiles[(i, j)] = np.asarray(grid[r0:r1, c0:c1], dtype=int).copy()
    return tiles


def _assert_macro_d4_symmetry(tiles: Dict[Pos, np.ndarray], size: int) -> None:
    for orbit in _position_orbits(size):
        rep = orbit[0]
        rep_tile = tiles[rep]
        for pos in orbit:
            transformed = [_transform_tile(rep_tile, op) for op in _ops_between(rep, pos, size)]
            assert any(np.array_equal(candidate, tiles[pos]) for candidate in transformed), "tiles must respect macro D4 symmetry"


def _draw_n() -> int:
    if not _N_BAG:
        _N_BAG.extend([3, 5, 5, 3, 5, 3])
        random.shuffle(_N_BAG)
    return _N_BAG.pop()


def _draw_center_mask_flag() -> bool:
    if not _CENTER_MASK_BAG:
        _CENTER_MASK_BAG.extend([False, False, True, False, True])
        random.shuffle(_CENTER_MASK_BAG)
    return _CENTER_MASK_BAG.pop()


def _tile_size_for_n(size: int) -> int:
    return random.choice((5, 6, 7) if size == 3 else (3, 4))


def _render_macro_grid(tiles: Dict[Pos, np.ndarray], size: int, tile_size: int) -> np.ndarray:
    side = size * tile_size + (size - 1)
    grid = np.full((side, side), BG, dtype=int)
    for k in range(1, size):
        idx = k * tile_size + (k - 1)
        grid[idx, :] = SEP
        grid[:, idx] = SEP

    for i in range(size):
        for j in range(size):
            r0 = i * (tile_size + 1)
            c0 = j * (tile_size + 1)
            grid[r0:r0 + tile_size, c0:c0 + tile_size] = tiles[(i, j)]
    return grid


def _make_tile(size: int, stabilizer_ops: Sequence[str], colors: Sequence[int]) -> np.ndarray | None:
    required = set(stabilizer_ops)
    pixel_orbits = _group_orbits(size, stabilizer_ops)

    for _ in range(MAX_TILE_ATTEMPTS):
        orbit_colors = [random.choice(colors) for _ in pixel_orbits]
        if set(orbit_colors) != set(colors):
            continue

        tile = np.zeros((size, size), dtype=int)
        for orbit, color in zip(pixel_orbits, orbit_colors):
            for r, c in orbit:
                tile[r, c] = color

        if set(_symmetry_ops(tile)) != required:
            continue
        if not _tile_visual_quality(tile, colors, required):
            continue
        return tile
    return None


def _tile_visual_quality(tile: np.ndarray, colors: Sequence[int], required: set[str]) -> bool:
    flat_colors = set(map(int, tile.flat))
    if flat_colors != set(map(int, colors)):
        return False
    if len({tuple(map(int, row)) for row in tile}) < 2:
        return False
    if len({tuple(map(int, tile[:, c])) for c in range(tile.shape[1])}) < 2:
        return False
    if np.max(np.bincount(tile.ravel(), minlength=10)) >= tile.size - 1:
        return False
    for op in OPS:
        if op in required:
            continue
        if np.count_nonzero(_transform_tile(tile, op) != tile) < 2:
            return False
    return True


def _build_complete_tiles(size: int, tile_size: int) -> Dict[Pos, np.ndarray] | None:
    tiles: Dict[Pos, np.ndarray] = {}
    for orbit in _position_orbits(size):
        rep = orbit[0]
        stabilizer_ops = _stabilizer_ops(size, rep)
        color_count = random.choice((2, 3))
        colors = random.sample(COLOR_POOL, color_count)
        base = _make_tile(tile_size, stabilizer_ops, colors)
        if base is None:
            return None

        for pos in orbit:
            ops = _ops_between(rep, pos, size)
            assert ops, "positions in one orbit must be connected by some D4 operation"
            tiles[pos] = _transform_tile(base, ops[0])
    return tiles


def _choose_nontrivial_orbit(tiles: Dict[Pos, np.ndarray], orbits: Sequence[Sequence[Pos]]) -> Sequence[Pos] | None:
    options = []
    for orbit in orbits:
        if len(orbit) <= 1:
            continue
        for a in orbit:
            for b in orbit:
                if a == b:
                    continue
                if not np.array_equal(tiles[a], tiles[b]):
                    options.append(tuple(orbit))
                    break
            else:
                continue
            break
    return random.choice(options) if options else None


def _choose_mask_cells(tile: np.ndarray) -> List[Pos]:
    all_cells = [(r, c) for r in range(tile.shape[0]) for c in range(tile.shape[1])]
    random.shuffle(all_cells)

    colors_seen: set[int] = set()
    chosen: List[Pos] = []
    for r, c in all_cells:
        value = int(tile[r, c])
        if value in colors_seen and len(chosen) >= 2:
            continue
        colors_seen.add(value)
        chosen.append((r, c))
        if len(chosen) == min(3, tile.size):
            break

    count = random.randint(1, min(3, len(chosen)))
    random.shuffle(chosen)
    return chosen[:count]


def _choose_self_symmetric_mask_cells(tile: np.ndarray) -> List[Pos]:
    eligible_orbits = [
        orbit
        for orbit in _group_orbits(tile.shape[0], OPS)
        if len(orbit) > 1 and len({int(tile[r, c]) for r, c in orbit}) == 1
    ]
    assert eligible_orbits, "center tile must have at least one nontrivial D4 pixel orbit to mask"
    random.shuffle(eligible_orbits)
    count = random.randint(1, min(3, len(eligible_orbits)))
    chosen = []
    for orbit in eligible_orbits[:count]:
        chosen.append(random.choice(list(orbit)))
    return chosen


def _mask_tile(tile: np.ndarray, cells: Sequence[Pos]) -> np.ndarray:
    out = np.asarray(tile, dtype=int).copy()
    for r, c in cells:
        out[r, c] = MASK
    return out


def _corrupt_tiles(
    full_tiles: Dict[Pos, np.ndarray],
    size: int,
    center_masked: bool,
) -> Tuple[Dict[Pos, np.ndarray], Dict[Pos, str]]:
    inp_tiles = {pos: tile.copy() for pos, tile in full_tiles.items()}
    states = {pos: "complete" for pos in full_tiles}
    orbits = _position_orbits(size)
    non_center_orbits = [orbit for orbit in orbits if len(orbit) > 1]

    forced_orbit = _choose_nontrivial_orbit(full_tiles, non_center_orbits)
    if forced_orbit is None:
        raise AssertionError("expected at least one nontrivial orbit")

    forced_blank_pos = random.choice(list(forced_orbit))
    forced_mask_pos = random.choice([pos for pos in forced_orbit if pos != forced_blank_pos])

    for orbit in non_center_orbits:
        max_corrupt = len(orbit) - 1
        if orbit == forced_orbit:
            corrupt_count = random.randint(2, max_corrupt) if max_corrupt >= 2 else 1
        else:
            corrupt_count = random.randint(0, max_corrupt)
        if corrupt_count == 0:
            continue

        corrupt_positions = random.sample(list(orbit), corrupt_count)
        if orbit == forced_orbit:
            if forced_blank_pos not in corrupt_positions:
                corrupt_positions[0] = forced_blank_pos
            if forced_mask_pos not in corrupt_positions:
                if len(corrupt_positions) == 1:
                    corrupt_positions.append(forced_mask_pos)
                else:
                    replace_idx = 0 if corrupt_positions[0] != forced_blank_pos else 1
                    corrupt_positions[replace_idx] = forced_mask_pos
            corrupt_positions = list(dict.fromkeys(corrupt_positions))
            while len(corrupt_positions) > max_corrupt:
                corrupt_positions.pop()

        kind_by_pos: Dict[Pos, str] = {}
        for pos in corrupt_positions:
            kind_by_pos[pos] = random.choice(("blank", "mask"))
        if orbit == forced_orbit:
            kind_by_pos[forced_blank_pos] = "blank"
            kind_by_pos[forced_mask_pos] = "mask"

        if len(kind_by_pos) == len(orbit):
            drop = random.choice(list(kind_by_pos))
            del kind_by_pos[drop]

        for pos, kind in kind_by_pos.items():
            if kind == "blank":
                inp_tiles[pos][:] = BG
            else:
                inp_tiles[pos] = _mask_tile(inp_tiles[pos], _choose_mask_cells(full_tiles[pos]))
            states[pos] = kind

    center = (size // 2, size // 2)
    if center_masked:
        inp_tiles[center] = _mask_tile(inp_tiles[center], _choose_self_symmetric_mask_cells(full_tiles[center]))
        states[center] = "center_mask"

    if not any(state == "blank" for state in states.values()):
        orbit = random.choice(non_center_orbits)
        candidates = [pos for pos in orbit if states[pos] == "complete"]
        pos = random.choice(candidates)
        inp_tiles[pos][:] = BG
        states[pos] = "blank"

    if not any(state in {"mask", "center_mask"} for state in states.values()):
        orbit = random.choice(non_center_orbits)
        candidates = [pos for pos in orbit if states[pos] == "complete"]
        pos = random.choice(candidates)
        inp_tiles[pos] = _mask_tile(inp_tiles[pos], _choose_mask_cells(full_tiles[pos]))
        states[pos] = "mask"

    return inp_tiles, states


def _semantic_checks(
    inp_tiles: Dict[Pos, np.ndarray],
    out_tiles: Dict[Pos, np.ndarray],
    states: Dict[Pos, str],
    size: int,
) -> bool:
    _assert_macro_d4_symmetry(out_tiles, size)

    saw_blank = False
    saw_mask = False
    saw_nontrivial = False

    for orbit in _position_orbits(size):
        complete_positions = [pos for pos in orbit if _is_complete_tile(inp_tiles[pos])]
        center_orbit = len(orbit) == 1 and orbit[0] == (size // 2, size // 2)

        if not center_orbit:
            if not complete_positions:
                return False
        else:
            if not complete_positions and states[orbit[0]] != "center_mask":
                return False

        for pos in orbit:
            state = states[pos]
            observed = inp_tiles[pos]
            expected = out_tiles[pos]

            if state == "blank":
                saw_blank = True
                if not np.all(observed == BG):
                    return False
            elif state == "mask":
                saw_mask = True
                unknown = observed == MASK
                if not np.any(unknown):
                    return False
                if not np.array_equal(observed[~unknown], expected[~unknown]):
                    return False
            elif state == "center_mask":
                saw_mask = True
                if pos != (size // 2, size // 2):
                    return False
                unknown = observed == MASK
                if not np.any(unknown):
                    return False
                for r, c in zip(*np.where(unknown)):
                    pixel_orbit = next(orb for orb in _group_orbits(observed.shape[0], OPS) if (int(r), int(c)) in orb)
                    orbit_vals = {
                        int(observed[rr, cc])
                        for rr, cc in pixel_orbit
                        if int(observed[rr, cc]) not in {BG, MASK}
                    }
                    if len(orbit_vals) != 1:
                        return False
                repaired = _fill_by_group_symmetry(observed, OPS)
                if not np.array_equal(repaired, expected):
                    return False
            else:
                if not np.array_equal(observed, expected):
                    return False

        for pos in orbit:
            if states[pos] not in {"blank", "mask"}:
                continue
            for src in complete_positions:
                if src == pos:
                    continue
                if not np.array_equal(out_tiles[src], out_tiles[pos]):
                    saw_nontrivial = True
                    break
            if saw_nontrivial:
                break

    return saw_blank and saw_mask and saw_nontrivial


def _gen_attempt() -> np.ndarray | None:
    for _ in range(MAX_LAYOUT_ATTEMPTS):
        size = _draw_n()
        tile_size = _tile_size_for_n(size)
        full_tiles = _build_complete_tiles(size, tile_size)
        if full_tiles is None:
            continue

        center_masked = _draw_center_mask_flag()
        try:
            inp_tiles, states = _corrupt_tiles(full_tiles, size, center_masked)
        except AssertionError:
            continue

        inp = _render_macro_grid(inp_tiles, size, tile_size)
        expected = _render_macro_grid(full_tiles, size, tile_size)

        if not _semantic_checks(inp_tiles, full_tiles, states, size):
            continue
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    return None
