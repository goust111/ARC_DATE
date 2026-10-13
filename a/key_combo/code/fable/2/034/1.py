from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, int(n) + 1) if int(n) % d == 0]


def _normalize_shift(dr: int, dc: int, tile_h: int, tile_w: int) -> tuple[int, int]:
    return int(dr) % int(tile_h), int(dc) % int(tile_w)


def _periodic_grid(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    tile_h, tile_w = tile.shape
    rows, cols = shape
    assert rows % tile_h == 0 and cols % tile_w == 0
    return np.tile(tile, (rows // tile_h, cols // tile_w))


def _candidate_tiles(grid: np.ndarray) -> list[dict]:
    rows, cols = grid.shape
    candidates: list[dict] = []
    for tile_h in _divisors(rows):
        for tile_w in _divisors(cols):
            tile = np.empty((tile_h, tile_w), dtype=int)
            total_matches = 0
            min_margin = None
            valid = True
            for rr in range(tile_h):
                for cc in range(tile_w):
                    cells = grid[rr::tile_h, cc::tile_w].ravel()
                    colors, counts = np.unique(cells, return_counts=True)
                    order = np.argsort(-counts, kind="stable")
                    best_idx = int(order[0])
                    tile[rr, cc] = int(colors[best_idx])
                    total_matches += int(counts[best_idx])
                    second = int(counts[int(order[1])]) if len(order) > 1 else 0
                    margin = int(counts[best_idx]) - second
                    min_margin = margin if min_margin is None else min(min_margin, margin)
                    if margin <= 0:
                        valid = False
            if not valid:
                continue
            flat = [int(v) for v in tile.ravel()]
            if len(set(flat)) != tile_h * tile_w:
                continue
            errors = int(grid.size - total_matches)
            candidates.append(
                {
                    "shape": (int(tile_h), int(tile_w)),
                    "tile": tile.copy(),
                    "area": int(tile_h * tile_w),
                    "errors": errors,
                    "min_margin": int(min_margin),
                }
            )
    candidates.sort(key=lambda item: (item["errors"], item["area"], -item["min_margin"], item["shape"]))
    return candidates


def _infer_tile(grid: np.ndarray) -> np.ndarray:
    candidates = _candidate_tiles(grid)
    assert candidates, "failed to infer a unique periodic tile"
    return candidates[0]["tile"].copy()


def _decode_defects(grid: np.ndarray, tile: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int]]:
    repaired = _periodic_grid(tile, grid.shape)
    positions = {(int(tile[r, c])): (int(r), int(c)) for r in range(tile.shape[0]) for c in range(tile.shape[1])}
    vectors: list[tuple[int, int]] = []
    total_r = 0
    total_c = 0

    for r_raw, c_raw in np.argwhere(grid != repaired):
        r = int(r_raw)
        c = int(c_raw)
        expected = int(repaired[r, c])
        actual = int(grid[r, c])
        assert actual in positions, "defect color must come from another tile position"
        exp_r, exp_c = positions[expected]
        act_r, act_c = positions[actual]
        vector = (int(act_r - exp_r), int(act_c - exp_c))
        vectors.append(vector)
        total_r += vector[0]
        total_c += vector[1]

    return repaired, vectors, (int(total_r), int(total_c))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each wrong-color defect as a tile-internal offset, then cyclically shift the repaired periodic pattern by the summed offset.

    Input structure:
    - The whole grid is tiled by repetitions of one small rectangular color tile.
    - Every tile position has a unique color, so each color identifies one tile coordinate.
    - A few cells are defects where the expected tile color was replaced by the color from another tile position.

    Rule:
    - Infer the smallest periodic tile from the majority-consistent repetitions.
    - Repair the grid by restoring the dominant color for every tile residue class.
    - For each defect, compare the expected tile color with the observed wrong color, convert that color substitution into the offset from expected tile position to actual tile position, sum all such offsets, and apply that summed vector as a cyclic translation of the fully repaired pattern.

    Key insight:
    - The defects are not noise to erase: because tile colors are unique within the tile, every wrong color encodes one precise offset, and the answer is the repaired pattern shifted by the total encoded instruction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    tile = _infer_tile(g)
    repaired, _vectors, shift = _decode_defects(g, tile)
    return np.roll(repaired, shift=shift, axis=(0, 1))


_ALL_TILE_SHAPES = tuple(
    shape
    for shape in (
        (2, 2),
        (2, 3),
        (3, 2),
        (2, 4),
        (4, 2),
        (3, 3),
        (1, 4),
        (4, 1),
        (1, 5),
        (5, 1),
        (1, 6),
        (6, 1),
        (1, 7),
        (7, 1),
        (1, 8),
        (8, 1),
        (1, 9),
        (9, 1),
    )
    if shape[0] * shape[1] <= len(Color.NOT_BLACK)
)
_TWO_D_TILE_SHAPES = tuple(shape for shape in _ALL_TILE_SHAPES if shape[0] > 1 and shape[1] > 1)
_ROW_TILE_SHAPES = tuple(shape for shape in _ALL_TILE_SHAPES if shape[0] > 1)
_COL_TILE_SHAPES = tuple(shape for shape in _ALL_TILE_SHAPES if shape[1] > 1)

_MODE_PROFILES = (
    {"count": 2, "kind": "both"},
    {"count": 2, "kind": "col"},
    {"count": 2, "kind": "row"},
    {"count": 3, "kind": "both"},
)

_GEN_CALL_COUNT = 0
_GEN_MAX_ATTEMPTS = 400
_MIN_DEFECT_GAP = 3


def _tile_shape_pool(kind: str) -> tuple[tuple[int, int], ...]:
    if kind == "both":
        return _TWO_D_TILE_SHAPES
    if kind == "row":
        return _ROW_TILE_SHAPES
    if kind == "col":
        return _COL_TILE_SHAPES
    raise ValueError(f"unknown profile kind: {kind}")


def _choose_shape(rng: np.random.Generator, kind: str) -> tuple[int, int]:
    pool = _tile_shape_pool(kind)
    return tuple(map(int, pool[int(rng.integers(len(pool)))]))


def _choose_repeat_count(rng: np.random.Generator, tile_dim: int) -> int:
    return int(rng.integers(2, min(6, 30 // int(tile_dim) + 1)))


def _build_tile(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    count = int(shape[0] * shape[1])
    colors = rng.choice(np.array(Color.NOT_BLACK, dtype=int), size=count, replace=False)
    rng.shuffle(colors)
    return colors.reshape(shape).astype(int)


def _all_residues(shape: tuple[int, int]) -> list[tuple[int, int]]:
    return [(r, c) for r in range(shape[0]) for c in range(shape[1])]


def _sample_residues(
    rng: np.random.Generator, shape: tuple[int, int], count: int
) -> list[tuple[int, int]]:
    residues = _all_residues(shape)
    rng.shuffle(residues)
    return [(int(r), int(c)) for r, c in residues[:count]]


def _vector_profile_ok(
    shape: tuple[int, int],
    kind: str,
    vectors: list[tuple[int, int]],
) -> bool:
    tile_h, tile_w = shape
    total_r = sum(dr for dr, _dc in vectors)
    total_c = sum(dc for _dr, dc in vectors)
    total_eff = _normalize_shift(total_r, total_c, tile_h, tile_w)
    if total_eff == (0, 0):
        return False

    if kind == "both" and not (total_eff[0] != 0 and total_eff[1] != 0):
        return False
    if kind == "row" and not (total_eff[0] != 0 and total_eff[1] == 0):
        return False
    if kind == "col" and not (total_eff[0] == 0 and total_eff[1] != 0):
        return False

    single_effects = {_normalize_shift(dr, dc, tile_h, tile_w) for dr, dc in vectors}
    if total_eff in single_effects:
        return False
    if len(single_effects) == 1:
        return False
    if kind == "both":
        if all(dr == 0 for dr, _dc in vectors):
            return False
        if all(dc == 0 for _dr, dc in vectors):
            return False
    return True


def _sample_target_residues(
    rng: np.random.Generator,
    shape: tuple[int, int],
    source_residues: list[tuple[int, int]],
    kind: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    all_positions = _all_residues(shape)
    for _ in range(300):
        targets: list[tuple[int, int]] = []
        vectors: list[tuple[int, int]] = []
        for src_r, src_c in source_residues:
            options = [(r, c) for r, c in all_positions if (r, c) != (src_r, src_c)]
            tgt_r, tgt_c = options[int(rng.integers(len(options)))]
            targets.append((int(tgt_r), int(tgt_c)))
            vectors.append((int(tgt_r - src_r), int(tgt_c - src_c)))
        if _vector_profile_ok(shape, kind, vectors):
            return targets, vectors
    return None


def _occurrence_options(
    shape: tuple[int, int],
    tile_shape: tuple[int, int],
    residues: list[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    rows, cols = shape
    tile_h, tile_w = tile_shape
    options: list[list[tuple[int, int]]] = []
    for res_r, res_c in residues:
        coords = [(r, c) for r in range(res_r, rows, tile_h) for c in range(res_c, cols, tile_w)]
        options.append(coords)
    return options


def _choose_defect_cells(
    rng: np.random.Generator,
    shape: tuple[int, int],
    tile_shape: tuple[int, int],
    residues: list[tuple[int, int]],
    min_gap: int,
) -> list[tuple[int, int]] | None:
    options = _occurrence_options(shape, tile_shape, residues)
    for coords in options:
        rng.shuffle(coords)

    chosen: list[tuple[int, int]] = []

    def backtrack(index: int) -> bool:
        if index == len(options):
            return True
        for coord in options[index]:
            if all(abs(coord[0] - rr) + abs(coord[1] - cc) >= min_gap for rr, cc in chosen):
                chosen.append((int(coord[0]), int(coord[1])))
                if backtrack(index + 1):
                    return True
                chosen.pop()
        return False

    return chosen.copy() if backtrack(0) else None


def _apply_defects(
    repaired: np.ndarray,
    tile: np.ndarray,
    cells: list[tuple[int, int]],
    targets: list[tuple[int, int]],
) -> np.ndarray:
    inp = repaired.copy()
    for (r, c), (target_r, target_c) in zip(cells, targets):
        inp[int(r), int(c)] = int(tile[int(target_r), int(target_c)])
    return inp


def _expected_output(tile: np.ndarray, shape: tuple[int, int], vectors: list[tuple[int, int]]) -> np.ndarray:
    repaired = _periodic_grid(tile, shape)
    total_r = sum(dr for dr, _dc in vectors)
    total_c = sum(dc for _dr, dc in vectors)
    return np.roll(repaired, shift=(int(total_r), int(total_c)), axis=(0, 1))


def _instance_valid(
    inp: np.ndarray,
    tile: np.ndarray,
    vectors: list[tuple[int, int]],
    expected_shape: tuple[int, int],
) -> bool:
    repaired = _periodic_grid(tile, inp.shape)
    out = _expected_output(tile, inp.shape, vectors)
    if np.array_equal(inp, repaired):
        return False
    if np.array_equal(out, repaired):
        return False
    if np.array_equal(inp, out):
        return False
    if int(np.count_nonzero(inp != repaired)) != len(vectors):
        return False
    if len(np.unique(inp)) != tile.size:
        return False

    candidates = _candidate_tiles(inp)
    if not candidates:
        return False
    best = candidates[0]
    if tuple(best["shape"]) != tuple(expected_shape):
        return False
    if not np.array_equal(best["tile"], tile):
        return False
    if best["errors"] != len(vectors):
        return False
    if len(candidates) > 1:
        second = candidates[1]
        if (second["errors"], second["area"]) == (best["errors"], best["area"]):
            return False

    tile_h, tile_w = tile.shape
    total_eff = _normalize_shift(
        sum(dr for dr, _dc in vectors),
        sum(dc for _dr, dc in vectors),
        tile_h,
        tile_w,
    )
    if total_eff == (0, 0):
        return False
    for vector in vectors:
        if _normalize_shift(vector[0], vector[1], tile_h, tile_w) == total_eff:
            return False

    solved = main(inp)
    return np.array_equal(solved, out)


def _generate_attempt(rng: np.random.Generator, profile: dict) -> np.ndarray | None:
    kind = str(profile["kind"])
    defect_count = int(profile["count"])

    tile_shape = _choose_shape(rng, kind)
    tile = _build_tile(rng, tile_shape)
    repeat_h = _choose_repeat_count(rng, tile_shape[0])
    repeat_w = _choose_repeat_count(rng, tile_shape[1])
    shape = (int(tile_shape[0] * repeat_h), int(tile_shape[1] * repeat_w))
    repaired = _periodic_grid(tile, shape)

    residues = _sample_residues(rng, tile_shape, defect_count)
    made = _sample_target_residues(rng, tile_shape, residues, kind)
    if made is None:
        return None
    targets, vectors = made

    cells = _choose_defect_cells(rng, shape, tile_shape, residues, min_gap=_MIN_DEFECT_GAP)
    if cells is None:
        cells = _choose_defect_cells(rng, shape, tile_shape, residues, min_gap=2)
    if cells is None:
        return None

    inp = _apply_defects(repaired, tile, cells, targets)
    if not _instance_valid(inp, tile, vectors, tile_shape):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    profile = _MODE_PROFILES[_GEN_CALL_COUNT % len(_MODE_PROFILES)]
    _GEN_CALL_COUNT += 1
    rng = np.random.default_rng()

    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _generate_attempt(rng, profile)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-defect translation puzzle")
