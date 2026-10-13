from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

try:
    from arc_utils import bbox, neighbors4
except Exception:
    def neighbors4(r: int, c: int):
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1

    def bbox(mask: np.ndarray):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)


BG = 0
BLUE = 1
RED = 2
SEP = 5

GEN_ATTEMPTS = 500
COMPRESSED_MIN = 10
COMPRESSED_MAX = 12
SQUARE_MIN = 4
SQUARE_MAX = 6

_ORBIT_CACHE: dict[int, list[tuple[tuple[int, int], ...]]] = {}


def _compress_grid(grid: np.ndarray):
    row_sep = np.all(grid == SEP, axis=1)
    col_sep = np.all(grid == SEP, axis=0)
    kept_rows = np.flatnonzero(~row_sep)
    kept_cols = np.flatnonzero(~col_sep)
    compressed = grid[np.ix_(kept_rows, kept_cols)].copy()
    return compressed, kept_rows, kept_cols, row_sep, col_sep


def _rotate_local(coord: tuple[int, int], size: int, turns: int = 1) -> tuple[int, int]:
    r, c = map(int, coord)
    for _ in range(turns % 4):
        r, c = c, size - 1 - r
    return int(r), int(c)


def _complete_rotational_mask(mask: np.ndarray) -> np.ndarray:
    size = int(mask.shape[0])
    assert mask.shape == (size, size)
    out = mask.copy()
    for r, c in np.argwhere(mask):
        cell = (int(r), int(c))
        for turns in (1, 2, 3):
            rr, cc = _rotate_local(cell, size, turns)
            out[rr, cc] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress separator-split coordinates, complete the hidden blue motif to fourfold rotational symmetry, then mark separator pixels touching new blue cells.

    Input structure:
    - The background is black, and some complete rows and columns are solid grey separators.
    - All non-separator blue pixels lie inside one square region only after those separator rows and columns are removed.
    - In the compressed grid, that blue region is an incomplete fourfold-rotational pattern.

    Rule:
    - Remove every full grey row and full grey column to obtain the compressed grid, while keeping the mapping back to original coordinates.
    - Find the minimal square bounding box containing all blue pixels in the compressed grid.
    - Rotate every existing blue pixel around that square's geometric center by 90, 180, and 270 degrees, adding the missing rotated blue pixels.
    - Map the completed compressed grid back to the original size by reinserting the grey separators.
    - Recolor a grey separator pixel red iff one of its 4-neighbors is a blue pixel added during the completion step rather than already present in the input.

    Key insight:
    - The apparent pattern is broken by separator rows and columns; only after coordinate compression does the hidden rotational symmetry become visible and completable.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    compressed, kept_rows, kept_cols, row_sep, col_sep = _compress_grid(g)
    blue_mask = compressed == BLUE
    blue_box = bbox(blue_mask)
    assert blue_box is not None, "expected blue pixels in the compressed grid"

    r0, c0, r1, c1 = blue_box
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    assert side_h == side_w, "compressed blue bbox must be square"

    local = blue_mask[r0 : r1 + 1, c0 : c1 + 1]
    completed_local = _complete_rotational_mask(local)

    completed_mask = blue_mask.copy()
    completed_mask[r0 : r1 + 1, c0 : c1 + 1] = completed_local
    added_mask = completed_mask & ~blue_mask

    out = g.copy()
    new_blue_orig = np.zeros_like(g, dtype=bool)
    for cr, cc in np.argwhere(completed_mask):
        rr = int(kept_rows[int(cr)])
        cc0 = int(kept_cols[int(cc)])
        out[rr, cc0] = BLUE
        if added_mask[int(cr), int(cc)]:
            new_blue_orig[rr, cc0] = True

    sep_mask = row_sep[:, None] | col_sep[None, :]
    H, W = g.shape
    for r, c in np.argwhere(sep_mask):
        rr0 = int(r)
        cc0 = int(c)
        for rr, cc in neighbors4(rr0, cc0):
            if 0 <= rr < H and 0 <= cc < W and new_blue_orig[rr, cc]:
                out[rr0, cc0] = RED
                break

    return out


def _coords_to_mask(size: int, coords: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def _all_orbits(size: int) -> list[tuple[tuple[int, int], ...]]:
    cached = _ORBIT_CACHE.get(size)
    if cached is not None:
        return cached

    seen: set[frozenset[tuple[int, int]]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for r in range(size):
        for c in range(size):
            orbit = frozenset(_rotate_local((r, c), size, turns) for turns in range(4))
            if orbit not in seen:
                seen.add(orbit)
                out.append(tuple(sorted(orbit)))
    _ORBIT_CACHE[size] = out
    return out


def _choose_target_orbits(rng: np.random.Generator, size: int):
    orbits = _all_orbits(size)
    four_orbits = [orbit for orbit in orbits if len(orbit) == 4]
    center_orbit = next((orbit for orbit in orbits if len(orbit) == 1), None)
    corner_orbit = next(orbit for orbit in four_orbits if (0, 0) in orbit)
    edge_orbits = [
        orbit
        for orbit in four_orbits
        if orbit != corner_orbit
        and any(r in (0, size - 1) or c in (0, size - 1) for r, c in orbit)
    ]
    inner_orbits = [orbit for orbit in four_orbits if orbit not in edge_orbits and orbit != corner_orbit]

    for _ in range(200):
        chosen = [corner_orbit]
        remaining = edge_orbits.copy()
        rng.shuffle(remaining)
        if remaining:
            chosen.append(remaining.pop())

        pool = remaining + inner_orbits.copy()
        rng.shuffle(pool)
        extra_count = int(rng.integers(0, min(2, len(pool)) + 1))
        chosen.extend(pool[:extra_count])

        if center_orbit is not None and rng.random() < 0.45:
            chosen.append(center_orbit)

        target = set().union(*map(set, chosen))
        if 8 <= len(target) <= 16:
            return chosen, target

    raise RuntimeError("failed to choose target orbits")


def _choose_visible_subset(
    rng: np.random.Generator,
    size: int,
    chosen_orbits: list[tuple[tuple[int, int], ...]],
    target: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    min_missing = max(1, (len(target) + 3) // 4)
    for _ in range(400):
        visible: set[tuple[int, int]] = set()
        missing = 0
        for orbit in chosen_orbits:
            orbit_cells = list(orbit)
            if len(orbit_cells) == 1:
                visible.add(orbit_cells[0])
                continue
            keep_count = int(rng.choice(np.array([1, 2, 2, 3], dtype=int)))
            picks = rng.choice(len(orbit_cells), size=keep_count, replace=False)
            for idx in np.atleast_1d(picks):
                visible.add(orbit_cells[int(idx)])
            missing += len(orbit_cells) - keep_count

        if missing < min_missing:
            continue

        visible_mask = _coords_to_mask(size, visible)
        if bbox(visible_mask) != (0, 0, size - 1, size - 1):
            continue
        if np.array_equal(_complete_rotational_mask(visible_mask), visible_mask):
            continue
        return visible

    raise RuntimeError("failed to choose visible subset")


def _choose_boundaries(
    rng: np.random.Generator,
    compressed_len: int,
    count: int,
    square_start: int,
    square_size: int,
) -> list[int]:
    all_boundaries = list(range(1, compressed_len))
    inside = list(range(square_start + 1, square_start + square_size))
    first = int(rng.choice(np.array(inside, dtype=int)))
    remaining = [b for b in all_boundaries if b != first]
    extras = []
    if count > 1:
        extras = list(map(int, rng.choice(np.array(remaining, dtype=int), size=count - 1, replace=False)))
    return sorted([first, *extras])


def _expand_indices(compressed_len: int, boundaries: list[int]):
    kept: list[int] = []
    separators: list[int] = []
    boundary_set = set(map(int, boundaries))
    orig = 0
    for i in range(compressed_len):
        kept.append(orig)
        orig += 1
        if i + 1 in boundary_set:
            separators.append(orig)
            orig += 1
    return np.array(kept, dtype=int), np.array(separators, dtype=int), orig


def _planned_output(grid: np.ndarray, meta: dict) -> np.ndarray:
    out = grid.copy()
    new_blue = np.zeros_like(grid, dtype=bool)

    kept_rows = meta["kept_rows"]
    kept_cols = meta["kept_cols"]
    sr, sc, size = meta["square"]
    target = meta["target"]
    visible = meta["visible"]

    for lr, lc in target:
        rr = int(kept_rows[sr + lr])
        cc = int(kept_cols[sc + lc])
        out[rr, cc] = BLUE
        if (lr, lc) not in visible:
            new_blue[rr, cc] = True

    sep_mask = meta["row_sep"][:, None] | meta["col_sep"][None, :]
    H, W = grid.shape
    for r, c in np.argwhere(sep_mask):
        rr0 = int(r)
        cc0 = int(c)
        for rr, cc in neighbors4(rr0, cc0):
            if 0 <= rr < H and 0 <= cc < W and new_blue[rr, cc]:
                out[rr0, cc0] = RED
                break
    return out


def _raw_rotation_shortcut_matches(grid: np.ndarray, out: np.ndarray) -> bool:
    in_blue = grid == BLUE
    blue_box = bbox(in_blue)
    assert blue_box is not None
    r0, c0, r1, c1 = blue_box
    if (r1 - r0) != (c1 - c0):
        return False

    local = in_blue[r0 : r1 + 1, c0 : c1 + 1]
    raw_completed = np.zeros_like(in_blue, dtype=bool)
    raw_completed[r0 : r1 + 1, c0 : c1 + 1] = _complete_rotational_mask(local)
    return np.array_equal(raw_completed, out == BLUE)


def _has_old_blue_next_to_grey_separator(inp: np.ndarray, out: np.ndarray) -> bool:
    _, _, _, row_sep, col_sep = _compress_grid(inp)
    sep_mask = row_sep[:, None] | col_sep[None, :]
    H, W = inp.shape
    for r, c in np.argwhere(inp == BLUE):
        rr0 = int(r)
        cc0 = int(c)
        for rr, cc in neighbors4(rr0, cc0):
            if 0 <= rr < H and 0 <= cc < W and sep_mask[rr, cc] and out[rr, cc] == SEP:
                return True
    return False


def _validate_generated(inp: np.ndarray, planned: np.ndarray) -> bool:
    out = main(inp)
    if not np.array_equal(out, planned):
        return False

    comp_in, kept_rows, kept_cols, row_sep, col_sep = _compress_grid(inp)
    comp_out = out[np.ix_(kept_rows, kept_cols)]
    if not (2 <= int(row_sep.sum()) <= 4 and 2 <= int(col_sep.sum()) <= 4):
        return False

    in_blue = comp_in == BLUE
    out_blue = comp_out == BLUE
    blue_box = bbox(in_blue)
    if blue_box is None:
        return False
    r0, c0, r1, c1 = blue_box
    size = r1 - r0 + 1
    if size != c1 - c0 + 1 or not (SQUARE_MIN <= size <= SQUARE_MAX):
        return False

    added = out_blue & ~in_blue
    if int(added.sum()) < max(1, (int(out_blue.sum()) + 3) // 4):
        return False

    if not np.any(out == RED):
        return False

    sep_mask = row_sep[:, None] | col_sep[None, :]
    if np.all(out[sep_mask] == RED):
        return False

    if not _has_old_blue_next_to_grey_separator(inp, out):
        return False

    row_sep_pos = np.flatnonzero(row_sep)
    col_sep_pos = np.flatnonzero(col_sep)
    if not np.any((row_sep_pos > kept_rows[r0]) & (row_sep_pos < kept_rows[r1])):
        return False
    if not np.any((col_sep_pos > kept_cols[c0]) & (col_sep_pos < kept_cols[c1])):
        return False

    if _raw_rotation_shortcut_matches(inp, out):
        return False

    return True


def _gen_attempt(rng: np.random.Generator):
    comp_h = int(rng.integers(COMPRESSED_MIN, COMPRESSED_MAX + 1))
    comp_w = int(rng.integers(COMPRESSED_MIN, COMPRESSED_MAX + 1))
    size = int(rng.integers(SQUARE_MIN, SQUARE_MAX + 1))
    sr = int(rng.integers(0, comp_h - size + 1))
    sc = int(rng.integers(0, comp_w - size + 1))

    chosen_orbits, target = _choose_target_orbits(rng, size)
    visible = _choose_visible_subset(rng, size, chosen_orbits, target)

    row_sep_count = int(rng.integers(2, 5))
    col_sep_count = int(rng.integers(2, 5))
    row_boundaries = _choose_boundaries(rng, comp_h, row_sep_count, sr, size)
    col_boundaries = _choose_boundaries(rng, comp_w, col_sep_count, sc, size)

    kept_rows, sep_rows, orig_h = _expand_indices(comp_h, row_boundaries)
    kept_cols, sep_cols, orig_w = _expand_indices(comp_w, col_boundaries)
    if orig_h > 30 or orig_w > 30:
        return None

    grid = np.full((orig_h, orig_w), BG, dtype=int)
    grid[sep_rows, :] = SEP
    grid[:, sep_cols] = SEP

    for lr, lc in visible:
        rr = int(kept_rows[sr + lr])
        cc = int(kept_cols[sc + lc])
        grid[rr, cc] = BLUE

    row_sep = np.zeros(orig_h, dtype=bool)
    col_sep = np.zeros(orig_w, dtype=bool)
    row_sep[sep_rows] = True
    col_sep[sep_cols] = True

    meta = {
        "kept_rows": kept_rows,
        "kept_cols": kept_cols,
        "row_sep": row_sep,
        "col_sep": col_sep,
        "square": (sr, sc, size),
        "target": set(target),
        "visible": set(visible),
    }
    planned = _planned_output(grid, meta)
    if not _validate_generated(grid, planned):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-rotation compression input")
