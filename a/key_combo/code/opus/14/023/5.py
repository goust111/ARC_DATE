from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

MAX_GENERATION_ATTEMPTS = 600
TEMPLATES = [
    np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=int,
    ),
]


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("expected a non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _extract_sprite(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray]:
    sprite_mask = grid == BLUE
    sprite_cells = int(np.count_nonzero(sprite_mask))
    assert 3 <= sprite_cells <= 5, "expected a 3-5 cell blue sprite"
    comps = _components4(sprite_mask)
    assert len(comps) == 1, "expected exactly one 4-connected blue sprite"
    box = _bbox(sprite_mask)
    patch = np.where(_crop_bbox(sprite_mask.astype(int), box), BLUE, BLACK).astype(int)
    return patch, box, comps[0]


def _candidate_rows(grid: np.ndarray) -> list[int]:
    return [r for r in range(grid.shape[0]) if int(grid[r, 0]) == RED and int(grid[r, -1]) == RED]


def _candidate_cols(grid: np.ndarray) -> list[int]:
    return [c for c in range(grid.shape[1]) if int(grid[0, c]) == RED and int(grid[-1, c]) == RED]


def _valid_rows(grid: np.ndarray, *, treat_all_candidates: bool = False) -> list[int]:
    rows = _candidate_rows(grid)
    if treat_all_candidates:
        return rows
    valid = []
    for r in rows:
        interior = grid[r, 1:-1]
        if np.all((interior == BLACK) | (interior == BLUE)):
            valid.append(r)
    return valid


def _valid_cols(grid: np.ndarray, *, treat_all_candidates: bool = False) -> list[int]:
    cols = _candidate_cols(grid)
    if treat_all_candidates:
        return cols
    valid = []
    for c in cols:
        interior = grid[1:-1, c]
        if np.all((interior == BLACK) | (interior == BLUE)):
            valid.append(c)
    return valid


def _tile_horizontally(out: np.ndarray, patch: np.ndarray, row: int) -> None:
    h, w = patch.shape
    _, width = out.shape
    for c0 in range(1, width - 1, w):
        for dr in range(h):
            rr = row + dr
            if not (0 <= rr < out.shape[0]):
                continue
            for dc in range(w):
                if patch[dr, dc] != BLUE:
                    continue
                cc = c0 + dc
                if 1 <= cc <= width - 2 and int(out[rr, cc]) == BLACK:
                    out[rr, cc] = BLUE


def _tile_vertically(out: np.ndarray, patch: np.ndarray, col: int) -> None:
    h, w = patch.shape
    height, _ = out.shape
    for r0 in range(1, height - 1, h):
        for dr in range(h):
            rr = r0 + dr
            if not (1 <= rr <= height - 2):
                continue
            for dc in range(w):
                if patch[dr, dc] != BLUE:
                    continue
                cc = col + dc
                if 0 <= cc < out.shape[1] and int(out[rr, cc]) == BLACK:
                    out[rr, cc] = BLUE


def _solve(
    grid: np.ndarray,
    *,
    force_rotate: bool | None = None,
    treat_all_candidates: bool = False,
    swap_modes: bool = False,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    sprite, sprite_box, sprite_coords = _extract_sprite(g)
    row_candidates = _candidate_rows(g)
    col_candidates = _candidate_cols(g)
    valid_rows = _valid_rows(g, treat_all_candidates=treat_all_candidates)
    valid_cols = _valid_cols(g, treat_all_candidates=treat_all_candidates)
    h_count = len(valid_rows)
    v_count = len(valid_cols)

    assert h_count >= 1 and v_count >= 1, "expected at least one valid corridor in each direction"
    assert h_count != v_count, "expected unequal horizontal and vertical valid-corridor counts"

    horizontal_major = h_count > v_count
    rotate = horizontal_major if force_rotate is None else bool(force_rotate)
    patch = np.rot90(sprite, -1) if rotate else sprite.copy()

    out = g.copy()
    out[out == BLUE] = BLACK

    tile_horizontal = horizontal_major if not swap_modes else not horizontal_major
    if tile_horizontal:
        for r in valid_rows:
            _tile_horizontally(out, patch, r)
        for c in valid_cols:
            out[1:-1, c] = GREEN
    else:
        for c in valid_cols:
            _tile_vertically(out, patch, c)
        for r in valid_rows:
            out[r, 1:-1] = GREEN

    meta = {
        "sprite_box": sprite_box,
        "sprite_coords": sprite_coords,
        "sprite": sprite,
        "rotated": rotate,
        "horizontal_major": horizontal_major,
        "valid_rows": valid_rows,
        "valid_cols": valid_cols,
        "row_candidates": row_candidates,
        "col_candidates": col_candidates,
        "candidate_row_count": len(row_candidates),
        "candidate_col_count": len(col_candidates),
    }
    return (out, meta) if return_meta else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count clear edge-to-edge corridors, rotate the blue sprite from that count comparison, then tile or recolor the corridors accordingly.

    Input structure:
    - The grid has a black background, red border endpoints, optional yellow blockers, and one small 4-connected blue sprite in the interior.
    - A horizontal corridor candidate is a row whose left and right border cells are red; a vertical candidate is a column whose top and bottom border cells are red.
    - Corridor interiors may be empty or blocked by yellow cells; the blue sprite is not treated as a blocker.

    Rule:
    - Keep only candidate rows and columns whose interior cells contain only black or blue, and count valid horizontals `H` and valid verticals `V`.
    - If `H > V`, rotate the cropped blue sprite 90 degrees clockwise; if `V > H`, keep it unchanged.
    - The direction with the larger valid-corridor count is the tiling direction: for each such corridor, start one step past its border endpoint and repeat the rotated sprite's bounding box tightly along that corridor, clipping only the overrun past the far endpoint.
    - Valid corridors in the smaller-count direction have their entire interior recolored green; red endpoints and yellow blockers remain unchanged, and the original sprite is erased.

    Key insight:
    - One corridor-count comparison drives two coupled decisions at once: it chooses both the sprite orientation and which corridor family receives sprite tiling instead of a flat fill.
    """
    return _solve(grid)


def _random_sprite() -> np.ndarray:
    patch = TEMPLATES[random.randrange(len(TEMPLATES))].copy()
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    patch = np.rot90(patch, -random.randrange(4))
    return np.where(patch > 0, BLUE, BLACK).astype(int)


def _sample_separated(values: list[int], count: int, min_gap: int) -> list[int] | None:
    if count == 0:
        return []
    if len(values) < count:
        return None
    for _ in range(200):
        pool = values[:]
        random.shuffle(pool)
        chosen: list[int] = []
        for v in pool:
            if all(abs(v - old) >= min_gap for old in chosen):
                chosen.append(v)
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _sample_subset(values: list[int], count: int) -> list[int] | None:
    if count > len(values):
        return None
    return sorted(random.sample(values, count))


def _pick_sprite_origin(
    size: int,
    sprite_shape: tuple[int, int],
    forbidden_rows: set[int],
    forbidden_cols: set[int],
) -> tuple[int, int] | None:
    sh, sw = sprite_shape
    row_starts = list(range(2, size - sh))
    col_starts = list(range(2, size - sw))
    random.shuffle(row_starts)
    random.shuffle(col_starts)
    for r0 in row_starts:
        rows = set(range(r0, r0 + sh))
        if rows & forbidden_rows:
            continue
        for c0 in col_starts:
            cols = set(range(c0, c0 + sw))
            if cols & forbidden_cols:
                continue
            return r0, c0
    return None


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        out, meta = _solve(grid, return_meta=True)
    except AssertionError:
        return False

    if not np.array_equal(main(grid), out):
        return False

    if not np.any(out == BLUE) or not np.any(out == GREEN):
        return False

    r0, c0, r1, c1 = meta["sprite_box"]
    if np.any(out[r0 : r1 + 1, c0 : c1 + 1] == BLUE):
        return False

    if meta["candidate_row_count"] == len(meta["valid_rows"]) and meta["candidate_col_count"] == len(meta["valid_cols"]):
        return False
    if meta["candidate_row_count"] == meta["candidate_col_count"]:
        return False

    alt_rotation = _solve(grid, force_rotate=not meta["rotated"])
    if np.array_equal(alt_rotation, out):
        return False

    alt_modes = _solve(grid, swap_modes=True)
    if np.array_equal(alt_modes, out):
        return False

    alt_candidates = _solve(grid, treat_all_candidates=True)
    if np.array_equal(alt_candidates, out):
        return False

    if meta["horizontal_major"]:
        if any(int(out[r, 1]) != BLUE and int(out[r, 1]) != GREEN for r in meta["valid_rows"]):
            return False
    else:
        if any(int(out[1, c]) != BLUE and int(out[1, c]) != GREEN for c in meta["valid_cols"]):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        size = random.randint(12, 16)
        sprite = _random_sprite()
        sh, sw = sprite.shape
        horizontal_major = bool(random.getrandbits(1))
        tiled_sprite = np.rot90(sprite, -1) if horizontal_major else sprite
        th, tw = tiled_sprite.shape

        if horizontal_major:
            h_valid = random.randint(2, 3)
            v_valid = random.randint(1, h_valid - 1)
            invalid_rows = random.randint(1, 2)
            invalid_cols = random.randint(0, 1)

            valid_row_pool = list(range(1, size - th))
            valid_rows = _sample_separated(valid_row_pool, h_valid, th + 1)
            if valid_rows is None:
                continue

            majority_band_rows = set()
            for r in valid_rows:
                majority_band_rows.update(range(r, min(size - 1, r + th)))

            invalid_row_pool = [r for r in range(1, size - 1) if r not in majority_band_rows]
            invalid_row_list = _sample_subset(invalid_row_pool, invalid_rows)
            if invalid_row_list is None:
                continue

            valid_cols = _sample_subset(list(range(1, size - 1)), v_valid)
            if valid_cols is None:
                continue

            invalid_col_pool = [c for c in range(1, size - 1) if c not in valid_cols]
            invalid_col_list = _sample_subset(invalid_col_pool, invalid_cols)
            if invalid_col_list is None:
                continue

            blocker_rows_for_invalid_cols = [
                r
                for r in range(1, size - 1)
                if r not in valid_rows and r not in majority_band_rows
            ]
            if invalid_cols > 0 and not blocker_rows_for_invalid_cols:
                continue

            forbidden_rows = set(valid_rows) | set(invalid_row_list) | majority_band_rows
            forbidden_cols = set(valid_cols) | set(invalid_col_list)
        else:
            v_valid = random.randint(2, 3)
            h_valid = random.randint(1, v_valid - 1)
            invalid_cols = random.randint(1, 2)
            invalid_rows = random.randint(0, 1)

            valid_col_pool = list(range(1, size - tw))
            valid_cols = _sample_separated(valid_col_pool, v_valid, tw + 1)
            if valid_cols is None:
                continue

            majority_band_cols = set()
            for c in valid_cols:
                majority_band_cols.update(range(c, min(size - 1, c + tw)))

            invalid_col_pool = [c for c in range(1, size - 1) if c not in majority_band_cols]
            invalid_col_list = _sample_subset(invalid_col_pool, invalid_cols)
            if invalid_col_list is None:
                continue

            valid_rows = _sample_subset(list(range(1, size - 1)), h_valid)
            if valid_rows is None:
                continue

            invalid_row_pool = [r for r in range(1, size - 1) if r not in valid_rows]
            invalid_row_list = _sample_subset(invalid_row_pool, invalid_rows)
            if invalid_row_list is None:
                continue

            blocker_cols_for_invalid_rows = [
                c
                for c in range(1, size - 1)
                if c not in valid_cols and c not in majority_band_cols
            ]
            if invalid_rows > 0 and not blocker_cols_for_invalid_rows:
                continue

            forbidden_rows = set(valid_rows) | set(invalid_row_list)
            forbidden_cols = set(valid_cols) | set(invalid_col_list) | majority_band_cols

        all_rows = set(valid_rows) | set(invalid_row_list)
        all_cols = set(valid_cols) | set(invalid_col_list)
        if len(all_rows) == len(all_cols):
            continue

        sprite_origin = _pick_sprite_origin(size, (sh, sw), forbidden_rows, forbidden_cols)
        if sprite_origin is None:
            continue
        sr, sc = sprite_origin
        sprite_rows = set(range(sr, sr + sh))
        sprite_cols = set(range(sc, sc + sw))

        grid = np.zeros((size, size), dtype=int)
        for r in all_rows:
            grid[r, 0] = RED
            grid[r, -1] = RED
        for c in all_cols:
            grid[0, c] = RED
            grid[-1, c] = RED

        for r in invalid_row_list:
            if horizontal_major:
                blocker_cols = [c for c in range(1, size - 1) if c not in valid_cols and c not in sprite_cols]
            else:
                blocker_cols = [c for c in blocker_cols_for_invalid_rows if c not in sprite_cols]
            if not blocker_cols:
                break
            count = min(len(blocker_cols), random.randint(1, 2))
            for c in random.sample(blocker_cols, count):
                grid[r, c] = YELLOW
        else:
            for c in invalid_col_list:
                if horizontal_major:
                    blocker_rows = [r for r in blocker_rows_for_invalid_cols if r not in sprite_rows]
                else:
                    blocker_rows = [r for r in range(1, size - 1) if r not in valid_rows and r not in sprite_rows]
                if not blocker_rows:
                    break
                count = min(len(blocker_rows), random.randint(1, 2))
                for r in random.sample(blocker_rows, count):
                    grid[r, c] = YELLOW
            else:
                grid[sr : sr + sh, sc : sc + sw] = np.maximum(grid[sr : sr + sh, sc : sc + sw], sprite)
                if _semantic_checks(grid):
                    return grid

    raise RuntimeError("failed to generate a valid corridor-rotation-tiling input")
