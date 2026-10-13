from __future__ import annotations

import random
from typing import Iterable

import numpy as np


EMPTY = 0
BLUE = 1
RED = 2
YELLOW = 4
PINK = 6

CELL_SIZE = 3
STRIDE = 4
CELL_ROWS = 6
CELL_COLS = 5
GRID_SHAPE = (23, 19)
ROW_STARTS = [i * STRIDE for i in range(CELL_ROWS)]
COL_STARTS = [i * STRIDE for i in range(CELL_COLS)]
SEP_ROWS = [r + CELL_SIZE for r in ROW_STARTS[:-1]]
SEP_COLS = [c + CELL_SIZE for c in COL_STARTS[:-1]]

INSTR_R0 = 1
INSTR_C0 = 12
INSTR_CELLS = {(0, 3), (0, 4), (1, 3), (1, 4)}
SPRITE_ROWS = (0, 1)
SPRITE_COLS = (0, 1, 2)

MAX_GENERATION_ATTEMPTS = 400

SPRITE_BASES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox_key(coords: np.ndarray) -> tuple[int, int]:
    return int(coords[:, 0].min()), int(coords[:, 1].min())


def _cell_origin(cell_r: int, cell_c: int) -> tuple[int, int]:
    return ROW_STARTS[cell_r], COL_STARTS[cell_c]


def _decode_mask(grid: np.ndarray) -> np.ndarray:
    block = grid[INSTR_R0 : INSTR_R0 + 6, INSTR_C0 : INSTR_C0 + 6]
    mask = np.zeros((3, 3), dtype=bool)
    for mr in range(3):
        for mc in range(3):
            sub = block[2 * mr : 2 * mr + 2, 2 * mc : 2 * mc + 2]
            if np.all(sub == PINK):
                mask[mr, mc] = True
            else:
                assert np.all(sub == EMPTY), "instruction blocks must be fully pink or fully empty"
    assert mask.any(), "decoded mask must contain at least one active position"
    return mask


def _build_terrain(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    terrain = grid.copy()
    for cell_r, r0 in enumerate(ROW_STARTS):
        for cell_c, c0 in enumerate(COL_STARTS):
            if (cell_r, cell_c) in INSTR_CELLS:
                continue
            cell = grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE]
            terrain_cell = terrain[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE]
            if np.any(cell == RED):
                terrain_cell[terrain_cell == BLUE] = EMPTY
                continue
            if np.all(cell[mask] == BLUE):
                terrain_cell[:] = PINK
            else:
                terrain_cell[terrain_cell == BLUE] = EMPTY

    terrain[terrain == RED] = EMPTY
    return terrain


def _occupancy(shape: tuple[int, int], objects: list[np.ndarray]) -> np.ndarray:
    occ = np.full(shape, -1, dtype=int)
    for idx, coords in enumerate(objects):
        assert np.all(occ[coords[:, 0], coords[:, 1]] == -1), "sprites must not overlap"
        occ[coords[:, 0], coords[:, 1]] = idx
    return occ


def _movable_down(objects: list[np.ndarray], occ: np.ndarray, terrain: np.ndarray) -> set[int]:
    h, _w = terrain.shape
    movable = set(range(len(objects)))
    changed = True
    while changed:
        changed = False
        for idx in list(movable):
            shifted = objects[idx] + np.array([1, 0], dtype=int)
            if np.any(shifted[:, 0] >= h):
                movable.remove(idx)
                changed = True
                continue
            if np.any(terrain[shifted[:, 0], shifted[:, 1]] == PINK):
                movable.remove(idx)
                changed = True
                continue

            labels = occ[shifted[:, 0], shifted[:, 1]]
            blockers = {int(label) for label in labels if int(label) not in {-1, idx}}
            if any(blocker not in movable for blocker in blockers):
                movable.remove(idx)
                changed = True

        claims: dict[tuple[int, int], list[int]] = {}
        for idx in movable:
            shifted = objects[idx] + np.array([1, 0], dtype=int)
            for r, c in shifted:
                claims.setdefault((int(r), int(c)), []).append(idx)

        losers = set()
        for group in claims.values():
            if len(group) <= 1:
                continue
            winner = min(group, key=lambda i: (int(objects[i][:, 1].min()), i))
            losers.update(idx for idx in group if idx != winner)
        if losers:
            movable.difference_update(losers)
            changed = True

    return movable


def _simulate_sprites(grid: np.ndarray, terrain: np.ndarray) -> list[np.ndarray]:
    sprites = _components_mask(grid == RED)
    sprites = [coords.copy() for coords in sorted(sprites, key=_bbox_key)]
    assert len(sprites) == 2, "expected exactly two red sprites"

    for _step in range(grid.shape[0]):
        occ = _occupancy(grid.shape, sprites)
        movable = _movable_down(sprites, occ, terrain)
        if not movable:
            break
        for idx in movable:
            sprites[idx][:, 0] += 1
    else:
        raise AssertionError("gravity failed to settle inside the grid height")

    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the instruction mask into solid obstacle cells, then let the top red sprites fall through the resulting terrain.

    Input structure:
    - A 23x19 lattice is split by yellow full-row/full-column grid lines into 6x5 cells of size 3x3.
    - The upper-right 2x2 macro-cells are merged into a 6x6 instruction region containing pink 2x2 blocks that encode a 3x3 binary mask.
    - Every non-instruction cell contains either blue pixels on black, or one of two small red 4-connected sprites placed in the top two cell rows.

    Rule:
    - Decode the 6x6 instruction area as a 3x3 mask: each 2x2 sub-block is active iff it is fully pink.
    - For each non-instruction 3x3 cell without a red sprite, if every active mask position is blue then replace the whole cell by solid pink; otherwise erase its blue pixels to black.
    - Treat the two red sprites as rigid bodies and move them downward synchronously one row at a time through black/yellow space until the next step would hit pink terrain, a sprite that has already settled, or the grid bottom.
    - Yellow grid lines stay visible unless a falling red sprite passes through and covers them; the instruction region stays unchanged.

    Key insight:
    - The mask does not recolor matching blue pixels directly; instead it turns whole cells into solid collision terrain, and that decoded terrain determines where the red sprites finally come to rest.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == GRID_SHAPE, f"expected {GRID_SHAPE}, got {g.shape}"

    mask = _decode_mask(g)
    terrain = _build_terrain(g, mask)
    sprites = _simulate_sprites(g, terrain)

    out = terrain.copy()
    for coords in sprites:
        assert not np.any(out[coords[:, 0], coords[:, 1]] == PINK), "sprites cannot overlap obstacles"
        out[coords[:, 0], coords[:, 1]] = RED
    return out


def _instruction_block(mask: np.ndarray) -> np.ndarray:
    return np.kron(mask.astype(int), np.full((2, 2), PINK, dtype=int))


def _make_base_grid(mask: np.ndarray) -> np.ndarray:
    grid = np.zeros(GRID_SHAPE, dtype=int)
    for sep in SEP_ROWS:
        grid[sep, :] = YELLOW
    for sep in SEP_COLS:
        grid[:, sep] = YELLOW

    grid[0, 11:19] = YELLOW
    grid[7, 11:19] = YELLOW
    grid[0:8, 11] = YELLOW
    grid[0:8, 18] = YELLOW
    grid[INSTR_R0 : INSTR_R0 + 6, INSTR_C0 : INSTR_C0 + 6] = _instruction_block(mask)
    return grid


def _random_mask(rng: np.random.Generator) -> np.ndarray:
    bit_count = int(rng.integers(2, 6))
    flat = np.zeros(9, dtype=bool)
    flat[rng.choice(9, size=bit_count, replace=False)] = True
    return flat.reshape(3, 3)


def _random_sprite_shape(rng: np.random.Generator) -> np.ndarray:
    base = np.array(random.choice(SPRITE_BASES), dtype=int)
    h = int(base[:, 0].max()) + 1
    w = int(base[:, 1].max()) + 1
    dr = int(rng.integers(0, CELL_SIZE - h + 1))
    dc = int(rng.integers(0, CELL_SIZE - w + 1))
    return base + np.array([dr, dc], dtype=int)


def _global_sprite_coords(cell_r: int, cell_c: int, local_coords: np.ndarray) -> np.ndarray:
    r0, c0 = _cell_origin(cell_r, cell_c)
    return local_coords + np.array([r0, c0], dtype=int)


def _matching_cell(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cell = np.zeros((CELL_SIZE, CELL_SIZE), dtype=int)
    cell[mask] = BLUE
    extra_positions = np.argwhere(~mask)
    if len(extra_positions):
        extra_count = int(rng.integers(0, len(extra_positions) + 1))
        if extra_count:
            chosen = extra_positions[rng.choice(len(extra_positions), size=extra_count, replace=False)]
            cell[chosen[:, 0], chosen[:, 1]] = BLUE
    return cell


def _nonmatching_cell(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    cell = np.zeros((CELL_SIZE, CELL_SIZE), dtype=int)
    mask_idx = np.flatnonzero(mask.reshape(-1))
    off_idx = np.flatnonzero(~mask.reshape(-1))

    if rng.random() < 0.12:
        return cell

    if rng.random() < 0.45 and len(mask_idx) >= 2:
        missing = int(rng.choice(mask_idx))
        keep = [idx for idx in mask_idx if idx != missing]
        if keep:
            cell.reshape(-1)[keep] = BLUE
        if len(off_idx):
            extra_count = int(rng.integers(0, min(4, len(off_idx)) + 1))
            if extra_count:
                chosen = rng.choice(off_idx, size=extra_count, replace=False)
                cell.reshape(-1)[chosen] = BLUE
    else:
        blocked = int(rng.choice(mask_idx))
        allowed = np.array([idx for idx in range(9) if idx != blocked], dtype=int)
        count = int(rng.integers(1, len(allowed) + 1))
        chosen = rng.choice(allowed, size=count, replace=False)
        cell.reshape(-1)[chosen] = BLUE

    assert not np.all(cell[mask] == BLUE), "non-matching cells must miss at least one mask position"
    return cell


def _all_content_cells() -> list[tuple[int, int]]:
    return [
        (cell_r, cell_c)
        for cell_r in range(CELL_ROWS)
        for cell_c in range(CELL_COLS)
        if (cell_r, cell_c) not in INSTR_CELLS
    ]


def _pick_sprite_layout(
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int, np.ndarray]], bool]:
    shared_column = bool(rng.integers(0, 2))
    if shared_column:
        col = int(rng.choice(SPRITE_COLS))
        cols = [col, col]
    else:
        cols = list(rng.choice(SPRITE_COLS, size=2, replace=False))

    for _ in range(50):
        upper = _random_sprite_shape(rng)
        lower = _random_sprite_shape(rng)
        upper_global = _global_sprite_coords(SPRITE_ROWS[0], cols[0], upper)
        lower_global = _global_sprite_coords(SPRITE_ROWS[1], cols[1], lower)
        if shared_column:
            if set(map(int, upper_global[:, 1])) & set(map(int, lower_global[:, 1])):
                return [(0, cols[0], upper), (1, cols[1], lower)], True
        else:
            return [(0, cols[0], upper), (1, cols[1], lower)], False
    raise RuntimeError("failed to place compatible sprite layouts")


def _sample_matching_cells(
    rng: np.random.Generator, sprite_cols: Iterable[int], sprite_branch: bool
) -> set[tuple[int, int]]:
    sprite_cols = list(sprite_cols)
    matching = set()
    for col in sorted(set(sprite_cols)):
        lo = 4 if sprite_branch and sprite_cols.count(col) == 2 else 2
        row = int(rng.integers(lo, CELL_ROWS))
        matching.add((row, col))

    candidates = [
        cell
        for cell in _all_content_cells()
        if cell not in matching and cell[0] >= 1
    ]
    extra_count = int(rng.integers(2, 8))
    extra_count = min(extra_count, len(candidates))
    if extra_count:
        chosen_idx = rng.choice(len(candidates), size=extra_count, replace=False)
        for idx in np.atleast_1d(chosen_idx):
            matching.add(candidates[int(idx)])
    return matching


def _paint_cell(grid: np.ndarray, cell_r: int, cell_c: int, patch: np.ndarray) -> None:
    r0, c0 = _cell_origin(cell_r, cell_c)
    grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = patch


def _validate_generated(
    inp: np.ndarray,
    mask: np.ndarray,
    matching_cells: set[tuple[int, int]],
    sprite_specs: list[tuple[int, int, np.ndarray]],
) -> bool:
    if inp.shape != GRID_SHAPE:
        return False

    if not np.array_equal(_decode_mask(inp), mask):
        return False

    red_components = sorted(_components_mask(inp == RED), key=_bbox_key)
    if len(red_components) != 2:
        return False

    for cell_r, cell_c in _all_content_cells():
        if (cell_r, cell_c) in INSTR_CELLS:
            continue
        r0, c0 = _cell_origin(cell_r, cell_c)
        cell = inp[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE]
        if np.any(cell == RED):
            continue
        is_match = np.all(cell[mask] == BLUE)
        if is_match != ((cell_r, cell_c) in matching_cells):
            return False

    out = main(inp)
    if np.any(out == BLUE):
        return False
    if not np.array_equal(out[INSTR_R0 : INSTR_R0 + 6, INSTR_C0 : INSTR_C0 + 6], inp[INSTR_R0 : INSTR_R0 + 6, INSTR_C0 : INSTR_C0 + 6]):
        return False

    for cell_r, cell_c in matching_cells:
        r0, c0 = _cell_origin(cell_r, cell_c)
        if not np.all(out[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] == PINK):
            return False

    final_red = _components_mask(out == RED)
    if not final_red:
        return False

    terrain = out.copy()
    terrain[terrain == RED] = EMPTY
    for coords in final_red:
        shifted = coords + np.array([1, 0], dtype=int)
        if np.any(shifted[:, 0] >= GRID_SHAPE[0]):
            continue
        blocked = terrain[shifted[:, 0], shifted[:, 1]] == PINK
        red_below = out[shifted[:, 0], shifted[:, 1]] == RED
        if not np.any(blocked | red_below):
            return False

    initial_min_rows = sorted(int(coords[:, 0].min()) for coords in red_components)
    final_min_rows = sorted(int(coords[:, 0].min()) for coords in final_red)
    if max(final_min_rows) <= max(initial_min_rows):
        return False

    # Keep at least one obstacle not directly under a sprite column so mask decoding matters globally.
    sprite_cols = {cell_c for _, cell_c, _ in sprite_specs}
    if not any(cell_c not in sprite_cols for _cell_r, cell_c in matching_cells):
        return False

    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    mask = _random_mask(rng)
    grid = _make_base_grid(mask)

    sprite_specs, shared_branch = _pick_sprite_layout(rng)
    sprite_cells = {(cell_r, cell_c) for cell_r, cell_c, _coords in sprite_specs}
    matching_cells = _sample_matching_cells(rng, [cell_c for _cell_r, cell_c, _coords in sprite_specs], shared_branch)

    for cell_r, cell_c in _all_content_cells():
        if (cell_r, cell_c) in sprite_cells:
            continue
        patch = _matching_cell(mask, rng) if (cell_r, cell_c) in matching_cells else _nonmatching_cell(mask, rng)
        _paint_cell(grid, cell_r, cell_c, patch)

    for cell_r, cell_c, local_coords in sprite_specs:
        r0, c0 = _cell_origin(cell_r, cell_c)
        patch = np.zeros((CELL_SIZE, CELL_SIZE), dtype=int)
        patch[local_coords[:, 0], local_coords[:, 1]] = RED
        grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = patch

    if not _validate_generated(grid, mask, matching_cells, sprite_specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mask-gravity terrain input")
