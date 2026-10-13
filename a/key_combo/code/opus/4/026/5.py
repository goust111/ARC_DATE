from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
ORANGE = 7

META_SIZE = 3
CELL_SIZE = 3
GRID_SIZE = META_SIZE * CELL_SIZE + (META_SIZE + 1)
MAX_ATTEMPTS = 300


def _pattern_key(patch: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in patch.reshape(-1))


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(int(v) for v in coords[0])}
    stack = [next(iter(seen))]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]):
                nxt = (nr, nc)
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == int(mask.sum())


def _corridor_centers(active_mask: np.ndarray) -> set[tuple[int, int]]:
    centers: set[tuple[int, int]] = set()
    for r in range(META_SIZE):
        if bool(active_mask[r, 0]) and not bool(active_mask[r, 1]) and bool(active_mask[r, 2]):
            centers.add((r, 1))
    for c in range(META_SIZE):
        if bool(active_mask[0, c]) and not bool(active_mask[1, c]) and bool(active_mask[2, c]):
            centers.add((1, c))
    return centers


def _build_blueprint_catalog() -> dict[str, list[np.ndarray]]:
    catalog = {"horizontal": [], "vertical": [], "both": []}
    for bits in range(1 << (META_SIZE * META_SIZE)):
        if bits.bit_count() not in {4, 5}:
            continue
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        has_h = any(bool(mask[r, 0]) and not bool(mask[r, 1]) and bool(mask[r, 2]) for r in range(3))
        has_v = any(bool(mask[0, c]) and not bool(mask[1, c]) and bool(mask[2, c]) for c in range(3))
        if not (has_h or has_v):
            continue
        if has_h and has_v:
            catalog["both"].append(mask)
        elif has_h:
            catalog["horizontal"].append(mask)
        else:
            catalog["vertical"].append(mask)
    return catalog


def _build_majority_catalog() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1 << (CELL_SIZE * CELL_SIZE)):
        count = bits.bit_count()
        if count not in {2, 3, 4}:
            continue
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        if _is_connected(mask):
            masks.append(mask)
    return masks


BLUEPRINT_CATALOG = _build_blueprint_catalog()
MAJORITY_CATALOG = _build_majority_catalog()


def _separator_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_seps = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == grid[r, 0])]
    col_seps = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == grid[0, c])]
    assert len(row_seps) == META_SIZE + 1 and len(col_seps) == META_SIZE + 1

    row_steps = [b - a for a, b in zip(row_seps, row_seps[1:])]
    col_steps = [b - a for a, b in zip(col_seps, col_seps[1:])]
    assert len(set(row_steps)) == 1 and len(set(col_steps)) == 1
    assert row_steps[0] == CELL_SIZE + 1 and col_steps[0] == CELL_SIZE + 1

    row_blocks = [(row_seps[i] + 1, row_seps[i + 1] - 1) for i in range(META_SIZE)]
    col_blocks = [(col_seps[i] + 1, col_seps[i + 1] - 1) for i in range(META_SIZE)]
    assert all(r1 - r0 + 1 == CELL_SIZE for r0, r1 in row_blocks)
    assert all(c1 - c0 + 1 == CELL_SIZE for c0, c1 in col_blocks)
    return row_blocks, col_blocks


def _extract_cells(
    grid: np.ndarray,
    row_blocks: list[tuple[int, int]],
    col_blocks: list[tuple[int, int]],
) -> np.ndarray:
    cells = np.zeros((META_SIZE, META_SIZE, CELL_SIZE, CELL_SIZE), dtype=int)
    for mr, (r0, r1) in enumerate(row_blocks):
        for mc, (c0, c1) in enumerate(col_blocks):
            cells[mr, mc] = grid[r0 : r1 + 1, c0 : c1 + 1]
    return cells


def _cell_bounds(
    row_blocks: list[tuple[int, int]],
    col_blocks: list[tuple[int, int]],
    mr: int,
    mc: int,
) -> tuple[int, int, int, int]:
    r0, r1 = row_blocks[mr]
    c0, c1 = col_blocks[mc]
    return r0, r1, c0, c1


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE)
    row_blocks, col_blocks = _separator_layout(g)
    cells = _extract_cells(g, row_blocks, col_blocks)

    sigs = [_pattern_key(cells[mr, mc]) for mr in range(3) for mc in range(3)]
    counts = Counter(sigs)
    assert sorted(counts.values()) == [2, 7], "expected 7 majority cells and 2 minority cells"

    majority_key, minority_key = sorted(counts, key=lambda key: counts[key], reverse=True)
    majority_patch = np.array(majority_key, dtype=int).reshape(3, 3)
    minority_patch = np.array(minority_key, dtype=int).reshape(3, 3)

    assert set(np.unique(majority_patch)).issubset({BLACK, BLUE})
    assert set(np.unique(minority_patch)).issubset({BLACK, RED})

    observed = np.zeros((3, 3), dtype=bool)
    for mr in range(3):
        for mc in range(3):
            observed[mr, mc] = _pattern_key(cells[mr, mc]) == minority_key

    active_mask = minority_patch == RED
    assert int(active_mask.sum()) in {4, 5}
    assert int(observed.sum()) == 2
    assert np.all(~observed | active_mask)

    return {
        "grid": g,
        "row_blocks": row_blocks,
        "col_blocks": col_blocks,
        "majority_patch": majority_patch,
        "minority_patch": minority_patch,
        "observed": observed,
        "active_mask": active_mask,
    }


def _render_output(parsed: dict) -> np.ndarray:
    out = parsed["grid"].copy()
    row_blocks = parsed["row_blocks"]
    col_blocks = parsed["col_blocks"]
    majority_patch = parsed["majority_patch"]
    minority_patch = parsed["minority_patch"]
    active_mask = parsed["active_mask"]

    for mr in range(3):
        for mc in range(3):
            r0, r1, c0, c1 = _cell_bounds(row_blocks, col_blocks, mr, mc)
            out[r0 : r1 + 1, c0 : c1 + 1] = minority_patch if bool(active_mask[mr, mc]) else majority_patch

    for mr, mc in _corridor_centers(active_mask):
        r0, r1, c0, c1 = _cell_bounds(row_blocks, col_blocks, mr, mc)
        cell = out[r0 : r1 + 1, c0 : c1 + 1]
        cell[majority_patch == BLACK] = GREEN

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the minority-cell fractal blueprint, then gate-fill majority corridor cells with green.

    Input structure:
    - The grid is a 3x3 arrangement of 3x3 inner cells separated by one-pixel orange lines.
    - Seven cells share one blue-on-black majority pattern, while two cells share one red-on-black minority pattern.
    - The red pixels inside a minority cell form a 3x3 blueprint whose active positions name where minority cells should appear in the outer 3x3 cell layout.

    Rule:
    - Read either minority cell's 3x3 interior as a binary blueprint: red positions are active, black positions are inactive.
    - Replace every active outer cell position with the minority pattern; inactive positions stay as the majority pattern.
    - After that completion, any row or column with minority cells at both ends and a majority cell in the middle is a corridor.
    - In each corridor's middle majority cell, recolor only the black background pixels to green, keeping the blue majority pixels unchanged.

    Key insight:
    - The green fill is not triggered by the original layout alone; it only appears after using the minority cell as a self-similar blueprint that reveals which middle cells are true minority corridors.
    """
    parsed = _parse_input(np.asarray(grid, dtype=int))
    return _render_output(parsed)


def _choose_blueprint() -> np.ndarray:
    category = random.choice(["horizontal", "vertical", "both"])
    return random.choice(BLUEPRINT_CATALOG[category]).copy()


def _choose_majority_mask(blueprint: np.ndarray) -> np.ndarray:
    candidates = []
    for mask in MAJORITY_CATALOG:
        if np.sum(mask != blueprint) >= 2:
            candidates.append(mask)
    return random.choice(candidates).copy()


def _observed_positions(active_mask: np.ndarray) -> list[tuple[int, int]]:
    coords = sorted((int(r), int(c)) for r, c in np.argwhere(active_mask))
    return [coords[0], coords[-1]]


def _render_input_from_parts(majority_patch: np.ndarray, minority_patch: np.ndarray) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), ORANGE, dtype=int)
    active_mask = minority_patch == RED
    observed = set(_observed_positions(active_mask))

    for mr in range(3):
        for mc in range(3):
            r0 = 1 + mr * (CELL_SIZE + 1)
            c0 = 1 + mc * (CELL_SIZE + 1)
            patch = minority_patch if (mr, mc) in observed else majority_patch
            grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = patch
    return grid


def _validate_generated_instance(grid: np.ndarray) -> None:
    parsed = _parse_input(grid)
    majority_patch = parsed["majority_patch"]
    minority_patch = parsed["minority_patch"]
    observed = parsed["observed"]
    active_mask = parsed["active_mask"]
    centers = _corridor_centers(active_mask)

    assert int(active_mask.sum()) in {4, 5}
    assert int(observed.sum()) == 2
    assert int((active_mask & ~observed).sum()) in {2, 3}
    assert centers, "blueprint completion must create at least one corridor"
    assert np.any(majority_patch == BLACK), "corridor fill must have black pixels to recolor"
    assert set(np.unique(majority_patch)).issubset({BLACK, BLUE})
    assert set(np.unique(minority_patch)).issubset({BLACK, RED})

    expected = _render_output(parsed)
    actual = main(grid)
    assert np.array_equal(actual, expected)

    row_blocks = parsed["row_blocks"]
    col_blocks = parsed["col_blocks"]
    for mr in range(3):
        for mc in range(3):
            r0, r1, c0, c1 = _cell_bounds(row_blocks, col_blocks, mr, mc)
            patch = actual[r0 : r1 + 1, c0 : c1 + 1]
            if bool(active_mask[mr, mc]):
                assert np.array_equal(patch, minority_patch)
            elif (mr, mc) in centers:
                expected_patch = majority_patch.copy()
                expected_patch[expected_patch == BLACK] = GREEN
                assert np.array_equal(patch, expected_patch)
            else:
                assert np.array_equal(patch, majority_patch)


def _gen_attempt() -> np.ndarray:
    blueprint = _choose_blueprint()
    majority_mask = _choose_majority_mask(blueprint)
    majority_patch = np.where(majority_mask, BLUE, BLACK)
    minority_patch = np.where(blueprint, RED, BLACK)
    grid = _render_input_from_parts(majority_patch, minority_patch)
    _validate_generated_instance(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            return _gen_attempt()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid corridor-blueprint instance")
