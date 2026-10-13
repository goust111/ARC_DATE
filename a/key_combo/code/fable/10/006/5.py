from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
ROWS_RANGE = (2, 4)
COLS_RANGE = (2, 4)
CELL_SIDE_RANGE = (4, 6)
TEMPLATE_COLOR_COUNT_RANGE = (2, 3)
TEMPLATE_FG_RANGE = (7, 12)
MAX_ATTEMPTS = 500

_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _orientation_arrays(arr: np.ndarray) -> list[tuple[int, np.ndarray]]:
    variants = [
        (0, arr),
        (1, np.rot90(arr, 1)),
        (2, np.rot90(arr, 2)),
        (3, np.rot90(arr, 3)),
        (4, np.fliplr(arr)),
        (5, np.rot90(np.fliplr(arr), 1)),
        (6, np.rot90(np.fliplr(arr), 2)),
        (7, np.rot90(np.fliplr(arr), 3)),
    ]
    unique: list[tuple[int, np.ndarray]] = []
    seen: set[bytes] = set()
    for idx, variant in variants:
        key = np.asarray(variant, dtype=int).tobytes()
        if key not in seen:
            seen.add(key)
            unique.append((idx, np.asarray(variant, dtype=int).copy()))
    return unique


def _separator_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates: list[int] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        row_hits = [r for r in range(h) if np.all(grid[r, :] == color)]
        col_hits = [c for c in range(w) if np.all(grid[:, c] == color)]
        if row_hits and col_hits:
            candidates.append(color)
    assert len(candidates) == 1, f"expected one separator color, got {candidates}"
    return candidates[0]


def _bands(size: int, lines: Iterable[int]) -> list[tuple[int, int]]:
    ordered = sorted(int(x) for x in lines)
    edges = [-1, *ordered, size]
    bands = [(a + 1, b) for a, b in zip(edges, edges[1:])]
    assert all(start < end for start, end in bands), bands
    return bands


def _cell_is_residue(cell: np.ndarray, variant: np.ndarray) -> bool:
    visible = cell != BG
    return bool(np.any(visible) and np.array_equal(cell[visible], variant[visible]))


def _cell_regions(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    sep = _separator_color(grid)
    row_lines = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == sep)]
    col_lines = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == sep)]
    row_bands = _bands(grid.shape[0], row_lines)
    col_bands = _bands(grid.shape[1], col_lines)
    heights = {r1 - r0 for r0, r1 in row_bands}
    widths = {c1 - c0 for c0, c1 in col_bands}
    assert len(heights) == 1 and len(widths) == 1
    assert next(iter(heights)) == next(iter(widths))
    return sep, row_bands, col_bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair every separator-grid cell by completing the uniquely oriented residue of the full template.

    Input structure:
    - Full horizontal and vertical separator lines split a black-background board into equal square cells.
    - Exactly one cell contains the complete multicolor template on that square cell canvas.
    - Every other cell contains a non-empty strict subset of one rotated or reflected copy of the same template.

    Rule:
    - Ignore the separator lines and extract the unique cell with the largest number of non-black pixels as the template.
    - Enumerate all rotations and reflections of that full cell-sized template.
    - For each incomplete cell, find the unique orientation whose colored pixels include every visible residue pixel at the same cell coordinates, then fill the whole cell with that orientation.
    - Keep the separator lines unchanged.

    Key insight:
    - The residue pixels are not noise; their positions inside the cell square are an orientation signature that selects which transformed copy must be restored.
    """
    g = np.asarray(grid, dtype=int)
    sep, row_bands, col_bands = _cell_regions(g)
    _ = sep

    cells: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    counts: list[int] = []
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            cell = g[r0:r1, c0:c1].copy()
            assert not np.any(cell == _separator_color(g))
            cells.append(((r0, r1, c0, c1), cell))
            counts.append(int(np.count_nonzero(cell != BG)))

    max_count = max(counts)
    template_indices = [i for i, count in enumerate(counts) if count == max_count]
    assert len(template_indices) == 1, "expected one uniquely complete template cell"
    template = cells[template_indices[0]][1]
    variants = _orientation_arrays(template)

    out = g.copy()
    for idx, ((r0, r1, c0, c1), cell) in enumerate(cells):
        if idx == template_indices[0]:
            fill = template
        else:
            matches = [variant for _variant_id, variant in variants if _cell_is_residue(cell, variant)]
            assert len(matches) == 1, "each residue must select exactly one orientation"
            fill = matches[0]
        out[r0:r1, c0:c1] = fill
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _connected_shape(side: int, area: int) -> list[tuple[int, int]]:
    cells = {(random.randrange(side), random.randrange(side))}
    while len(cells) < area:
        r, c = random.choice(tuple(cells))
        neighbors = [
            (r - 1, c),
            (r + 1, c),
            (r, c - 1),
            (r, c + 1),
        ]
        random.shuffle(neighbors)
        placed = False
        for rr, cc in neighbors:
            if 0 <= rr < side and 0 <= cc < side and (rr, cc) not in cells:
                cells.add((rr, cc))
                placed = True
                break
        if not placed:
            frontier = []
            for rr, cc in cells:
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                raise RuntimeError("could not grow connected shape")
            cells.add(random.choice(frontier))
    return list(cells)


def _random_template(side: int, sep_color: int) -> np.ndarray | None:
    color_count = random.randint(*TEMPLATE_COLOR_COUNT_RANGE)
    colors = random.sample([c for c in PALETTE if c != sep_color], color_count)
    area_hi = min(TEMPLATE_FG_RANGE[1], side * side - 3)
    area_lo = min(area_hi, max(TEMPLATE_FG_RANGE[0], side + 2))
    fg_count = random.randint(area_lo, area_hi)
    coords = _connected_shape(side, fg_count)

    arr = np.zeros((side, side), dtype=int)
    shuffled = coords[:]
    random.shuffle(shuffled)
    for color, (r, c) in zip(colors, shuffled):
        arr[r, c] = color
    for r, c in shuffled[len(colors):]:
        arr[r, c] = random.choice(colors)

    box = _bbox(arr != BG)
    assert box is not None
    r0, c0, r1, c1 = box
    if (r1 - r0 + 1) < 3 or (c1 - c0 + 1) < 3:
        return None
    if len(_orientation_arrays(arr)) != 8:
        return None
    return arr


def _compatible_ids(residue: np.ndarray, variants: list[tuple[int, np.ndarray]]) -> list[int]:
    hits = []
    for variant_id, variant in variants:
        if _cell_is_residue(residue, variant):
            hits.append(int(variant_id))
    return hits


def _unique_residue(variant_id: int, variants: list[tuple[int, np.ndarray]]) -> np.ndarray | None:
    variant_map = {idx: arr for idx, arr in variants}
    target = variant_map[variant_id]
    coords = [tuple(map(int, rc)) for rc in np.argwhere(target != BG)]
    if len(coords) < 4:
        return None

    selected: list[tuple[int, int]] = []
    survivors = {idx for idx, _arr in variants if idx != variant_id}
    unused = coords[:]

    while survivors:
        best_coord = None
        best_removed: set[int] = set()
        random.shuffle(unused)
        for r, c in unused:
            removed = {
                other_id
                for other_id in survivors
                if int(variant_map[other_id][r, c]) != int(target[r, c])
            }
            if len(removed) > len(best_removed):
                best_coord = (r, c)
                best_removed = removed
        if best_coord is None or not best_removed:
            return None
        selected.append(best_coord)
        unused.remove(best_coord)
        survivors -= best_removed

    max_keep = len(coords) - 2
    if len(selected) > max_keep:
        return None
    wanted = random.randint(max(2, len(selected)), max_keep)
    random.shuffle(unused)
    selected.extend(unused[: wanted - len(selected)])

    residue = np.zeros_like(target)
    for r, c in selected:
        residue[r, c] = target[r, c]
    if _compatible_ids(residue, variants) != [variant_id]:
        return None
    return residue


def _grid_shape(row_cells: int, col_cells: int, side: int) -> tuple[int, int]:
    return row_cells * side + (row_cells - 1), col_cells * side + (col_cells - 1)


def _compose_grid(
    row_cells: int,
    col_cells: int,
    side: int,
    sep_color: int,
    cell_contents: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    h, w = _grid_shape(row_cells, col_cells, side)
    grid = np.zeros((h, w), dtype=int)
    for rr in range(1, row_cells):
        grid[rr * side + rr - 1, :] = sep_color
    for cc in range(1, col_cells):
        grid[:, cc * side + cc - 1] = sep_color
    for br in range(row_cells):
        for bc in range(col_cells):
            r0 = br * (side + 1)
            c0 = bc * (side + 1)
            grid[r0:r0 + side, c0:c0 + side] = cell_contents[(br, bc)]
    return grid


def _choose_orientations(count: int) -> list[int]:
    distinct = random.randint(3, min(5, count, 8))
    rotation_id = random.choice([0, 1, 2, 3])
    reflection_id = random.choice([4, 5, 6, 7])
    rest_pool = [idx for idx in range(8) if idx not in {rotation_id, reflection_id}]
    chosen = [rotation_id, reflection_id]
    chosen.extend(random.sample(rest_pool, distinct - 2))
    random.shuffle(chosen)
    assignments = chosen[:]
    while len(assignments) < count:
        assignments.append(random.choice(chosen))
    random.shuffle(assignments)
    return assignments


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    template_pos: tuple[int, int],
    completion_ids: dict[tuple[int, int], int],
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    sep, row_bands, col_bands = _cell_regions(inp)
    if sep == BG:
        return False

    cells_in: dict[tuple[int, int], np.ndarray] = {}
    counts = {}
    for br, (r0, r1) in enumerate(row_bands):
        for bc, (c0, c1) in enumerate(col_bands):
            cell = inp[r0:r1, c0:c1]
            cells_in[(br, bc)] = cell
            counts[(br, bc)] = int(np.count_nonzero(cell != BG))

    template_count = counts[template_pos]
    if sum(count == template_count for count in counts.values()) != 1:
        return False

    template = cells_in[template_pos]
    variants = _orientation_arrays(template)
    if len(variants) != 8:
        return False

    used_ids = set(completion_ids.values())
    if len(used_ids) < 3:
        return False
    if not any(idx < 4 for idx in used_ids):
        return False
    if not any(idx >= 4 for idx in used_ids):
        return False

    variant_map = {idx: arr for idx, arr in variants}
    if not any(not np.array_equal(arr, template) for arr in variant_map.values()):
        return False

    for pos, orientation_id in completion_ids.items():
        residue = cells_in[pos]
        if counts[pos] >= template_count:
            return False
        hits = _compatible_ids(residue, variants)
        if hits != [orientation_id]:
            return False
        if np.array_equal(residue, variant_map[orientation_id]):
            return False
        if not np.array_equal(expected[
            pos[0] * (template.shape[0] + 1): pos[0] * (template.shape[0] + 1) + template.shape[0],
            pos[1] * (template.shape[1] + 1): pos[1] * (template.shape[1] + 1) + template.shape[1],
        ], variant_map[orientation_id]):
            return False

    if not any(orientation_id != 0 for orientation_id in completion_ids.values()):
        return False
    return True


def _attempt() -> tuple[np.ndarray, np.ndarray] | None:
    row_cells = random.randint(*ROWS_RANGE)
    col_cells = random.randint(*COLS_RANGE)
    if row_cells * col_cells < 4:
        return None

    side = random.randint(*CELL_SIDE_RANGE)
    if max(_grid_shape(row_cells, col_cells, side)) > 30:
        return None

    sep_color = random.choice(PALETTE)
    template = _random_template(side, sep_color)
    if template is None:
        return None

    variants = _orientation_arrays(template)
    if len(variants) != 8:
        return None
    variant_map = {idx: arr for idx, arr in variants}

    template_pos = (random.randrange(row_cells), random.randrange(col_cells))
    other_positions = [
        (br, bc)
        for br in range(row_cells)
        for bc in range(col_cells)
        if (br, bc) != template_pos
    ]
    assignments = _choose_orientations(len(other_positions))

    input_cells: dict[tuple[int, int], np.ndarray] = {template_pos: template.copy()}
    output_cells: dict[tuple[int, int], np.ndarray] = {template_pos: template.copy()}
    completion_ids: dict[tuple[int, int], int] = {}

    for pos, orientation_id in zip(other_positions, assignments):
        residue = _unique_residue(orientation_id, variants)
        if residue is None:
            return None
        input_cells[pos] = residue
        output_cells[pos] = variant_map[orientation_id].copy()
        completion_ids[pos] = orientation_id

    inp = _compose_grid(row_cells, col_cells, side, sep_color, input_cells)
    expected = _compose_grid(row_cells, col_cells, side, sep_color, output_cells)

    output_key = (expected.shape, expected.tobytes())
    if output_key in _SEEN_OUTPUTS:
        return None
    if not _validate_instance(inp, expected, template_pos, completion_ids):
        return None

    _SEEN_OUTPUTS.add(output_key)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        sample = _attempt()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid residue-orientation repair input")
