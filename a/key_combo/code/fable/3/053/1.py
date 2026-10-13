from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
TEMPLATE_COLORS = [1, 2, 3, 6, 7, 8, 9]
GENERATION_RETRIES = 400
MAX_GRID_SIDE = 30

_ANCHOR_CACHE: dict[int, list[tuple[int, int]]] = {}


def _bands(separators: list[int], size: int) -> list[tuple[int, int]]:
    separators = sorted(int(s) for s in separators)
    starts = [0] + [s + 1 for s in separators]
    ends = separators + [int(size)]
    bands = [(lo, hi) for lo, hi in zip(starts, ends) if lo < hi]
    assert bands
    return bands


def _dihedral_variants(tile: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(tile, k=k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def _yellow_pos(tile: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(tile == YELLOW)
    assert coords.shape == (1, 2)
    return tuple(map(int, coords[0]))


def _slot_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    h, w = map(int, grid.shape)
    row_seps = [r for r in range(h) if np.all(grid[r, :] == GREY)]
    col_seps = [c for c in range(w) if np.all(grid[:, c] == GREY)]

    row_bands = _bands(row_seps, h)
    col_bands = _bands(col_seps, w)
    row_sizes = {hi - lo for lo, hi in row_bands}
    col_sizes = {hi - lo for lo, hi in col_bands}
    assert len(row_sizes) == 1 and len(col_sizes) == 1
    slot_size = int(next(iter(row_sizes)))
    assert slot_size == int(next(iter(col_sizes)))
    return row_bands, col_bands, slot_size


def _variant_map(template_slot: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    mapping: dict[tuple[int, int], np.ndarray] = {}
    for variant in _dihedral_variants(template_slot):
        pos = _yellow_pos(variant)
        assert pos not in mapping
        mapping[pos] = variant
    assert len(mapping) == 8
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the template into each slot using the orientation whose yellow anchor lands on the marker.

    Input structure:
    - Full grey divider rows and columns split the canvas into equal square slots.
    - Exactly one slot contains a multicolor template on black background with exactly one yellow anchor pixel.
    - Every other slot is otherwise black and contains exactly one yellow marker pixel.

    Rule:
    - Derive the slot lattice from the full grey divider lines and locate the unique template slot.
    - Enumerate the template slot's eight dihedral orientations (four rotations, each with or without mirroring).
    - For every marker-only slot, choose the unique orientation whose yellow pixel lands on that marker position and copy the whole oriented template into the slot.

    Key insight:
    - The yellow pixel is not preserved as a standalone marker; its local position uniquely indexes the template orientation, so one marker pixel determines the full transformed slot.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_bands, col_bands, _slot_size = _slot_layout(g)

    template_slot: np.ndarray | None = None
    marker_slots: list[tuple[tuple[int, int], tuple[int, int, int, int]]] = []

    for br, (r0, r1) in enumerate(row_bands):
        for bc, (c0, c1) in enumerate(col_bands):
            slot = g[r0:r1, c0:c1]
            non_black = int(np.count_nonzero(slot != BLACK))
            yellow_coords = np.argwhere(slot == YELLOW)
            assert yellow_coords.shape == (1, 2)
            if non_black > 1:
                assert template_slot is None
                template_slot = slot.copy()
            else:
                assert non_black == 1
                assert np.array_equal(slot, (slot == YELLOW).astype(int) * YELLOW)
                marker_slots.append((tuple(map(int, yellow_coords[0])), (r0, r1, c0, c1)))

    assert template_slot is not None
    variants = _variant_map(template_slot)

    out = g.copy()
    for marker_pos, (r0, r1, c0, c1) in marker_slots:
        assert marker_pos in variants
        out[r0:r1, c0:c1] = variants[marker_pos]
    return out


def _neighbors4(size: int, cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < size and 0 <= cc < size:
            out.append((rr, cc))
    return out


def _anchor_candidates(size: int) -> list[tuple[int, int]]:
    if size not in _ANCHOR_CACHE:
        candidates = []
        probe = np.zeros((size, size), dtype=int)
        for r in range(size):
            for c in range(size):
                probe.fill(BLACK)
                probe[r, c] = YELLOW
                positions = {_yellow_pos(variant) for variant in _dihedral_variants(probe)}
                if len(positions) == 8:
                    candidates.append((r, c))
        assert candidates
        _ANCHOR_CACHE[size] = candidates
    return _ANCHOR_CACHE[size]


def _grow_mask(size: int, anchor: tuple[int, int], target_cells: int) -> list[tuple[int, int]]:
    cells = [anchor]
    chosen = {anchor}
    while len(cells) < target_cells:
        frontier = []
        for cell in cells:
            for nbr in _neighbors4(size, cell):
                if nbr not in chosen and nbr not in frontier:
                    frontier.append(nbr)
        assert frontier
        pick = random.choice(frontier)
        chosen.add(pick)
        cells.append(pick)
    return cells


def _paint_template(size: int, cells: list[tuple[int, int]], anchor: tuple[int, int]) -> np.ndarray:
    template = np.full((size, size), BLACK, dtype=int)
    others = [cell for cell in cells if cell != anchor]
    n_colors = random.randint(2, min(4, len(others)))
    palette = random.sample(TEMPLATE_COLORS, n_colors)
    mode = random.choice(["rows", "cols", "diag", "distance", "growth"])
    offset = random.randrange(n_colors)

    for idx, (r, c) in enumerate(others):
        if mode == "rows":
            color = palette[(r + offset) % n_colors]
        elif mode == "cols":
            color = palette[(c + offset) % n_colors]
        elif mode == "diag":
            color = palette[(r + c + offset) % n_colors]
        elif mode == "distance":
            dist = abs(r - anchor[0]) + abs(c - anchor[1])
            color = palette[(dist + offset) % n_colors]
        else:
            color = palette[(idx + offset) % n_colors]
        template[r, c] = color

    if len(others) >= len(palette):
        forced = others.copy()
        random.shuffle(forced)
        for color, cell in zip(palette, forced):
            template[cell] = color

    template[anchor] = YELLOW
    return template


def _make_template(slot_size: int) -> np.ndarray | None:
    area = slot_size * slot_size
    min_cells = max(slot_size + 2, 7)
    max_cells = min(area - 4, int(round(area * 0.62)))
    if min_cells > max_cells:
        return None

    for _ in range(80):
        anchor = random.choice(_anchor_candidates(slot_size))
        target_cells = random.randint(min_cells, max_cells)
        cells = _grow_mask(slot_size, anchor, target_cells)
        rows = {r for r, _c in cells}
        cols = {c for _r, c in cells}
        if len(rows) < 3 or len(cols) < 3:
            continue

        template = _paint_template(slot_size, cells, anchor)
        if len(set(map(int, template.flat)) - {BLACK, YELLOW}) < 2:
            continue
        positions = {_yellow_pos(variant) for variant in _dihedral_variants(template)}
        if len(positions) != 8:
            continue
        return template
    return None


def _layout_choices(slot_size: int) -> list[tuple[int, int]]:
    choices = []
    for rows in range(1, 5):
        for cols in range(1, 6):
            if rows * cols < 3 or rows * cols > 8:
                continue
            height = rows * slot_size + (rows - 1)
            width = cols * slot_size + (cols - 1)
            if height <= MAX_GRID_SIDE and width <= MAX_GRID_SIDE:
                choices.append((rows, cols))
    assert choices
    return choices


def _slot_slices(rows: int, cols: int, slot_size: int) -> list[tuple[tuple[int, int], tuple[slice, slice]]]:
    slots = []
    for br in range(rows):
        for bc in range(cols):
            r0 = br * (slot_size + 1)
            c0 = bc * (slot_size + 1)
            slots.append(((br, bc), (slice(r0, r0 + slot_size), slice(c0, c0 + slot_size))))
    return slots


def _empty_grid(rows: int, cols: int, slot_size: int) -> np.ndarray:
    height = rows * slot_size + (rows - 1)
    width = cols * slot_size + (cols - 1)
    grid = np.full((height, width), BLACK, dtype=int)
    for br in range(1, rows):
        grid[br * slot_size + (br - 1), :] = GREY
    for bc in range(1, cols):
        grid[:, bc * slot_size + (bc - 1)] = GREY
    return grid


def _choose_variants(variants: list[tuple[tuple[int, int], np.ndarray, bool]], count: int) -> list[tuple[tuple[int, int], np.ndarray, bool]]:
    if count == 1:
        return [random.choice(variants)]

    mirrored = [item for item in variants if item[2]]
    plain = [item for item in variants if not item[2]]
    chosen = [random.choice(plain), random.choice(mirrored)]
    used = {item[0] for item in chosen}
    remainder = [item for item in variants if item[0] not in used]
    chosen.extend(random.sample(remainder, count - len(chosen)))
    random.shuffle(chosen)
    return chosen


def _gen_attempt() -> np.ndarray | None:
    slot_size = random.randint(5, 7)
    rows, cols = random.choice(_layout_choices(slot_size))
    template = _make_template(slot_size)
    if template is None:
        return None

    variant_entries = []
    for k in range(4):
        rotated = np.rot90(template, k=k)
        variant_entries.append((_yellow_pos(rotated), rotated, False))
        mirrored = np.fliplr(rotated)
        variant_entries.append((_yellow_pos(mirrored), mirrored, True))

    if len({pos for pos, _tile, _mir in variant_entries}) != 8:
        return None

    grid = _empty_grid(rows, cols, slot_size)
    expected = grid.copy()
    slots = _slot_slices(rows, cols, slot_size)
    source_index = random.randrange(len(slots))
    target_count = len(slots) - 1
    chosen_variants = _choose_variants(variant_entries, target_count)
    target_iter = iter(chosen_variants)

    for idx, ((_br, _bc), (rs, cs)) in enumerate(slots):
        if idx == source_index:
            grid[rs, cs] = template
            expected[rs, cs] = template
        else:
            pos, variant, _mirrored = next(target_iter)
            slot = np.full((slot_size, slot_size), BLACK, dtype=int)
            slot[pos] = YELLOW
            grid[rs, cs] = slot
            expected[rs, cs] = variant

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid marker-anchored template-orientation input")
