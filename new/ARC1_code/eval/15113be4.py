import numpy as np
from arc_utils import paste as _gen_paste


EMPTY = 0
BLUE = 1
YELLOW = 4


def _decode_mask(g: np.ndarray) -> tuple[int, np.ndarray]:
    colors = set(map(int, np.unique(g)))
    accent_colors = colors - {EMPTY, BLUE, YELLOW}
    assert len(accent_colors) == 1
    accent = accent_colors.pop()

    accent_pixels = g == accent
    coords = np.argwhere(accent_pixels)
    assert coords.size > 0
    rb0, cb0 = coords.min(axis=0)
    rb1, cb1 = coords.max(axis=0)

    h, w = g.shape
    candidates: list[tuple[int, int, np.ndarray]] = []
    for r0 in range(max(0, int(rb1) - 5), min(int(rb0), h - 6) + 1):
        for c0 in range(max(0, int(cb1) - 5), min(int(cb0), w - 6) + 1):
            window = g[r0 : r0 + 6, c0 : c0 + 6]
            if not np.all((window == EMPTY) | (window == accent)):
                continue

            localized = np.zeros_like(accent_pixels)
            localized[r0 : r0 + 6, c0 : c0 + 6] = window == accent
            if not np.array_equal(localized, accent_pixels):
                continue

            mask = np.zeros((3, 3), dtype=bool)
            valid = True
            for mr in range(3):
                for mc in range(3):
                    block = window[2 * mr : 2 * mr + 2, 2 * mc : 2 * mc + 2]
                    if np.all(block == accent):
                        mask[mr, mc] = True
                    elif not np.all(block == EMPTY):
                        valid = False
            if valid:
                candidates.append((r0, c0, mask))

    assert len(candidates) == 1
    _, _, mask = candidates[0]
    assert np.any(mask)
    return accent, mask


def _cell_starts(length: int) -> range:
    assert length >= 3
    assert (length + 1) % 4 == 0
    return range(0, length, 4)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the blue pixels matching the accent-coded 3x3 mask inside every qualifying lattice cell.

    Input structure
    - The grid is a lattice of 3x3 cells separated by yellow lines.
    - One non-blue, non-yellow accent color appears only as 2x2 blocks inside a 6x6 instruction region.
    - Ordinary candidate cells contain only empty and blue pixels.

    Rule
    - Decode the instruction region into a 3x3 mask: each solid 2x2 accent block marks the corresponding mask pixel.
    - For every 3x3 lattice cell containing blue pixels at all mask positions, recolor those mask-position pixels to the accent color.
    - Preserve the instruction region, yellow lattice, empty pixels, and all non-matching blue pixels.

    Key insight
    - The accent shape is a scaled-up code for which positions in a 3x3 blue pattern should be recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    accent, mask = _decode_mask(g)
    out = g.copy()

    for r0 in _cell_starts(g.shape[0]):
        for c0 in _cell_starts(g.shape[1]):
            cell = g[r0 : r0 + 3, c0 : c0 + 3]
            if np.all((cell == EMPTY) | (cell == BLUE)) and np.all(cell[mask] == BLUE):
                out_cell = out[r0 : r0 + 3, c0 : c0 + 3]
                out_cell[mask] = accent

    return out

# --------------- generator (o5x) ---------------

# CFG: colors and fixed lattice geometry.
_GEN_EMPTY = 0
_GEN_BLUE = 1
_GEN_YELLOW = 4
_GEN_ACCENTS = np.array([2, 3, 5, 6, 7, 8, 9], dtype=int)
_GEN_GRID_SIZE, _GEN_CELL_SIZE, _GEN_CELL_STRIDE = 23, 3, 4
_GEN_CELL_PIXELS = _GEN_CELL_SIZE * _GEN_CELL_SIZE
_GEN_CELL_STARTS = list(range(0, _GEN_GRID_SIZE, _GEN_CELL_STRIDE))
_GEN_SEPARATORS = [r + _GEN_CELL_SIZE for r in _GEN_CELL_STARTS[:-1]]

# CFG: instruction-region encoding.
_GEN_BLOCK_SCALE = 2
_GEN_INSTRUCTION_SIZE = _GEN_CELL_SIZE * _GEN_BLOCK_SCALE
_GEN_INSTRUCTION_ORIGINS = (1, _GEN_GRID_SIZE - _GEN_INSTRUCTION_SIZE - 1)
_GEN_INSTRUCTION_CELLS_PER_SIDE = 2
_GEN_MASK_BITS_RANGE = (3, 6)

# CFG: content-cell sampling and validation.
_GEN_MATCH_COUNT_RANGE = (2, 5)
_GEN_TOTAL_BLUE_RANGE = (96, 142)
_GEN_NONMATCH_BLUE_RANGE = (1, _GEN_CELL_PIXELS - 1)
_GEN_EMPTY_CELL_CHANCE_DENOMINATOR, _GEN_MAX_ATTEMPTS = 2, 300


def _gen_mask() -> np.ndarray:
    mask = np.zeros(_GEN_CELL_PIXELS, dtype=bool)
    bit_count = int(np.random.randint(*_GEN_MASK_BITS_RANGE))
    mask[np.random.choice(_GEN_CELL_PIXELS, size=bit_count, replace=False)] = True
    return mask.reshape(_GEN_CELL_SIZE, _GEN_CELL_SIZE)


def _gen_instruction_block(mask: np.ndarray, accent: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.full((_GEN_BLOCK_SCALE, _GEN_BLOCK_SCALE), int(accent)))


def _gen_corner() -> tuple[int, int, set[tuple[int, int]]]:
    top = bool(np.random.randint(2))
    left = bool(np.random.randint(2))
    ir = _GEN_INSTRUCTION_ORIGINS[0] if top else _GEN_INSTRUCTION_ORIGINS[1]
    ic = _GEN_INSTRUCTION_ORIGINS[0] if left else _GEN_INSTRUCTION_ORIGINS[1]
    far_start = len(_GEN_CELL_STARTS) - _GEN_INSTRUCTION_CELLS_PER_SIDE
    blocked_rows = range(0, _GEN_INSTRUCTION_CELLS_PER_SIDE) if top else range(far_start, len(_GEN_CELL_STARTS))
    blocked_cols = range(0, _GEN_INSTRUCTION_CELLS_PER_SIDE) if left else range(far_start, len(_GEN_CELL_STARTS))
    instruction_cells = {(r, c) for r in blocked_rows for c in blocked_cols}
    return ir, ic, instruction_cells


def _gen_content_cells(instruction_cells: set[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    return [
        (ri, ci, r0, c0)
        for ri, r0 in enumerate(_GEN_CELL_STARTS)
        for ci, c0 in enumerate(_GEN_CELL_STARTS)
        if (ri, ci) not in instruction_cells
    ]


def _gen_allocate_counts(matching: np.ndarray, mask_count: int, total_blues: int) -> np.ndarray:
    n = len(matching)
    mins = np.full(n, _GEN_NONMATCH_BLUE_RANGE[0], dtype=int)
    nonmatching = np.flatnonzero(~matching)
    if nonmatching.size and np.random.randint(_GEN_EMPTY_CELL_CHANCE_DENOMINATOR):
        mins[int(np.random.choice(nonmatching))] = 0
    mins[matching] = int(mask_count)

    maxs = np.full(n, _GEN_NONMATCH_BLUE_RANGE[1], dtype=int)
    maxs[matching] = _GEN_CELL_PIXELS
    counts = mins.copy()
    extra = int(total_blues) - int(counts.sum())
    caps = maxs - counts
    while extra > 0:
        open_cells = np.flatnonzero(caps > 0)
        if open_cells.size == 0:
            return counts
        idx = int(np.random.choice(open_cells))
        counts[idx] += 1
        caps[idx] -= 1
        extra -= 1
    return counts


def _gen_cell_pattern(mask: np.ndarray, count: int, is_matching: bool) -> np.ndarray:
    cell = np.zeros((_GEN_CELL_SIZE, _GEN_CELL_SIZE), dtype=int)
    flat = cell.reshape(-1)
    mask_positions = np.flatnonzero(mask.reshape(-1))

    if is_matching:
        flat[mask_positions] = _GEN_BLUE
        remaining = np.flatnonzero(~mask.reshape(-1))
        take = int(count) - len(mask_positions)
        if take > 0:
            flat[np.random.choice(remaining, size=take, replace=False)] = _GEN_BLUE
        return cell

    blocked = int(np.random.choice(mask_positions))
    allowed = np.array([p for p in range(_GEN_CELL_PIXELS) if p != blocked], dtype=int)
    if count > 0:
        flat[np.random.choice(allowed, size=int(count), replace=False)] = _GEN_BLUE
    return cell


def _gen_base_output(instr_r: int, instr_c: int, mask: np.ndarray, accent: int) -> np.ndarray:
    out = np.zeros((_GEN_GRID_SIZE, _GEN_GRID_SIZE), dtype=int)
    for sep in _GEN_SEPARATORS:
        out[sep, :] = _GEN_YELLOW
        out[:, sep] = _GEN_YELLOW

    br0, bc0 = instr_r - 1, instr_c - 1
    br1, bc1 = instr_r + _GEN_INSTRUCTION_SIZE, instr_c + _GEN_INSTRUCTION_SIZE
    out[[br0, br1], bc0 : bc1 + 1] = _GEN_YELLOW
    out[br0 : br1 + 1, [bc0, bc1]] = _GEN_YELLOW
    _gen_paste(out, _gen_instruction_block(mask, accent), instr_r, instr_c)
    return out


def _gen_matching_flags(cell_count: int) -> tuple[np.ndarray, set[int]]:
    match_count = int(np.random.randint(*_GEN_MATCH_COUNT_RANGE))
    match_indices = set(map(int, np.random.choice(cell_count, size=match_count, replace=False)))
    matching = np.array([i in match_indices for i in range(cell_count)], dtype=bool)
    return matching, match_indices


def _gen_render_content(out, content_cells, mask, accent, matching, counts) -> None:
    for i, (_, _, r0, c0) in enumerate(content_cells):
        cell = _gen_cell_pattern(mask, int(counts[i]), bool(matching[i]))
        if matching[i]:
            cell = cell.copy()
            cell[mask] = accent
        _gen_paste(out, cell, r0, c0)


def _gen_input_from_output(out, content_cells, mask, match_indices) -> np.ndarray:
    inp = out.copy()
    for i in match_indices:
        _, _, r0, c0 = content_cells[i]
        inp_cell = inp[r0 : r0 + _GEN_CELL_SIZE, c0 : c0 + _GEN_CELL_SIZE]
        inp_cell[mask] = _GEN_BLUE
    return inp


def _gen_attempt() -> np.ndarray | None:
    mask = _gen_mask()
    mask_count = int(mask.sum())
    accent = int(np.random.choice(_GEN_ACCENTS))
    instr_r, instr_c, instruction_cells = _gen_corner()
    content_cells = _gen_content_cells(instruction_cells)

    out = _gen_base_output(instr_r, instr_c, mask, accent)
    matching, match_indices = _gen_matching_flags(len(content_cells))
    total_blues = int(np.random.randint(*_GEN_TOTAL_BLUE_RANGE))
    counts = _gen_allocate_counts(matching, mask_count, total_blues)
    _gen_render_content(out, content_cells, mask, accent, matching, counts)

    inp = _gen_input_from_output(out, content_cells, mask, match_indices)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 15113be4 input")
