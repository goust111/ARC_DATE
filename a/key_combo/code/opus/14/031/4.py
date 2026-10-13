import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

SOLID_COLORS = (BLUE, GREEN, YELLOW, GREY)
PARITY_ORDER = ((0, 0), (0, 1), (1, 0), (1, 1))
GENERATION_RETRIES = 500

_SEEN_OUTPUT_SIGNATURES: set[bytes] = set()
_PARITY_CURSOR = 0


def _open_runs(length: int, separators: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    prev = -1
    for sep in sorted(int(x) for x in separators):
        if prev + 1 <= sep - 1:
            runs.append((prev + 1, sep - 1))
        prev = sep
    if prev + 1 <= length - 1:
        runs.append((prev + 1, length - 1))
    return runs


def _run_index(runs: list[tuple[int, int]], value: int) -> int:
    for idx, (lo, hi) in enumerate(runs):
        if lo <= value <= hi:
            return idx
    raise ValueError(f"value {value} not inside any run")


def _cell_bounds(cell_size: int, count: int) -> list[tuple[int, int]]:
    return [(1 + idx * (cell_size + 1), (idx + 1) * (cell_size + 1) - 1) for idx in range(count)]


def _parity_class_size(rows: int, cols: int, parity: tuple[int, int]) -> int:
    pr, pc = parity
    return sum(1 for r in range(rows) for c in range(cols) if (r % 2, c % 2) == (pr, pc))


def _cell_offsets(coords: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    return sorted((int(r - r0), int(c - c0)) for r, c in coords)


def _is_solid_cell(patch: np.ndarray) -> bool:
    colors = {int(x) for x in patch.flat}
    return len(colors) == 1 and next(iter(colors)) in SOLID_COLORS


def _build_expected(
    inp: np.ndarray,
    row_cells: list[tuple[int, int]],
    col_cells: list[tuple[int, int]],
    source_cell: tuple[int, int],
    offsets: list[tuple[int, int]],
) -> np.ndarray:
    out = np.array(inp, copy=True)
    source_row, source_col = source_cell
    source_r0 = row_cells[source_row][0]
    source_c0 = col_cells[source_col][0]
    parity = (source_row % 2, source_col % 2)

    for dr, dc in offsets:
        out[source_r0 + dr, source_c0 + dc] = BLACK

    for row_idx, (r0, r1) in enumerate(row_cells):
        for col_idx, (c0, c1) in enumerate(col_cells):
            if (row_idx, col_idx) == source_cell:
                continue
            if (row_idx % 2, col_idx % 2) != parity:
                continue
            patch = inp[r0 : r1 + 1, c0 : c1 + 1]
            if not _is_solid_cell(patch):
                continue
            for dr, dc in offsets:
                rr = r0 + dr
                cc = c0 + dc
                if rr <= r1 and cc <= c1:
                    out[rr, cc] = BLACK

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve the template's hole pattern into every same-parity solid cell of the separator grid.

    Input structure:
    - Black background surrounds a lattice of full red separator rows and columns.
    - The separator lattice partitions the grid into open rectangular cells indexed from the upper left.
    - Most open cells are either empty or filled by one solid monochrome block in blue, green, yellow, or grey.
    - Exactly one open cell contains an orange connected template on black instead of a solid block.

    Rule:
    - Detect the full red separator lines and derive the open cell row and column runs between them.
    - Find the unique cell containing the orange template, and record the template pixel offsets from that cell's upper-left corner.
    - Compute the parity pair `(row_index % 2, col_index % 2)` of the template cell.
    - For every cell with the same parity pair, if the cell is a solid monochrome block, recolor the template offsets to black to punch holes there; cells of other parities stay unchanged.
    - Remove the orange template itself by turning its orange pixels to black, while preserving empty cells and separator lines.

    Key insight:
    - The template carries two independent clues: its cell position selects the parity class to edit, and its internal orange shape specifies the hole pattern to carve.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    row_lines = np.flatnonzero(np.all(g == RED, axis=1))
    col_lines = np.flatnonzero(np.all(g == RED, axis=0))
    if len(row_lines) == 0 or len(col_lines) == 0:
        raise ValueError("missing separator lines")

    row_cells = _open_runs(int(g.shape[0]), row_lines)
    col_cells = _open_runs(int(g.shape[1]), col_lines)
    if not row_cells or not col_cells:
        raise ValueError("separator lines leave no cells")

    template_coords = np.argwhere(g == ORANGE)
    if len(template_coords) == 0:
        raise ValueError("missing orange template")

    source_rows = {_run_index(row_cells, int(r)) for r, _c in template_coords}
    source_cols = {_run_index(col_cells, int(c)) for _r, c in template_coords}
    if len(source_rows) != 1 or len(source_cols) != 1:
        raise ValueError("template must be contained in exactly one cell")

    source_row = source_rows.pop()
    source_col = source_cols.pop()
    source_r0 = row_cells[source_row][0]
    source_c0 = col_cells[source_col][0]
    offsets = _cell_offsets(template_coords, source_r0, source_c0)

    return _build_expected(g, row_cells, col_cells, (source_row, source_col), offsets)


def _random_connected_offsets(cell_size: int) -> list[tuple[int, int]]:
    target_size = random.randint(2, min(4, cell_size * cell_size - 1))
    for _ in range(200):
        start = (random.randrange(cell_size), random.randrange(cell_size))
        coords = {start}
        while len(coords) < target_size:
            frontier = []
            for r, c in coords:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < cell_size and 0 <= nc < cell_size and (nr, nc) not in coords:
                        frontier.append((nr, nc))
            if not frontier:
                break
            coords.add(random.choice(frontier))
        if len(coords) != target_size:
            continue
        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        if max(rows) == min(rows) and max(cols) == min(cols):
            continue
        if len(coords) >= 3 and random.random() < 0.6:
            if max(rows) == min(rows) or max(cols) == min(cols):
                continue
        return sorted(coords)
    raise RuntimeError("failed to sample a connected template")


def _choose_source_cell(rows: int, cols: int) -> tuple[int, int]:
    global _PARITY_CURSOR

    preferred = PARITY_ORDER[_PARITY_CURSOR % len(PARITY_ORDER)]
    _PARITY_CURSOR += 1

    candidate_parities = [preferred] + [parity for parity in PARITY_ORDER if parity != preferred]
    for parity in candidate_parities:
        if _parity_class_size(rows, cols, parity) >= 3:
            candidates = [(r, c) for r in range(rows) for c in range(cols) if (r % 2, c % 2) == parity]
            return random.choice(candidates)
    raise RuntimeError("no valid template parity class")


def _assign_states(rows: int, cols: int, source_cell: tuple[int, int]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    source_parity = (source_cell[0] % 2, source_cell[1] % 2)
    target_cells = [(r, c) for r in range(rows) for c in range(cols) if (r, c) != source_cell and (r % 2, c % 2) == source_parity]
    other_cells = [(r, c) for r in range(rows) for c in range(cols) if (r, c) != source_cell and (r % 2, c % 2) != source_parity]

    if len(target_cells) < 2 or len(other_cells) < 2:
        raise RuntimeError("insufficient cells for both parity groups")

    max_target = len(target_cells) if len(target_cells) == 2 else len(target_cells) - 1
    max_other = len(other_cells) if len(other_cells) == 2 else len(other_cells) - 1
    target_count = random.randint(2, max_target)
    other_count = random.randint(2, max_other)

    target_solids = set(random.sample(target_cells, target_count))
    other_solids = set(random.sample(other_cells, other_count))
    return target_solids, other_solids


def _assign_colors(
    target_solids: set[tuple[int, int]],
    other_solids: set[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    colors: dict[tuple[int, int], int] = {}
    shared_color = random.choice(SOLID_COLORS)
    target_anchor = random.choice(tuple(target_solids))
    other_anchor = random.choice(tuple(other_solids))
    colors[target_anchor] = shared_color
    colors[other_anchor] = shared_color

    for cell in sorted(target_solids | other_solids):
        if cell in colors:
            continue
        colors[cell] = random.choice(SOLID_COLORS)

    used = set(colors.values())
    if len(used) == 1:
        changeable = [cell for cell in colors if cell not in {target_anchor, other_anchor}]
        if changeable:
            cell = random.choice(changeable)
            alternatives = [color for color in SOLID_COLORS if color != shared_color]
            colors[cell] = random.choice(alternatives)
    return colors


def _render_input(
    rows: int,
    cols: int,
    cell_size: int,
    source_cell: tuple[int, int],
    template_offsets: list[tuple[int, int]],
    solid_colors: dict[tuple[int, int], int],
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = rows * cell_size + rows + 1
    width = cols * cell_size + cols + 1
    grid = np.full((height, width), BLACK, dtype=int)

    row_lines = np.arange(0, height, cell_size + 1)
    col_lines = np.arange(0, width, cell_size + 1)
    grid[row_lines, :] = RED
    grid[:, col_lines] = RED

    row_cells = _cell_bounds(cell_size, rows)
    col_cells = _cell_bounds(cell_size, cols)

    for (row_idx, col_idx), color in solid_colors.items():
        r0, r1 = row_cells[row_idx]
        c0, c1 = col_cells[col_idx]
        grid[r0 : r1 + 1, c0 : c1 + 1] = color

    source_r0 = row_cells[source_cell[0]][0]
    source_c0 = col_cells[source_cell[1]][0]
    for dr, dc in template_offsets:
        grid[source_r0 + dr, source_c0 + dc] = ORANGE

    return grid, row_cells, col_cells


def _expected_changed_cells(
    row_cells: list[tuple[int, int]],
    col_cells: list[tuple[int, int]],
    source_cell: tuple[int, int],
    offsets: list[tuple[int, int]],
    target_solids: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    changed: set[tuple[int, int]] = set()
    source_r0 = row_cells[source_cell[0]][0]
    source_c0 = col_cells[source_cell[1]][0]
    for dr, dc in offsets:
        changed.add((source_r0 + dr, source_c0 + dc))

    for cell in target_solids:
        r0 = row_cells[cell[0]][0]
        c0 = col_cells[cell[1]][0]
        for dr, dc in offsets:
            changed.add((r0 + dr, c0 + dc))
    return changed


def _validate_instance(
    inp: np.ndarray,
    out: np.ndarray,
    row_cells: list[tuple[int, int]],
    col_cells: list[tuple[int, int]],
    source_cell: tuple[int, int],
    offsets: list[tuple[int, int]],
    target_solids: set[tuple[int, int]],
    other_solids: set[tuple[int, int]],
    solid_colors: dict[tuple[int, int], int],
) -> bool:
    expected = _build_expected(inp, row_cells, col_cells, source_cell, offsets)
    if not np.array_equal(out, expected):
        return False

    if len(target_solids) < 2 or len(other_solids) < 2:
        return False

    all_cells = {(r, c) for r in range(len(row_cells)) for c in range(len(col_cells))}
    empty_cells = all_cells - {source_cell} - target_solids - other_solids
    if not empty_cells:
        return False

    changed = {tuple(map(int, coord)) for coord in np.argwhere(inp != out)}
    expected_changed = _expected_changed_cells(row_cells, col_cells, source_cell, offsets, target_solids)
    if changed != expected_changed:
        return False

    source_patch = out[
        row_cells[source_cell[0]][0] : row_cells[source_cell[0]][1] + 1,
        col_cells[source_cell[1]][0] : col_cells[source_cell[1]][1] + 1,
    ]
    if np.any(source_patch == ORANGE):
        return False

    shared = {solid_colors[cell] for cell in target_solids} & {solid_colors[cell] for cell in other_solids}
    if not shared:
        return False

    for cell in target_solids:
        r0, r1 = row_cells[cell[0]]
        c0, c1 = col_cells[cell[1]]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        expected_patch = np.full((r1 - r0 + 1, c1 - c0 + 1), solid_colors[cell], dtype=int)
        for dr, dc in offsets:
            expected_patch[dr, dc] = BLACK
        if not np.array_equal(patch, expected_patch):
            return False

    for cell in other_solids:
        r0, r1 = row_cells[cell[0]]
        c0, c1 = col_cells[cell[1]]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(patch == solid_colors[cell]):
            return False

    return True


def _output_signature(grid: np.ndarray) -> bytes:
    return grid.tobytes() + repr(grid.shape).encode()


def _gen_attempt() -> np.ndarray | None:
    rows = random.choice((4, 5))
    cols = random.choice((4, 5))
    cell_size = random.choice((3, 4))

    source_cell = _choose_source_cell(rows, cols)
    target_solids, other_solids = _assign_states(rows, cols, source_cell)
    solid_colors = _assign_colors(target_solids, other_solids)
    template_offsets = _random_connected_offsets(cell_size)

    inp, row_cells, col_cells = _render_input(rows, cols, cell_size, source_cell, template_offsets, solid_colors)
    expected = _build_expected(inp, row_cells, col_cells, source_cell, template_offsets)

    try:
        out = main(inp)
    except Exception:
        return None

    if not _validate_instance(
        inp,
        out,
        row_cells,
        col_cells,
        source_cell,
        template_offsets,
        target_solids,
        other_solids,
        solid_colors,
    ):
        return None

    signature = _output_signature(expected)
    if signature in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_OUTPUT_SIGNATURES.add(signature)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid parity-gated punch-holes input")
