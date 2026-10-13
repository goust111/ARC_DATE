from __future__ import annotations

import itertools
import random
import sys

import numpy as np

ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

from arc_utils import colors_present, get_bg, paste


MAX_SIDE = 30
PALETTE = tuple(range(10))
MIN_CELL_SIDE = 5
MAX_CELL_SIDE = 7
MAX_GEN_ATTEMPTS = 400


def _candidate_separators(grid: np.ndarray, bg: int) -> list[int]:
    candidates: list[int] = []
    height, width = map(int, grid.shape)
    for r in range(height):
        value = int(grid[r, 0])
        if value != bg and np.all(grid[r, :] == value):
            candidates.append(value)
    for c in range(width):
        value = int(grid[0, c])
        if value != bg and np.all(grid[:, c] == value):
            candidates.append(value)
    return sorted(set(candidates))


def _split_spans(separator_indices: list[int], limit: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for index in sorted(separator_indices):
        assert index > start, "separator lines must leave non-empty cells"
        spans.append((start, index - 1))
        start = index + 1
    assert start < limit, "separator lines cannot consume the full canvas"
    spans.append((start, limit - 1))
    return spans


def _parse_with_separator(grid: np.ndarray, bg: int, sep: int) -> list[dict[str, object]] | None:
    sep_rows = [int(r) for r in range(grid.shape[0]) if np.all(grid[r, :] == sep)]
    sep_cols = [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == sep)]
    if not (sep_rows or sep_cols):
        return None

    try:
        row_spans = _split_spans(sep_rows, int(grid.shape[0]))
        col_spans = _split_spans(sep_cols, int(grid.shape[1]))
    except AssertionError:
        return None
    if len(row_spans) * len(col_spans) < 3:
        return None

    cell_height = row_spans[0][1] - row_spans[0][0] + 1
    cell_width = col_spans[0][1] - col_spans[0][0] + 1
    if cell_height != cell_width or cell_height < 3:
        return None

    cells: list[dict[str, object]] = []
    for r0, r1 in row_spans:
        if r1 - r0 + 1 != cell_height:
            return None
        for c0, c1 in col_spans:
            if c1 - c0 + 1 != cell_width:
                return None
            cell = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
            if np.any(cell == sep):
                return None
            cell_colors = colors_present(cell, exclude={bg})
            if len(cell_colors) != 1:
                return None
            color = int(cell_colors[0])
            if color == sep:
                return None
            mask = cell == color
            if not np.any(mask) or not np.any(cell == bg):
                return None

            left_sig = tuple(bool(v) for v in mask[:, 0])
            right_sig = tuple(bool(v) for v in mask[:, -1])
            if not (any(left_sig) and any(right_sig)):
                return None

            cells.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "cell": cell,
                    "color": color,
                    "left_sig": left_sig,
                    "right_sig": right_sig,
                }
            )
    return cells


def _parse_cells(grid: np.ndarray) -> tuple[int, int, list[dict[str, object]]]:
    bg = int(get_bg(grid))
    candidates = _candidate_separators(grid, bg)
    assert candidates, "expected at least one separator candidate"

    valid: list[tuple[int, list[dict[str, object]]]] = []
    for sep in candidates:
        cells = _parse_with_separator(grid, bg, sep)
        if cells is not None:
            valid.append((sep, cells))

    assert len(valid) == 1, "separator structure must be uniquely determined"
    sep, cells = valid[0]
    return bg, sep, cells


def _successors(cells: list[dict[str, object]]) -> list[int]:
    assert len({int(cell["color"]) for cell in cells}) == len(cells), "sprite colors must be unique"
    left_map: dict[tuple[bool, ...], int] = {}
    for index, cell in enumerate(cells):
        signature = cell["left_sig"]
        assert signature not in left_map, "left-edge signatures must be unique"
        left_map[signature] = index

    successors: list[int] = []
    for cell in cells:
        signature = cell["right_sig"]
        assert signature in left_map, "every right-edge signature must match some left edge"
        successors.append(left_map[signature])

    assert len(set(successors)) == len(cells), "each cell must have a unique predecessor"

    order: list[int] = []
    current = 0
    for _ in range(len(cells)):
        assert current not in order, "successor links must form one cycle"
        order.append(current)
        current = successors[current]
    assert current == 0 and len(order) == len(cells), "all cells must belong to one cyclic chain"
    return successors


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden cyclic chain from edge matches, then fill each cell's background with its successor sprite's color.

    Input structure:
    - Full-row and/or full-column separator lines split the canvas into equal square cells.
    - Each cell contains one monochrome sprite of a unique color on a shared background.
    - Every sprite touches both the left and right border of its own cell, while the cell layout is unrelated to the hidden chain order.

    Rule:
    - Compare each cell's right-edge foreground pattern with every other cell's left-edge foreground pattern, row by row.
    - The unique exact match defines that cell's successor; all cells together form one cycle, with the tail wrapping to the head.
    - Recolor only the background pixels inside each cell to the successor sprite's color, preserving sprite pixels, separator lines, and cell positions.

    Key insight:
    - "Adjacent" means adjacent in the hidden edge-matching chain, not adjacent on the page, so the layout is a decoy and only the border-contact signatures determine the borrowed color.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    bg, _sep, cells = _parse_cells(grid)
    successors = _successors(cells)

    out = grid.copy()
    for index, cell in enumerate(cells):
        r0, c0, r1, c1 = cell["bbox"]
        patch = np.array(cell["cell"], copy=True)
        fill_color = int(cells[successors[index]]["color"])
        patch[patch == bg] = fill_color
        paste(out, patch, r0, c0)

    return out


def _signature_pool(side: int) -> list[tuple[int, ...]]:
    pool: list[tuple[int, ...]] = []
    for size in range(1, min(3, side) + 1):
        pool.extend(tuple(combo) for combo in itertools.combinations(range(side), size))
    return pool


def _choose_signatures(side: int, count: int) -> list[tuple[int, ...]]:
    pool = _signature_pool(side)
    for _ in range(200):
        signatures = random.sample(pool, count)
        lengths = {len(sig) for sig in signatures}
        touches_top_or_bottom = any(0 in sig or side - 1 in sig for sig in signatures)
        has_interior_only = any(all(0 < row < side - 1 for row in sig) for sig in signatures)
        if len(lengths) >= 2 and touches_top_or_bottom and has_interior_only:
            return signatures
    return random.sample(pool, count)


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    side = int(mask.shape[0])
    candidates: set[tuple[int, int]] = set()
    rows, cols = np.where(mask)
    for r, c in zip(rows.tolist(), cols.tolist()):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < side and 0 <= nc < side and not mask[nr, nc] and 0 < nc < side - 1:
                candidates.add((nr, nc))
    return list(candidates)


def _make_sprite(side: int, left_sig: tuple[int, ...], right_sig: tuple[int, ...]) -> np.ndarray:
    left_rows = sorted(left_sig)
    right_rows = sorted(right_sig)
    anchor_rows = sorted(set(left_rows) | set(right_rows))
    hub_col = random.randint(1, side - 2)
    hub_row = random.choice(anchor_rows)

    mask = np.zeros((side, side), dtype=bool)
    low = min(anchor_rows + [hub_row])
    high = max(anchor_rows + [hub_row])
    mask[low : high + 1, hub_col] = True
    for row in left_rows:
        mask[row, : hub_col + 1] = True
    for row in right_rows:
        mask[row, hub_col:] = True

    max_area = side * side - max(4, side - 1)
    target_area = random.randint(int(mask.sum()), min(max_area, int(mask.sum()) + side + 2))
    while int(mask.sum()) < target_area:
        frontier = _frontier(mask)
        if not frontier:
            break
        r, c = random.choice(frontier)
        mask[r, c] = True

    assert np.array_equal(np.where(mask[:, 0])[0], np.array(left_rows))
    assert np.array_equal(np.where(mask[:, -1])[0], np.array(right_rows))
    assert int(mask.sum()) < side * side, "sprite must leave background cells"
    return mask


def _factorizations(count: int) -> list[tuple[int, int]]:
    return [(rows, count // rows) for rows in range(1, count + 1) if count % rows == 0]


def _choose_layout(count: int) -> tuple[int, int, int]:
    weighted: list[tuple[int, int]] = []
    for rows, cols in _factorizations(count):
        max_side = min(
            MAX_CELL_SIDE,
            (MAX_SIDE - (rows - 1)) // rows,
            (MAX_SIDE - (cols - 1)) // cols,
        )
        if max_side < MIN_CELL_SIDE:
            continue
        weight = 3 if rows > 1 and cols > 1 else 1
        weighted.extend([(rows, cols)] * weight)
    rows, cols = random.choice(weighted)
    max_side = min(
        MAX_CELL_SIDE,
        (MAX_SIDE - (rows - 1)) // rows,
        (MAX_SIDE - (cols - 1)) // cols,
    )
    side = random.randint(MIN_CELL_SIDE, max_side)
    return rows, cols, side


def _spatial_neighbor_pairs(rows: int, cols: int) -> set[frozenset[int]]:
    pairs: set[frozenset[int]] = set()
    for r in range(rows):
        for c in range(cols):
            index = r * cols + c
            if r + 1 < rows:
                pairs.add(frozenset((index, (r + 1) * cols + c)))
            if c + 1 < cols:
                pairs.add(frozenset((index, r * cols + c + 1)))
    return pairs


def _choose_placement(rows: int, cols: int, count: int) -> list[int]:
    spatial_pairs = _spatial_neighbor_pairs(rows, cols)
    for _ in range(300):
        placement = random.sample(range(count), count)
        chain_neighbors = 0
        for chain_index in range(count):
            succ = (chain_index + 1) % count
            pos_a = placement.index(chain_index)
            pos_b = placement.index(succ)
            if frozenset((pos_a, pos_b)) in spatial_pairs:
                chain_neighbors += 1
        if chain_neighbors < count:
            return placement
    return random.sample(range(count), count)


def _build_expected(
    bg: int,
    sep: int,
    rows: int,
    cols: int,
    side: int,
    placement: list[int],
    sprites: list[np.ndarray],
    colors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    height = rows * side + (rows - 1)
    width = cols * side + (cols - 1)
    inp = np.full((height, width), sep, dtype=int)
    out = np.full((height, width), sep, dtype=int)

    for pos_index, chain_index in enumerate(placement):
        grid_r = pos_index // cols
        grid_c = pos_index % cols
        top = grid_r * (side + 1)
        left = grid_c * (side + 1)

        color = colors[chain_index]
        succ_color = colors[(chain_index + 1) % len(colors)]
        sprite = sprites[chain_index]

        cell_in = np.full((side, side), bg, dtype=int)
        cell_in[sprite] = color
        cell_out = np.full((side, side), succ_color, dtype=int)
        cell_out[sprite] = color

        paste(inp, cell_in, top, left)
        paste(out, cell_out, top, left)

    return inp, out


def _independent_expected(grid: np.ndarray) -> np.ndarray:
    bg, _sep, cells = _parse_cells(grid)
    cell_count = len(cells)
    left_map = {cell["left_sig"]: idx for idx, cell in enumerate(cells)}
    out = grid.copy()
    for idx, cell in enumerate(cells):
        r0, c0, r1, c1 = cell["bbox"]
        patch = np.array(cell["cell"], copy=True)
        patch[patch == bg] = int(cells[left_map[cell["right_sig"]]]["color"])
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    valid = 0
    for perm_tail in itertools.permutations(range(1, cell_count)):
        perm = (0,) + perm_tail
        ok = True
        for i, idx in enumerate(perm):
            nxt = perm[(i + 1) % cell_count]
            if cells[idx]["right_sig"] != cells[nxt]["left_sig"]:
                ok = False
                break
        if ok:
            valid += 1
    assert valid == 1, "the chain should be unique once the start cell is fixed"
    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.choice([3, 4, 4, 5])
    rows, cols, side = _choose_layout(count)
    bg, sep, *sprite_colors = random.sample(PALETTE, count + 2)

    signatures = _choose_signatures(side, count)
    sprites = [
        _make_sprite(side, signatures[i - 1], signatures[i])
        for i in range(count)
    ]
    placement = _choose_placement(rows, cols, count)
    inp, expected = _build_expected(
        bg=bg,
        sep=sep,
        rows=rows,
        cols=cols,
        side=side,
        placement=placement,
        sprites=sprites,
        colors=sprite_colors,
    )

    try:
        actual = main(inp)
        oracle = _independent_expected(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if not np.array_equal(actual, oracle):
        return None
    if max(inp.shape) > MAX_SIDE or max(actual.shape) > MAX_SIDE:
        return None
    if len({tuple(sig) for sig in signatures}) != count:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
