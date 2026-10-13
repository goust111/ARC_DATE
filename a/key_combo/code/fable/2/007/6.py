from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_fable_2_007_6",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()
Color = _ARC_UTILS.Color
get_bg = _ARC_UTILS.get_bg
neighbors4 = _ARC_UTILS.neighbors4


_SLOT_SIZE = 3
_SLOT_STRIDE = 4
_DIR_CELLS = [(r, c) for r in range(3) for c in range(3)]
_PALETTE = tuple(range(10))

_GEN_ROWS_RANGE = (4, 6)
_GEN_COLS_RANGE = (4, 6)
_GEN_SYMBOL_COUNT_RANGE = (4, 7)
_GEN_PROTOTYPE_AREA_RANGE = (4, 8)
_GEN_REQUIRE_CENTER_PROB = 0.75
_GEN_INCLUDE_CENTER_SYMBOL_PROB = 0.75
_GEN_ATTEMPTS = 400


def _slot_bands(length: int) -> list[tuple[int, int]]:
    assert length % _SLOT_STRIDE == 1, "expected 3x3 slots with 1-pixel separators"
    return [(start + 1, start + 4) for start in range(0, length - 1, _SLOT_STRIDE)]


def _parse_slots(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    row_bands = _slot_bands(g.shape[0])
    col_bands = _slot_bands(g.shape[1])

    assert np.all(g[::4, :] == bg), "row separators must stay background"
    assert np.all(g[:, ::4] == bg), "column separators must stay background"

    occupied: dict[tuple[int, int], dict[str, object]] = {}
    for slot_r, (r0, r1) in enumerate(row_bands):
        for slot_c, (c0, c1) in enumerate(col_bands):
            block = g[r0:r1, c0:c1]
            mask = block != bg
            if not np.any(mask):
                continue
            colors = {int(color) for color in block[mask]}
            assert len(colors) == 1, "each occupied slot must be monochromatic"
            occupied[(slot_r, slot_c)] = {
                "color": colors.pop(),
                "mask": mask.copy(),
            }
    assert occupied, "expected at least one occupied slot"
    return bg, row_bands, col_bands, occupied


def _paint_slot(
    grid: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    slot: tuple[int, int],
    mask: np.ndarray,
    color: int,
) -> None:
    r0, r1 = row_bands[slot[0]]
    c0, c1 = col_bands[slot[1]]
    block = grid[r0:r1, c0:c1]
    block[mask] = int(color)


def _prototype_from_slots(occupied: dict[tuple[int, int], dict[str, object]]) -> np.ndarray:
    prototype = np.zeros((_SLOT_SIZE, _SLOT_SIZE), dtype=bool)
    for info in occupied.values():
        prototype |= np.asarray(info["mask"], dtype=bool)
    assert np.any(prototype), "prototype must contain at least one cell"
    return prototype


def _mask_missing_cell(mask: np.ndarray, prototype: np.ndarray) -> tuple[int, int]:
    assert np.all(mask <= prototype), "slot mask must be a subset of the prototype"
    missing = prototype & ~mask
    assert int(missing.sum()) == 1, "each slot must miss exactly one prototype cell"
    cell = np.argwhere(missing)[0]
    return int(cell[0]), int(cell[1])


def _move_from_missing(cell: tuple[int, int]) -> tuple[int, int]:
    return int(cell[0] - 1), int(cell[1] - 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each incomplete 3x3 slot symbol, then move it into the adjacent slot indicated by its missing cell.

    Input structure:
    - The grid is a table of 3x3 symbol slots separated and bordered by one-pixel background lines.
    - Some slots are occupied by monochromatic 3x3 masks; every occupied mask is the same hidden prototype with exactly one cell removed.
    - The full prototype never appears directly in the input and must be recovered from the union of all occupied masks.

    Rule:
    - Infer the prototype as the union of all occupied slot masks.
    - For each occupied slot, find the unique prototype cell missing from that mask; its position inside the 3x3 slot gives a direction among the eight neighbors or the center.
    - Repaint that symbol as the full prototype in its original color and place it in the adjacent slot in that direction; a center missing cell means the completed symbol stays in place.
    - Clear every vacated source slot to background and keep all separator lines unchanged.

    Key insight:
    - The apparent defect is actually an instruction: the hidden prototype emerges only from combining all incomplete copies, and the one absent cell inside each copy encodes its translation direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, row_bands, col_bands, occupied = _parse_slots(g)
    prototype = _prototype_from_slots(occupied)

    out = np.full_like(g, bg)
    used_targets: set[tuple[int, int]] = set()
    for (slot_r, slot_c), info in occupied.items():
        color = int(info["color"])
        mask = np.asarray(info["mask"], dtype=bool)
        missing_cell = _mask_missing_cell(mask, prototype)
        dr, dc = _move_from_missing(missing_cell)
        target = (slot_r + dr, slot_c + dc)
        assert 0 <= target[0] < len(row_bands) and 0 <= target[1] < len(col_bands), "move leaves the table"
        assert target not in used_targets, "completed symbols must land in distinct slots"
        used_targets.add(target)
        _paint_slot(out, row_bands, col_bands, target, prototype, color)

    return out


def _random_prototype() -> np.ndarray:
    require_center = random.random() < _GEN_REQUIRE_CENTER_PROB
    target_area = random.randint(*_GEN_PROTOTYPE_AREA_RANGE)
    for _ in range(100):
        start = (1, 1) if require_center else random.choice(_DIR_CELLS)
        cells = {start}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < 3 and 0 <= nc < 3 and (nr, nc) not in cells:
                        frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target_area:
            continue
        mask = np.zeros((_SLOT_SIZE, _SLOT_SIZE), dtype=bool)
        for cell in cells:
            mask[cell] = True
        if require_center and not mask[1, 1]:
            continue
        return mask
    raise RuntimeError("failed to sample a prototype")


def _sample_missing_cells(prototype: np.ndarray, count: int) -> list[tuple[int, int]]:
    proto_cells = [tuple(map(int, cell)) for cell in np.argwhere(prototype)]
    moving_cells = [cell for cell in proto_cells if cell != (1, 1)]
    assert len(proto_cells) >= 2 and moving_cells, "prototype must allow at least one movement direction"

    chosen: list[tuple[int, int]] = []
    include_center = prototype[1, 1] and random.random() < _GEN_INCLUDE_CENTER_SYMBOL_PROB
    if include_center:
        chosen.append((1, 1))
        chosen.append(random.choice(moving_cells))
    else:
        chosen.extend(random.sample(proto_cells, 2))
        if chosen[0] == chosen[1]:
            raise AssertionError("expected two distinct missing cells")

    weighted_pool = moving_cells * 3 + ([(1, 1)] if prototype[1, 1] else [])
    while len(chosen) < count:
        chosen.append(random.choice(weighted_pool))

    random.shuffle(chosen)
    assert len(set(chosen)) >= 2, "need multiple missing-cell directions to reveal the prototype"
    assert any(cell != (1, 1) for cell in chosen), "need at least one moving symbol"
    return chosen


def _candidate_pairs(rows: int, cols: int, move: tuple[int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    dr, dc = move
    pairs = []
    for target_r in range(rows):
        for target_c in range(cols):
            source_r = target_r - dr
            source_c = target_c - dc
            if 0 <= source_r < rows and 0 <= source_c < cols:
                source = (source_r, source_c)
                target = (target_r, target_c)
                if move != (0, 0) and source == target:
                    continue
                pairs.append((source, target))
    random.shuffle(pairs)
    return pairs


def _assign_slots(rows: int, cols: int, missing_cells: list[tuple[int, int]]):
    moves = [_move_from_missing(cell) for cell in missing_cells]
    candidates = [_candidate_pairs(rows, cols, move) for move in moves]
    order = sorted(range(len(moves)), key=lambda idx: len(candidates[idx]))

    assignment: list[tuple[tuple[int, int], tuple[int, int]] | None] = [None] * len(moves)
    used_sources: set[tuple[int, int]] = set()
    used_targets: set[tuple[int, int]] = set()

    def search(depth: int) -> bool:
        if depth == len(order):
            return True
        idx = order[depth]
        for source, target in candidates[idx]:
            if source in used_sources or target in used_targets:
                continue
            if source in used_targets or target in used_sources:
                continue
            assignment[idx] = (source, target)
            used_sources.add(source)
            used_targets.add(target)
            if search(depth + 1):
                return True
            used_sources.remove(source)
            used_targets.remove(target)
            assignment[idx] = None
        return False

    if not search(0):
        return None
    return assignment


def _render_scene(
    rows: int,
    cols: int,
    bg: int,
    prototype: np.ndarray,
    colors: list[int],
    missing_cells: list[tuple[int, int]],
    assignments: list[tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    shape = (rows * _SLOT_STRIDE + 1, cols * _SLOT_STRIDE + 1)
    inp = np.full(shape, bg, dtype=int)
    out = np.full(shape, bg, dtype=int)
    row_bands = _slot_bands(shape[0])
    col_bands = _slot_bands(shape[1])
    specs: list[dict[str, object]] = []

    for color, missing_cell, pair in zip(colors, missing_cells, assignments):
        source, target = pair
        source_mask = prototype.copy()
        source_mask[missing_cell] = False
        _paint_slot(inp, row_bands, col_bands, source, source_mask, color)
        _paint_slot(out, row_bands, col_bands, target, prototype, color)
        specs.append(
            {
                "color": int(color),
                "source": source,
                "target": target,
                "missing": missing_cell,
            }
        )
    return inp, out, specs


def _semantic_check(
    inp: np.ndarray,
    out: np.ndarray,
    prototype: np.ndarray,
    specs: list[dict[str, object]],
) -> bool:
    try:
        bg_in, row_bands_in, col_bands_in, occupied_in = _parse_slots(inp)
        bg_out, row_bands_out, col_bands_out, occupied_out = _parse_slots(out)
        if bg_in != bg_out or row_bands_in != row_bands_out or col_bands_in != col_bands_out:
            return False
        if len(occupied_in) != len(specs) or len(occupied_out) != len(specs):
            return False
        if not np.array_equal(_prototype_from_slots(occupied_in), prototype):
            return False

        by_color_in = {int(info["color"]): (slot, np.asarray(info["mask"], dtype=bool)) for slot, info in occupied_in.items()}
        by_color_out = {int(info["color"]): (slot, np.asarray(info["mask"], dtype=bool)) for slot, info in occupied_out.items()}
        if len(by_color_in) != len(specs) or len(by_color_out) != len(specs):
            return False

        moving_seen = False
        distinct_missing = set()
        for spec in specs:
            color = int(spec["color"])
            source = spec["source"]
            target = spec["target"]
            missing = spec["missing"]
            expected_source = prototype.copy()
            expected_source[missing] = False

            slot_in, mask_in = by_color_in[color]
            slot_out, mask_out = by_color_out[color]
            if slot_in != source or slot_out != target:
                return False
            if not np.array_equal(mask_in, expected_source):
                return False
            if not np.array_equal(mask_out, prototype):
                return False

            dr, dc = _move_from_missing(missing)
            if (source[0] + dr, source[1] + dc) != target:
                return False
            if (dr, dc) != (0, 0):
                moving_seen = True
            distinct_missing.add(tuple(map(int, missing)))

        if not moving_seen or len(distinct_missing) < 2:
            return False
        return np.array_equal(main(inp), out)
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*_GEN_ROWS_RANGE)
    cols = random.randint(*_GEN_COLS_RANGE)
    max_symbols = min(_GEN_SYMBOL_COUNT_RANGE[1], (rows * cols * 2) // 5)
    if max_symbols < _GEN_SYMBOL_COUNT_RANGE[0]:
        return None
    symbol_count = random.randint(_GEN_SYMBOL_COUNT_RANGE[0], max_symbols)

    prototype = _random_prototype()
    missing_cells = _sample_missing_cells(prototype, symbol_count)
    assignments = _assign_slots(rows, cols, missing_cells)
    if assignments is None:
        return None

    bg = random.choice(_PALETTE)
    colors = random.sample([color for color in _PALETTE if color != bg], symbol_count)
    inp, out, specs = _render_scene(rows, cols, bg, prototype, colors, missing_cells, assignments)
    if not _semantic_check(inp, out, prototype, specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
