from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
Slot = tuple[int, int]

MACRO_SIDE = 3
TILE_SIDE = 3
GRID_SIDE = MACRO_SIDE * TILE_SIDE + (MACRO_SIDE - 1)
SEP_COLOR = 5
YELLOW = 4
PALETTE = [0, 1, 2, 3, 6, 7, 8, 9]
MAX_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.ravel(grid))).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not coords:
        return False
    seen = {coords[0]}
    stack = [coords[0]]
    cells = set(coords)
    while stack:
        r, c = stack.pop()
        for nbr in neighbors4(r, c):
            if nbr in cells and nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return len(seen) == len(cells)


def _build_shape_library() -> dict[Coord, list[np.ndarray]]:
    by_anchor: dict[Coord, list[np.ndarray]] = {(r, c): [] for r in range(TILE_SIDE) for c in range(TILE_SIDE)}
    for area in range(3, 6):
        for cells in itertools.combinations(range(TILE_SIDE * TILE_SIDE), area):
            mask = np.zeros((TILE_SIDE, TILE_SIDE), dtype=bool)
            rr, cc = np.unravel_index(cells, (TILE_SIDE, TILE_SIDE))
            mask[rr, cc] = True
            if not _is_connected(mask):
                continue
            box = bbox(mask)
            assert box is not None
            r0, c0, r1, c1 = box
            if (r1 - r0 + 1) == 1 and (c1 - c0 + 1) == 1:
                continue
            for anchor in zip(rr.tolist(), cc.tolist()):
                by_anchor[anchor].append(mask.copy())
    return by_anchor


SHAPES_BY_ANCHOR = _build_shape_library()
ROW_BLOCKS = [(0, 2), (4, 6), (8, 10)]
COL_BLOCKS = [(0, 2), (4, 6), (8, 10)]
ROW_LINES = [3, 7]
COL_LINES = [3, 7]
ALL_SLOTS = [(r, c) for r in range(MACRO_SIDE) for c in range(MACRO_SIDE)]


def _segments_from_lines(size: int, lines: list[int]) -> list[tuple[int, int]] | None:
    if len(lines) != MACRO_SIDE - 1:
        return None
    bounds = [-1, *sorted(lines), size]
    segments = []
    for i in range(MACRO_SIDE):
        start = bounds[i] + 1
        end = bounds[i + 1] - 1
        if end - start + 1 != TILE_SIDE:
            return None
        segments.append((start, end))
    return segments


def _find_layout(grid: np.ndarray):
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]
        cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]
        row_blocks = _segments_from_lines(grid.shape[0], rows)
        col_blocks = _segments_from_lines(grid.shape[1], cols)
        if row_blocks is not None and col_blocks is not None:
            candidates.append((color, row_blocks, col_blocks, rows, cols))
    assert len(candidates) == 1, "expected one 3x3 macro-grid split by full separator lines"
    return candidates[0]


def _parse_scene(grid: np.ndarray):
    sep_color, row_blocks, col_blocks, row_lines, col_lines = _find_layout(grid)
    interior_mask = np.ones_like(grid, dtype=bool)
    interior_mask[row_lines, :] = False
    interior_mask[:, col_lines] = False
    bg = get_bg(grid[interior_mask])

    records = {}
    for mr, (r0, r1) in enumerate(row_blocks):
        for mc, (c0, c1) in enumerate(col_blocks):
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if np.all(patch == bg):
                continue

            yellow_cells = np.argwhere(patch == YELLOW)
            assert yellow_cells.shape[0] == 1, "each sprite must contain exactly one yellow pointer"
            visible_colors = set(map(int, np.unique(patch))) - {bg, YELLOW}
            assert len(visible_colors) == 1, "each sprite must have one base color plus yellow"
            color = visible_colors.pop()

            mask = patch != bg
            target = tuple(map(int, yellow_cells[0]))
            records[(mr, mc)] = {"mask": mask.copy(), "color": color, "target": target}

    occupied = set(records)
    targets = [record["target"] for record in records.values()]
    assert occupied, "expected at least one occupied macro slot"
    assert len(set(targets)) == len(targets), "pointer targets must be distinct"
    assert set(targets) == occupied, "pointers must permute the occupied macro slots"
    return bg, sep_color, row_blocks, col_blocks, row_lines, col_lines, records


def _blank_scene(
    shape: tuple[int, int],
    bg: int,
    sep_color: int,
    row_lines: list[int],
    col_lines: list[int],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    out[row_lines, :] = int(sep_color)
    out[:, col_lines] = int(sep_color)
    return out


def _patch_from_record(record: dict[str, object], bg: int, *, mark_yellow: bool) -> np.ndarray:
    patch = np.full((TILE_SIDE, TILE_SIDE), int(bg), dtype=int)
    patch[record["mask"]] = int(record["color"])
    if mark_yellow:
        tr, tc = record["target"]
        patch[tr, tc] = YELLOW
    return patch


def _place_patch(
    canvas: np.ndarray,
    row_block: tuple[int, int],
    col_block: tuple[int, int],
    patch: np.ndarray,
) -> None:
    r0, r1 = row_block
    c0, c1 = col_block
    canvas[r0 : r1 + 1, c0 : c1 + 1] = patch


def _render_output(
    shape: tuple[int, int],
    bg: int,
    sep_color: int,
    row_blocks: list[tuple[int, int]],
    col_blocks: list[tuple[int, int]],
    row_lines: list[int],
    col_lines: list[int],
    records: dict[Slot, dict[str, object]],
) -> np.ndarray:
    out = _blank_scene(shape, bg, sep_color, row_lines, col_lines)
    for record in records.values():
        target_r, target_c = record["target"]
        patch = _patch_from_record(record, bg, mark_yellow=False)
        _place_patch(out, row_blocks[target_r], col_blocks[target_c], patch)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move every marked sprite to the macro slot named by its yellow local coordinate and restore the yellow cell to the sprite color.

    Input structure:
    - Two full separator rows and two full separator columns split the canvas into a 3x3 array of 3x3 macro slots.
    - Some macro slots contain a single-color sprite drawn on the background, with exactly one sprite cell replaced by yellow.
    - Empty macro slots contain only background, and the separator lines keep their structural color.

    Rule:
    - In each occupied macro slot, read the yellow cell's local `(row, col)` inside that 3x3 slot as the destination macro-slot address.
    - Treat the yellow cell as part of the sprite footprint and restore it to the sprite's base color before moving.
    - Move all sprites simultaneously into their destination slots, keeping each sprite's local shape unchanged and leaving vacated slots as background.

    Key insight:
    - The yellow pixel is not output content; it is a coordinate pointer written in the sprite's local frame, so the whole scene is a simultaneous permutation of occupied macro slots.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, sep_color, row_blocks, col_blocks, row_lines, col_lines, records = _parse_scene(g)
    return _render_output(g.shape, bg, sep_color, row_blocks, col_blocks, row_lines, col_lines, records)


def _sample_slot_subset(count: int) -> list[Slot] | None:
    for _ in range(200):
        slots = random.sample(ALL_SLOTS, count)
        rows = {r for r, _ in slots}
        cols = {c for _, c in slots}
        if len(rows) < 2 or len(cols) < 2:
            continue
        return slots
    return None


def _cycle_is_informative(order: list[Slot]) -> bool:
    targets = order[1:] + order[:1]
    row_change = any(src[0] != dst[0] for src, dst in zip(order, targets))
    col_change = any(src[1] != dst[1] for src, dst in zip(order, targets))
    diagonal_change = any(src[0] != dst[0] and src[1] != dst[1] for src, dst in zip(order, targets))
    return row_change and col_change and diagonal_change


def _sample_cycle(slots: list[Slot]) -> dict[Slot, Slot] | None:
    for _ in range(300):
        order = random.sample(slots, len(slots))
        if _cycle_is_informative(order):
            return {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}
    return None


def _choose_mask(anchor: Slot, used_masks: set[bytes]) -> np.ndarray:
    options = SHAPES_BY_ANCHOR[anchor][:]
    random.shuffle(options)
    for mask in options:
        key = mask.tobytes()
        if key not in used_masks:
            used_masks.add(key)
            return mask.copy()
    mask = random.choice(SHAPES_BY_ANCHOR[anchor])
    used_masks.add(mask.tobytes())
    return mask.copy()


def _render_input(records: dict[Slot, dict[str, object]], bg: int) -> np.ndarray:
    inp = _blank_scene((GRID_SIDE, GRID_SIDE), bg, SEP_COLOR, ROW_LINES, COL_LINES)
    for slot, record in records.items():
        patch = _patch_from_record(record, bg, mark_yellow=True)
        _place_patch(inp, ROW_BLOCKS[slot[0]], COL_BLOCKS[slot[1]], patch)
    return inp


def _expected_output(records: dict[Slot, dict[str, object]], bg: int) -> np.ndarray:
    return _render_output(
        (GRID_SIDE, GRID_SIDE),
        bg,
        SEP_COLOR,
        ROW_BLOCKS,
        COL_BLOCKS,
        ROW_LINES,
        COL_LINES,
        records,
    )


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(PALETTE)
    count = random.choice([3, 3, 4, 4, 5])
    slots = _sample_slot_subset(count)
    if slots is None:
        return None

    mapping = _sample_cycle(slots)
    if mapping is None:
        return None

    colors = random.sample([color for color in PALETTE if color != bg], count)
    used_masks: set[bytes] = set()
    records: dict[Slot, dict[str, object]] = {}
    for slot, color in zip(slots, colors):
        target = mapping[slot]
        records[slot] = {
            "mask": _choose_mask(target, used_masks),
            "color": int(color),
            "target": target,
        }

    inp = _render_input(records, bg)
    out = _expected_output(records, bg)
    if np.array_equal(inp, out):
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pointer-ring transport input")
