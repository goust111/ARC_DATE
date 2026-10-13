from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
ARROW = 8
PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
ARROW_DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

BASE_SHAPES = (
    ((0, 0), (0, 1), (1, 0)),  # L3
    ((0, 0), (0, 1), (0, 2)),  # I3
    ((0, 0), (0, 1), (1, 0), (1, 1)),  # O4
    ((0, 0), (1, 0), (2, 0), (2, 1)),  # L4
    ((0, 1), (1, 0), (1, 1), (1, 2)),  # T4
    ((0, 0), (0, 1), (1, 1), (1, 2)),  # Z4
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),  # P5
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),  # plus5
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),  # L5
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),  # T5
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),  # U5
    ((0, 1), (0, 2), (1, 0), (1, 1), (2, 0)),  # skew5
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (2, 3)),  # bent6
    ((0, 1), (0, 2), (1, 0), (1, 1), (2, 1), (2, 2)),  # zig6
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)),  # cap7
    ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1)),  # branch7
)

GRID_SIZE_MIN = 16
GRID_SIZE_MAX = 24
PAIR_COUNT_MIN = 1
PAIR_COUNT_MAX = 3
EXTRA_OBJECT_MAX = 2
HANDLE_LEN_RANGE = (3, 5)
HEAD_GAP_RANGE = (3, 5)
PAIR_PLACEMENT_ATTEMPTS = 200
EXTRA_PLACEMENT_ATTEMPTS = 120
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _normalize(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rotate90(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize([(-c, r) for r, c in cells])


def _reflect_h(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return _normalize([(r, -c) for r, c in cells])


def _transform_shape(cells: tuple[tuple[int, int], ...], rotation: int, reflect: bool) -> tuple[tuple[int, int], ...]:
    out = tuple(cells)
    if reflect:
        out = _reflect_h(out)
    for _ in range(rotation % 4):
        out = _rotate90(out)
    return _normalize(out)


def _sample_shape() -> tuple[tuple[int, int], ...]:
    base = random.choice(BASE_SHAPES)
    return _transform_shape(base, rotation=random.randrange(4), reflect=bool(random.getrandbits(1)))


def _translate(cells: tuple[tuple[int, int], ...], dr: int, dc: int) -> tuple[tuple[int, int], ...]:
    return tuple((r + dr, c + dc) for r, c in cells)


def _bbox(cells: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _component_cells(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    neighbors = DIR4 if connectivity == 4 else DIR8
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_objects(g: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    cell_to_object: dict[tuple[int, int], int] = {}

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (BG, ARROW) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not _in_bounds(g.shape, nr, nc) or bool(seen[nr, nc]):
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            obj = {
                "color": color,
                "coords": tuple(sorted(coords)),
                "bbox": _bbox(coords),
            }
            idx = len(objects)
            objects.append(obj)
            for coord in coords:
                cell_to_object[coord] = idx
    return objects, cell_to_object


def _parse_arrow(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> dict:
    pts = tuple(sorted(coords))
    n = len(pts)
    assert 4 <= n <= 6, "arrow must have handle length 3-5 plus one tip"

    row_counts: dict[int, int] = {}
    col_counts: dict[int, int] = {}
    for r, c in pts:
        row_counts[r] = row_counts.get(r, 0) + 1
        col_counts[c] = col_counts.get(c, 0) + 1

    handle_rows = [r for r, count in row_counts.items() if count == n - 1]
    handle_cols = [c for c, count in col_counts.items() if count == n - 1]

    if len(handle_rows) == 1 and not handle_cols:
        row = handle_rows[0]
        handle = sorted((r, c) for r, c in pts if r == row)
        tip_candidates = [(r, c) for r, c in pts if r != row]
        assert len(tip_candidates) == 1
        tip = tip_candidates[0]
        cols = [c for _r, c in handle]
        assert cols == list(range(cols[0], cols[-1] + 1)), "horizontal handle must be contiguous"
        assert abs(tip[0] - row) == 1, "horizontal arrow tip must be vertically offset by one"
        if tip[1] == cols[-1] + 1:
            direction = (0, 1)
            head = (row, cols[-1])
            tail = (row, cols[0])
        elif tip[1] == cols[0] - 1:
            direction = (0, -1)
            head = (row, cols[0])
            tail = (row, cols[-1])
        else:
            raise AssertionError("horizontal arrow tip must extend one step beyond the head")
        return {
            "coords": pts,
            "direction": direction,
            "head": head,
            "tail": tail,
            "tip": tip,
        }

    if len(handle_cols) == 1 and not handle_rows:
        col = handle_cols[0]
        handle = sorted((r, c) for r, c in pts if c == col)
        tip_candidates = [(r, c) for r, c in pts if c != col]
        assert len(tip_candidates) == 1
        tip = tip_candidates[0]
        rows = [r for r, _c in handle]
        assert rows == list(range(rows[0], rows[-1] + 1)), "vertical handle must be contiguous"
        assert abs(tip[1] - col) == 1, "vertical arrow tip must be horizontally offset by one"
        if tip[0] == rows[-1] + 1:
            direction = (1, 0)
            head = (rows[-1], col)
            tail = (rows[0], col)
        elif tip[0] == rows[0] - 1:
            direction = (-1, 0)
            head = (rows[0], col)
            tail = (rows[-1], col)
        else:
            raise AssertionError("vertical arrow tip must extend one step beyond the head")
        return {
            "coords": pts,
            "direction": direction,
            "head": head,
            "tail": tail,
            "tip": tip,
        }

    raise AssertionError("azure component is not a valid simplified arrow")


def _find_arrows(g: np.ndarray) -> list[dict]:
    arrows = []
    for comp in _component_cells(g == ARROW, connectivity=8):
        arrows.append(_parse_arrow(comp))
    assert arrows, "expected at least one arrow"
    return arrows


def _scan_for_target(
    g: np.ndarray,
    cell_to_object: dict[tuple[int, int], int],
    head: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[int, tuple[int, int]]:
    dr, dc = direction
    r, c = head[0] + dr, head[1] + dc
    while _in_bounds(g.shape, r, c):
        color = int(g[r, c])
        if color == BG:
            r += dr
            c += dc
            continue
        if color == ARROW:
            r += dr
            c += dc
            continue
        return cell_to_object[(r, c)], (r, c)
    raise AssertionError("every arrow must point to some colored object")


def _projection_shift(
    obj: dict,
    tail: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = obj["bbox"]
    dr, dc = direction
    if (dr, dc) == (0, 1):
        return 0, int(tail[1] - 1 - c1)
    if (dr, dc) == (0, -1):
        return 0, int(tail[1] + 1 - c0)
    if (dr, dc) == (1, 0):
        return int(tail[0] - 1 - r1), 0
    assert (dr, dc) == (-1, 0)
    return int(tail[0] + 1 - r0), 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each arrow's target object to the arrow tail, then erase the arrow.

    Input structure:
    - Black background.
    - Small monochrome colored objects.
    - Azure arrows made of a 3-5 cell straight handle plus one diagonally offset tip.

    Rule:
    - Extend the arrow from its head along the handle axis to find the first colored object.
    - Translate that object's exact shape back to the opposite side of the arrow.
    - The copied shape keeps its orientation; only its axis-aligned position changes so
      that the copied edge facing the arrow sits flush against the tail.
    - Remove all azure arrow pixels. Original target objects stay in place.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects, cell_to_object = _find_objects(g)
    assert objects, "expected at least one colored object"
    arrows = _find_arrows(g)

    out = g.copy()
    out[g == ARROW] = BG

    planned: dict[tuple[int, int], int] = {}
    for arrow in arrows:
        obj_idx, _hit = _scan_for_target(g, cell_to_object, arrow["head"], arrow["direction"])
        obj = objects[obj_idx]
        shift_r, shift_c = _projection_shift(obj, arrow["tail"], arrow["direction"])

        for r, c in obj["coords"]:
            nr, nc = int(r + shift_r), int(c + shift_c)
            assert _in_bounds(g.shape, nr, nc), "projected copy must remain in bounds"
            assert int(g[nr, nc]) in (BG, ARROW), "projected copy must land on empty space"
            assert (nr, nc) not in planned, "projected copies must not overlap"
            planned[(nr, nc)] = int(obj["color"])

    for (r, c), color in planned.items():
        out[r, c] = color
    return out


def _neighbors8(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if _in_bounds(shape, nr, nc):
                    out.add((nr, nc))
    return out


def _cells_ok(
    cells: set[tuple[int, int]],
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
) -> bool:
    return all(_in_bounds(shape, r, c) and (r, c) not in forbidden for r, c in cells)


def _build_arrow_cells(
    head: tuple[int, int],
    direction: tuple[int, int],
    handle_len: int,
    tip_side: int,
) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    hr, hc = head
    dr, dc = direction
    if dr == 0:
        handle = [(hr, hc - dc * i) for i in range(handle_len)]
        tip = (hr + tip_side, hc + dc)
    else:
        handle = [(hr - dr * i, hc) for i in range(handle_len)]
        tip = (hr + dr, hc + tip_side)
    return tuple(handle + [tip]), handle[-1]


def _copy_cells_for_pair(
    target_cells: tuple[tuple[int, int], ...],
    direction: tuple[int, int],
    tail: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = _bbox(target_cells)
    dr, dc = direction
    if (dr, dc) == (0, 1):
        shift = (0, int(tail[1] - 1 - c1))
    elif (dr, dc) == (0, -1):
        shift = (0, int(tail[1] + 1 - c0))
    elif (dr, dc) == (1, 0):
        shift = (int(tail[0] - 1 - r1), 0)
    else:
        shift = (int(tail[0] + 1 - r0), 0)
    return _translate(target_cells, shift[0], shift[1])


def _line_entry(target_cells: tuple[tuple[int, int], ...], direction: tuple[int, int], axis: int) -> int:
    dr, dc = direction
    if dr == 0:
        cols = [c for r, c in target_cells if r == axis]
        return min(cols) if dc == 1 else max(cols)
    rows = [r for r, c in target_cells if c == axis]
    return min(rows) if dr == 1 else max(rows)


def _corridor_cells(
    head: tuple[int, int],
    direction: tuple[int, int],
    entry_value: int,
) -> set[tuple[int, int]]:
    dr, dc = direction
    cells: set[tuple[int, int]] = set()
    r, c = head
    while True:
        r += dr
        c += dc
        if (dr == 0 and c == entry_value) or (dc == 0 and r == entry_value):
            break
        cells.add((r, c))
    return cells


def _place_pair(
    grid_shape: tuple[int, int],
    direction: tuple[int, int],
    color: int,
    input_forbidden: set[tuple[int, int]],
    output_forbidden: set[tuple[int, int]],
    reserved_clear: set[tuple[int, int]],
) -> dict | None:
    h, w = grid_shape
    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        target_shape = _sample_shape()
        th = 1 + max(r for r, _c in target_shape)
        tw = 1 + max(c for _r, c in target_shape)

        handle_len = random.randint(*HANDLE_LEN_RANGE)
        head_gap = random.randint(*HEAD_GAP_RANGE)
        tip_side = random.choice((-1, 1))

        top = random.randint(0, h - th)
        left = random.randint(0, w - tw)
        target_cells = _translate(target_shape, top, left)
        target_set = set(target_cells)
        target_block = _neighbors8(target_set, grid_shape)
        if not _cells_ok(target_set, grid_shape, input_forbidden | output_forbidden | reserved_clear):
            continue

        if direction[0] == 0:
            candidate_axes = sorted({r for r, _c in target_cells})
        else:
            candidate_axes = sorted({c for _r, c in target_cells})
        random.shuffle(candidate_axes)

        for axis in candidate_axes:
            entry = _line_entry(target_cells, direction, axis)
            if direction[0] == 0:
                head = (axis, int(entry - head_gap * direction[1]))
                if not (0 < head[0] < h - 1):
                    continue
            else:
                head = (int(entry - head_gap * direction[0]), axis)
                if not (0 < head[1] < w - 1):
                    continue

            arrow_cells, tail = _build_arrow_cells(head, direction, handle_len, tip_side)
            arrow_set = set(arrow_cells)
            if not _cells_ok(arrow_set, grid_shape, input_forbidden | output_forbidden | reserved_clear | target_block):
                continue

            copy_cells = _copy_cells_for_pair(target_cells, direction, tail)
            copy_set = set(copy_cells)
            if not _cells_ok(copy_set, grid_shape, output_forbidden):
                continue
            if _neighbors8(copy_set, grid_shape) & target_set:
                continue
            if copy_set & arrow_set:
                continue

            corridor = _corridor_cells(head, direction, entry)
            if not _cells_ok(corridor, grid_shape, input_forbidden | reserved_clear | target_set | arrow_set):
                continue
            if corridor & copy_set:
                continue

            return {
                "color": color,
                "target_cells": tuple(sorted(target_cells)),
                "copy_cells": tuple(sorted(copy_cells)),
                "arrow_cells": tuple(sorted(arrow_cells)),
                "corridor": corridor,
            }
    return None


def _place_extra_object(
    grid_shape: tuple[int, int],
    color: int,
    input_forbidden: set[tuple[int, int]],
    output_forbidden: set[tuple[int, int]],
    reserved_clear: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...] | None:
    h, w = grid_shape
    for _ in range(EXTRA_PLACEMENT_ATTEMPTS):
        shape = _sample_shape()
        sh = 1 + max(r for r, _c in shape)
        sw = 1 + max(c for _r, c in shape)
        top = random.randint(0, h - sh)
        left = random.randint(0, w - sw)
        cells = _translate(shape, top, left)
        cell_set = set(cells)
        if not _cells_ok(cell_set, grid_shape, input_forbidden | output_forbidden | reserved_clear):
            continue
        return tuple(sorted(cells))
    return None


def _render_pair_input(
    shape: tuple[int, int],
    pairs: list[dict],
    extras: list[tuple[int, tuple[tuple[int, int], ...]]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BG, dtype=int)
    expected = np.full(shape, BG, dtype=int)

    for color, cells in extras:
        for r, c in cells:
            inp[r, c] = color
            expected[r, c] = color

    for pair in pairs:
        color = int(pair["color"])
        for r, c in pair["target_cells"]:
            inp[r, c] = color
            expected[r, c] = color
        for r, c in pair["arrow_cells"]:
            inp[r, c] = ARROW
        for r, c in pair["copy_cells"]:
            expected[r, c] = color

    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    w = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    grid_shape = (h, w)

    pair_count = random.randint(PAIR_COUNT_MIN, PAIR_COUNT_MAX)
    extra_count = random.randint(0, EXTRA_OBJECT_MAX)
    if pair_count == 1 and extra_count == 0:
        extra_count = 1

    total_objects = pair_count + extra_count
    colors = random.sample(PALETTE, total_objects)
    directions = [random.choice(ARROW_DIRECTIONS) for _ in range(pair_count)]

    input_cells: set[tuple[int, int]] = set()
    input_forbidden: set[tuple[int, int]] = set()
    output_cells: set[tuple[int, int]] = set()
    output_forbidden: set[tuple[int, int]] = set()
    reserved_clear: set[tuple[int, int]] = set()
    pairs: list[dict] = []
    extras: list[tuple[int, tuple[tuple[int, int], ...]]] = []

    for i in range(pair_count):
        placed = _place_pair(
            grid_shape=grid_shape,
            direction=directions[i],
            color=colors[i],
            input_forbidden=input_forbidden,
            output_forbidden=output_forbidden,
            reserved_clear=reserved_clear,
        )
        if placed is None:
            return None

        target_set = set(placed["target_cells"])
        arrow_set = set(placed["arrow_cells"])
        copy_set = set(placed["copy_cells"])
        input_cells |= target_set | arrow_set
        input_forbidden |= _neighbors8(target_set | arrow_set, grid_shape)
        output_cells |= target_set | copy_set
        output_forbidden |= _neighbors8(target_set | copy_set, grid_shape)
        reserved_clear |= set(placed["corridor"])
        pairs.append(placed)

    for j in range(extra_count):
        color = colors[pair_count + j]
        cells = _place_extra_object(
            grid_shape=grid_shape,
            color=color,
            input_forbidden=input_forbidden,
            output_forbidden=output_forbidden,
            reserved_clear=reserved_clear,
        )
        if cells is None:
            return None
        cell_set = set(cells)
        extras.append((color, cells))
        input_cells |= cell_set
        output_cells |= cell_set
        input_forbidden |= _neighbors8(cell_set, grid_shape)
        output_forbidden |= _neighbors8(cell_set, grid_shape)

    inp, expected = _render_pair_input(grid_shape, pairs, extras)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid arrow-projection instance")
