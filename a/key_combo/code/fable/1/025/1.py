from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


Box = tuple[int, int, int, int]
Coord = tuple[int, int]
DIRS: tuple[Coord, ...] = ((0, 1), (0, -1), (1, 0), (-1, 0))


@dataclass(frozen=True)
class Mover:
    color: int
    full_mask: np.ndarray
    body_box: Box
    direction: Coord


@dataclass(frozen=True)
class PlannedObject:
    color: int
    input_box: Box
    stem_cells: tuple[Coord, ...]
    direction: Coord
    travel: int
    final_box: Box | None


def _bg_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(values, minlength=12)
    return int(np.argmax(counts))


def _bbox(mask: np.ndarray) -> Box | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _box_h(box: Box) -> int:
    return int(box[2] - box[0] + 1)


def _box_w(box: Box) -> int:
    return int(box[3] - box[1] + 1)


def _box_slice(box: Box) -> tuple[slice, slice]:
    return slice(box[0], box[2] + 1), slice(box[1], box[3] + 1)


def _box_in_bounds(shape: tuple[int, int], box: Box) -> bool:
    h, w = shape
    return 0 <= box[0] <= box[2] < h and 0 <= box[1] <= box[3] < w


def _shift_box(box: Box, dr: int, dc: int, dist: int) -> Box:
    return (
        int(box[0] + dr * dist),
        int(box[1] + dc * dist),
        int(box[2] + dr * dist),
        int(box[3] + dc * dist),
    )


def _boxes_overlap(a: Box, b: Box) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _paint_box(grid: np.ndarray, box: Box, color: int) -> None:
    rs, cs = _box_slice(box)
    grid[rs, cs] = int(color)


def _paint_mask(mask: np.ndarray, box: Box, value: bool = True) -> None:
    rs, cs = _box_slice(box)
    mask[rs, cs] = bool(value)


def _paint_cells(grid: np.ndarray, cells: tuple[Coord, ...], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _cells_free(occupancy: np.ndarray, cells: tuple[Coord, ...]) -> bool:
    h, w = occupancy.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if occupancy[r, c]:
            return False
    return True


def _box_free(occupancy: np.ndarray, box: Box) -> bool:
    if not _box_in_bounds(occupancy.shape, box):
        return False
    rs, cs = _box_slice(box)
    return not bool(np.any(occupancy[rs, cs]))


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            nxt = (nr, nc)
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == int(mask.sum())


def _is_solid_rectangle(mask: np.ndarray) -> bool:
    box = _bbox(mask)
    if box is None:
        return False
    return int(mask.sum()) == _box_h(box) * _box_w(box)


def _largest_solid_rectangle(mask: np.ndarray) -> Box:
    h, w = mask.shape
    best_box: Box | None = None
    best_area = 0
    tied = False

    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0]:
                continue
            for r1 in range(r0, h):
                for c1 in range(c0, w):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if area < best_area:
                        continue
                    if not np.all(mask[r0 : r1 + 1, c0 : c1 + 1]):
                        continue
                    if area > best_area:
                        best_area = area
                        best_box = (r0, c0, r1, c1)
                        tied = False
                    elif best_box != (r0, c0, r1, c1):
                        tied = True

    assert best_box is not None and not tied, "moving body must be the unique largest solid rectangle"
    return best_box


def _stem_direction(stem_mask: np.ndarray, body_box: Box) -> Coord:
    rows, cols = np.where(stem_mask)
    assert rows.size > 0, "expected a non-empty stem"
    unique_rows = sorted(set(map(int, rows)))
    unique_cols = sorted(set(map(int, cols)))
    r0, c0, r1, c1 = body_box

    if len(unique_rows) == 1:
        row = unique_rows[0]
        cols_sorted = sorted(map(int, cols))
        assert cols_sorted == list(range(cols_sorted[0], cols_sorted[-1] + 1)), "stem must be contiguous"
        assert r0 <= row <= r1, "horizontal stem must attach to the body side"
        if cols_sorted[-1] < c0:
            return (0, -1)
        if c1 < cols_sorted[0]:
            return (0, 1)
        raise AssertionError("horizontal stem must lie outside the body")

    if len(unique_cols) == 1:
        col = unique_cols[0]
        rows_sorted = sorted(map(int, rows))
        assert rows_sorted == list(range(rows_sorted[0], rows_sorted[-1] + 1)), "stem must be contiguous"
        assert c0 <= col <= c1, "vertical stem must attach to the body side"
        if rows_sorted[-1] < r0:
            return (-1, 0)
        if r1 < rows_sorted[0]:
            return (1, 0)
        raise AssertionError("vertical stem must lie outside the body")

    raise AssertionError("stem must be a straight one-cell-wide segment")


def _parse_scene(grid: np.ndarray) -> tuple[int, tuple[int, np.ndarray, Box], list[Mover]]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    anchors: list[tuple[int, np.ndarray, Box]] = []
    movers: list[Mover] = []

    for color in colors:
        mask = g == color
        assert _connected(mask), "each object color must form one 4-connected component"
        if _is_solid_rectangle(mask):
            box = _bbox(mask)
            assert box is not None
            anchors.append((color, mask, box))
            continue

        body_box = _largest_solid_rectangle(mask)
        body_mask = np.zeros_like(mask, dtype=bool)
        _paint_mask(body_mask, body_box, True)
        stem_mask = mask & ~body_mask
        direction = _stem_direction(stem_mask, body_box)
        movers.append(Mover(color=color, full_mask=mask.copy(), body_box=body_box, direction=direction))

    assert len(anchors) == 1, "expected exactly one anchor object"
    return bg, anchors[0], movers


def _candidate_shift(current: np.ndarray, bg: int, mover: Mover, settled_mask: np.ndarray) -> int | None:
    blockers = (current != bg) & ~mover.full_mask
    dr, dc = mover.direction
    shift = 1

    while True:
        moved_box = _shift_box(mover.body_box, dr, dc, shift)
        if not _box_in_bounds(current.shape, moved_box):
            return None

        rs, cs = _box_slice(moved_box)
        if np.any(blockers[rs, cs]):
            if shift == 1:
                return None
            overlap = blockers[rs, cs]
            if not np.array_equal(overlap, settled_mask[rs, cs] & overlap):
                return None
            return shift - 1
        shift += 1


def _simulate(grid: np.ndarray, *, require_unique: bool = False) -> tuple[np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    bg, anchor, movers = _parse_scene(g)
    _anchor_color, anchor_mask, _anchor_box = anchor

    current = g.copy()
    settled_mask = anchor_mask.copy()
    remaining = list(movers)
    order: list[int] = []

    while remaining:
        candidates: list[tuple[Mover, int]] = []
        for mover in remaining:
            shift = _candidate_shift(current, bg, mover, settled_mask)
            if shift is not None:
                candidates.append((mover, shift))

        if not candidates:
            break
        if len(candidates) != 1:
            if require_unique:
                raise AssertionError("expected a unique movable object at each step")
            break

        mover, shift = candidates[0]
        dr, dc = mover.direction
        current[mover.full_mask] = bg
        final_box = _shift_box(mover.body_box, dr, dc, shift)
        _paint_box(current, final_box, mover.color)
        _paint_mask(settled_mask, final_box, True)
        remaining = [other for other in remaining if other.color != mover.color]
        order.append(int(mover.color))

    return current, order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each rectangular body along its own stem direction until it first docks against the already settled anchor chain.

    Input structure
    - A black-background grid contains one solid rectangular anchor object with no stem.
    - Every other foreground color forms one 4-connected object made of a solid rectangular body plus a straight one-cell-wide stem.
    - Colors are object identities only: the roles come from geometry, not from color order.

    Rule
    - Repeatedly find the unique object whose rectangular body can translate along the stem direction, with its own stem erased, until the next step would hit the already settled structure.
    - Erase that object's stem, move only its rectangular body to the last free position before collision, and add the landed rectangle to the settled structure.
    - Stop when no unique movable object remains; any object that never finds a settled docking face stays unchanged.

    Key insight
    - The stem is only a direction pointer: travel distance is determined by the first collision with the current anchor chain, and each newly landed rectangle becomes a new anchor for later objects.
    """
    out, _order = _simulate(np.asarray(grid, dtype=int), require_unique=False)
    return out


def _adjacent_boxes(shape: tuple[int, int], support: Box, body_h: int, body_w: int, direction: Coord) -> list[Box]:
    h, w = shape
    sr0, sc0, sr1, sc1 = support
    dr, dc = direction
    boxes: list[Box] = []

    if (dr, dc) == (0, 1):
        c1 = sc0 - 1
        c0 = c1 - body_w + 1
        for r0 in range(sr0 - body_h + 1, sr1 + 1):
            box = (r0, c0, r0 + body_h - 1, c1)
            if _box_in_bounds((h, w), box):
                boxes.append(box)
    elif (dr, dc) == (0, -1):
        c0 = sc1 + 1
        c1 = c0 + body_w - 1
        for r0 in range(sr0 - body_h + 1, sr1 + 1):
            box = (r0, c0, r0 + body_h - 1, c1)
            if _box_in_bounds((h, w), box):
                boxes.append(box)
    elif (dr, dc) == (1, 0):
        r1 = sr0 - 1
        r0 = r1 - body_h + 1
        for c0 in range(sc0 - body_w + 1, sc1 + 1):
            box = (r0, c0, r1, c0 + body_w - 1)
            if _box_in_bounds((h, w), box):
                boxes.append(box)
    else:
        r0 = sr1 + 1
        r1 = r0 + body_h - 1
        for c0 in range(sc0 - body_w + 1, sc1 + 1):
            box = (r0, c0, r1, c0 + body_w - 1)
            if _box_in_bounds((h, w), box):
                boxes.append(box)

    return boxes


def _stem_options(box: Box, direction: Coord, stem_len: int) -> list[tuple[Coord, ...]]:
    r0, c0, r1, c1 = box
    dr, dc = direction
    options: list[tuple[Coord, ...]] = []
    if (dr, dc) == (0, 1):
        for r in range(r0, r1 + 1):
            options.append(tuple((r, c1 + d) for d in range(1, stem_len + 1)))
    elif (dr, dc) == (0, -1):
        for r in range(r0, r1 + 1):
            options.append(tuple((r, c0 - d) for d in range(1, stem_len + 1)))
    elif (dr, dc) == (1, 0):
        for c in range(c0, c1 + 1):
            options.append(tuple((r1 + d, c) for d in range(1, stem_len + 1)))
    else:
        for c in range(c0, c1 + 1):
            options.append(tuple((r0 - d, c) for d in range(1, stem_len + 1)))
    return options


def _corridor_clear(occupancy: np.ndarray, input_box: Box, direction: Coord, travel: int) -> bool:
    dr, dc = direction
    for dist in range(1, travel + 1):
        if not _box_free(occupancy, _shift_box(input_box, dr, dc, dist)):
            return False
    return True


def _render_input(
    shape: tuple[int, int],
    anchor_box: Box,
    anchor_color: int,
    chain: list[PlannedObject],
    decoys: list[PlannedObject],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    _paint_box(grid, anchor_box, anchor_color)
    for plan in chain + decoys:
        _paint_box(grid, plan.input_box, plan.color)
        _paint_cells(grid, plan.stem_cells, plan.color)
    return grid


def _render_output(
    shape: tuple[int, int],
    anchor_box: Box,
    anchor_color: int,
    chain: list[PlannedObject],
    decoys: list[PlannedObject],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    _paint_box(grid, anchor_box, anchor_color)
    for plan in chain:
        assert plan.final_box is not None
        _paint_box(grid, plan.final_box, plan.color)
    for plan in decoys:
        _paint_box(grid, plan.input_box, plan.color)
        _paint_cells(grid, plan.stem_cells, plan.color)
    return grid


def _input_occupancy(shape: tuple[int, int], anchor_box: Box, chain: list[PlannedObject], decoys: list[PlannedObject]) -> np.ndarray:
    return _render_input(shape, anchor_box, 1, chain, decoys) != 0


def _output_occupancy(shape: tuple[int, int], anchor_box: Box, chain: list[PlannedObject], decoys: list[PlannedObject]) -> np.ndarray:
    return _render_output(shape, anchor_box, 1, chain, decoys) != 0


def _path_occupancy(shape: tuple[int, int], chain: list[PlannedObject]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for plan in chain:
        dr, dc = plan.direction
        for dist in range(plan.travel + 1):
            _paint_mask(mask, _shift_box(plan.input_box, dr, dc, dist), True)
        for cell in plan.stem_cells:
            mask[cell] = True
    return mask


def _validate_scene(
    shape: tuple[int, int],
    anchor_box: Box,
    anchor_color: int,
    chain: list[PlannedObject],
    decoys: list[PlannedObject],
) -> bool:
    inp = _render_input(shape, anchor_box, anchor_color, chain, decoys)
    expected = _render_output(shape, anchor_box, anchor_color, chain, decoys)
    try:
        out, order = _simulate(inp, require_unique=True)
    except AssertionError:
        return False
    return order == [plan.color for plan in chain] and np.array_equal(out, expected)


def _choose_anchor(shape: tuple[int, int]) -> Box:
    h, w = shape
    anchor_h = random.randint(3, 5)
    anchor_w = random.randint(3, 5)
    r0 = random.randint(4, h - anchor_h - 5)
    c0 = random.randint(4, w - anchor_w - 5)
    return (r0, c0, r0 + anchor_h - 1, c0 + anchor_w - 1)


def _sample_chain_plan(
    shape: tuple[int, int],
    anchor_box: Box,
    anchor_color: int,
    chain: list[PlannedObject],
    color: int,
) -> PlannedObject | None:
    occupancy = _input_occupancy(shape, anchor_box, chain, [])
    final_boxes = [anchor_box] + [plan.final_box for plan in chain if plan.final_box is not None]
    support = anchor_box if not chain else chain[-1].final_box
    assert support is not None

    directions = list(DIRS)
    random.shuffle(directions)
    if chain and random.random() < 0.7:
        prev_dir = chain[-1].direction
        orth = [d for d in directions if d[0] != prev_dir[0] and d[1] != prev_dir[1]]
        if orth:
            directions = orth + [d for d in directions if d not in orth]

    for direction in directions:
        body_h = random.randint(2, 4)
        body_w = random.randint(2, 4)
        candidates = _adjacent_boxes(shape, support, body_h, body_w, direction)
        random.shuffle(candidates)

        for final_box in candidates:
            if any(_boxes_overlap(final_box, other) for other in final_boxes):
                continue
            if not _box_free(occupancy, final_box):
                continue

            max_back = 0
            while _box_in_bounds(shape, _shift_box(final_box, -direction[0], -direction[1], max_back + 1)):
                max_back += 1
            travel_options = list(range(3, min(8, max_back) + 1))
            random.shuffle(travel_options)

            for travel in travel_options:
                input_box = _shift_box(final_box, -direction[0], -direction[1], travel)
                if not _box_free(occupancy, input_box):
                    continue
                if not _corridor_clear(occupancy, input_box, direction, travel):
                    continue

                stem_lengths = list(range(1, min(4, travel - 2) + 1))
                random.shuffle(stem_lengths)
                for stem_len in stem_lengths:
                    stem_options = _stem_options(input_box, direction, stem_len)
                    random.shuffle(stem_options)
                    for stem_cells in stem_options:
                        if not _cells_free(occupancy, stem_cells):
                            continue
                        plan = PlannedObject(
                            color=color,
                            input_box=input_box,
                            stem_cells=stem_cells,
                            direction=direction,
                            travel=travel,
                            final_box=final_box,
                        )
                        prefix = chain + [plan]
                        if _validate_scene(shape, anchor_box, anchor_color, prefix, []):
                            return plan
    return None


def _sample_decoy(
    shape: tuple[int, int],
    anchor_box: Box,
    anchor_color: int,
    chain: list[PlannedObject],
    decoys: list[PlannedObject],
    color: int,
) -> PlannedObject | None:
    occupancy = _input_occupancy(shape, anchor_box, chain, decoys)
    reserved = occupancy | _output_occupancy(shape, anchor_box, chain, [])
    reserved |= _path_occupancy(shape, chain)
    h, w = shape

    for _ in range(300):
        direction = random.choice(DIRS)
        body_h = random.randint(2, 4)
        body_w = random.randint(2, 4)
        box = (
            random.randint(1, h - body_h - 2),
            random.randint(1, w - body_w - 2),
            0,
            0,
        )
        input_box = (box[0], box[1], box[0] + body_h - 1, box[1] + body_w - 1)
        if not _box_free(reserved, input_box):
            continue

        stem_lengths = list(range(1, 4))
        random.shuffle(stem_lengths)
        for stem_len in stem_lengths:
            options = _stem_options(input_box, direction, stem_len)
            random.shuffle(options)
            for stem_cells in options:
                if not _cells_free(reserved, stem_cells):
                    continue
                plan = PlannedObject(
                    color=color,
                    input_box=input_box,
                    stem_cells=stem_cells,
                    direction=direction,
                    travel=0,
                    final_box=None,
                )
                if _validate_scene(shape, anchor_box, anchor_color, chain, decoys + [plan]):
                    return plan
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        shape = (random.randint(16, 24), random.randint(16, 24))
        chain_len = random.choices([1, 2, 3, 4], weights=[1, 3, 3, 2])[0]
        decoy_len = random.choices([0, 1, 2], weights=[2, 3, 1])[0]
        n_colors = 1 + chain_len + decoy_len
        palette = random.sample(range(1, 10), n_colors)
        anchor_color = palette[0]
        anchor_box = _choose_anchor(shape)

        chain: list[PlannedObject] = []
        ok = True
        for color in palette[1 : 1 + chain_len]:
            plan = _sample_chain_plan(shape, anchor_box, anchor_color, chain, color)
            if plan is None:
                ok = False
                break
            chain.append(plan)
        if not ok:
            continue

        decoys: list[PlannedObject] = []
        for color in palette[1 + chain_len :]:
            plan = _sample_decoy(shape, anchor_box, anchor_color, chain, decoys, color)
            if plan is not None:
                decoys.append(plan)

        if not _validate_scene(shape, anchor_box, anchor_color, chain, decoys):
            continue

        grid = _render_input(shape, anchor_box, anchor_color, chain, decoys)
        out = main(grid)
        if np.array_equal(grid, out):
            continue
        return grid

    raise RuntimeError("failed to generate a valid chain-anchor slide scene")
