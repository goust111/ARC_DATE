from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


GEN_RETRIES = 400
FRAME_COUNT_RANGE = (2, 4)
INNER_SIZE_RANGE = (5, 8)
GAP_RANGE = (2, 4)
OUTER_MARGIN_RANGE = (1, 3)
LATCH_COUNT_RANGE = (1, 4)
COLOR_POOL = tuple(range(10))


def _color_counts(grid: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(grid, return_counts=True)
    return {int(v): int(n) for v, n in zip(vals, counts)}


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _perimeter_cells(bbox: BBox) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    out: list[Coord] = []
    for c in range(c0, c1 + 1):
        out.append((r0, c))
        out.append((r1, c))
    for r in range(r0 + 1, r1):
        out.append((r, c0))
        out.append((r, c1))
    return out


def _door_candidates(bbox: BBox) -> list[tuple[str, Coord]]:
    r0, c0, r1, c1 = bbox
    out: list[tuple[str, Coord]] = []
    for c in range(c0 + 1, c1):
        out.append(("top", (r0, c)))
        out.append(("bottom", (r1, c)))
    for r in range(r0 + 1, r1):
        out.append(("left", (r, c0)))
        out.append(("right", (r, c1)))
    return out


def _segment(start: Coord, end: Coord) -> list[Coord]:
    r0, c0 = start
    r1, c1 = end
    assert r0 == r1 or c0 == c1
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _in_box(cell: Coord, bbox: BBox) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _nested_strict(inner: BBox, outer: BBox) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and inner[2] < outer[2] and inner[3] < outer[3]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    out: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(cells, dtype=int))
    return out


def _roles(grid: np.ndarray) -> tuple[int, int, int, int]:
    counts = _color_counts(grid)
    bg = max(counts, key=counts.get)
    colors = [c for c in counts if c != bg]
    assert len(colors) == 3, "expected bg, frame, latch, and anchor roles"

    anchors = [c for c in colors if counts[c] == 1]
    assert len(anchors) == 1, "expected exactly one anchor pixel"
    anchor_color = anchors[0]

    others = [c for c in colors if c != anchor_color]
    frame_color = max(others, key=lambda c: counts[c])
    latch_color = min(others, key=lambda c: counts[c])
    assert counts[frame_color] > counts[latch_color] >= 2
    return int(bg), int(frame_color), int(latch_color), int(anchor_color)


def _detect_frames(grid: np.ndarray, frame_color: int, latch_color: int) -> list[dict]:
    mask = (grid == int(frame_color)) | (grid == int(latch_color))
    comps = _components(mask)
    assert comps, "expected nested frame components"

    frames = []
    for coords in comps:
        bbox = _bbox_from_coords(coords)
        cell_set = {tuple(map(int, p)) for p in coords}
        perimeter = _perimeter_cells(bbox)
        missing = [cell for cell in perimeter if cell not in cell_set]
        assert len(missing) == 1, "each frame must have exactly one empty door"
        door = missing[0]
        latches = [cell for cell in perimeter if cell in cell_set and int(grid[cell]) == int(latch_color)]
        assert latches, "each frame must contain at least one fake-door latch"
        frames.append(
            {
                "bbox": bbox,
                "door": door,
                "latches": latches,
                "cells": cell_set,
            }
        )

    frames.sort(key=lambda item: (item["bbox"][2] - item["bbox"][0] + 1) * (item["bbox"][3] - item["bbox"][1] + 1))
    for inner, outer in zip(frames, frames[1:]):
        assert _nested_strict(inner["bbox"], outer["bbox"]), "frames must form a strict containment chain"
    return frames


def _door_side(bbox: BBox, door: Coord) -> str:
    r0, c0, r1, c1 = bbox
    r, c = door
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    if r == r0:
        return "top"
    assert r == r1
    return "bottom"


def _route_through_frame(start: Coord, bbox: BBox, door: Coord) -> tuple[list[Coord], Coord, str]:
    r0, c0, r1, c1 = bbox
    door_r, door_c = door
    side = _door_side(bbox, door)

    if side == "left":
        bend = (door_r, start[1])
        outside = (door_r, c0 - 1)
        cells = _segment(start, bend) + _segment(bend, outside)[1:]
    elif side == "right":
        bend = (door_r, start[1])
        outside = (door_r, c1 + 1)
        cells = _segment(start, bend) + _segment(bend, outside)[1:]
    elif side == "top":
        bend = (start[0], door_c)
        outside = (r0 - 1, door_c)
        cells = _segment(start, bend) + _segment(bend, outside)[1:]
    else:
        bend = (start[0], door_c)
        outside = (r1 + 1, door_c)
        cells = _segment(start, bend) + _segment(bend, outside)[1:]
    return cells, outside, side


def _extend_to_edge(start: Coord, side: str, shape: tuple[int, int]) -> list[Coord]:
    height, width = shape
    r, c = start
    if side == "left":
        return _segment(start, (r, 0))
    if side == "right":
        return _segment(start, (r, width - 1))
    if side == "top":
        return _segment(start, (0, c))
    return _segment(start, (height - 1, c))


def _paint_cells(out: np.ndarray, cells: list[Coord], color: int, allowed: set[int]) -> None:
    for r, c in cells:
        assert int(out[r, c]) in allowed, "path may only overwrite empty cells and the anchor"
        out[r, c] = int(color)


def _trace_solution(anchor: Coord, frames: list[dict], shape: tuple[int, int]) -> list[Coord]:
    current = anchor
    path = [anchor]
    for i, frame in enumerate(frames):
        cells, current, side = _route_through_frame(current, frame["bbox"], frame["door"])
        blocked = frames[i - 1]["bbox"] if i else None
        door = frame["door"]
        outside = current
        new_cells = cells[1:]
        if side in {"left", "right"}:
            assert anchor[0] != door[0] or i > 0
        else:
            assert anchor[1] != door[1] or i > 0
        for cell in new_cells:
            if cell == door or cell == outside:
                pass
            else:
                r, c = cell
                r0, c0, r1, c1 = frame["bbox"]
                assert r0 < r < r1 and c0 < c < c1, "internal route left the current room/corridor"
            if blocked is not None:
                assert not _in_box(cell, blocked), "route must stay in the outer corridor"
        path.extend(new_cells)

    tail = _extend_to_edge(current, side, shape)[1:]
    path.extend(tail)
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the anchor-colored escape route from the innermost room through the true door of each nested frame.

    Input structure:
    - The grid contains a strict containment chain of one-pixel-thick axis-aligned rectangular frames.
    - Each frame border has several one-cell interruptions: exactly one is a fully empty door, while the others
      are occupied by latch-colored pixels and therefore remain blocked fake doors.
    - A single anchor pixel sits inside the innermost room, and there is at least one empty corridor cell between
      consecutive frames and between the outermost frame and the canvas edge.

    Rule:
    - Keep the frames, latches, and anchor, and draw the route in the anchor color using only 4-neighbor moves.
    - For a left/right true door, first move vertically to the door row, then move horizontally straight through
      the door into the next outer corridor; for a top/bottom true door, first move horizontally to the door column,
      then move vertically straight through the door.
    - Repeat this layer by layer from the innermost frame to the outermost one, then continue straight in the final
      exit direction until the route reaches the canvas edge. The empty door cells are painted when the route crosses them.

    Key insight:
    - The geometry is a nested-room maze whose answer is determined entirely by the ordered sequence of true doors;
      fake gaps are disambiguated only because they are visibly latched rather than truly empty.
    """
    g = np.asarray(grid, dtype=int)
    bg, frame_color, latch_color, anchor_color = _roles(g)
    anchor = tuple(map(int, np.argwhere(g == int(anchor_color))[0]))
    frames = _detect_frames(g, frame_color, latch_color)

    path = _trace_solution(anchor, frames, g.shape)

    out = g.copy()
    allowed = {int(bg), int(anchor_color)}
    _paint_cells(out, path, anchor_color, allowed)
    return out


def _sample_bboxes(rng: random.Random) -> tuple[int, int, list[BBox]] | None:
    frame_count = rng.randint(*FRAME_COUNT_RANGE)
    inner_h = rng.randint(*INNER_SIZE_RANGE)
    inner_w = rng.randint(*INNER_SIZE_RANGE)
    gaps = [
        (
            rng.randint(*GAP_RANGE),
            rng.randint(*GAP_RANGE),
            rng.randint(*GAP_RANGE),
            rng.randint(*GAP_RANGE),
        )
        for _ in range(frame_count - 1)
    ]
    margin_top = rng.randint(*OUTER_MARGIN_RANGE)
    margin_left = rng.randint(*OUTER_MARGIN_RANGE)
    margin_bottom = rng.randint(*OUTER_MARGIN_RANGE)
    margin_right = rng.randint(*OUTER_MARGIN_RANGE)

    total_top = sum(g[0] for g in gaps)
    total_left = sum(g[1] for g in gaps)
    total_bottom = sum(g[2] for g in gaps)
    total_right = sum(g[3] for g in gaps)

    outer_h = inner_h + total_top + total_bottom
    outer_w = inner_w + total_left + total_right
    height = outer_h + margin_top + margin_bottom
    width = outer_w + margin_left + margin_right
    if height > 30 or width > 30:
        return None

    inner = (
        margin_top + total_top,
        margin_left + total_left,
        margin_top + total_top + inner_h - 1,
        margin_left + total_left + inner_w - 1,
    )
    bboxes = [inner]
    for top_gap, left_gap, bottom_gap, right_gap in gaps:
        prev = bboxes[-1]
        bboxes.append(
            (
                prev[0] - top_gap,
                prev[1] - left_gap,
                prev[2] + bottom_gap,
                prev[3] + right_gap,
            )
        )
    return height, width, bboxes


def _sample_anchor(rng: random.Random, inner: BBox) -> Coord:
    r0, c0, r1, c1 = inner
    return rng.randint(r0 + 1, r1 - 1), rng.randint(c0 + 1, c1 - 1)


def _valid_route_cells(
    cells: list[Coord],
    door: Coord,
    outside: Coord,
    frame_bbox: BBox,
    inner_block: BBox | None,
    occupied: set[Coord],
    shape: tuple[int, int],
) -> bool:
    height, width = shape
    for cell in cells[1:]:
        r, c = cell
        if not (0 <= r < height and 0 <= c < width):
            return False
        if cell in occupied:
            return False
        if cell != door and cell != outside:
            r0, c0, r1, c1 = frame_bbox
            if not (r0 < r < r1 and c0 < c < c1):
                return False
        if inner_block is not None and _in_box(cell, inner_block):
            return False
    return True


def _pick_true_doors(
    rng: random.Random,
    bboxes: list[BBox],
    anchor: Coord,
    shape: tuple[int, int],
) -> tuple[list[Coord], list[Coord]] | None:
    current = anchor
    occupied = {anchor}
    door_cells: list[Coord] = []
    full_path = [anchor]
    side_bias = Counter()

    for i, bbox in enumerate(bboxes):
        inner_block = bboxes[i - 1] if i else None
        candidates_by_side: dict[str, list[tuple[Coord, list[Coord], Coord]]] = {s: [] for s in ("left", "right", "top", "bottom")}
        for side, door in _door_candidates(bbox):
            if side in {"left", "right"} and current[0] == door[0]:
                continue
            if side in {"top", "bottom"} and current[1] == door[1]:
                continue
            cells, outside, _ = _route_through_frame(current, bbox, door)
            if _valid_route_cells(cells, door, outside, bbox, inner_block, occupied, shape):
                candidates_by_side[side].append((door, cells, outside))

        available_sides = [side for side, items in candidates_by_side.items() if items]
        if not available_sides:
            return None
        min_used = min(side_bias[side] for side in available_sides)
        preferred = [side for side in available_sides if side_bias[side] == min_used]
        side = rng.choice(preferred)
        door, cells, current = rng.choice(candidates_by_side[side])

        occupied.update(cells[1:])
        full_path.extend(cells[1:])
        door_cells.append(door)
        side_bias[side] += 1

    last_side = _door_side(bboxes[-1], door_cells[-1])
    tail = _extend_to_edge(current, last_side, shape)
    if any(cell in occupied for cell in tail[1:]):
        return None
    full_path.extend(tail[1:])
    return door_cells, full_path


def _pick_latches(rng: random.Random, bbox: BBox, true_door: Coord) -> list[Coord]:
    side = _door_side(bbox, true_door)
    candidates = [(s, cell) for s, cell in _door_candidates(bbox) if cell != true_door]
    same_axis = [cell for s, cell in candidates if (side in {"left", "right"}) == (s in {"left", "right"})]
    other_axis = [cell for s, cell in candidates if cell not in same_axis]

    chosen: list[Coord] = []
    if same_axis:
        chosen.append(rng.choice(same_axis))
    total = rng.randint(*LATCH_COUNT_RANGE)
    pool = [cell for _s, cell in candidates if cell not in chosen]
    rng.shuffle(pool)
    if other_axis and len(chosen) < total:
        extra = rng.choice(other_axis)
        if extra not in chosen:
            chosen.append(extra)
    for cell in pool:
        if len(chosen) >= total:
            break
        if abs(cell[0] - true_door[0]) + abs(cell[1] - true_door[1]) == 1:
            continue
        chosen.append(cell)
    return chosen


def _render_grid(
    shape: tuple[int, int],
    bg: int,
    frame_color: int,
    latch_color: int,
    anchor_color: int,
    bboxes: list[BBox],
    doors: list[Coord],
    latches: list[list[Coord]],
    anchor: Coord,
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for bbox, door, latch_cells in zip(bboxes, doors, latches):
        for cell in _perimeter_cells(bbox):
            if cell != door:
                grid[cell] = int(frame_color)
        for cell in latch_cells:
            grid[cell] = int(latch_color)
    grid[anchor] = int(anchor_color)
    return grid


def _oracle_output(shape: tuple[int, int], doors: list[Coord], bboxes: list[BBox], anchor: Coord, base_grid: np.ndarray) -> np.ndarray:
    frames = [{"bbox": bbox, "door": door, "latches": []} for bbox, door in zip(bboxes, doors)]
    path = _trace_solution(anchor, frames, shape)
    out = base_grid.copy()
    color = int(base_grid[anchor])
    for cell in path:
        out[cell] = color
    return out


def _validate_generated(
    grid: np.ndarray,
    bboxes: list[BBox],
    doors: list[Coord],
    latches: list[list[Coord]],
    anchor: Coord,
) -> bool:
    expected = _oracle_output(grid.shape, doors, bboxes, anchor, grid)
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    _, frame_color, latch_color, anchor_color = _roles(grid)
    frames = _detect_frames(grid, frame_color, latch_color)
    if len(frames) != len(bboxes):
        return False
    if tuple(map(int, np.argwhere(grid == int(anchor_color))[0])) != anchor:
        return False
    if np.array_equal(actual, grid):
        return False
    if len(set(map(tuple, doors))) != len(doors):
        return False
    for frame, bbox, true_door, latch_cells in zip(frames, bboxes, doors, latches):
        if frame["bbox"] != bbox:
            return False
        if frame["door"] != true_door:
            return False
        if not latch_cells:
            return False
        if set(frame["latches"]) != set(latch_cells):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sampled = _sample_bboxes(rng)
    if sampled is None:
        return None
    height, width, bboxes = sampled
    anchor = _sample_anchor(rng, bboxes[0])
    doors_and_path = _pick_true_doors(rng, bboxes, anchor, (height, width))
    if doors_and_path is None:
        return None
    doors, _path = doors_and_path

    latches = [_pick_latches(rng, bbox, door) for bbox, door in zip(bboxes, doors)]
    if any(not items for items in latches):
        return None

    bg, frame_color, latch_color, anchor_color = rng.sample(COLOR_POOL, 4)
    grid = _render_grid(
        (height, width),
        bg,
        frame_color,
        latch_color,
        anchor_color,
        bboxes,
        doors,
        latches,
        anchor,
    )

    if not _validate_generated(grid, bboxes, doors, latches, anchor):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-room escape input")
