from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        _in_bounds(g.shape, nr, nc)
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return out


def _is_complete_square(grid: np.ndarray, obj: dict) -> bool:
    r0, c0, r1, c1 = obj["bbox"]
    color = int(obj["color"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h < 2:
        return False
    if not np.all(grid[r0, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r1, c0 : c1 + 1] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c0] == color):
        return False
    if not np.all(grid[r0 : r1 + 1, c1] == color):
        return False
    return True


def _square_trace_paths(
    bbox: tuple[int, int, int, int], shape: tuple[int, int]
) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    corners = [
        ((r0, c0), (-1, -1)),
        ((r0, c1), (-1, 1)),
        ((r1, c0), (1, -1)),
        ((r1, c1), (1, 1)),
    ]
    paths: list[list[tuple[int, int]]] = []
    for (cr, cc), (dr, dc) in corners:
        rr = cr + dr
        cc2 = cc + dc
        path: list[tuple[int, int]] = []
        while _in_bounds(shape, rr, cc2):
            path.append((rr, cc2))
            rr += dr
            cc2 += dc
        paths.append(path)
    return paths


def _trace_owner_map(
    squares: list[dict], shape: tuple[int, int]
) -> dict[tuple[int, int], set[int]]:
    owners: dict[tuple[int, int], set[int]] = defaultdict(set)
    for idx, square in enumerate(squares):
        for path in _square_trace_paths(square["bbox"], shape):
            for cell in path:
                owners[cell].add(idx)
    return owners


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect structurally complete squares, then emit diagonal traces from their outside corner seeds.

    Input structure:
    - The grid has a black background and several monochrome 4-connected foreground objects.
    - Some objects are complete squares: their bounding box is square and every border cell of that box is present.
    - Other objects are incomplete shapes or non-square blobs that must stay unchanged.

    Rule:
    - Keep every original object pixel unchanged.
    - For each complete square, take the four corner cells and step one cell outward along the matching diagonal.
    - From each seed, continue along the same diagonal to the grid boundary, coloring background cells with the square's color.
    - If trace cells from different complete squares land on the same background cell, paint that cell grey.

    Key insight:
    - The corner marks are not the final answer; only objects that first pass the complete-square test are allowed to launch the diagonal scans.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_components(g, bg)
    squares = [obj for obj in components if _is_complete_square(g, obj)]

    owners = _trace_owner_map(squares, g.shape)
    out = g.copy()
    for (r, c), src_ids in owners.items():
        if int(g[r, c]) != bg:
            continue
        if len(src_ids) >= 2:
            out[r, c] = GREY
        else:
            square_id = next(iter(src_ids))
            out[r, c] = int(squares[square_id]["color"])
    return out


def _all_neighbors8(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIRS8]


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and mask[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _make_square_cells(
    bbox: tuple[int, int, int, int], filled: bool
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if filled or r in (r0, r1) or c in (c0, c1):
                cells.add((r, c))
    return cells


def _mask_complete_square(mask: np.ndarray) -> bool:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    if not coords or _component_count(mask) != 1:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h < 2:
        return False
    return bool(
        np.all(mask[r0, c0 : c1 + 1])
        and np.all(mask[r1, c0 : c1 + 1])
        and np.all(mask[r0 : r1 + 1, c0])
        and np.all(mask[r0 : r1 + 1, c1])
    )


def _rotate_mask(mask: np.ndarray) -> np.ndarray:
    k = random.randint(0, 3)
    out = np.rot90(mask, k)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.asarray(out, dtype=bool)


def _random_distractor_mask() -> np.ndarray:
    kind = random.choice(["missing_corner", "rect", "ell", "notch"])
    if kind == "missing_corner":
        size = random.randint(3, 4)
        mask = np.zeros((size, size), dtype=bool)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        corner = random.choice([(0, 0), (0, -1), (-1, 0), (-1, -1)])
        mask[corner] = False
    elif kind == "rect":
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        while h == w:
            w = random.randint(2, 4)
        mask = np.ones((h, w), dtype=bool)
    elif kind == "ell":
        size = random.randint(3, 4)
        mask = np.zeros((size, size), dtype=bool)
        arm_h = random.randint(2, size)
        arm_w = random.randint(2, size)
        mask[:, 0] = True
        mask[-1, :arm_w] = True
        mask[size - arm_h :, 0] = True
    else:
        size = random.randint(3, 4)
        mask = np.ones((size, size), dtype=bool)
        notch_r = random.randint(0, size - 2)
        notch_c = random.randint(0, size - 2)
        mask[notch_r : notch_r + 2, notch_c : notch_c + 2] = False
        if not mask.any():
            mask[0, 0] = True
            mask[-1, 0] = True
            mask[-1, -1] = True

    mask = _rotate_mask(mask)
    if _component_count(mask) != 1 or _mask_complete_square(mask):
        return _random_distractor_mask()
    return mask


def _place_cells(
    grid: np.ndarray, cells: set[tuple[int, int]], color: int
) -> None:
    for r, c in cells:
        grid[r, c] = color


def _add_object_buffer(
    object_buffer: set[tuple[int, int]],
    cells: set[tuple[int, int]],
    shape: tuple[int, int],
) -> None:
    for cell in cells:
        for nr, nc in _all_neighbors8(cell):
            if _in_bounds(shape, nr, nc):
                object_buffer.add((nr, nc))
        object_buffer.add(cell)


def _can_place_square(
    bbox: tuple[int, int, int, int],
    filled: bool,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    object_buffer: set[tuple[int, int]],
    trace_cells: set[tuple[int, int]],
    allow_trace_overlap: bool,
) -> tuple[bool, set[tuple[int, int]], set[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    if r0 < 3 or c0 < 3 or r1 > shape[0] - 4 or c1 > shape[1] - 4:
        return False, set(), set()
    cells = _make_square_cells(bbox, filled)
    if any(cell in object_buffer for cell in cells):
        return False, set(), set()
    if any(cell in trace_cells for cell in cells):
        return False, set(), set()

    paths = _square_trace_paths(bbox, shape)
    path_cells = {cell for path in paths for cell in path}
    if any(cell in occupied for cell in path_cells):
        return False, set(), set()
    if not allow_trace_overlap and any(cell in trace_cells for cell in path_cells):
        return False, set(), set()
    return True, cells, path_cells


def _can_place_mask(
    mask: np.ndarray,
    top: int,
    left: int,
    shape: tuple[int, int],
    object_buffer: set[tuple[int, int]],
    trace_cells: set[tuple[int, int]],
) -> tuple[bool, set[tuple[int, int]]]:
    cells: set[tuple[int, int]] = set()
    h, w = mask.shape
    if top < 0 or left < 0 or top + h > shape[0] or left + w > shape[1]:
        return False, set()
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            cell = (top + r, left + c)
            if cell in object_buffer or cell in trace_cells:
                return False, set()
            cells.add(cell)
    return True, cells


def _forced_cross_square_pair(
    shape: tuple[int, int], branch: str
) -> list[tuple[tuple[int, int, int, int], bool]] | None:
    h, w = shape
    for _ in range(200):
        size_a = random.randint(2, 4)
        size_b = random.randint(2, 4)
        filled_a = size_a == 2 or random.random() < 0.45
        filled_b = size_b == 2 or random.random() < 0.45
        t_a = random.randint(2, 5)
        t_b = random.randint(2, 5)

        if branch == "cross_below":
            rx = random.randint(7, h - 4)
            cx = random.randint(5, w - 6)
            a_bbox = (
                rx - t_a - size_a,
                cx - t_a - size_a,
                rx - t_a - 1,
                cx - t_a - 1,
            )
            b_bbox = (
                rx - t_b - size_b,
                cx + t_b + 1,
                rx - t_b - 1,
                cx + t_b + size_b,
            )
        else:
            rx = random.randint(3, h - 8)
            cx = random.randint(5, w - 6)
            a_bbox = (
                rx + t_a + 1,
                cx - t_a - size_a,
                rx + t_a + size_a,
                cx - t_a - 1,
            )
            b_bbox = (
                rx + t_b + 1,
                cx + t_b + 1,
                rx + t_b + size_b,
                cx + t_b + size_b,
            )

        boxes = [a_bbox, b_bbox]
        ok = True
        for r0, c0, r1, c1 in boxes:
            if r0 < 3 or c0 < 3 or r1 > h - 4 or c1 > w - 4:
                ok = False
                break
        if not ok:
            continue
        if not (
            a_bbox[2] + 1 < b_bbox[0]
            or b_bbox[2] + 1 < a_bbox[0]
            or a_bbox[3] + 1 < b_bbox[1]
            or b_bbox[3] + 1 < a_bbox[1]
        ):
            continue
        return [(a_bbox, filled_a), (b_bbox, filled_b)]
    return None


def _random_square_candidate(shape: tuple[int, int]) -> tuple[tuple[int, int, int, int], bool]:
    size = random.randint(2, 4)
    r0 = random.randint(3, shape[0] - size - 4)
    c0 = random.randint(3, shape[1] - size - 4)
    filled = size == 2 or random.random() < 0.4
    return (r0, c0, r0 + size - 1, c0 + size - 1), filled


def _semantic_ok(
    grid: np.ndarray,
    complete_count: int,
    total_objects: int,
    expect_intersection: bool,
) -> bool:
    comps = _find_components(grid, BLACK)
    if len(comps) != total_objects:
        return False
    squares = [obj for obj in comps if _is_complete_square(grid, obj)]
    if len(squares) != complete_count:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if not np.array_equal(out[grid != BLACK], grid[grid != BLACK]):
        return False

    grey_count = int(np.count_nonzero(out == GREY))
    if expect_intersection:
        if grey_count < 1 or grey_count > 8:
            return False
    else:
        if grey_count != 0:
            return False

    changed = int(np.count_nonzero(out != grid))
    if changed < complete_count * 8:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(16, 22)
    w = random.randint(16, 22)
    total_objects = random.randint(3, 5)
    complete_count = 2 if total_objects == 3 else random.choice([2, 3])
    complete_count = min(complete_count, total_objects - 1)
    distractor_count = total_objects - complete_count
    if distractor_count < 1:
        return None

    branch = random.choices(
        ["none", "cross_below", "cross_above"], weights=[3, 2, 2], k=1
    )[0]
    colors = random.sample(PALETTE, total_objects)

    grid = np.full((h, w), BLACK, dtype=int)
    occupied: set[tuple[int, int]] = set()
    object_buffer: set[tuple[int, int]] = set()
    trace_cells: set[tuple[int, int]] = set()
    square_specs: list[dict] = []

    square_colors = colors[:complete_count]
    if branch != "none":
        pair = _forced_cross_square_pair((h, w), branch)
        if pair is None:
            return None
        for idx, (bbox, filled) in enumerate(pair):
            ok, cells, path_cells = _can_place_square(
                bbox,
                filled,
                (h, w),
                occupied,
                object_buffer,
                trace_cells,
                allow_trace_overlap=(idx == 1),
            )
            if not ok:
                return None
            occupied |= cells
            trace_cells |= path_cells
            _add_object_buffer(object_buffer, cells, (h, w))
            square_specs.append(
                {
                    "bbox": bbox,
                    "filled": filled,
                    "color": square_colors[len(square_specs)],
                    "cells": cells,
                }
            )

    while len(square_specs) < complete_count:
        placed = False
        for _ in range(300):
            bbox, filled = _random_square_candidate((h, w))
            ok, cells, path_cells = _can_place_square(
                bbox,
                filled,
                (h, w),
                occupied,
                object_buffer,
                trace_cells,
                allow_trace_overlap=False,
            )
            if not ok:
                continue
            occupied |= cells
            trace_cells |= path_cells
            _add_object_buffer(object_buffer, cells, (h, w))
            square_specs.append(
                {
                    "bbox": bbox,
                    "filled": filled,
                    "color": square_colors[len(square_specs)],
                    "cells": cells,
                }
            )
            placed = True
            break
        if not placed:
            return None

    distractor_specs: list[dict] = []
    for color in colors[complete_count:]:
        placed = False
        for _ in range(300):
            mask = _random_distractor_mask()
            mh, mw = mask.shape
            top = random.randint(1, h - mh - 1)
            left = random.randint(1, w - mw - 1)
            ok, cells = _can_place_mask(mask, top, left, (h, w), object_buffer, trace_cells)
            if not ok:
                continue
            occupied |= cells
            _add_object_buffer(object_buffer, cells, (h, w))
            distractor_specs.append({"color": color, "cells": cells})
            placed = True
            break
        if not placed:
            return None

    for square in square_specs:
        _place_cells(grid, square["cells"], square["color"])
    for distractor in distractor_specs:
        _place_cells(grid, distractor["cells"], distractor["color"])

    if not _semantic_ok(
        grid,
        complete_count=complete_count,
        total_objects=total_objects,
        expect_intersection=(branch != "none"),
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
