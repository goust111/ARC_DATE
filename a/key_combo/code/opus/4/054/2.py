from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


BG = 0
PATH = 5
BOX_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
GEN_RNG = np.random.default_rng()


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class BoxSpec:
    color: int
    bbox: BBox
    adjacent_path: tuple[Coord, Coord, Coord]
    orientation: str
    contact_index: int
    middle_path: Coord


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _same_color_components(grid: np.ndarray, forbidden: set[int]) -> list[tuple[int, np.ndarray]]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, np.ndarray]] = []
    forbidden = {int(v) for v in forbidden}
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in forbidden:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for nr, nc in neighbors4(rr, cc):
                    if (
                        in_bounds(grid.shape, nr, nc)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append((color, np.asarray(cells, dtype=int)))
    return comps


def _path_neighbors(path_set: set[Coord]) -> dict[Coord, tuple[Coord, ...]]:
    return {
        cell: tuple(n for n in neighbors4(*cell) if n in path_set)
        for cell in path_set
    }


def _ordered_path(grid: np.ndarray) -> tuple[list[Coord], dict[Coord, tuple[Coord, ...]]]:
    path_coords = [tuple(map(int, coord)) for coord in np.argwhere(grid == PATH)]
    assert path_coords, "expected a grey path"
    path_set = set(path_coords)
    nbrs = _path_neighbors(path_set)
    endpoints = [cell for cell, nn in nbrs.items() if len(nn) == 1]
    assert len(endpoints) == 2, "expected exactly two path endpoints"
    start = min(endpoints)

    ordered = [start]
    prev = None
    cur = start
    while True:
        nxt = [n for n in nbrs[cur] if n != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "expected a simple non-branching path"
        prev, cur = cur, nxt[0]
        ordered.append(cur)

    assert len(ordered) == len(path_coords), "expected one 4-connected path component"
    return ordered, nbrs


def _local_orientation(ordered_path: list[Coord], index_map: dict[Coord, int], cell: Coord) -> str:
    idx = int(index_map[cell])
    assert 0 < idx < len(ordered_path) - 1, "contact must not be an endpoint"
    prev_cell = ordered_path[idx - 1]
    next_cell = ordered_path[idx + 1]
    if prev_cell[0] == cell[0] == next_cell[0]:
        return "h"
    if prev_cell[1] == cell[1] == next_cell[1]:
        return "v"
    raise AssertionError("contact must be on a straight path segment")


def _box_side_kind(box_cells: set[Coord], bbox: BBox) -> str | None:
    r0, c0, r1, c1 = bbox
    rel = {(r - r0, c - c0) for r, c in box_cells}
    if rel == {(0, 0), (0, 1), (0, 2)}:
        return "top"
    if rel == {(2, 0), (2, 1), (2, 2)}:
        return "bottom"
    if rel == {(0, 0), (1, 0), (2, 0)}:
        return "left"
    if rel == {(0, 2), (1, 2), (2, 2)}:
        return "right"
    return None


def _extract_boxes(grid: np.ndarray, ordered_path: list[Coord], nbrs: dict[Coord, tuple[Coord, ...]]) -> list[BoxSpec]:
    path_set = set(ordered_path)
    index_map = {cell: i for i, cell in enumerate(ordered_path)}
    boxes: list[BoxSpec] = []
    for color, coords in _same_color_components(grid, forbidden={BG, PATH}):
        assert int(coords.shape[0]) == 8, "expected each foreground object to be a hollow 3x3 box"
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        assert (r1 - r0, c1 - c0) == (2, 2), "expected 3x3 boxes"
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert int(sub[1, 1]) == BG, "expected hollow box center"
        ring = {(rr, cc) for rr in range(3) for cc in range(3) if (rr, cc) != (1, 1)}
        assert all(int(sub[rr, cc]) == color for rr, cc in ring), "expected monochrome box border"

        box_set = {tuple(map(int, cell)) for cell in coords}
        adjacent_path = {
            path_cell
            for box_cell in box_set
            for path_cell in neighbors4(*box_cell)
            if path_cell in path_set
        }
        adjacent_box_cells = {
            box_cell
            for box_cell in box_set
            for path_cell in neighbors4(*box_cell)
            if path_cell in path_set
        }
        assert len(adjacent_path) == 3, "expected each box to share one full side with the path"
        assert len(adjacent_box_cells) == 3, "expected a single adjacent box side"
        side_kind = _box_side_kind(adjacent_box_cells, bbox)
        assert side_kind is not None, "expected adjacency along one box side"

        rows = {r for r, _ in adjacent_path}
        cols = {c for _, c in adjacent_path}
        if len(rows) == 1:
            orientation = "h"
            ordered_adj = tuple(sorted(adjacent_path, key=lambda rc: rc[1]))
            cvals = [c for _, c in ordered_adj]
            assert cvals == list(range(cvals[0], cvals[0] + 3)), "expected three consecutive adjacent path cells"
            assert side_kind in {"top", "bottom"}
        elif len(cols) == 1:
            orientation = "v"
            ordered_adj = tuple(sorted(adjacent_path, key=lambda rc: rc[0]))
            rvals = [r for r, _ in ordered_adj]
            assert rvals == list(range(rvals[0], rvals[0] + 3)), "expected three consecutive adjacent path cells"
            assert side_kind in {"left", "right"}
        else:
            raise AssertionError("expected straight path adjacency")

        middle_path = ordered_adj[1]
        assert _local_orientation(ordered_path, index_map, middle_path) == orientation
        contact_index = min(index_map[cell] for cell in ordered_adj)
        boxes.append(
            BoxSpec(
                color=int(color),
                bbox=bbox,
                adjacent_path=ordered_adj,
                orientation=orientation,
                contact_index=int(contact_index),
                middle_path=middle_path,
            )
        )

    assert boxes, "expected at least one box"
    middle_cells = [box.middle_path for box in boxes]
    assert len(set(middle_cells)) == len(middle_cells), "expected distinct contact runs"
    return boxes


def _transformed_patch(color: int, orientation: str) -> np.ndarray:
    patch = np.full((3, 3), BG, dtype=int)
    patch[1, 1] = int(color)
    if orientation == "h":
        patch[0, 1] = int(color)
        patch[1, 0] = int(color)
        patch[1, 2] = int(color)
        patch[2, 1] = int(color)
        return patch
    if orientation == "v":
        patch[0, 0] = int(color)
        patch[0, 2] = int(color)
        patch[2, 0] = int(color)
        patch[2, 2] = int(color)
        return patch
    raise ValueError(f"unknown orientation: {orientation}")


def _apply_with_forced_orientation(grid: np.ndarray, orientation: str) -> np.ndarray:
    ordered_path, nbrs = _ordered_path(grid)
    boxes = sorted(_extract_boxes(grid, ordered_path, nbrs), key=lambda box: box.contact_index)
    out = np.array(grid, dtype=int, copy=True)
    for box in boxes:
        r0, c0, r1, c1 = box.bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = _transformed_patch(box.color, orientation)
    out[np.asarray(grid) == PATH] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Traverse a snake path and use its local direction to choose how nearby hollow boxes are rewritten.

    Input structure:
    - A black background contains one grey 4-connected snake path with exactly two endpoints.
    - Several 3x3 hollow boxes of non-black, non-grey colors are placed near the snake.
    - Each box shares exactly one full side with a straight 3-pixel run of the snake, and that run has a unique middle pixel.

    Rule:
    - Pick the endpoint with smaller row index, breaking ties by smaller column index, and traverse the snake one pixel at a time.
    - Whenever traversal reaches the straight run adjacent to a box, inspect the middle path pixel of that run.
    - If that middle pixel lies on a horizontal path segment, replace the box by a same-color plus sign inside its 3x3 bbox.
    - If that middle pixel lies on a vertical path segment, replace the box by a same-color diamond mask inside its 3x3 bbox.
    - If transformed boxes would overlap, later boxes in traversal order overwrite earlier ones. Erase the grey snake from the final output.

    Key insight:
    - The snake is a two-layer controller: traversal order supplies processing order, while the local horizontal or vertical geometry next to each box selects which 3x3 mask to draw.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    ordered_path, nbrs = _ordered_path(g)
    boxes = sorted(_extract_boxes(g, ordered_path, nbrs), key=lambda box: box.contact_index)

    out = g.copy()
    for box in boxes:
        r0, c0, r1, c1 = box.bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = _transformed_patch(box.color, box.orientation)

    out[g == PATH] = BG
    return out


def _build_snake_path(shape: tuple[int, int], row_count: int, seg_len: int, gap: int, start_left: bool) -> list[Coord]:
    h, w = map(int, shape)
    path_h = 1 + (row_count - 1) * gap
    assert h >= path_h + 6 and w >= seg_len + 6
    r0 = int(GEN_RNG.integers(3, h - path_h - 2))
    c0 = int(GEN_RNG.integers(3, w - seg_len - 2))
    c1 = c0 + seg_len - 1
    rows = [r0 + i * gap for i in range(row_count)]

    ordered: list[Coord] = []
    for i, row in enumerate(rows):
        left_to_right = start_left if i % 2 == 0 else not start_left
        cols = list(range(c0, c1 + 1)) if left_to_right else list(range(c1, c0 - 1, -1))
        if not ordered:
            ordered.extend((row, c) for c in cols)
            continue
        prev_row = rows[i - 1]
        connector_col = ordered[-1][1]
        for rr in range(prev_row + 1, row + 1):
            ordered.append((rr, connector_col))
        ordered.extend((row, c) for c in cols[1:])

    assert len(set(ordered)) == len(ordered), "snake path must be simple"
    return [(int(r), int(c)) for r, c in ordered]


def _adjacent_side_candidates(shape: tuple[int, int], ordered_path: list[Coord]) -> list[dict]:
    h, w = map(int, shape)
    path_set = set(ordered_path)
    nbrs = _path_neighbors(path_set)
    index_map = {cell: i for i, cell in enumerate(ordered_path)}
    candidates: list[dict] = []
    seen_bboxes: set[BBox] = set()

    ring_cells = {(rr, cc) for rr in range(3) for cc in range(3) if (rr, cc) != (1, 1)}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            bbox = (r0, c0, r0 + 2, c0 + 2)
            box_cells = {(r0 + rr, c0 + cc) for rr, cc in ring_cells}
            if box_cells & path_set:
                continue
            adjacent_path = {
                path_cell
                for box_cell in box_cells
                for path_cell in neighbors4(*box_cell)
                if path_cell in path_set
            }
            if len(adjacent_path) != 3:
                continue
            adjacent_box_cells = {
                box_cell
                for box_cell in box_cells
                for path_cell in neighbors4(*box_cell)
                if path_cell in path_set
            }
            if len(adjacent_box_cells) != 3:
                continue
            side_kind = _box_side_kind(adjacent_box_cells, bbox)
            if side_kind is None:
                continue
            rows = {r for r, _ in adjacent_path}
            cols = {c for _, c in adjacent_path}
            if len(rows) == 1:
                orientation = "h"
                ordered_adj = tuple(sorted(adjacent_path, key=lambda rc: rc[1]))
                cvals = [c for _, c in ordered_adj]
                if cvals != list(range(cvals[0], cvals[0] + 3)):
                    continue
                if side_kind not in {"top", "bottom"}:
                    continue
            elif len(cols) == 1:
                orientation = "v"
                ordered_adj = tuple(sorted(adjacent_path, key=lambda rc: rc[0]))
                rvals = [r for r, _ in ordered_adj]
                if rvals != list(range(rvals[0], rvals[0] + 3)):
                    continue
                if side_kind not in {"left", "right"}:
                    continue
            else:
                continue

            middle = ordered_adj[1]
            idx = index_map[middle]
            if not (0 < idx < len(ordered_path) - 1):
                continue
            try:
                if _local_orientation(ordered_path, index_map, middle) != orientation:
                    continue
            except AssertionError:
                continue
            if bbox in seen_bboxes:
                continue
            seen_bboxes.add(bbox)
            candidates.append(
                {
                    "bbox": bbox,
                    "adjacent_path": ordered_adj,
                    "orientation": orientation,
                    "contact_index": int(min(index_map[cell] for cell in ordered_adj)),
                    "middle_path": middle,
                }
            )
    return candidates


def _bboxes_separated(a: BBox, b: BBox, pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _choose_boxes(candidates: list[dict]) -> list[dict] | None:
    horiz = [cand for cand in candidates if cand["orientation"] == "h"]
    vert = [cand for cand in candidates if cand["orientation"] == "v"]
    if not horiz or not vert:
        return None

    target_count = int(GEN_RNG.integers(3, 6))
    target_count = min(target_count, len(candidates))
    orders = [
        [horiz[int(GEN_RNG.integers(len(horiz)))], vert[int(GEN_RNG.integers(len(vert)))]]
        for _ in range(18)
    ]
    all_candidates = candidates[:]
    for _ in range(18):
        GEN_RNG.shuffle(all_candidates)
        orders.append(all_candidates[:2])

    for seed_selection in orders:
        selected: list[dict] = []
        used_middle: set[Coord] = set()
        for cand in seed_selection:
            if cand["middle_path"] in used_middle:
                continue
            if all(_bboxes_separated(cand["bbox"], prev["bbox"], pad=1) for prev in selected):
                selected.append(cand)
                used_middle.add(cand["middle_path"])

        pool = candidates[:]
        GEN_RNG.shuffle(pool)
        for cand in pool:
            if len(selected) >= target_count:
                break
            if cand["middle_path"] in used_middle:
                continue
            if all(_bboxes_separated(cand["bbox"], prev["bbox"], pad=1) for prev in selected):
                selected.append(cand)
                used_middle.add(cand["middle_path"])

        if len(selected) < 3:
            continue
        if {cand["orientation"] for cand in selected} != {"h", "v"}:
            continue
        return selected
    return None


def _render_input(shape: tuple[int, int], ordered_path: list[Coord], boxes: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for r, c in ordered_path:
        grid[r, c] = PATH
    colors = list(GEN_RNG.choice(np.asarray(BOX_COLORS, dtype=int), size=len(boxes), replace=False))
    for color, spec in zip(colors, boxes):
        r0, c0, r1, c1 = spec["bbox"]
        patch = np.full((3, 3), BG, dtype=int)
        for rr, cc in itertools.product(range(3), range(3)):
            if (rr, cc) != (1, 1):
                patch[rr, cc] = int(color)
        grid[r0 : r1 + 1, c0 : c1 + 1] = patch
    return grid


def _count_orientations(boxes: list[BoxSpec]) -> tuple[int, int]:
    h_count = sum(box.orientation == "h" for box in boxes)
    v_count = sum(box.orientation == "v" for box in boxes)
    return int(h_count), int(v_count)


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        ordered_path, nbrs = _ordered_path(grid)
        boxes = _extract_boxes(grid, ordered_path, nbrs)
        output = main(grid)
    except Exception:
        return False

    if not (20 <= len(ordered_path) <= 30):
        return False
    if len(boxes) < 3 or len(boxes) > 5:
        return False

    h_count, v_count = _count_orientations(boxes)
    if h_count == 0 or v_count == 0:
        return False

    if np.any(output == PATH):
        return False

    forced_h = _apply_with_forced_orientation(grid, "h")
    forced_v = _apply_with_forced_orientation(grid, "v")
    if np.array_equal(output, forced_h) or np.array_equal(output, forced_v):
        return False

    if np.array_equal(grid, output):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    path_options = [(4, 4, 2), (4, 5, 2), (4, 6, 2), (4, 4, 3), (4, 5, 3), (5, 4, 2)]
    row_count, seg_len, gap = path_options[int(GEN_RNG.integers(len(path_options)))]
    min_h = 1 + (row_count - 1) * gap + 6
    min_w = seg_len + 6
    shape_options = [(h, w) for h in range(max(14, min_h), 19) for w in range(max(14, min_w), 19)]
    if not shape_options:
        return None
    h, w = shape_options[int(GEN_RNG.integers(len(shape_options)))]
    start_left = bool(GEN_RNG.integers(0, 2))

    try:
        ordered_path = _build_snake_path((h, w), row_count, seg_len, gap, start_left)
    except Exception:
        return None
    candidates = _adjacent_side_candidates((h, w), ordered_path)
    chosen = _choose_boxes(candidates)
    if chosen is None:
        return None
    grid = _render_input((h, w), ordered_path, chosen)
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid snake-box input")
